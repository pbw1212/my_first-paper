"""Compare complementary pair evidence and historical live-edge reach on a fixed proposal."""

import argparse
from itertools import groupby
import json
from pathlib import Path
import random

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from build_dept3_pilot import DAY, ROOT, make_windows, mask, simulate, read_rows, save_rows
from score_dept3_candidates import roles
from thesis_holdout import (DEPT, EXTERNAL, SOURCE, OUT as OLD, Window, fit, frozen_models, load,
                            load_events, models, sha, write_json)

OUT=ROOT/"runs/temporal_pair_evidence"
NEW=ROOT/"data/raw/email-Eu-core-temporal-Dept2.txt.gz"
PLAN=dict(primary="event-level reciprocal rank of the exact pair (zero outside the fixed top 64)",
          secondary=["exact top1","exact top5","exact top32","both in pool"],
          comparison=["old fusion","direct","radius","additive","static pair features","time-shuffled replay"],
          model="ordered replay and static pair features; fixed ridge 0.02, no target labels",
          external="Dept2; 12 equally spaced full 14-day windows, 15%-80% of elapsed span, include empty windows",
          sampling="60 dual and 20 ordinary events per external window, cap 60000 uniform source draws",
          history="six preceding 14-day windows; 24 live-edge draws per window; decay 2 windows half-life",
          development="train on new conditional Dept3 simulations in weeks 11-25 using past folds; diagnose on Dept1",
          seed=93171)


def live_reach(groups,n):
    state=[1<<i for i in range(n)]
    for group in groups:
        updates={}
        for u,v in group:
            updates[v]=updates.get(v,0)|state[u]
        for v,more in updates.items():
            state[v]|=more
    width=(n+7)//8
    buf=b"".join(z.to_bytes(width,"little") for z in state)
    return np.unpackbits(np.frombuffer(buf,dtype=np.uint8).reshape(n,width),axis=1,bitorder="little")[:,:n].T


def history_kernels(raw,g):
    n=len(g["nodes"])
    ix={v:i for i,v in enumerate(g["nodes"])}
    width=14*DAY
    paired=np.zeros((2,n,n))
    total=0.
    for k in range(6):
        end=g["start"]-k*width
        if end-width<raw[0][0]:
            continue
        events=[(t,ix[u],ix[v]) for t,u,v in raw if end-width<=t<end]
        groups=[[(u,v) for _,u,v in batch] for _,batch in groupby(events,key=lambda e:e[0])]
        rng=random.Random(PLAN["seed"]+g["start"]+k)
        weight=2**(-k/2)
        for _ in range(24):
            live=[[(u,v) for u,v in batch if rng.random()<.3] for batch in groups]
            live=[b for b in live if b]
            paired[0]+=weight*live_reach(live,n)
            rng.shuffle(live)
            paired[1]+=weight*live_reach(live,n)
            total+=weight
    paired/=max(total,1.)
    for matrix in paired:
        np.fill_diagonal(matrix,0)
    return paired


def scale(x):
    sigma=x.std(axis=0)
    return (x-x.mean(axis=0))/np.where(sigma>1e-9,sigma,1.)


def evidence(p,i,j,early):
    # p[a,v] quantifies support from possible source a for visible member v.
    a,b=p[i],p[j]
    best=np.maximum(a,b)
    extra_a=np.maximum(a-b,0).mean(axis=1)
    extra_b=np.maximum(b-a,0).mean(axis=1)
    return [best.mean(axis=1),best.min(axis=1),best[:,early].mean(axis=1),
            np.minimum(extra_a,extra_b),np.maximum(extra_a,extra_b)]


class PairEvidence:
    def __init__(self,raw,g,forward,node):
        self.window=Window(g,raw,forward,node)
        self.reach=history_kernels(raw,g)
        n=len(g["nodes"])
        ix=self.window.hist[1]
        mat=np.zeros((n,n))
        for u,v,w in g["edges"]:
            mat[ix[u],ix[v]]=np.log1p(w)
        walk=csr_matrix(mat / np.maximum(mat.sum(axis=1,keepdims=True),1e-12))
        two=walk@walk
        self.diffusion=(walk*.5+two*.3+(two@walk)*.2).toarray()

    def design(self,o):
        w=self.window
        nodes,i,j,_,old,d,add,tie=w.design(o)
        ix=w.hist[1]
        rows=np.array([ix[v] for v in nodes])
        cols=np.array([ix[v] for group in o["observed"] for v in group])
        early=np.arange(len(cols))<len(o["observed"][0])
        sub=np.ix_(rows,cols)
        dist=w.features.history.distance[sub]
        radius=-np.minimum(dist[i],dist[j]).max(axis=1)
        affinity=1/np.maximum(1,dist.astype(float))
        affinity[dist>=100]=0
        static=[*old.T,radius,*evidence(affinity,i,j,early)]
        walk=np.log1p(self.diffusion[sub]/np.maximum(self.diffusion[:,cols].mean(axis=0),1e-6))
        static.extend(evidence(walk,i,j,early))
        x=scale(np.column_stack(static))
        designs={"static":x}
        for tag,r in zip(("ordered","shuffled"),self.reach):
            prior=np.maximum(r[:,cols].mean(axis=0),.01)
            signal=np.log1p(r[sub]/prior)
            mass=np.log1p(r[rows].sum(axis=1))
            extra=evidence(signal,i,j,early)+[np.minimum(mass[i],mass[j]),np.maximum(mass[i],mass[j])]
            designs[tag]=np.column_stack((x,scale(np.column_stack(extra))))
        return nodes,i,j,designs,dict(direct=d,radius=radius,additive=add),tie,old


def draw(g,stream,seed,dual_target,ordinary_target):
    rng=random.Random(seed)
    obs,truth=[],[]
    senders={u for _,contacts in stream for u,_ in contacts}
    ordinary=dual=0
    attempts=0
    for attempts in range(1,60001):
        s=sorted(rng.sample(g["nodes"],2))
        trial=rng.getrandbits(63)
        need=ordinary<ordinary_target
        if senders.isdisjoint(s) or (not need and not senders.issuperset(s)):
            continue
        tr=simulate(stream,s,trial)
        m=mask(tr)
        if not m:
            continue
        q,v=m
        lab=dict(sources=s,trajectory=tr,cut_batches=q)
        both=dual<dual_target and len(roles(lab)[1])==2
        if not (both or need):
            continue
        sid=g["window"]+"-"+str(len(obs))
        obs.append(dict(id=sid,window=g["window"],observed=v,k=2))
        truth.append(dict(lab,id=sid,seed=trial,window=g["window"],dual=both,ordinary=need))
        dual+=int(both)
        ordinary+=int(need)
        if dual==dual_target and ordinary==ordinary_target:
            break
    return obs,truth,dict(dual=dual,ordinary=ordinary,attempts=attempts)


def train():
    OUT.mkdir(exist_ok=True)
    write_json(OUT/"protocol.json",PLAN)
    raw=load_events(DEPT)
    gs,streams=make_windows(raw,"train",.15,.6)
    nm,fm,_=models("Dept3")
    rows={k:[] for k in ("ordered","static","shuffled")}
    counts={}
    for week,g in enumerate(gs,1):
        if week<=10:
            continue
        fold="rolling_"+str(((week-1)//5)*5)
        obs,labs,c=draw(g,streams[g["window"]],90600+week,55,0)
        c["inside"]=0
        if obs:
            w=PairEvidence(raw,g,fm[fold],nm[fold])
        for o,lab in zip(obs,labs):
            nodes,i,j,designs,*_=w.design(o)
            if not set(lab["sources"])<=set(nodes):
                continue
            a,b=sorted(nodes.index(v) for v in lab["sources"])
            y=int(np.flatnonzero((i==a)&(j==b))[0])
            for tag,x in designs.items():
                rows[tag].append((x,y))
            c["inside"]+=1
        counts[g["window"]]=c
        print(g["window"],c,flush=True)
    f={k:fit(r,r[0][0].shape[1]) for k,r in rows.items()}
    write_json(OUT/"model.json",dict(weights=f,training=counts,protocol=sha(OUT/"protocol.json"),
               predecessor=sha(OLD/"frozen.json")))


def cohort(name,raw):
    if name in ("Dept1","CollegeMsg"):
        folder=OLD/name
        obs=[o for o in read_rows(folder/"public.jsonl") if o["mask"]==50]
        return read_rows(folder/"graphs.jsonl"),obs,read_rows(folder/"truth.jsonl")
    allgraphs,streams=make_windows(raw,"external",.15,.8)
    gs=[allgraphs[p] for p in np.linspace(0,len(allgraphs)-1,12).astype(int)]
    inputs,labs,flow=[],[],{}
    for week,g in enumerate(gs,1):
        o,t,c=draw(g,streams[g["window"]],95100+week,60,20)
        inputs.extend(o)
        labs.extend(t)
        flow[g["window"]]=c
    save_rows(OUT/"external_public.jsonl",inputs)
    save_rows(OUT/"external_truth.jsonl",labs)
    write_json(OUT/"external_sampling.json",flow)
    return gs,inputs,labs


def metrics(rows,tag):
    ranks=np.array([r[tag] if r[tag] is not None else np.inf for r in rows])
    return dict(mrr=float(np.mean(1/ranks)),top1=int(sum(ranks==1)),top5=int(sum(ranks<=5)),
                hit32=int(sum(ranks<=32)))


def evaluate(name):
    learned=load(OUT/"model.json")
    assert load(OUT/"protocol.json")==PLAN and learned["protocol"]==sha(OUT/"protocol.json")
    assert learned["predecessor"]==sha(OLD/"frozen.json")
    f=frozen_models()["CollegeMsg" if name=="CollegeMsg" else "Dept3"]
    source={"Dept1":EXTERNAL,"CollegeMsg":SOURCE,"Dept2":NEW}[name]
    raw=load_events(source)
    gs,inputs,labs=cohort(name,raw)
    predictions=[]
    tags=["direct","radius","additive","fusion",*learned["weights"]]
    for g in gs:
        items=[o for o in inputs if o["window"]==g["window"]]
        if not items:
            continue
        w=PairEvidence(raw,g,f["forward"],f["node"])
        for o in items:
            nodes,i,j,designs,scores,tie,old=w.design(o)
            scores["fusion"]=old@f["fusion"]
            for tag,weights in learned["weights"].items():
                scores[tag]=designs[tag]@weights
            orders={k:np.lexsort((tie,-scores["additive"],-v)).tolist() for k,v in scores.items()}
            predictions.append(dict(id=o["id"],window=o["window"],nodes=nodes,orders=orders))
    save_rows(OUT/(name+"_predictions.jsonl"),predictions)
    truth={r["id"]:r for r in labs}
    rows=[]
    for p in predictions:
        lab=truth[p["id"]]
        nodes=p["nodes"]
        i,j=np.triu_indices(len(nodes),1)
        y=None
        if set(lab["sources"])<=set(nodes):
            a,b=sorted(nodes.index(v) for v in lab["sources"])
            y=int(np.flatnonzero((i==a)&(j==b))[0])
        row=dict(id=lab["id"],window=lab["window"],dual=lab["dual"],ordinary=lab["ordinary"],inside=y is not None)
        for tag,order in p["orders"].items():
            row[tag]=order.index(y)+1 if y is not None else None
        rows.append(row)
    assert len(rows)==len(labs)==len({r["id"] for r in rows})
    if name in ("Dept1","CollegeMsg"):
        before={r["id"]:r for r in read_rows(OLD/name/"event_results.jsonl") if r["mask"]==50}
        assert all(r["radius"]==before[r["id"]]["radius64"] and r["fusion"]==before[r["id"]]["fusion64"] for r in rows)
    save_rows(OUT/(name+"_ranks.jsonl"),rows)
    report={}
    for subgroup in ("dual","ordinary"):
        rr=[r for r in rows if r[subgroup]]
        report[subgroup]=dict(n=len(rr),inside=sum(r["inside"] for r in rr),
                              scores={k:metrics(rr,k) for k in tags})
    write_json(OUT/(name+"_summary.json"),dict(report,model=sha(OUT/"model.json"),raw=sha(source)))
    print(name,report,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("phase",choices=("train","development","college","external"))
    phase=ap.parse_args().phase
    with threadpool_limits(limits=1):
        if phase=="train":
            train()
        else:
            evaluate({"development":"Dept1","college":"CollegeMsg","external":"Dept2"}[phase])
