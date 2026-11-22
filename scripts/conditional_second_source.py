"""Development-only comparison: use unexplained observations to propose a partner."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from temporal_pair_evidence import (DEPT, EXTERNAL, SOURCE, OLD, OUT, PairEvidence, cohort, draw, evidence, fit,
    frozen_models, load_events, make_windows, metrics, models, read_rows, save_rows, scale, write_json)
from thesis_holdout import CONDITIONAL, Window
from run_dept3_candidates import rank_pool
from conditional_candidates import direct_scores


class PartnerWindow:
    def __init__(self,g,raw,node,forward):
        self.base=Window(g,raw,forward,node)
        old=self.base.hist[3]
        recent=self.base.hist[4]
        p=csr_matrix(.45*old+.55*recent)
        self.walk=(p*.5+(p@p)*.3+(p@p@p)*.2).toarray()
        self.mean=np.maximum(self.walk.mean(axis=0),1e-5)

    def event(self,o):
        w=self.base
        vals,order,*_=rank_pool(w.node,w.features.event(o),False)
        ix=w.hist[1]
        ids=np.array([ix[v] for v in order])
        cols=np.array([ix[v] for b in o["observed"] for v in b])
        signal=self.walk[np.ix_(ids,cols)]
        signal=signal/(signal+self.mean[cols])
        score=np.array([vals[v] for v in order])
        static=scale(np.column_stack((score,signal.mean(axis=1),signal.max(axis=1),
                                     signal[:,:len(o["observed"][0])].mean(axis=1))))
        return order,score,signal,static


def designs(signal,static,anchor):
    explained=signal[anchor]
    residual=1-explained
    gain=np.maximum(signal-explained,0.)
    features=np.column_stack((signal@residual/max(.001,residual.sum()),gain.mean(axis=1),
                              gain.max(axis=1),np.maximum(explained-signal,0.).mean(axis=1)))
    return {"independent":static,"residual":np.column_stack((static,scale(features)))}


def train(name,raw):
    nm,fm,_=models(name)
    gs,streams=make_windows(raw,"train",.15,.6,width=(14 if name=="Dept3" else 7)*86400)
    if name=="CollegeMsg":
        inputs=read_rows(CONDITIONAL/"conditional_observations.jsonl")
        truth={r["id"]:r for r in read_rows(CONDITIONAL/"conditional_truth.jsonl")}
    rows={"independent":[],"residual":[]}
    for week,g in enumerate(gs,1):
        if name=="Dept3":
            if week<=10:
                continue
            tag="rolling_"+str((week-1)//5*5)
            obs,labs,_=draw(g,streams[g["window"]],90600+week,55,0)
        else:
            if week<=6:
                continue
            tag=str((week-1)//2*2)
            obs=[o for o in inputs if o["window"]==g["window"]]
            labs=[truth[o["id"]] for o in obs]
        if not obs:
            continue
        w=PartnerWindow(g,raw,nm[tag],fm[tag])
        for o,lab in zip(obs,labs):
            order,_,signal,static=w.event(o)
            a,b=[order.index(v) for v in lab["sources"]]
            indices=np.arange(len(order))
            for anchor,y in ((a,b),(b,a)):
                for method,x in designs(signal,static,anchor).items():
                    rows[method].append((x[indices!=anchor],y-int(y>anchor)))
    return {tag:fit(rr,rr[0][0].shape[1]) for tag,rr in rows.items()}


def evaluate(name,weights):
    source=EXTERNAL if name=="Dept1" else SOURCE
    model=frozen_models()["Dept3" if name=="Dept1" else name]
    raw=load_events(source)
    gs,obs,labs=cohort(name,raw)
    truth={r["id"]:r for r in labs}
    import json
    replay_weights=json.loads((OUT/"model.json").read_text())["weights"]["ordered"]
    previous={r["id"]:r for r in read_rows(OUT/(name+"_ranks.jsonl"))}
    records=[]
    for g in gs:
        items=[o for o in obs if o["window"]==g["window"]]
        if not items:
            continue
        w=PartnerWindow(g,raw,model["node"],model["forward"])
        f=w.base
        replay=PairEvidence(raw,g,model["forward"],model["node"])
        for o in items:
            nodes,values,signal,static=w.event(o)
            indices=np.arange(len(nodes))
            size=min(64,len(nodes))
            preserve=min(32,size)
            pools={"original":indices[:size]}
            for method,weights_for_method in weights.items():
                mixture=np.zeros(len(nodes))
                prior=values[:preserve]-logsumexp(values[:preserve])
                for a in range(preserve):
                    score=designs(signal,static,a)[method]@weights_for_method
                    score[a]=-np.inf
                    mixture+=np.exp(prior[a]+score-logsumexp(score))
                partners=indices[preserve:][np.argsort(-mixture[preserve:],kind="stable")[:size-preserve]]
                pools[method]=np.r_[indices[:preserve],partners]
            # Same previously frozen direct+node scoring function on all three proposed pools.
            lab=truth[o["id"]]
            record=dict(id=o["id"],window=g["window"],dual=lab["dual"],ordinary=lab["ordinary"])
            ix=f.hist[1]
            vis=np.array([ix[v] for b in o["observed"] for v in b])
            ends=np.cumsum([len(b) for b in o["observed"]])
            for method,pool in pools.items():
                pool=np.sort(pool)
                people=[nodes[k] for k in pool]
                assert len(set(people))==size and set(nodes[:preserve])<=set(people)
                idp=np.array([ix[v] for v in people])
                i,j=np.triu_indices(size,1)
                pairs=np.column_stack((idp[i],idp[j]))
                d=direct_scores(f.base,f.csr.indptr,f.csr.indices,f.csr.data,pairs,vis,ends)
                x=scale(np.column_stack((values[pool[i]],values[pool[j]],d)))
                order=np.lexsort((np.arange(len(i)),-(values[pool[i]]+values[pool[j]]),-(x@model["fusion"])))
                rank=None
                if set(lab["sources"])<=set(people):
                    a,b=sorted(people.index(v) for v in lab["sources"])
                    y=np.flatnonzero((i==a)&(j==b))[0]
                    rank=int(np.flatnonzero(order==y)[0])+1
                record[method]=rank
                sub=np.ix_(idp,vis)
                dist=f.features.history.distance[sub]
                affinity=np.where(dist>=100,0.,1/np.maximum(1,dist))
                early=np.arange(len(vis))<len(o["observed"][0])
                walk=np.log1p(replay.diffusion[sub]/np.maximum(replay.diffusion[:,vis].mean(axis=0),1e-6))
                radius=-np.minimum(dist[i],dist[j]).max(axis=1)
                static_design=scale(np.column_stack([*x.T,radius,*evidence(affinity,i,j,early),*evidence(walk,i,j,early)]))
                reach=replay.reach[0]
                signal_from_replay=np.log1p(reach[sub]/np.maximum(reach[:,vis].mean(axis=0),.01))
                mass=np.log1p(reach[idp].sum(axis=1))
                extra=scale(np.column_stack([*evidence(signal_from_replay,i,j,early),
                    np.minimum(mass[i],mass[j]),np.maximum(mass[i],mass[j])]))
                rerank=np.lexsort((np.arange(len(i)),-(values[pool[i]]+values[pool[j]]),
                                   -(np.column_stack((static_design,extra))@replay_weights)))
                record[method+"_replay"]=None if rank is None else int(np.flatnonzero(rerank==y)[0])+1
                if method=="original":
                    assert record["original_replay"]==previous[o["id"]]["ordered"]
            records.append(record)
    result={}
    for group in ("dual","ordinary"):
        rr=[r for r in records if r[group]]
        result[group]={k:dict(metrics(rr,k),both=sum(r[k] is not None for r in rr),
                              replay=metrics(rr,k+"_replay"),
                              gained=sum(r[k] is not None and r["original"] is None for r in rr),
                              lost=sum(r[k] is None and r["original"] is not None for r in rr))
                       for k in ("original","independent","residual")}
    save_rows(OUT/(name+"_partner_ranks.jsonl"),records)
    print(name,result,flush=True)
    return result


if __name__=="__main__":
    with threadpool_limits(limits=1):
        weights={k:train(k,load_events(p)) for k,p in (("Dept3",DEPT),("CollegeMsg",SOURCE))}
        write_json(OUT/"partner_weights.json",weights)
        write_json(OUT/"partner_dev_summary.json",{
            name:evaluate(name,weights["Dept3" if name=="Dept1" else name]) for name in ("Dept1","CollegeMsg")})
