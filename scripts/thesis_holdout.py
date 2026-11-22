"""Freeze train-only pair models, draw time holdouts, and rank from public inputs."""

import argparse
import hashlib
import json
import platform
import random
from collections import Counter
from itertools import combinations

import numba
import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from build_dept3_pilot import (DAY, ROOT, RAW_SHA, SOURCE as DEPT, SPLITS, build_split,
                               make_windows, mask, read_rows, save_rows, simulate)
from collegemsg_candidates import CollegeHistory
from collegemsg_data import CFG, OUT as COLLEGE, SOURCE
from collegemsg_forward import simulate_sparse
from conditional_candidates import OUT as CONDITIONAL, direct_scores
from dept3_forward_model import kernels, prefix_graph
from probe_cascade_yield import load_events
from run_dept3_candidates import Features, rank_pool, write_json
from run_dept3_forward import design_from_kernel
from score_dept3_candidates import roles

OUT = ROOT / "runs/thesis_holdout"
SOURCES = {"Dept3":DEPT, "CollegeMsg":SOURCE}
EXTERNAL = ROOT/"data/raw/email-Eu-core-temporal-Dept1.txt.gz"
TRANSFER = dict(source="Dept1", model="Dept3", span=[.15,.8], width_days=14,
    selection="8 indices evenly spaced over all complete windows, rounded down",seed=84100,
    per_window={"dual":25,"ordinary":12},attempt_cap=300000,
    sha256="d914eb16d63cf48db86bb197fd939231092013937d04e80c8e0298bd20248fbf")
PROTOCOL = dict(test=[0.8,1.0], fraction=[25,50,75], main_fraction=50,
    per_window={"dual":40,"ordinary":20}, attempt_cap=300000, seed={"Dept3":82100,"CollegeMsg":83100},
    width_days={"Dept3":14,"CollegeMsg":7}, primary="prefix128 vs direct at 32 nodes, dual cohort, Hit@32",
    fusion_contrast="three-feature listwise fusion vs direct and two-feature ablation at 64 nodes",
    fit="original frozen node and forward folds, conditional training trajectories, ridge 0.02, event-standardized inputs",
    sensitivity="same 50%-selected dual trajectories eligible with two visible lineages at all three masks",
    inference="historical graph and visible batches only; no test fitting; time-window cluster bootstrap descriptive")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def models(network):
    if network=="CollegeMsg":
        return load(COLLEGE/"candidate_models.json")["node"], load(COLLEGE/"forward_models.json"), "12"
    return (load(ROOT/"runs/dept3_candidates/models.json")["models"],
            load(ROOT/"runs/dept3_forward/models.json")["folds"], "full2000")


class Window:
    def __init__(self,g,raw,forward,node):
        self.features = Features(g,raw,CollegeHistory)
        self.hist = prefix_graph(g,raw)
        self.base,self.edges = kernels(self.hist, forward)
        self.csr = csr_matrix(self.edges)
        self.forward,self.node = forward,node

    def design(self,o):
        vals, ordered, _, wide = rank_pool(self.node,self.features.event(o),True)
        nodes = ordered[:64]
        ix = self.hist[1]
        i,j = np.triu_indices(len(nodes),1)
        idx = np.array([ix[v] for v in nodes])
        pairs = np.column_stack((idx[i],idx[j]))
        visible = np.array([ix[v] for batch in o["observed"] for v in batch])
        ends = np.cumsum([len(b) for b in o["observed"]])
        d = direct_scores(self.base,self.csr.indptr,self.csr.indices,self.csr.data,pairs,visible,ends)
        scores = np.array([vals[v] for v in nodes])
        x = np.column_stack((scores[i],scores[j],d))
        sigma = x.std(axis=0)
        x = (x-x.mean(axis=0))/np.where(sigma>1e-10,sigma,1.)
        # The first 32 preserve the previous additive tie order.
        positions = {tuple(sorted(p)):k for k,p in enumerate(wide)}
        tie = np.array([positions.get(tuple(sorted((nodes[a],nodes[b]))),len(wide)+n)
                        for n,(a,b) in enumerate(zip(i,j))])
        return nodes,i,j,pairs,x,d,scores[i]+scores[j],tie


def fit(rows,dim):
    feats = [x[:,:dim] for x,_ in rows]
    def objective(w):
        loss,grad = 0.,np.zeros(dim)
        for x,(_,y) in zip(feats,rows):
            s = x@w
            z = logsumexp(s)
            loss += z-s[y]
            grad += np.exp(s-z)@x-x[y]
        return loss/len(rows)+.01*np.dot(w,w), grad/len(rows)+.02*w
    opt = minimize(objective,np.zeros(dim),jac=True,method="L-BFGS-B")
    assert opt.success and np.isfinite(opt.x).all()
    return opt.x.tolist()


def freeze():
    OUT.mkdir(exist_ok=True)
    write_json(OUT/"protocol.json",PROTOCOL)
    frozen={}
    for name,source in SOURCES.items():
        raw = load_events(source)
        assert sha(source)==(RAW_SHA if name=="Dept3" else CFG["raw_sha256"])
        node,forward,last = models(name)
        if name=="Dept3":
            graphs,obs,truth,_ = build_split(raw,"train",*SPLITS["train"],target=2000)
            cuts = [(10,15),(15,20),(20,25)]
        else:
            graphs=read_rows(COLLEGE/"public/graphs.jsonl")
            obs=read_rows(CONDITIONAL/"conditional_observations.jsonl")
            truth=read_rows(CONDITIONAL/"conditional_truth.jsonl")
            cuts=[(6,8),(8,10),(10,12)]
        labels={t["id"]:set(t["sources"]) for t in truth if len(roles(t)[1])==2}
        eligible,counts=[],{}
        for left,right in cuts:
            tag = f"rolling_{left}" if name=="Dept3" else str(left)
            for g in graphs:
                week=int(g["window"].split("-")[1])
                if not g["window"].startswith("train") or not left<week<=right:
                    continue
                w=Window(g,raw,forward[tag],node[tag])
                n=inside=0
                for o in obs:
                    if o["window"]!=g["window"] or o["id"] not in labels:
                        continue
                    nodes,i,j,_,x,*_=w.design(o)
                    n+=1
                    a=labels[o["id"]]
                    if a<=set(nodes):
                        a,b=sorted(nodes.index(v) for v in a)
                        target=int(np.flatnonzero((i==a)&(j==b))[0])
                        eligible.append((x,target))
                        inside+=1
                counts[g["window"]]=[n,inside]
        frozen[name]=dict(node=node[last],forward=forward[last],node_only=fit(eligible,2),
                          fusion=fit(eligible,3),training=counts,sha256=sha(source))
        print(name,"training [conditional, in pool]",counts,flush=True)
    write_json(OUT/"frozen.json",frozen)
    write_json(OUT/"freeze_manifest.json",dict(frozen=sha(OUT/"frozen.json"),protocol=sha(OUT/"protocol.json"),
               python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
               sklearn=sklearn.__version__,numba=numba.__version__))


def frozen_models():
    mark=load(OUT/"freeze_manifest.json")
    assert load(OUT/"protocol.json")==PROTOCOL and mark["protocol"]==sha(OUT/"protocol.json")
    assert mark["frozen"]==sha(OUT/"frozen.json")
    return load(OUT/"frozen.json")


def masked(tr,f):
    groups=[[v for v,_ in b["joins"]] for b in tr]
    limit=sum(map(len,groups))*f//100
    cut=count=0
    for batch in groups:
        if count+len(batch)>limit:
            break
        cut+=1
        count+=len(batch)
    if cut and sum(map(len,groups[cut:]))>=3:
        return cut,groups[cut:]


def datasets(external):
    return {"Dept1":EXTERNAL} if external else SOURCES


def windows(name,raw):
    if name=="Dept1":
        graphs,streams=make_windows(raw,"external",*TRANSFER["span"],width=TRANSFER["width_days"]*DAY)
        return [graphs[k] for k in np.linspace(0,len(graphs)-1,8).astype(int)],streams
    return make_windows(raw,"test",*PROTOCOL["test"],width=PROTOCOL["width_days"][name]*DAY)


def generate(external=False):
    frozen=frozen_models()
    if external:
        write_json(OUT/"transfer_protocol.json",dict(TRANSFER, frozen=sha(OUT/"frozen.json")))
    for name,source in datasets(external).items():
        assert sha(source)==(TRANSFER["sha256"] if external else frozen[name]["sha256"])
        quotas=(TRANSFER if external else PROTOCOL)["per_window"]
        folder=OUT/name
        folder.mkdir(exist_ok=True)
        raw=load_events(source)
        graphs,streams=windows(name,raw)
        save_rows(folder/"graphs.jsonl",graphs)
        public,truth,flow=[],[],{}
        for week,g in enumerate(graphs,1):
            key=g["window"]
            rng=random.Random((TRANSFER["seed"] if external else PROTOCOL["seed"][name])+week)
            stream=streams[key]
            senders={u for _,contacts in stream for u,_ in contacts}
            c=Counter(ordinary=0,dual=0,attempts=0)
            for attempt in range(1,PROTOCOL["attempt_cap"]+1):
                c["attempts"]+=1
                s=sorted(rng.sample(g["nodes"],2))
                seed=rng.getrandbits(63)
                need=c["ordinary"]<quotas["ordinary"]
                if senders.isdisjoint(s) or (not need and not senders.issuperset(s)):
                    continue
                tr=simulate(stream,s,seed)
                m=mask(tr)
                if not m:
                    continue
                cut,observed=m
                label=dict(sources=s,trajectory=tr,cut_batches=cut)
                dual=len(roles(label)[1])==2 and c["dual"]<quotas["dual"]
                if not (dual or need):
                    continue
                sid=f"{name}-{week:02}-{len(truth)+1:05}"
                c["ordinary"]+=int(need)
                c["dual"]+=int(dual)
                truth.append(dict(label,id=sid,window=key,seed=seed,ordinary=need,dual=dual))
                for f in PROTOCOL["fraction"]:
                    view=masked(tr,f)
                    if view and (f==50 or dual):
                        public.append(dict(id=sid,window=key,mask=f,observed=view[1],k=2))
                if all(c[k]==n for k,n in quotas.items()):
                    break
            flow[key]=dict(c)
            print(name,key,flow[key],flush=True)
        save_rows(folder/"public.jsonl",public)
        save_rows(folder/"truth.jsonl",truth)
        write_json(folder/"generation.json",flow)


def predict(external=False):
    frozen=frozen_models()
    if external:
        assert load(OUT/"transfer_protocol.json")==dict(TRANSFER,frozen=sha(OUT/"frozen.json"))
    numba.set_num_threads(min(8,numba.get_num_threads()))
    for name,source in datasets(external).items():
        folder=OUT/name
        raw=load_events(source)
        f=frozen["Dept3" if external else name]
        inputs=read_rows(folder/"public.jsonl")
        for g in read_rows(folder/"graphs.jsonl"):
            out=[]
            w=Window(g,raw,f["forward"],f["node"])
            for o in inputs:
                if o["window"]!=g["window"]:
                    continue
                nodes,i,j,pairs,x,d,add,tie=w.design(o)
                narrow=np.flatnonzero(j<32)
                rank={}
                def save(method,score,indices):
                    order=np.lexsort((tie[indices],-add[indices],-score))
                    rank[method]=indices[order].tolist()
                allpairs=np.arange(len(i))
                for method,score in (("add64",add),("direct64",d),("node64",x[:,:2]@f["node_only"]),
                                     ("fusion64",x@f["fusion"])):
                    save(method,score,allpairs)
                save("direct32",d[narrow],narrow)
                # Directed maximum distance from the nearest candidate to each observed member.
                distances=w.features.history.distance
                ix=w.hist[1]
                dvis=distances[np.ix_([ix[v] for v in nodes],[ix[v] for b in o["observed"] for v in b])]
                radius=np.minimum(dvis[i],dvis[j]).max(axis=1)
                save("radius64",-radius,allpairs)
                if o["mask"]==50:
                    _,_,*args=design_from_kernel(o,ix,w.base,w.edges,f["forward"],128)
                    s=simulate_sparse(w.base,w.csr.indptr,w.csr.indices,w.csr.data,pairs[narrow],*args)
                    assert np.isfinite(s).all() and np.allclose(s[:,0],d[narrow],atol=1e-10,rtol=0)
                    save("prefix32",s[:,2],narrow)
                out.append(dict(id=o["id"],window=o["window"],mask=o["mask"],nodes=nodes,orders=rank))
            save_rows(folder/(g["window"]+"_rankings.jsonl"),out)
            print(name,g["window"],"views scored",len(out),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("phase",choices=["freeze","generate","predict","external_generate","external_predict"])
    args=ap.parse_args()
    with threadpool_limits(limits=1,user_api="blas"):
        {"freeze":freeze,"generate":generate,"predict":predict,"external_generate":lambda:generate(True),
         "external_predict":lambda:predict(True)}[args.phase]()
