"""Compare joint two-source simulation under the observation mask with snapshot coverage."""

import argparse
import hashlib
import zipfile
from collections import defaultdict

import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import ndtr
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from temporal_pair_evidence import (DAY, DEPT, OUT as PREV, ROOT, PairEvidence, draw, fit,
    frozen_models, load, load_events, make_windows, metrics, models, read_rows, save_rows, scale,
    sha, write_json)
from build_dept3_pilot import mask, simulate
from score_dept3_candidates import roles

OUT=ROOT/"runs/masked_pair_benchmark"
RAW=ROOT/"data/raw/ia-radoslaw-email.zip"
PLAN=dict(network="Manufacturing emails; unchanged Dept3 proposal and parameters fitted on Dept3 only",
          span=[.30,1.],width_days=14,dual_target=35,ordinary_target=15,
          primary="dual cohort exact pair MRR, zero outside pool; secondary Top1/5/32, ordinary cohort",
          contrasts=["joint_tail vs ordered","joint_tail vs joint_full","joint_tail vs radius","joint_tail vs exhaustive snapshot"],
          history="30 equiprobable draws cycled over at most six complete preceding 14-day windows",
          BO="20 clusters and 50 EI queries, RBF GP, best queried pair first; Hamming peak stopped at first decline",
          seed=196712,training="same 770 conditional Dept3 events and 758 eligible pair rows as previous study")


def company():
    with zipfile.ZipFile(RAW) as z:
        items=[tuple(map(int,line.split())) for line in z.read("ia-radoslaw-email.edges").splitlines()
               if line.strip() and not line.startswith(b"%")]
    assert len(items)==82927 and all(w==1 for _,_,w,_ in items)
    raw=sorted({(t,u,v) for u,v,w,t in items if u!=v})
    return raw


NEVER=2_000_000_000


@njit
def arrival(n,t,u,v,live):
    arr=np.full((n,n),NEVER,np.int32)
    for a in range(n):
        arr[a,a]=0
    for k in range(len(t)):
        if live[k]:
            for a in range(n):
                if arr[a,u[k]]<t[k] and arr[a,v[k]]>t[k]:
                    arr[a,v[k]]=t[k]
    return arr


def historical_arrivals(raw,g):
    width=14*DAY
    ix={v:i for i,v in enumerate(g["nodes"])}
    slices=[]
    for k in range(6):
        end=g["start"]-width*k
        start=end-width
        if start<raw[0][0]:
            continue
        triples=[(t-start+1,ix[u],ix[v]) for t,u,v in raw if start<=t<end]
        slices.append(np.array(triples,dtype=np.int32).reshape(-1,3).T)
    assert slices
    rng=np.random.default_rng(PLAN["seed"]+g["start"])
    draws=[]
    for rep in range(30):
        t,u,v=slices[rep%len(slices)]
        draws.append(arrival(len(ix),t,u,v,rng.random(len(t))<.3))
    return np.stack(draws)


@njit
def joint_bank(arr,aa,bb):
    reps,n,_=arr.shape
    P=len(aa)
    # Counts of full and censored membership; pair and draw indexed size trajectories.
    counts=np.zeros((2,P,n),np.int16)
    sizes=np.zeros((2,P,reps),np.int16)
    coverage=np.zeros((P,reps,21),np.int16)
    # Arrival at 0 means seed. Positive arrivals fall in bins 1..20.
    buckets=np.minimum(21,(arr.astype(np.int64)+(14*86400//20)-1)//(14*86400//20))
    for p in range(P):
        a,b=aa[p],bb[p]
        for s in range(reps):
            t=np.minimum(arr[s,a],arr[s,b])
            for v in range(n):
                bucket=min(buckets[s,a,v],buckets[s,b,v])
                if bucket<=20:
                    coverage[p,s,bucket]+=1
            for j in range(1,21):
                coverage[p,s,j]+=coverage[p,s,j-1]
            added=np.sort(t[(t>0)&(t<NEVER)])
            size=len(added)
            if size<4:
                continue
            cut=added[size//2]
            hidden=np.searchsorted(added,cut)
            if hidden==0 or size-hidden<3:
                continue
            sizes[0,p,s]=size
            sizes[1,p,s]=size-hidden
            for v in range(n):
                if 0<t[v]<NEVER:
                    counts[0,p,v]+=1
                    if t[v]>=cut:
                        counts[1,p,v]+=1
    return counts,sizes,coverage,buckets.astype(np.uint8)


@njit
def peak_scores(pids,aa,bb,coverage,buckets,obs):
    out=np.zeros(len(pids))
    n=buckets.shape[1]
    for k,p in enumerate(pids):
        a,b=aa[p],bb[p]
        for s in range(buckets.shape[0]):
            overlap=np.zeros(22,np.int16)
            for v in obs:
                overlap[min(buckets[s,a,v],buckets[s,b,v])]+=1
            peak=-1.
            hits=0
            for j in range(21):
                hits+=overlap[j]
                score=1.-(coverage[p,s,j]+len(obs)-2*hits)/n
                if score<peak:
                    break
                peak=score
            out[k]+=peak
    return out/buckets.shape[0]


class Bank:
    def __init__(self,raw,g):
        self.ix={v:i for i,v in enumerate(g["nodes"])}
        n=len(self.ix)
        self.a,self.b=np.triu_indices(n,1)
        self.pair_id=np.zeros((n,n),int)
        self.pair_id[self.a,self.b]=self.pair_id[self.b,self.a]=np.arange(len(self.a))
        counts,self.sizes,self.cover,self.buckets=joint_bank(historical_arrivals(raw,g),self.a,self.b)
        eligible=(self.sizes[0]>0).sum(axis=1)
        prior=(counts.sum(axis=1)+1)/(eligible.sum()+2)
        prob=(counts+6*prior[:,None,:])/(eligible[None,:,None]+6)
        self.absence=np.log1p(-prob).sum(axis=2)
        self.logit=np.log(prob)-np.log1p(-prob)

    def design(self,o,nodes,i,j):
        idx=np.array([self.ix[v] for v in nodes])
        pids=self.pair_id[idx[i],idx[j]]
        vis=np.array([self.ix[v] for group in o["observed"] for v in group])
        designs=[]
        for h in (0,1):
            logit=self.logit[h,pids[:,None],vis]
            sizes=self.sizes[h,pids]
            align=np.exp(-np.abs(np.log((sizes+1)/(len(vis)+1))))*(sizes>0)
            feats=np.column_stack((logit.mean(axis=1),self.absence[h,pids]/len(vis),np.log(.02+align.mean(axis=1))))
            designs.append(scale(feats))
        return designs,pids,vis


def bo_geometry(n):
    a,b=np.triu_indices(n,1)
    x=np.zeros((len(a),n))
    x[np.arange(len(x)),a]=x[np.arange(len(x)),b]=1
    # A complete orthonormal graph Fourier basis preserves Euclidean distances
    # of the pair indicators, including their distances to k-means centroids.
    labels=KMeans(20,n_init=5,random_state=199).fit_predict(x)
    return x,[np.flatnonzero(labels==k) for k in range(20)]


def search_bo(x,groups,objective,rng):
    used=np.zeros(len(x),bool)
    chosen=[int(rng.choice(group)) for group in groups]
    used[chosen]=True
    length=1.
    for step in range(50):
        design=x[chosen]
        y=objective[chosen]
        y=(y-y.mean())/max(y.std(),1e-8)
        dist=np.maximum(0.,4.-2*design@design.T)
        def factor(ell):
            K=np.exp(-dist/(2*ell**2))+.05*np.eye(len(y))
            return cho_factor(K,lower=True)
        if step%10==0:
            def nll(ell):
                ch=factor(ell)
                return y@cho_solve(ch,y)/2+np.log(np.diag(ch[0])).sum()
            length=min((.35,.7,1.4,2.8),key=nll)
        ch=factor(length)
        options=[rng.choice(group[~used[group]]) for group in groups if np.any(~used[group])]
        K=np.exp(-(4.-2*design@x[options].T)/(2*length**2))
        mu=K.T@cho_solve(ch,y)
        sd=np.sqrt(np.maximum(1e-9,1-(solve_triangular(ch[0],K,lower=True)**2).sum(axis=0)))
        improve=mu-y.max()
        z=improve/sd
        ei=improve*ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)
        pick=options[np.argmax(ei)]
        chosen.append(int(pick))
        used[pick]=True
    return np.array(chosen)


def freeze():
    OUT.mkdir(exist_ok=True)
    write_json(OUT/"protocol.json",PLAN)
    raw=load_events(DEPT)
    gs,streams=make_windows(raw,"train",.15,.6)
    nm,fm,_=models("Dept3")
    rows=defaultdict(list)
    for week,g in enumerate(gs,1):
        if week<=10:
            continue
        obs,labs,_=draw(g,streams[g["window"]],90600+week,55,0)
        if not obs:
            continue
        tag=f"rolling_{(week-1)//5*5}"
        w=PairEvidence(raw,g,fm[tag],nm[tag])
        bank=Bank(raw,g)
        for o,lab in zip(obs,labs):
            nodes,i,j,ds,*_=w.design(o)
            if not set(lab["sources"])<=set(nodes):
                continue
            a,b=sorted(nodes.index(v) for v in lab["sources"])
            y=int(np.flatnonzero((i==a)&(j==b))[0])
            (full,tail),_,_=bank.design(o,nodes,i,j)
            for m,extra in (("joint_full",full),("joint_tail",tail)):
                rows[m].append((np.column_stack((ds["ordered"],extra)),y))
            rows["tail_only"].append((tail,y))
        print("training",week,flush=True)
    assert all(len(r)==758 for r in rows.values())
    write_json(OUT/"weights.json",{m:fit(r,r[0][0].shape[1]) for m,r in rows.items()})
    paths=[OUT/"weights.json",OUT/"protocol.json",PREV/"model.json",RAW,ROOT/"scripts/masked_pair_benchmark.py"]
    write_json(OUT/"lock.json",{p.relative_to(ROOT).as_posix():sha(p) for p in paths})


def predict():
    assert all(sha(ROOT/p)==s for p,s in load(OUT/"lock.json").items())
    raw=company()
    gs,streams=make_windows(raw,"company",*PLAN["span"])
    weights=load(OUT/"weights.json")
    prev=load(PREV/"model.json")["weights"]
    f=frozen_models()["Dept3"]
    public,truth,flow=[],[],{}
    for k,g in enumerate(gs):
        o,t,c=draw(g,streams[g["window"]],PLAN["seed"]+k,35,15)
        public.extend(o)
        truth.extend(t)
        flow[g["window"]]=c
    save_rows(OUT/"observations.jsonl",public)
    save_rows(OUT/"truth.jsonl",truth)
    write_json(OUT/"generation.json",flow)
    geo={}
    for g in gs:
        items=[o for o in public if o["window"]==g["window"]]
        if not items:
            continue
        bank=Bank(raw,g)
        w=PairEvidence(raw,g,f["forward"],f["node"])
        output=[]
        for o in items:
            nodes,i,j,ds,scores,tie,_=w.design(o)
            (full,tail),pids,vis=bank.design(o,nodes,i,j)
            scores.update({m:ds[m]@prev[m] for m in ("ordered","shuffled")})
            scores["joint_full"]=np.column_stack((ds["ordered"],full))@weights["joint_full"]
            scores["joint_tail"]=np.column_stack((ds["ordered"],tail))@weights["joint_tail"]
            scores["tail_only"]=tail@weights["tail_only"]
            peak=peak_scores(pids,bank.a,bank.b,bank.cover,bank.buckets,vis)
            scores["snapshot_all"]=peak
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(o["id"].encode()).digest()[:8],"little"))
            tie_random=rng.permutation(len(i))
            if len(nodes) not in geo:
                geo[len(nodes)]=bo_geometry(len(nodes))
            chosen=search_bo(*geo[len(nodes)],peak,rng)
            random70=rng.choice(len(i),len(chosen),replace=False)
            orders={m:np.lexsort((tie,-scores["additive"],-s)).tolist() for m,s in scores.items()}
            orders["snapshot_all"]=np.lexsort((tie_random,-peak)).tolist()
            for m,idx in (("bosoul70",chosen),("random70",random70)):
                orders[m]=idx[np.lexsort((tie_random[idx],-peak[idx]))].tolist()
            output.append(dict(id=o["id"],window=g["window"],nodes=nodes,orders=orders))
        save_rows(OUT/(g["window"]+"_predictions.jsonl"),output)
        print(g["window"],len(items),"scored",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["freeze","predict"])
    with threadpool_limits(limits=1):
        {"freeze":freeze,"predict":predict}[parser.parse_args().phase]()
