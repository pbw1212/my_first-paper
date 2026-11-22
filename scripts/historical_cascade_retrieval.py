"""Development pilot: compare a query suffix with labeled cascades simulated strictly in the past."""

from itertools import combinations

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix

from temporal_pair_evidence import OUT, OLD, DAY, EXTERNAL, load_events, read_rows, save_rows, metrics, write_json


@njit
def database(t,u,v,n,trials,seed):
    np.random.seed(seed)
    starts=[0]
    for k in range(1,len(t)):
        if t[k]!=t[k-1]:
            starts.append(k)
    starts.append(len(t))
    src=np.empty((trials,2),np.int32)
    visible=np.zeros((trials,n),np.int8)
    accepted=0
    for _ in range(trials):
        a=np.random.randint(n)
        b=np.random.randint(n-1)
        b+=b>=a
        label=np.zeros(n,np.int8)
        label[a],label[b]=1,2
        time=np.full(n,-1,np.int32)
        added=0
        for left,right in zip(starts[:-1],starts[1:]):
            for k in range(left,right):
                s,d=u[k],v[k]
                if label[s] and time[s]!=left and (not label[d] or time[d]==left) and np.random.random()<.3:
                    if not label[d]:
                        added+=1
                    label[d]|=label[s]
                    time[d]=left
        if added<4:
            continue
        times=np.sort(time[time>=0])
        first=times[added//2]
        hidden=np.sum(times<first)
        if hidden==0 or added-hidden<3:
            continue
        evidence=0
        for j in range(n):
            if time[j]>=first:
                evidence|=label[j]
        if evidence!=3:
            continue
        visible[accepted]=time>=first
        src[accepted,0],src[accepted,1]=a,b
        accepted+=1
    return src[:accepted].copy(),visible[:accepted].copy()


def reference(raw,g):
    ix={v:i for i,v in enumerate(g["nodes"])}
    sources,features=[],[]
    for k in range(6):
        end=g["start"]-14*DAY*k
        start=end-14*DAY
        if start<raw[0][0]:
            continue
        contacts=np.array([(t,ix[u],ix[v]) for t,u,v in raw if start<=t<end],dtype=np.int64)
        if len(contacts)==0:
            continue
        s,f=database(*contacts.T,len(ix),15000,3701+g["start"]+k)
        if len(s):
            sources.append(s)
            features.append(csr_matrix(f))
    from scipy.sparse import vstack
    return ix,np.vstack(sources),vstack(features).tocsr()


def test():
    folder=OLD/"Dept1"
    raw=load_events(EXTERNAL)
    public=[o for o in read_rows(folder/"public.jsonl") if o["mask"]==50]
    predictions={p["id"]:p for p in read_rows(OUT/"Dept1_predictions.jsonl")}
    lab={r["id"]:r for r in read_rows(folder/"truth.jsonl")}
    rr=[]
    sizes={}
    for g in read_rows(folder/"graphs.jsonl"):
        items=[o for o in public if o["window"]==g["window"]]
        if not items:
            continue
        ix,src,ref=reference(raw,g)
        sizes[g["window"]]=len(src)
        n=len(ix)
        idf=np.log1p(len(src)/(1+np.asarray(ref.sum(axis=0)).ravel()))
        norm=np.sqrt(np.asarray(ref@(idf**2)).ravel())
        for o in items:
            v=[ix[node] for batch in o["observed"] for node in batch]
            q=np.zeros(n)
            q[v]=idf[v]**2
            similarity=np.asarray(ref@q).ravel()/np.maximum(1e-8,norm*np.sqrt(q.sum()))
            # Fixed exponential kernel and a smoothed empirical source prior.
            weight=np.exp(12*(similarity-similarity.max()))
            weight*=len(weight)/weight.sum()
            posterior=np.bincount(src.ravel(),weights=np.repeat(weight,2),minlength=n)+1
            prior=np.bincount(src.ravel(),minlength=n)+1
            p=predictions[o["id"]]
            nodes=p["nodes"]
            i,j=np.triu_indices(len(nodes),1)
            ids=np.array([ix[v] for v in nodes])
            pair_prior=np.log(prior[ids[i]])+np.log(prior[ids[j]])
            pair_posterior=np.log(posterior[ids[i]])+np.log(posterior[ids[j]])
            old=np.argsort(p["orders"]["ordered"])+1
            base=-np.log(old)
            scores=dict(ordered=base,retrieval=pair_posterior,
                        with_prior=base+.5*pair_prior,with_retrieval=base+.5*pair_posterior)
            truth=lab[o["id"]]
            sources=set(truth["sources"])
            y=None
            if sources<=set(nodes):
                a,b=sorted(nodes.index(v) for v in sources)
                y=np.flatnonzero((i==a)&(j==b))[0]
            row=dict(id=o["id"],window=g["window"],dual=truth["dual"],ordinary=truth["ordinary"])
            for tag,s in scores.items():
                order=np.lexsort((old,-s))
                row[tag]=None if y is None else int(np.flatnonzero(order==y)[0])+1
            rr.append(row)
    report={c:{k:metrics([r for r in rr if r[c]],k) for k in scores} for c in ("dual","ordinary")}
    save_rows(OUT/"retrieval_dev_ranks.jsonl",rr)
    write_json(OUT/"retrieval_dev_summary.json",dict(report,bank_sizes=sizes))
    print(report)


if __name__=="__main__":
    test()
