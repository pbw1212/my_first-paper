"""Matched five-frame hard-label experiments with static and temporal history."""
import argparse
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from distilled_source_pairs import CHECKPOINTS, Packed, PairNet, PublicWindow
from factorial_fgssi import OUT, SEEDS
from fgssi_suffix import OUT as EARLIER, cohort, digest, load, raw_data, write_json
from five_frame_pairs import frames


class StaticWindow:
    """Use only undirected cumulative contact counts and five public membership increments."""
    def __init__(self,graph):
        self.nodes=sorted(graph["nodes"])
        self.index={v:i for i,v in enumerate(self.nodes)}
        n=len(self.nodes)
        counts=np.zeros((n,n),dtype=np.float32)
        for u,v,c in graph["edges"]:
            i,j=self.index[u],self.index[v]
            counts[i,j]+=c
            counts[j,i]+=c
        adj=counts>0
        dist=shortest_path(csr_matrix(adj),directed=False,unweighted=True)
        self.reach=np.divide(1.,dist,out=np.zeros_like(dist),where=(dist>0)&np.isfinite(dist))
        self.contact=np.log1p(counts)
        self.central=np.column_stack((np.log1p(adj.sum(1)),np.log1p(counts.sum(1)),self.reach.mean(1)))

    def design(self,event):
        groups=event["observed"]
        visible=[v for group in groups for v in group]
        nodes=sorted(set(self.nodes)-set(visible))
        i=[self.index[v] for v in nodes]
        j=[self.index[v] for v in visible]
        reach=self.reach[np.ix_(i,j)]
        contact=self.contact[np.ix_(i,j)]
        order=np.repeat(np.arange(len(groups)),list(map(len,groups)))
        previous=order[:,None]<order[None,:]
        redundancy=np.where(previous,self.reach[np.ix_(j,j)],0.).max(axis=0)
        weights=1/(1+redundancy)
        weights/=weights.sum()
        features=[self.central[i],reach@weights,contact@weights]
        for c in (reach,contact):
            for subset in (c,c[:,:len(groups[0])],c[:,-len(groups[-1]):]):
                features.extend((subset.mean(1),subset.max(1)))
        absolute=np.column_stack(features)
        relative=(rankdata(absolute,axis=0)-.5)/len(i)
        context=np.broadcast_to([np.log1p(len(nodes)),np.log1p(len(j)),len(groups),len(groups[0])/len(j)],(len(i),4))
        x=np.column_stack((absolute,relative,context)).astype(np.float32)
        weighted=reach*np.sqrt(weights)
        return (nodes,None,None),(x,np.minimum(2.,self.contact[np.ix_(i,i)]/3),(weighted@weighted.T).astype(np.float32))


def examples(name,kind,phase,raw):
    result=[]
    for graph,events,truth in cohort(name,phase):
        window=StaticWindow(graph) if kind=="static" else PublicWindow(graph,raw)
        for event,label in zip(events,truth,strict=True):
            base,features=window.design(frames(event))
            nodes=base[0]
            a,b=[nodes.index(v) for v in label["sources"]]
            soft=np.zeros(len(nodes),np.float32)
            soft[b]=1
            result.append((*features,a,b,soft))
    return result


def folder_for(name,kind):
    return OUT/"fits"/name/kind


def fit(name,kind):
    folder=folder_for(name,kind)
    folder.mkdir(parents=True,exist_ok=True)
    raw=raw_data(name)
    rows=examples(name,kind,"fit",raw)
    x=np.vstack([r[0] for r in rows])
    mean=x.mean(0);std=np.maximum(x.std(0),.025)
    train=Packed(rows,mean,std)
    del rows,x
    val=Packed(examples(name,kind,"cal",raw),mean,std)
    assert (len(train.x),len(val.x))==(1000,300)
    np.savez(folder/"scale.npz",mean=mean,std=std)
    for seed in SEEDS:
        for interaction in (False,True):
            tag=("pair" if interaction else "add")+str(seed)
            if kind=="temporal" and interaction and seed==314159:
                # The five-frame source model used this exact feature set, objective and seed.
                cfg=load(EARLIER/name/"five_selection.json")
                assert all(digest(EARLIER/name/f)==h for f,h in cfg["hashes"].items())
                with np.load(EARLIER/name/"five_scale.npz") as old:
                    assert np.array_equal(mean,old["mean"]) and np.array_equal(std,old["std"])
                saved=EARLIER/name/"five_hard.pt"
                net=PairNet(len(mean),True)
                write_json(folder/(tag+".json"),dict(seed=seed,epoch=cfg["selected"],curve=cfg["curve"],weights=str(saved),
                    sha256=digest(saved),parameters=sum(p.numel() for p in net.parameters()),fit_seconds=None))
                continue
            torch.manual_seed(seed)
            rng=np.random.default_rng(seed)
            net=PairNet(len(mean),interaction)
            opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.005)
            best=float("inf");curve={}
            saved=folder/(tag+".pt")
            start=time.perf_counter()
            for epoch in range(1,max(CHECKPOINTS)+1):
                net.train()
                for idx in np.array_split(rng.permutation(len(train.x)),int(np.ceil(len(train.x)/24))):
                    opt.zero_grad()
                    train.loss(net,idx,False).backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),5)
                    opt.step()
                if epoch in CHECKPOINTS:
                    net.eval()
                    with torch.no_grad():
                        value=sum(len(i)*val.loss(net,i,False).item() for i in np.array_split(np.arange(len(val.x)),15))/len(val.x)
                    curve[epoch]=value
                    if value<best:
                        best=value
                        torch.save(net.state_dict(),saved)
            write_json(folder/(tag+".json"),dict(seed=seed,epoch=min(curve,key=curve.get),curve=curve,weights=str(saved),
                sha256=digest(saved),parameters=sum(p.numel() for p in net.parameters()),fit_seconds=time.perf_counter()-start))
            print(name,kind,tag,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("network",choices=["company","Dept2"])
    parser.add_argument("history",choices=["static","temporal"])
    args=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        fit(args.network,args.history)
