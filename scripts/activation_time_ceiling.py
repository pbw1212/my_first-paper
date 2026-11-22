"""Exact seed-pair decision given all contact and activation times, with hidden parents."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.special import logsumexp

import fgssi_suffix as fg
from build_dept3_pilot import P, simulate
from prior_partner_ceiling import conditional
from silent_source_oracle import CompleteTrace
from source_prior_shift import NETWORKS, OUT as BASE, REGIMES, activity, windows

OUT = BASE.parent/"activation_time_ceiling"
NEVER = np.iinfo(np.int64).max
LOG_Q = np.log1p(-P)


def times_only(trajectory):
    return [(batch["t"],v) for batch in trajectory for v,_ in batch["joins"]]


class ActivationTimes(CompleteTrace):
    def __init__(self,graph,stream):
        super().__init__(graph,stream)
        self.n=len(self.nodes)
        arc=np.zeros((self.n,self.n),dtype=int)
        np.add.at(arc,(self.senders,self.targets),1)
        self.dyad=arc+arc.T

    def distribution(self,joins,w,law):
        t,u,v=self.times,self.senders,self.targets
        times=np.full(self.n,NEVER,dtype=np.int64)
        for when,node in joins:
            times[self.index[node]]=when
        unseen=np.flatnonzero(times==NEVER)
        before=t<times[v]
        active=t>times[u]
        # Groups with zero previously observed active senders must intersect the seed pair.
        on=t==times[v]
        basis=np.bincount(v[on & active],minlength=self.n)
        candidates=(times[u]==NEVER) & on
        origins={}
        for a,b in zip(u[candidates],v[candidates]):
            bucket=origins.setdefault(b,{})
            bucket[a]=bucket.get(a,0)+1
        joined=np.flatnonzero(times!=NEVER)
        required=[sorted(origins.get(b,{})) for b in joined if basis[b]==0]
        if not required or not all(required):
            raise ValueError("An observed activation must be explained by a source contact.")
        cover=np.array(min(required,key=len),dtype=int)
        first=np.repeat(cover,len(unseen))
        second=np.tile(unseen,len(cover))
        keep=(first!=second) & ((first<second) | ~np.isin(second,cover))
        a,b=np.minimum(first[keep],second[keep]),np.maximum(first[keep],second[keep])
        for witnesses in required:
            keep=np.isin(a,witnesses) | np.isin(b,witnesses)
            a,b=a[keep],b[keep]
        extra=np.bincount(u[before],minlength=self.n)-np.bincount(v[before & active],minlength=self.n)
        # The source-independent number of failed contacts is discarded.
        score=(extra[a]+extra[b]-self.dyad[a,b])*LOG_Q
        # Repeated attempts in a synchronous contact batch form a single success probability.
        # A target with no source contacts has a constant factor, which cancels.
        for target,incoming in origins.items():
            possible=np.zeros(self.n,dtype=int)
            for sender,n in incoming.items():
                possible[sender]=n
            trials=basis[target]+possible[a]+possible[b]
            assert np.all(trials>0)
            score+=np.log(-np.expm1(LOG_Q*trials))
        if law=="one":
            score+=np.log(w[a]+w[b])
        elif law=="both":
            score+=np.log(w[a])+np.log(w[b])
        else:
            assert law=="flat"
        logprob=score-logsumexp(score)
        return a,b,logprob,cover,unseen


def exact_list(a,b,p,cover,unseen,n,k):
    k=min(k,len(unseen))
    if len(unseen)<=k:
        return unseen,1.
    nodes=np.unique(np.r_[a,b])
    if len(nodes)<=k:
        return np.r_[nodes,np.setdiff1d(unseen,nodes)[:k-len(nodes)]],1.
    cover=np.intersect1d(cover,nodes)
    if len(cover)>14:
        # Sparse exact formulation for a dense first qualifying contact batch.
        m=len(a)
        ii=np.arange(m)
        offset=len(nodes)
        row=np.r_[ii,ii,m+ii,m+ii,np.repeat(2*m,offset)]
        col=np.r_[offset+ii,np.searchsorted(nodes,a),offset+ii,np.searchsorted(nodes,b),np.arange(offset)]
        vv=np.r_[np.ones(m),-np.ones(m),np.ones(m),-np.ones(m),np.ones(offset)]
        mat=coo_matrix((vv,(row,col)),shape=(2*m+1,offset+m)).tocsr()
        res=milp(np.r_[np.zeros(offset),-p],integrality=np.r_[np.ones(offset),np.zeros(m)],
                 bounds=Bounds(0,1),constraints=LinearConstraint(mat,np.r_[np.full(2*m,-np.inf),k],np.r_[np.zeros(2*m),k]),
                 options={"mip_rel_gap":0.})
        assert res.success
        selected=nodes[res.x[:offset]>.5]
        return selected,float(p[np.isin(a,selected)&np.isin(b,selected)].sum())
    outer=np.setdiff1d(nodes,cover)
    weight=np.zeros((len(cover),n))
    pos={v:i for i,v in enumerate(cover)}
    for i,j,v in zip(a,b,p):
        if i in pos:
            weight[pos[i],j]+=v
        if j in pos:
            weight[pos[j],i]+=v
    high=-1.
    chosen=None
    for size in range(max(1,k-len(outer)),min(k,len(cover))+1):
        for subset in combinations(range(len(cover)),size):
            inner=cover[list(subset)]
            link=weight[list(subset)].sum(0)
            extra=outer[np.lexsort((outer,-link[outer]))[:k-size]]
            value=link[inner].sum()/2+link[extra].sum()
            if value>high:
                high,chosen=value,np.r_[inner,extra]
    assert chosen is not None and len(chosen)==k
    return chosen,float(high)


def protocol():
    return dict(observations="all within-window contact times and directions, nonseed join times and identities; parents hidden",
                purpose="diagnostic at known source law; evaluate same complete traces as parent-labelled ceiling",
                budget=12,source_protocol=fg.digest(BASE/"protocol.json"),script=fg.digest(Path(__file__)))


def run(net):
    assert fg.load(OUT/"protocol.json")==protocol()
    raw,gs=windows(net,False)
    path=BASE/net/"test"
    assert fg.digest(path/"events.jsonl")==fg.load(path/"audit.json")["events"]
    records=fg.read_rows(path/"events.jsonl")
    parent={r["id"]:r for r in fg.read_rows(BASE/net/"ceiling/events.jsonl")}
    result=[]
    for g,stream in gs:
        entries=[r for r in records if r["window"]==g["window"]]
        if not entries:
            continue
        world=ActivationTimes(g,stream)
        w=activity(g,raw)
        for r in entries:
            trajectory=simulate(stream,r["sources"],r["seed"])
            a,b,lp,cover,unseen=world.distribution(times_only(trajectory),w,r["regime"])
            prob=np.exp(lp)
            selected,mass=exact_list(a,b,prob,cover,unseen,world.n,12)
            real=sorted(world.index[v] for v in r["sources"])
            truth=np.flatnonzero((a==real[0]) & (b==real[1]))
            assert len(truth)==1 and np.isclose(mass,prob[np.isin(a,selected)&np.isin(b,selected)].sum())
            lab=dict(trajectory=trajectory,sources=r["sources"])
            anchor,partners,cap=conditional(world,lab,r["regime"],w,False)
            assert np.isclose(parent[r["id"]]["cap"],cap)
            chosen={world.nodes[q] for q in selected}
            rb=sum(p for v,p in partners.items() if v in chosen) if anchor in chosen else 0.
            assert rb<=cap+1e-10
            result.append(dict(id=r["id"],window=g["window"],regime=r["regime"],emitters=r["emitters"],
                               compatible_pairs=len(a),cover_size=len(cover),time_mass=mass,parent_mass=cap,
                               time_list_parent_expectation=rb,hit=int(set(r["sources"])<=chosen),
                               true_log_prob=float(lp[truth[0]]),extra_identity_bits=float((np.log(partners[next(v for v in r["sources"] if v!=anchor)])+(-lp[truth[0]]))/np.log(2))))
    assert len(result)==len(parent)==len(records)
    target=OUT/net
    target.mkdir(exist_ok=True)
    fg.save_rows(target/"events.jsonl",result)
    report={}
    for law in REGIMES:
        group=[r for r in result if r["regime"]==law]
        report[law]={}
        for e in (0,1,2):
            sub=[r for r in group if not e or r["emitters"]==e]
            if not sub:
                continue
            fields=("time_mass","parent_mass","time_list_parent_expectation","hit","extra_identity_bits")
            stats={key:float(np.mean([r[key] for r in sub])) for key in fields}
            stats.update(n=len(sub),value_of_parents_pp=100*(stats["parent_mass"]-stats["time_list_parent_expectation"]),
                         unique_pair_fraction=float(np.mean([r["compatible_pairs"]==1 for r in sub])),
                         max_cover_size=max(r["cover_size"] for r in sub))
            report[law][("all","one","two")[e]]=stats
    fg.write_json(target/"summary.json",report)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze",*NETWORKS))
    cmd=parser.parse_args().action
    OUT.mkdir(exist_ok=True)
    if cmd=="freeze":
        if not (OUT/"protocol.json").exists():
            fg.write_json(OUT/"protocol.json",protocol())
        assert fg.load(OUT/"protocol.json")==protocol()
    else:
        run(cmd)
