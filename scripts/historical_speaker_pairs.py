"""Infer the first speaker and responder from completed past discussions and activity."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
import heldout_r_lists as held
from historical_thread_roles import historical_document
from joint_source_shortlist import best_subset
from public_thread_transfer import DAY,thread_forest
from real_trained_mixtures import LISTS,messages as old_messages
from test_reply_direction import groups

OUT=held.OUT/"speaker_pairs"
MASSES=[4.,32.,256.]
SHRINK=[1.,8.,64.,512.]
METHODS=("volume","roles","paired","matched_margins")


def months(name,late):
    msg=held.messages(name) if name in held.NAMES else old_messages(name,late)
    forest,_=thread_forest(msg)
    docs=[historical_document(root,thread,msg[root][1],msg[root][0])[:3] for root,thread in forest.items()]
    for g,_,vol,real in groups(name,late):
        nodes=g["nodes"]
        ix={v:i for i,v in enumerate(nodes)}
        c=np.zeros((len(ix),len(ix)))
        for t,a,b in docs:
            if g["start"]-98*DAY<=t<g["start"]-14*DAY and a in ix and b in ix:
                c[ix[a],ix[b]]+=1
        v=np.array([vol[n]+.5 for n in nodes])
        yield ix,c,v,real,g["window"]


def post(c,v,tau,kappa=None):
    base=v/v.sum()
    p=c.sum(1)+tau*base
    q=c.sum(0)+tau*base
    p/=p.sum()
    q/=q.sum()
    cond=np.broadcast_to(q,c.shape).copy()
    np.fill_diagonal(cond,0)
    cond/=cond.sum(1,keepdims=True)
    if kappa is not None:
        cond=(c+kappa*cond)/(c.sum(1,keepdims=True)+kappa)
    return p[:,None]*cond


def margins(joint):
    # Maximum-entropy coupling of the same two marginals with the diagonal excluded.
    r,c=joint.sum(1),joint.sum(0)
    col=c.copy()
    for _ in range(1000):
        row=r/(col.sum()-col)
        col=c/(row.sum()-row)
        if np.max(abs(row*(col.sum()-col)-r))<1e-12:
            break
    table=np.outer(row,col)
    np.fill_diagonal(table,0)
    assert np.max(abs(table.sum(1)-r))<1e-10 and np.max(abs(table.sum(0)-c))<1e-10
    return table


def legal(e,ix):
    visible={v for grp in e["observed"] for v in grp}
    ids=np.array([i for v,i in ix.items() if v not in visible],dtype=int)
    return ids,len(visible)>=3 and len(ids)>=12


def train():
    cfg=dict(primary="paired minus roles, all real eligible threads; separate test communities; fixed 12-person budget",
             controls="28-day message volume; maximum entropy legal ordered pair table with exactly matched role marginals",
             training="2017-2018 Python lists only, conditional ordered-pair NLL on scoreable two-reachable discussions; first select role smoothing then conditional smoothing",
             disjoint_tests=["2019 Python follow-up","GCC","FFmpeg","R-devel","R-package-devel"],
             tau=MASSES,kappa=SHRINK,sha256=fg.digest(Path(__file__)),dependencies={
                p.name:fg.digest(p) for p in [held.OUT/"protocol_v2.json",
                Path(__file__).with_name("test_reply_direction.py"),Path(__file__).with_name("joint_source_shortlist.py")]})
    totals=Counter()
    n=0
    for name in LISTS:
        for ix,c,v,real,_ in months(name,False):
            pairs={(t,k):post(c,v,t,k) for t in MASSES for k in [None,*SHRINK]}
            for e,src in real:
                ids,ready=legal(e,ix)
                if not ready or not set(src)<=set(ix):
                    continue
                a,b=[ix[s] for s in src]
                assert a in ids and b in ids
                for key,table in pairs.items():
                    totals[key]+=np.log(table[np.ix_(ids,ids)].sum()/table[a,b])
                n+=1
    tau=min(MASSES,key=lambda t:totals[t,None])
    kappa=min(SHRINK,key=lambda k:totals[tau,k])
    cfg.update(selected_tau=tau,selected_kappa=kappa,n=n,mean_nll={str(k):v/n for k,v in totals.items()})
    if (OUT/"protocol.json").exists():
        assert fg.load(OUT/"protocol.json")==cfg
    else:
        fg.write_json(OUT/"protocol.json",cfg)


def run(name,late,cfg):
    tag=name+("-2019" if late else "")
    rows=[]
    keys=[]
    for ix,c,v,real,window in months(name,late):
        keys.append(window)
        separate=post(c,v,cfg["selected_tau"])
        linked=post(c,v,cfg["selected_tau"],cfg["selected_kappa"])
        for e,src in real:
            ids,ready=legal(e,ix)
            r=dict(id=e["id"],window=window,ready=ready,roles={})
            if ready:
                tie=fg.tie_order(e["id"],len(ids))
                sorted_volume=fg.sorted_nodes(v[ids],tie)
                pair=linked[np.ix_(ids,ids)]
                pair/=pair.sum()
                tables=[separate[np.ix_(ids,ids)],pair,margins(pair)]
                choices=[sorted_volume[:12]]
                for mat in tables:
                    sym=(mat+mat.T)/mat.sum()
                    choices.append(best_subset(sym,sorted_volume,tie,12) if len(ids)>12 else sorted_volume)
                a,b=[ix.get(s,-1) for s in src]
                for method,sel in zip(METHODS,choices):
                    picked=ids[sel]
                    r["roles"][method]=int(a in picked)+2*int(b in picked)
            rows.append(r)
    path=(held.OUT/name/"events.jsonl" if name in held.NAMES else held.FIT/"early_speakers"/(tag+".jsonl"))
    old={r["id"]:r for r in fg.read_rows(path) if r.get("profile","real")=="real"}
    assert len(old)==len(rows) and all(old[r["id"]]["ready"]==r["ready"] and
        old[r["id"]]["hits"].get("volume",0)==int(r["roles"].get("volume",0)==3) for r in rows)
    fg.save_rows(OUT/(tag+".jsonl"),rows)
    if not rows:
        return dict(n=0)
    codes=np.array([[r["roles"].get(m,0) for m in METHODS] for r in rows])
    hits=(codes==3).astype(int)
    out=dict(n=len(rows),scored=sum(r["ready"] for r in rows),
             methods={m:dict(both=int(hits[:,i].sum()),first=int((codes[:,i]%2).sum()),second=int((codes[:,i]//2).sum()))
                      for i,m in enumerate(METHODS)},paired_differences={})
    at=np.array([keys.index(r["window"]) for r in rows])
    ct=np.bincount(at,minlength=len(keys))
    rng=np.random.default_rng(35417)
    nn=len(keys)
    bb=((rng.integers(nn,size=(10000,(nn+2)//3,1))+np.arange(3))%nn).reshape(10000,-1)[:,:nn]
    den=ct[bb].sum(1)
    ok=den>0
    for i in (0,1,3):
        d=hits[:,2]-hits[:,i]
        details=dict(pp=100*d.mean(),gained=int((d>0).sum()),lost=int((d<0).sum()))
        if nn>=12:
            dd=np.bincount(at,weights=d,minlength=nn)[bb].sum(1)
            details["ci95_pp"]=np.quantile(100*dd[ok]/den[ok],[.025,.975]).tolist()
        out["paired_differences"][METHODS[i]]=details
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("fit","test"))
    action=parser.parse_args().action
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(1):
        if action=="fit":
            train()
        else:
            cfg=fg.load(OUT/"protocol.json")
            assert cfg["sha256"]==fg.digest(Path(__file__))
            names=[(n,True) for n in LISTS]+[(n,False) for n in ("gcc","ffmpeg-devel",*held.NAMES)]
            results={n+("-2019" if late else ""):run(n,late,cfg) for n,late in names}
            fg.write_json(OUT/"summary.json",results)
