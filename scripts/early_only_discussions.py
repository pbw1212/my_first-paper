"""Evaluate the documented first two speakers when only their prefix is missing."""
import argparse
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
import heldout_libc_alpha as libc
import heldout_r_lists as r
from public_thread_transfer import DAY,thread_forest
from real_trained_mixtures import rank

OUT=r.OUT/"early_only"
NAMES=(*r.NAMES,libc.NAME)


def plan():
    return dict(targets=list(NAMES),months=r.KEYS,
                primary="frozen mixed_offset versus 28-day post count, all threads with a unique second and a later distinct speaker",
                eligibility="root confirmed; 14-day horizon; unique second arrival strictly later than root; at least one further distinct author",
                observations="hide initial two only; retain all later first arrivals on closed 98-day roster",
                scoring="at least one visible historical author and 12 legal candidates; all participants outside remain misses; shared abstentions",
                comparisons="all four frozen source models; old-mask common roots separately; also common scorable under both rules",
                tie="as in ordinary cohorts; no real labels for fitting",budget=12,
                hashes={p.name+str(i):fg.digest(p) for i,p in enumerate([Path(__file__),r.OUT/"protocol_v2.json",libc.OUT/"protocol.json"])})


def prefix(root,thread,msg):
    first,t0=msg[root][:2]
    arrivals={}
    for t,_,a in thread:
        if a!=first and t0<=t<t0+14*DAY:
            arrivals[a]=min(arrivals.get(a,t),t)
    sequence=sorted((t,a) for a,t in arrivals.items())
    if len(sequence)<2 or sequence[0][0]<=t0 or sequence[0][0]==sequence[1][0]:
        return None
    return [first,sequence[0][1]],[list(a for _,a in group) for _,group in groupby(sequence[1:],key=lambda x:x[0])]


def paired(z,keys):
    delta=np.array([v for _,v in z],float)
    if not len(delta):
        return dict(n=0)
    at=np.array([keys.index(k) for k,_ in z])
    counts=np.bincount(at,minlength=24)
    rng=np.random.default_rng(94183)
    blocks=((rng.integers(24,size=(10000,8,1))+np.arange(3))%24).reshape(-1,24)
    seeds=rng.integers(5,size=(10000,1,5))
    den=counts[blocks].sum(1)
    ok=den>0
    sums=np.array([delta[at==k].sum(0) for k in range(24)])
    sampled=sums[blocks[:,:,None],seeds].sum(1).mean(1)
    return dict(n=len(z),pp=100*delta.mean(),ci95_pp=np.quantile(100*sampled[ok]/den[ok],[.025,.975]).tolist())


def run(name):
    msg=libc.messages() if name==libc.NAME else r.messages(name)
    forest,_=thread_forest(msg)
    roots_by_month={}
    from datetime import datetime,timezone
    for root in forest:
        k=datetime.fromtimestamp(msg[root][1],timezone.utc).strftime("%Y-%m")
        roots_by_month.setdefault(k,[]).append(root)
    models=r.fitted_models("Dept2")
    rows=[]
    old_keys=set()
    for g,h,_,vol,ordinary,_ in r.calendar_windows(msg,name):
        ix={n:i for i,n in enumerate(g["nodes"])}
        hist=r.PublicWindow(g,h)
        w=r.activity(g,h)
        means=r.law_arrays(w)[3]
        previous={e["id"]:(src,e["observed"]) for e,src in ordinary}
        old_keys.update(previous)
        for root in sorted(roots_by_month.get(g["window"],[])):
            payload=prefix(root,forest[root],msg)
            if payload is None:
                continue
            src,batches=payload
            visible=[[v for v in grp if v in ix] for grp in batches]
            obs=[grp for grp in visible if grp]
            seen={v for grp in obs for v in grp}
            assert seen.isdisjoint(src)
            ready=bool(seen) and len(ix)-len(seen)>=12
            if root in previous:
                oldsrc,oldsuffix=previous[root]
                assert src==oldsrc and all(v in seen for grp in oldsuffix for v in grp)
            e=dict(id=root,window=g["window"],observed=obs,k=2)
            lists=rank(hist,e,models,w,ix,means,vol) if ready else {}
            assert all(len(set(v))==12 and set(v)<=ix.keys()-seen for v in lists.values())
            rows.append(dict(id=root,window=g["window"],sources=src,ready=ready,
                             reachable=set(src)<=ix.keys(),visible=len(seen),later=sum(map(len,batches)),
                             hits={m:int(set(src)<=set(v)) for m,v in lists.items()}))
    assert old_keys<={z["id"] for z in rows} and len(rows)==len({z["id"] for z in rows})
    fg.save_rows(OUT/(name+".jsonl"),rows)


def report(name):
    rows=fg.read_rows(OUT/(name+".jsonl"))
    base=(libc.OUT if name==libc.NAME else r.OUT/name)
    old={z["id"]:z for z in fg.read_rows(base/"events.jsonl") if z["profile"]=="real"}
    by_id={z["id"]:z for z in rows}
    assert set(old)<=set(by_id) and all(by_id[k]["sources"]==z["sources"] for k,z in old.items())
    def hit(z,m):
        return np.array([z["hits"].get(m if m=="volume" else f"{m}/{s}",0) for s in r.SEEDS])
    out=dict(eligible=len(rows),ready=sum(z["ready"] for z in rows),
             scoreable_and_reachable=sum(z["ready"] and z["reachable"] for z in rows),
             ordinary_eligible=len(old),num_visible=dict(Counter(str(z["visible"]) for z in rows)),
             methods={m:dict(hits_by_seed=np.stack([hit(z,m) for z in rows]).sum(0).tolist(),
                             both_pct=100*np.stack([hit(z,m) for z in rows]).mean()) for m in ("volume",*r.ARMS)})
    out["offset_minus_volume"]=paired([(z["window"],hit(z,"mixed_offset")-hit(z,"volume")) for z in rows],r.KEYS)
    out["offset_minus_plain"]=paired([(z["window"],hit(z,"mixed_offset")-hit(z,"mixed_plain")) for z in rows],r.KEYS)
    out["same_roots_more_observations"]={}
    for tag in ("all_common_roots","scored_under_both"):
        cases=[z for k,z in old.items() if tag=="all_common_roots" or (z["ready"] and by_id[k]["ready"])]
        out["same_roots_more_observations"][tag]=paired([(z["window"],hit(by_id[z["id"]],"mixed_offset")-hit(z,"mixed_offset")) for z in cases],r.KEYS)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("cmd",choices=["freeze","report",*NAMES])
    cmd=parser.parse_args().cmd
    OUT.mkdir(exist_ok=True)
    path=OUT/"protocol.json"
    if cmd=="freeze" and not path.exists():
        fg.write_json(path,plan())
    assert fg.load(path)==plan()
    if cmd=="report":
        fg.write_json(OUT/"summary.json",{n:report(n) for n in NAMES})
    elif cmd in NAMES:
        torch.set_num_threads(1)
        with threadpool_limits(1):
            run(cmd)
