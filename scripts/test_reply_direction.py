"""Isolate reply-edge orientation from the source activity prior on real discussions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
import heldout_r_lists as held
from calendar_thread_calibration import TEST
from historical_thread_roles import TARGET_MONTHS
from public_thread_transfer import DAY, MONTHS, marked_thread, thread_forest
from real_trained_mixtures import LISTS, OUT as OLD, messages as previous_messages, rank
from train_source_mixtures import SEEDS, PublicWindow, activity, fitted_models, law_arrays

OUT=held.OUT/"direction"
VARIANTS=("forward_out","reverse_out","forward_in","reverse_in")


def plan():
    paths=[Path(__file__).resolve(),held.OUT/"protocol.json",OLD/"protocol.json"]
    return dict(primary="reverse minus original contact history, at fixed outgoing-reply source prior",
                alternatives=list(VARIANTS),model="Dept2 mixed_offset, all 5 original seeds; no target fitting",
                controls="reverse member features only, reverse source prior only, both reversed, 28-day total posts",
                targets="two R communities 2021-22, previously examined Python/GCC/FFmpeg cohorts separately",
                accounting="all eligible real threads; scoreability unchanged; both-in-12 and per-role recall",
                hashes={p.relative_to(OUT.parents[2]).as_posix():fg.digest(p) for p in paths})


def groups(name,late):
    if name in held.NAMES:
        msg=held.messages(name)
        for g,h,_,v,real,_ in held.calendar_windows(msg,name):
            yield g,h,v,real
        return
    msg=previous_messages(name,late)
    forest,contacts=thread_forest(msg)
    for y,m in (TEST if late else MONTHS if name in LISTS else TARGET_MONTHS):
        start=int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        end=int(datetime(y+(m==12),m%12+1,1,tzinfo=timezone.utc).timestamp())
        past=[r for r in contacts if start-98*DAY<=r[0]<start]
        cnt=Counter((a,b) for _,a,b in past)
        nodes=sorted({v for uv in cnt for v in uv})
        roster=set(nodes)
        key=f"{y}-{m:02d}"
        g=dict(start=start,window=key,nodes=nodes,edges=[(*uv,c) for uv,c in sorted(cnt.items())])
        vol=Counter(a for a,t,_,_ in msg.values() if start-28*DAY<=t<start)
        real=[]
        for r,batch in sorted(forest.items()):
            if start<=msg[r][1]<end:
                _,payload=marked_thread(r,batch,msg)
                if payload:
                    src,obs=payload
                    visible=[[v for v in grp if v in roster] for grp in obs]
                    real.append((dict(id=r,window=key,observed=[grp for grp in visible if grp]),src))
        yield g,past,vol,real


def run(name,late,models):
    tag=name+("-2019" if late else "")
    new=name in held.NAMES
    path=held.OUT/name/"events.jsonl" if new else OLD/(tag+".jsonl")
    old={r["id"]:r for r in fg.read_rows(path) if not new or r["profile"]=="real"}
    rows=[]
    keys=[]
    for g,past,vol,real in groups(name,late):
        keys.append(g["window"])
        ix={v:i for i,v in enumerate(g["nodes"])}
        prepared=False
        for e,src in real:
            vis={v for grp in e["observed"] for v in grp}
            ready=len(vis)>=3 and len(ix)-len(vis)>=12
            previous=old[e["id"]]
            assert ready==previous["ready"]
            r=dict(id=e["id"],window=g["window"],ready=ready,role_hits={})
            if ready:
                if not prepared:
                    backward=sorted((t,v,u) for t,u,v in past)
                    rev=dict(g,edges=sorted((v,u,c) for u,v,c in g["edges"]))
                    histories=[PublicWindow(g,past),PublicWindow(rev,backward)]
                    priors=[activity(g,past),activity(rev,backward)]
                    means=[law_arrays(w)[3] for w in priors]
                    prepared=True
                for variant,(di,pr) in zip(VARIANTS,((0,0),(1,0),(0,1),(1,1))):
                    pred=rank(histories[di],e,models,priors[pr],ix,means[pr],vol)
                    for s in SEEDS:
                        k=f"mixed_offset/{s}"
                        code=int(src[0] in pred[k])+2*int(src[1] in pred[k])
                        r["role_hits"][f"{variant}/{s}"]=code
                        if variant==VARIANTS[0]:
                            assert int(code==3)==previous["hits"][k]
                v=pred["volume"]
                r["role_hits"]["volume"]=int(src[0] in v)+2*int(src[1] in v)
                assert int(r["role_hits"]["volume"]==3)==previous["hits"]["volume"]
            rows.append(r)
    assert len(rows)==len(old)==len({r["id"] for r in rows})
    fg.save_rows(OUT/(tag+".jsonl"),rows)
    result=dict(eligible=len(rows),scored=sum(r["ready"] for r in rows),methods={})
    if not rows:
        return result
    values={}
    for method in ("volume",*VARIANTS):
        codes=np.array([[r["role_hits"].get("volume" if method=="volume" else f"{method}/{s}",0) for s in SEEDS] for r in rows])
        values[method]=(codes==3).astype(int)
        result["methods"][method]=dict(both_pct=100*values[method].mean(),first_pct=100*(codes%2).mean(),second_pct=100*(codes//2).mean(),
                                       hits_by_seed=values[method].sum(0).tolist())
    diff=values["reverse_out"]-values["forward_out"]
    result["primary_pp"]=100*diff.mean()
    if len(keys)>=12:
        n=len(keys)
        at=np.array([keys.index(r["window"]) for r in rows])
        c=np.bincount(at,minlength=n)
        rng=np.random.default_rng(61033)
        blocks=((rng.integers(n,size=(10000,(n+2)//3,1))+np.arange(3))%n).reshape(10000,-1)[:,:n]
        seeds=rng.integers(5,size=(10000,1,5))
        counts=c[blocks].sum(1)
        ok=counts>0
        sums=np.array([diff[at==j].sum(0) for j in range(n)])
        d=sums[blocks[:,:,None],seeds].sum(1).mean(1)
        result["ci95_pp"]=np.quantile(100*d[ok]/counts[ok],[.025,.975]).tolist()
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","existing","heldout"))
    cmd=parser.parse_args().action
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=OUT/"protocol.json"
    if cmd=="freeze" and not cfg.exists():
        fg.write_json(cfg,plan())
    assert fg.load(cfg)==plan()
    if cmd!="freeze":
        torch.set_num_threads(1)
        with threadpool_limits(1):
            models=[row for row in fitted_models("Dept2") if row[0]=="mixed_offset"]
            names=([(n,False) for n in held.NAMES] if cmd=="heldout" else
                   [(n,False) for n in (*LISTS,"gcc","ffmpeg-devel")]+[(n,True) for n in LISTS])
            result={n+("-2019" if late else ""):run(n,late,models) for n,late in names}
            fg.write_json(OUT/(cmd+".json"),result)
