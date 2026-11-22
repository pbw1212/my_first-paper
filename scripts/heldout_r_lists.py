"""Frozen cross-community evaluation of source-prior transfer on two R mailing lists."""
import argparse
import calendar
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from historical_thread_roles import ARCHIVE_MONTHS, TARGET_MONTHS
from r_archive_headers import prepare
from public_thread_transfer import DAY, marked_thread, thread_forest
from real_trained_mixtures import rank
from train_source_mixtures import (ARMS, OUT as FIT, PROFILES, SEEDS, PublicWindow, activity,
                                  correlated_prior, draw_events, fitted_models, law_arrays)

OUT = FIT.parent/"heldout_r_lists"
NAMES = ("r-devel","r-package-devel")
ATTEMPTS = 800
SEED = 17628179
KEYS = [f"{y}-{m:02d}" for y,m in TARGET_MONTHS]


def urls(name):
    if name=="r-devel":
        files=[f"{y}-{calendar.month_name[m]}" for y,m in ARCHIVE_MONTHS]
    else:
        files=sorted({f"{y}q{(m-1)//3+1}" for y,m in ARCHIVE_MONTHS})
    return [f"https://stat.ethz.ch/pipermail/{name}/{f}.txt.gz" for f in files]


def lock():
    base=OUT.parents[1]
    paths=[Path(__file__).resolve(),FIT/"protocol.json",
           *(FIT/"Dept2/fit").glob("*.pt"),*(FIT/"Dept2/fit").glob("*.npz")]
    paths.extend(base/"scripts"/(n+".py") for n in ("real_trained_mixtures","train_source_mixtures",
                  "source_prior_shift","blind_source_prior","public_thread_archive","public_thread_transfer",
                  "build_dept3_pilot","distilled_source_pairs","five_frame_pairs","historical_thread_roles","r_archive_headers"))
    first=fg.load(OUT/"protocol.json")
    assert first["seed"]==SEED and first["attempts"]==ATTEMPTS and first["months"]==KEYS and first["profiles"]==PROFILES
    assert first["communities"]=={n:urls(n) for n in NAMES}
    return dict(previous_protocol=fg.digest(OUT/"protocol.json"),amendment="v1 parsed zero of 5760 messages; match opaque archive sender prefix; same compressed input and evaluation settings",
                communities={n:urls(n) for n in NAMES},months=KEYS,seed=SEED,attempts=ATTEMPTS,
                trained_on="Dept2 only; no target tuning",arms=list(ARMS),seeds=list(SEEDS),profiles=PROFILES,
                primary="mixed_offset minus mixed_plain for mostly_active, by community, both-in-12",
                secondary="mixed_offset minus uniform_offset; all four generator profiles; real early speakers versus volume",
                population="synthetic on each month's first 14 days; real roots over entire month with 14 days follow-up",
                history="98 days resolved nonself replies until month start; identical roster and features for real and simulated events",
                interval="paired circular 3-calendar-month blocks and five training seeds, 10000 replicates",
                hashes={p.relative_to(base).as_posix():fg.digest(p) for p in paths})


def fetch(name):
    folder=OUT/name/"headers_v2"
    folder.mkdir(parents=True,exist_ok=True)
    originals={r["url"]:r["sha256"] for r in fg.load(folder.parent/"archives.json")}
    def part(url):
        key=url.rsplit("/",1)[1].split(".")[0]
        dest=folder/(key+".jsonl.gz")
        manifest=folder/(key+".json")
        if manifest.exists() and dest.exists():
            old=fg.load(manifest)
            assert old["url"]==url and old["headers_sha256"]==fg.digest(dest)
            return old
        rows,meta=prepare((name,url,key,originals[url]))
        obj="".join(json.dumps(row,separators=(",",":"))+"\n" for row in rows).encode()
        tmp=dest.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(obj,mtime=0))
        tmp.replace(dest)
        meta["headers_sha256"]=fg.digest(dest)
        fg.write_json(manifest,meta)
        return meta
    with ThreadPoolExecutor(3) as pool:
        manifest=list(pool.map(part,urls(name)))
    fg.write_json(folder.parent/"archives_v2.json",manifest)


def messages(name):
    folder=OUT/name
    msg={}
    conflicts=set()
    manifest=fg.load(folder/"archives_v2.json")
    assert [r["url"] for r in manifest]==urls(name)
    for row in manifest:
        key=row["url"].rsplit("/",1)[1].split(".")[0]
        p=folder/"headers_v2"/(key+".jsonl.gz")
        assert fg.digest(p)==row["headers_sha256"]
        with gzip.open(p,"rt",encoding="utf8") as stream:
            for mid,*rest in map(json.loads,stream):
                value=tuple(rest)
                if mid in msg and msg[mid]!=value:
                    conflicts.add(mid)
                msg[mid]=value
    return {k:v for k,v in msg.items() if k not in conflicts}


def calendar_windows(msg,name):
    forest,contacts=thread_forest(msg)
    for j,(y,m) in enumerate(TARGET_MONTHS):
        start=int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        end=int(datetime(y+(m==12),m%12+1,1,tzinfo=timezone.utc).timestamp())
        history=[(t,u,v) for t,u,v in contacts if start-98*DAY<=t<start]
        counts=Counter((u,v) for _,u,v in history)
        nodes=sorted({n for edge in counts for n in edge})
        roster=set(nodes)
        g=dict(nodes=nodes,window=KEYS[j],start=start,edges=[(*uv,c) for uv,c in sorted(counts.items())])
        buckets=defaultdict(list)
        for t,u,v in contacts:
            if start<=t<start+14*DAY and u in roster and v in roster:
                buckets[t].append((u,v))
        volume=Counter(a for a,t,_,_ in msg.values() if start-28*DAY<=t<start)
        roots=[r for r in sorted(forest) if start<=msg[r][1]<end]
        real=[]
        for root in roots:
            _,payload=marked_thread(root,forest[root],msg)
            if payload:
                src,batches=payload
                obs=[[v for v in grp if v in roster] for grp in batches]
                real.append((dict(id=root,window=KEYS[j],k=2,observed=[grp for grp in obs if grp]),src))
        yield g,history,sorted(buckets.items()),volume,real,len(roots)


def real_or_sim(e,src,profile,hist,models,w,ix,means,volume):
    seen={v for grp in e["observed"] for v in grp}
    assert seen.isdisjoint(src)
    ready=len(seen)>=3 and len(ix)-len(seen)>=12
    lists=rank(hist,e,models,w,ix,means,volume) if ready else {}
    assert all(len(set(v))==12 and set(v)<=ix.keys()-seen for v in lists.values())
    return dict(id=e["id"],window=e["window"],profile=profile,ready=ready,
                reachable=set(src)<=ix.keys(),sources=src,
                hits={k:int(set(src)<=set(v)) for k,v in lists.items()})


def run(name):
    msg=messages(name)
    models=fitted_models("Dept2")
    rows,flow=[],[]
    for j,(g,history,stream,volume,real,roots) in enumerate(calendar_windows(msg,name)):
        hist=PublicWindow(g,history)
        ix={n:i for i,n in enumerate(g["nodes"])}
        w=activity(g,history)
        means=law_arrays(w)[3]
        edge=correlated_prior(history,g)
        for e,src in real:
            rows.append(real_or_sim(e,src,"real",hist,models,w,ix,means,volume))
        count={}
        for pi,p in enumerate(PROFILES):
            rng=np.random.default_rng([SEED,NAMES.index(name),j,pi])
            n=0
            for e,src,_,_ in draw_events(g,stream,w,edge,rng,p,ATTEMPTS,False):
                rows.append(real_or_sim(e,src,p,hist,models,w,ix,means,volume))
                n+=1
            count[p]=n
        flow.append(dict(window=KEYS[j],roster=len(ix),history_contacts=len(history),simulation_contacts=sum(len(b) for _,b in stream),
                         real_roots=roots,real_eligible=len(real),simulated_eligible=count))
    assert len({r["id"] for r in rows})==len(rows)
    fg.save_rows(OUT/name/"events.jsonl",rows)
    fg.write_json(OUT/name/"flow.json",flow)


def summary(rows):
    if not rows:
        return dict(n=0)
    vol=np.array([r["hits"].get("volume",0) for r in rows])[:,None]
    hits={arm:np.array([[r["hits"].get(f"{arm}/{s}",0) for s in SEEDS] for r in rows]) for arm in ARMS}
    at=np.array([KEYS.index(r["window"]) for r in rows])
    ct=np.bincount(at,minlength=len(KEYS))
    rng=np.random.default_rng(541711)
    blocks=((rng.integers(24,size=(10000,8,1))+np.arange(3))%24).reshape(-1,24)
    seeds=rng.integers(5,size=(10000,1,5))
    denom=ct[blocks].sum(1)
    ok=denom>0
    def diff(data):
        per=np.array([data[at==j].sum(0) for j in range(24)])
        b=per[blocks[:,:,None],seeds].sum(1).mean(1)
        return dict(pp=100*data.mean(),ci95=np.quantile(100*b[ok]/denom[ok],[.025,.975]).tolist(),
                    months_with_events=int(sum(ct>0)),valid_draws=int(ok.sum()))
    return dict(n=len(rows),scored=sum(r["ready"] for r in rows),scoreable_and_reachable=sum(r["ready"] and r["reachable"] for r in rows),
                volume_pct=100*vol.mean(),both_pct={k:100*v.mean() for k,v in hits.items()},
                seed_hits={k:v.sum(0).tolist() for k,v in hits.items()},volume_hits=int(vol.sum()),window_counts=ct.tolist(),
                offset_vs_plain=diff(hits["mixed_offset"]-hits["mixed_plain"]),
                offset_vs_simple=diff(hits["mixed_offset"]-hits["uniform_offset"]),
                offset_vs_volume=diff(hits["mixed_offset"]-vol))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["freeze","fetch","test","report"])
    parser.add_argument("name",nargs="?",choices=NAMES)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    cfg=OUT/"protocol_v2.json"
    if args.action=="freeze" and not cfg.exists():
        fg.write_json(cfg,lock())
    assert fg.load(cfg)==lock()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.action=="fetch":
            fetch(args.name)
        elif args.action=="test":
            run(args.name)
        elif args.action=="report":
            res={}
            for n in NAMES:
                rows=fg.read_rows(OUT/n/"events.jsonl")
                res[n]={p:summary([r for r in rows if r["profile"]==p]) for p in [*PROFILES,"real"]}
            fg.write_json(OUT/"summary.json",res)
