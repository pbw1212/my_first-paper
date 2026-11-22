"""Pre-lock the Python-trained role model before collecting a later GNU libc archive."""
import argparse
import calendar
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from threadpoolctl import threadpool_limits

import member_role_evidence as re
from check_early_only_discussions import reference
from public_thread_archive import prepare_month
from real_trained_mixtures import rank

fg=re.fg
task=re.task
OUT=re.OUT/"libc_2024"
MONTHS=[(2023,m) for m in range(9,13)]+[(2024,m) for m in range(1,13)]+[(2025,1)]


def config():
    paths=[Path(__file__),re.OUT/"protocol.json",Path(re.__file__),task.OUT/"protocol.json"]
    return dict(test="GNU libc January to December 2024; real early-only task; previous period 2021-22",
                urls=[task.libc.BASE+f"{y}-{calendar.month_name[m]}.txt.gz" for y,m in MONTHS],
                primary="role_suffix minus reply on all eligible discussions; secondaries vs roles, post volume and frozen Dept2 models",
                training="Python 2017-18 coefficients frozen; month-start history only; no target calibration",
                intervals="10000 paired circular blocks of three of the twelve months; paired seeds for neural comparators",
                hashes={p.resolve().relative_to(task.r.OUT.parents[1]).as_posix():fg.digest(p) for p in paths})


def fetch(url):
    label=url.rsplit("/",1)[1].split(".")[0]
    dst=OUT/"headers"/(label+".jsonl.gz")
    meta=dst.with_suffix(".json")
    if meta.exists():
        entry=fg.load(meta)
        assert entry["url"]==url and entry["headers_sha256"]==fg.digest(dst)
        return entry
    _,_,rows,entry=prepare_month(("libc-alpha",url,label))
    assert rows or not entry["messages"]
    tmp=dst.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress("".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows).encode(),mtime=0))
    tmp.replace(dst)
    entry["headers_sha256"]=fg.digest(dst)
    fg.write_json(meta,entry)
    return entry


def load_messages():
    records=fg.load(OUT/"archives.json")
    assert [v["url"] for v in records]==config()["urls"]
    msg={}
    conflict=set()
    for rec in records:
        path=OUT/"headers"/(rec["url"].rsplit("/",1)[1].split(".")[0]+".jsonl.gz")
        assert fg.digest(path)==rec["headers_sha256"]
        with gzip.open(path,"rt",encoding="utf8") as stream:
            for root,*tail in map(json.loads,stream):
                data=tuple(tail)
                if root in msg and msg[root]!=data:
                    conflict.add(root)
                msg[root]=data
    return {k:v for k,v in msg.items() if k not in conflict}


def forecast(msg):
    forest,contacts=re.thread_forest(msg)
    docs=[]
    roots=defaultdict(list)
    for root,thread in sorted(forest.items()):
        who,t=msg[root][:2]
        docs.append(re.historical_document(root,thread,t,who))
        roots[datetime.fromtimestamp(t,timezone.utc).strftime("%Y-%m")].append(root)
    cfg=fg.load(re.OUT/"protocol.json")["models"]
    group=task.r.fitted_models("Dept2")
    rows=[]
    flow=[]
    for m in range(1,13):
        key=f"2024-{m:02d}"
        t=int(datetime(2024,m,1,tzinfo=timezone.utc).timestamp())
        history=re.History(t,msg,contacts,docs)
        past=[i for i in contacts if t-98*re.DAY<=i[0]<t]
        edges=Counter((a,b) for _,a,b in past)
        g=dict(start=t,window=key,nodes=history.nodes,edges=[(*e,c) for e,c in sorted(edges.items())])
        hist=task.r.PublicWindow(g,past)
        w=task.r.activity(g,past)
        means=task.r.law_arrays(w)[3]
        vol=Counter(a for a,t0,_,_ in msg.values() if t-28*re.DAY<=t0<t)
        n0=len(rows)
        for root in roots[key]:
            payload=task.prefix(root,forest[root],msg)
            assert payload==reference(root,forest[root],msg)
            if not payload:
                continue
            gold,batches=payload
            obs=[[v for v in b if v in history.ix] for b in batches]
            obs=[b for b in obs if b]
            nodes,x=history.design(batches)
            out=dict(id=root,window=key,ready=x is not None,reachable=set(gold)<=set(nodes),hits={},source_roles={})
            if x is not None:
                tie=fg.tie_order(root,len(nodes))
                neural=rank(hist,dict(id=root,window=key,observed=obs,k=2),group,w,history.ix,means,vol)
                out["hits"]={k:int(set(gold)<=set(ls)) for k,ls in neural.items()}
                assert neural["volume"]==[nodes[i] for i in fg.sorted_nodes(x[0][:,0],tie)[:12]]
                for arm,dim in re.WIDTH.items():
                    p,q=[softmax(x[k][:,:dim]@cfg[arm][k]["theta"]) for k in (0,1)]
                    ids=re.choose(p,q,tie)[0] if len(nodes)>12 else np.arange(len(nodes))
                    sel={nodes[i] for i in ids}
                    assert len(sel)==12
                    out["source_roles"][arm]=int(gold[0] in sel)+2*int(gold[1] in sel)
                    out["hits"][arm]=int(set(gold)<=sel)
            rows.append(out)
        flow.append(dict(month=key,roots=len(roots[key]),eligible=len(rows)-n0))
    assert len(rows)==len({z["id"] for z in rows})
    return rows,flow


def summarize(rows):
    keys=[f"2024-{m:02d}" for m in range(1,13)]
    at=np.array([keys.index(z["window"]) for z in rows])
    counts=np.bincount(at,minlength=12)
    rng=np.random.default_rng(3811)
    blocks=((rng.integers(12,size=(10000,4,1))+np.arange(3))%12).reshape(10000,12)
    seeds=rng.integers(5,size=(10000,1,5))
    den=counts[blocks].sum(1)
    ok=den>0
    scores={}
    results=dict(n=len(rows),scored=sum(z["ready"] for z in rows),scored_reachable=sum(z["ready"] and z["reachable"] for z in rows),
                 both_pct={},contrasts={})
    for m in ["volume",*re.WIDTH,*task.r.ARMS]:
        scores[m]=np.array([[z["hits"].get(f"{m}/{s}" if m in task.r.ARMS else m,0) for s in task.r.SEEDS] for z in rows])
        results["both_pct"][m]=100*scores[m].mean()
    for m in ("roles","reply","volume","mixed_offset","uniform_offset"):
        diff=scores["role_suffix"]-scores[m]
        table=np.array([diff[at==i].sum(0) for i in range(12)])
        change=table[blocks[:,:,None],seeds].sum(1).mean(1)
        results["contrasts"][m]=dict(pp=100*diff.mean(),ci95_pp=np.quantile(100*change[ok]/den[ok],[.025,.975]).tolist(),
                                     gained=(diff>0).sum(0).tolist(),lost=(diff<0).sum(0).tolist())
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","fetch","test"))
    cmd=parser.parse_args().action
    OUT.mkdir(exist_ok=True)
    path=OUT/"protocol.json"
    if cmd=="freeze" and not path.exists():
        fg.write_json(path,config())
    assert fg.load(path)==config()
    if cmd=="fetch":
        (OUT/"headers").mkdir(exist_ok=True)
        with ThreadPoolExecutor(3) as pool:
            fg.write_json(OUT/"archives.json",list(pool.map(fetch,config()["urls"])))
    if cmd=="test":
        torch.set_num_threads(1)
        with threadpool_limits(1):
            rows,flow=forecast(load_messages())
            fg.save_rows(OUT/"events.jsonl",rows)
            fg.write_json(OUT/"flow.json",flow)
            fg.write_json(OUT/"summary.json",summarize(rows))
