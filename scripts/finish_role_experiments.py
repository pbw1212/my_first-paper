"""Held-out community replication and fixed-coefficient evidence removal for early speakers."""
import argparse
import calendar
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import io
import json
import mailbox
from email.parser import BytesHeaderParser
from pathlib import Path
import tempfile
import hashlib

import numpy as np
import torch
from scipy.special import softmax
from threadpoolctl import threadpool_limits

import member_role_evidence as re
import role_evidence_time_holdout as future
from check_early_only_discussions import reference
from public_thread_archive import download,parse_message,token
from real_trained_mixtures import rank
from role_evidence_archive_compat import TextHeaders

fg=re.fg
task=re.task
OUT=re.OUT/"closure"
SAMBA="samba-technical-2020"
COHORTS=(*task.NAMES,"libc-alpha-2024",SAMBA)
MONTHS=[(2019,m) for m in range(9,13)]+[(2020,m) for m in range(1,13)]+[(2021,1)]
BASE="https://lists.samba.org/archive/samba-technical/"


def spec():
    paths=[Path(__file__),Path(re.__file__),re.OUT/"protocol.json",future.OUT/"protocol.json",
           task.r.OUT/"protocol_v2.json",Path(reference.__code__.co_filename)]
    return dict(primary="New community: role_suffix minus refitted reply-only model; same both-in-12 endpoint on all eligible roots",
                mechanism="Compare role_suffix with the same coefficients and bias, setting its last two input features to zero",
                controls=["role frequency model","28-day post count","frozen Dept2 mixed_offset and uniform_offset"],
                new_community=SAMBA,urls=[BASE+f"{y}-{calendar.month_name[m]}.txt.gz" for y,m in MONTHS],
                population="2020 roots within verified reply trees, 14-day follow-up, first two distinct early addresses hidden; complete later suffix",
                prior_cohorts=list(COHORTS[:-1]),source_training="Python 2017-18 coefficients; early models and all other settings fixed",
                inference="retain shared abstention; no adjustment or model selection on new test labels; report paired three-month block intervals",
                archive_choice="LLVM January 2020 diagnostic: 973 of 974 header fragments used a shared list sender lacking Reply-To and X-Original-From. Samba January metadata had 28 distinct addresses among 121 fragments and no shared list sender. No hit rates were used in choosing the new community.",
                hashes={p.resolve().relative_to(task.r.OUT.parents[1]).as_posix():fg.digest(p) for p in paths})


def fetch(url):
    label=url.rsplit("/",1)[1].split(".")[0]
    dst=OUT/"headers"/(label+".jsonl.gz")
    meta=dst.with_suffix(".json")
    if meta.exists():
        entry=fg.load(meta)
        assert entry["url"]==url and entry["header_sha256"]==fg.digest(dst)
        return entry
    raw=download(url)
    data=gzip.decompress(raw)
    assert data.startswith(b"From ")
    rows=[]
    parser=BytesHeaderParser()
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/"mailbox"
        p.write_bytes(data)
        box=mailbox.mbox(p,create=False,factory=lambda fp:parser.parse(io.BytesIO(fp.read()),headersonly=True))
        try:
            total=len(box)
            for m in box:
                rec=parse_message(TextHeaders(m))
                if rec is not None:
                    rows.append(rec)
        finally:
            box.close()
    tmp=dst.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress("".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows).encode(),mtime=0))
    tmp.replace(dst)
    entry=dict(url=url,sha256=hashlib.sha256(raw).hexdigest(),header_sha256=fg.digest(dst),messages=total,parsed=len(rows))
    fg.write_json(meta,entry)
    return entry


def samba_messages():
    rows=fg.load(OUT/"archives.json")
    assert [r["url"] for r in rows]==spec()["urls"]
    msg={}
    conflicts=set()
    alias=token("samba-technical@lists.samba.org")[:20]
    for entry in rows:
        path=OUT/"headers"/(entry["url"].rsplit("/",1)[1].split(".")[0]+".jsonl.gz")
        assert entry["header_sha256"]==fg.digest(path)
        with gzip.open(path,"rt",encoding="utf8") as stream:
            for mid,*vals in map(json.loads,stream):
                if vals[0]==alias:
                    continue
                rest=tuple(vals)
                if mid in msg and msg[mid]!=rest:
                    conflicts.add(mid)
                msg[mid]=rest
    return {k:v for k,v in msg.items() if k not in conflicts}


def cohort(name):
    if name==SAMBA:
        return samba_messages(),[(2020,m) for m in range(1,13)],None
    if name=="libc-alpha-2024":
        return future.load_messages(),[(2024,m) for m in range(1,13)],fg.read_rows(future.OUT/"events.jsonl")
    msg=task.libc.messages() if name==task.libc.NAME else task.r.messages(name)
    return msg,task.r.TARGET_MONTHS,fg.read_rows(re.OUT/(name+".jsonl"))


def run(name):
    msg,months,archived=cohort(name)
    cfg=fg.load(re.OUT/"protocol.json")["models"]
    old={z["id"]:z for z in archived} if archived is not None else None
    forest,contacts=re.thread_forest(msg)
    docs=[]
    roots=defaultdict(list)
    for root,thread in sorted(forest.items()):
        first,t=msg[root][:2]
        docs.append(re.historical_document(root,thread,t,first))
        roots[datetime.fromtimestamp(t,timezone.utc).strftime("%Y-%m")].append(root)
    nets=[v for v in task.r.fitted_models("Dept2") if v[0] in ("mixed_offset","uniform_offset")] if name==SAMBA else []
    rows=[]
    flow=[]
    for y,m in months:
        key=f"{y}-{m:02d}"
        start=int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        h=re.History(start,msg,contacts,docs)
        before=len(rows)
        if nets:
            past=[z for z in contacts if start-98*re.DAY<=z[0]<start]
            edges=Counter((a,b) for _,a,b in past)
            g=dict(start=start,window=key,nodes=h.nodes,edges=[(*e,c) for e,c in sorted(edges.items())])
            neural=task.r.PublicWindow(g,past)
            w=task.r.activity(g,past)
            means=task.r.law_arrays(w)[3]
            vol=Counter(v for v,t,_,_ in msg.values() if start-28*re.DAY<=t<start)
        for root in roots[key]:
            payload=task.prefix(root,forest[root],msg)
            assert payload==reference(root,forest[root],msg)
            if not payload:
                continue
            gold,obs=payload
            nodes,x=h.design(obs)
            z=dict(id=root,window=key,reachable=set(gold)<=set(nodes),ready=x is not None,roles={})
            if x is not None:
                tie=fg.tie_order(root,len(nodes))
                def code(chosen):
                    sel={nodes[i] for i in chosen}
                    assert len(sel)==12
                    return int(gold[0] in sel)+2*int(gold[1] in sel)
                z["roles"]["volume"]=code(fg.sorted_nodes(x[0][:,0],tie)[:12])
                for arm,dim in [*re.WIDTH.items(),("neutral",6)]:
                    family="role_suffix" if arm=="neutral" else arm
                    p,q=[softmax(x[j][:,:dim]@np.asarray(cfg[family][j]["theta"])[:dim]) for j in (0,1)]
                    ids=re.choose(p,q,tie)[0] if len(nodes)>12 else np.arange(12)
                    z["roles"][arm]=code(ids)
                if nets:
                    bs=[[v for v in b if v in h.ix] for b in obs]
                    lists=rank(neural,dict(id=root,window=key,k=2,observed=[b for b in bs if b]),nets,w,h.ix,means,vol)
                    assert int(set(gold)<=set(lists["volume"]))==(z["roles"]["volume"]==3)
                    z["neural"]={k:int(set(gold)<=set(v)) for k,v in lists.items() if k!="volume"}
            if old is not None:
                src=old[root]
                saved=src["source_roles"] if name=="libc-alpha-2024" else src["roles"]
                assert all(z["roles"].get(k)==v for k,v in saved.items())
                assert bool(saved)==z["ready"]
            rows.append(z)
        flow.append(dict(month=key,verified_roots=len(roots[key]),eligible=len(rows)-before))
    assert len(rows)==len({z["id"] for z in rows}) and (old is None or len(rows)==len(old))
    fg.save_rows(OUT/(name+".jsonl"),rows)
    fg.write_json(OUT/(name+"-flow.json"),flow)


def summarize(name):
    rows=fg.read_rows(OUT/(name+".jsonl"))
    keys=[r["month"] for r in fg.load(OUT/(name+"-flow.json"))]
    codes={m:np.array([z["roles"].get(m,0) for z in rows]) for m in ("volume","roles","reply","neutral","role_suffix")}
    hit={m:np.repeat((c==3).astype(int)[:,None],5,axis=1) for m,c in codes.items()}
    if name==SAMBA:
        for m in ("uniform_offset","mixed_offset"):
            hit[m]=np.array([[r.get("neural",{}).get(f"{m}/{s}",0) for s in task.r.SEEDS] for r in rows])
    at=np.array([keys.index(z["window"]) for z in rows])
    n=len(keys)
    ct=np.bincount(at,minlength=n)
    rng=np.random.default_rng(150892)
    bb=((rng.integers(n,size=(10000,(n+2)//3,1))+np.arange(3))%n).reshape(10000,-1)[:,:n]
    seeds=rng.integers(5,size=(10000,1,5))
    den=ct[bb].sum(1)
    ok=den>0
    out=dict(n=len(rows),scored=sum(z["ready"] for z in rows),scored_reachable=sum(z["ready"] and z["reachable"] for z in rows),
             hit_counts={m:h.sum(0).tolist() for m,h in hit.items()},both_pct={m:100*h.mean() for m,h in hit.items()},contrasts={})
    for a,b in [("role_suffix",m) for m in hit if m!="role_suffix"]+[("roles","volume"),("neutral","reply")]:
        delta=hit[a]-hit[b]
        table=np.array([delta[at==j].sum(0) for j in range(n)])
        d=table[bb[:,:,None],seeds].sum(1).mean(1)
        out["contrasts"][a+"-"+b]=dict(pp=100*delta.mean(),ci95_pp=np.quantile(100*d[ok]/den[ok],[.025,.975]).tolist(),
                gained=(delta>0).sum(0).tolist(),lost=(delta<0).sum(0).tolist())
    assert np.array_equal(hit["role_suffix"]-hit["reply"],(hit["role_suffix"]-hit["neutral"])+(hit["neutral"]-hit["reply"]))
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","fetch","previous","new","report"))
    cmd=parser.parse_args().action
    OUT.mkdir(exist_ok=True)
    lock=OUT/"protocol.json"
    if cmd=="freeze" and not lock.exists():
        fg.write_json(lock,spec())
    assert fg.load(lock)==spec()
    if cmd=="fetch":
        (OUT/"headers").mkdir(exist_ok=True)
        with ThreadPoolExecutor(3) as pool:
            fg.write_json(OUT/"archives.json",list(pool.map(fetch,spec()["urls"])))
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if cmd in ("previous","new"):
            for n in (COHORTS[:-1] if cmd=="previous" else [SAMBA]):
                run(n)
        elif cmd=="report":
            fg.write_json(OUT/"summary.json",{n:summarize(n) for n in COHORTS})
