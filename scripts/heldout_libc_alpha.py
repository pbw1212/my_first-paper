"""Supplement the sparse R holdouts with an unchanged decision task on GNU libc."""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import gzip

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
import heldout_r_lists as r
from check_heldout_r_lists import audit
from historical_thread_roles import ARCHIVE_MONTHS
from public_thread_archive import prepare_month

OUT=r.OUT.parent/"heldout_libc_alpha"
NAME="libc-alpha"
SEED=21326541
BASE="https://sourceware.org/pipermail/libc-alpha/"


def spec():
    return dict(community=NAME,urls=[BASE+f"{y}-{calendar.month_name[m]}.txt.gz" for y,m in ARCHIVE_MONTHS],
                prior_protocol=fg.digest(r.OUT/"protocol_v2.json"),training="same frozen Dept2 weights",
                reason="add a separate community after the two R lists yielded only 7 and 71 mostly_active events",
                rules="identical 24 months, eligibility, history, profiles, draws, role proxy, four arms, shortlist and block bootstrap",
                primary="mostly_active mixed_offset minus mixed_plain",seed=SEED,
                hashes={p.name:fg.digest(p) for p in (Path(__file__),Path(r.__file__),Path(__file__).with_name("check_heldout_r_lists.py"))})


def fetch(url):
    key=url.rsplit("/",1)[1].split(".")[0]
    path=OUT/"headers"/(key+".jsonl.gz")
    meta=path.with_suffix(".json")
    if path.exists() and meta.exists():
        cfg=fg.load(meta)
        assert cfg["url"]==url and cfg["headers_sha256"]==fg.digest(path)
        return cfg
    _,_,rows,cfg=prepare_month((NAME,url,key))
    assert rows or not cfg["messages"]
    tmp=path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress("".join(json.dumps(v,separators=(",",":"))+"\n" for v in rows).encode(),mtime=0))
    tmp.replace(path)
    cfg["headers_sha256"]=fg.digest(path)
    fg.write_json(meta,cfg)
    return cfg


def messages():
    msg={}
    conflicts=set()
    manifest=fg.load(OUT/"archives.json")
    assert [v["url"] for v in manifest]==spec()["urls"]
    for record in manifest:
        key=record["url"].rsplit("/",1)[1].split(".")[0]
        path=OUT/"headers"/(key+".jsonl.gz")
        assert fg.digest(path)==record["headers_sha256"]
        with gzip.open(path,"rt",encoding="utf8") as stream:
            for mid,*values in map(json.loads,stream):
                v=tuple(values)
                if mid in msg and msg[mid]!=v:
                    conflicts.add(mid)
                msg[mid]=v
    return {k:v for k,v in msg.items() if k not in conflicts}


def evaluate():
    msg=messages()
    models=r.fitted_models("Dept2")
    rows,flow=[],[]
    for j,(g,history,stream,vol,real,roots) in enumerate(r.calendar_windows(msg,NAME)):
        hist=r.PublicWindow(g,history)
        ix={v:i for i,v in enumerate(g["nodes"])}
        w=r.activity(g,history)
        means=r.law_arrays(w)[3]
        link=r.correlated_prior(history,g)
        for e,src in real:
            rows.append(r.real_or_sim(e,src,"real",hist,models,w,ix,means,vol))
        counts={}
        for pi,p in enumerate(r.PROFILES):
            rng=np.random.default_rng([SEED,0,j,pi])
            before=len(rows)
            for e,src,_,_ in r.draw_events(g,stream,w,link,rng,p,r.ATTEMPTS,False):
                rows.append(r.real_or_sim(e,src,p,hist,models,w,ix,means,vol))
            counts[p]=len(rows)-before
        flow.append(dict(window=g["window"],roster=len(ix),real_roots=roots,real_eligible=len(real),simulated_eligible=counts))
    assert len({v["id"] for v in rows})==len(rows)
    fg.save_rows(OUT/"events.jsonl",rows)
    fg.write_json(OUT/"flow.json",flow)
    scores={p:r.summary([v for v in rows if v["profile"]==p]) for p in [*r.PROFILES,"real"]}
    fg.write_json(OUT/"summary.json",scores)
    fg.write_json(OUT/"verification.json",audit(NAME,models,msg,OUT,scores,SEED,0))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","fetch","test"))
    cmd=parser.parse_args().action
    OUT.mkdir(exist_ok=True)
    cfg=OUT/"protocol.json"
    if cmd=="freeze" and not cfg.exists():
        fg.write_json(cfg,spec())
    assert fg.load(cfg)==spec() and fg.load(r.OUT/"protocol_v2.json")==r.lock()
    if cmd=="fetch":
        (OUT/"headers").mkdir(exist_ok=True)
        with ThreadPoolExecutor(3) as pool:
            fg.write_json(OUT/"archives.json",list(pool.map(fetch,spec()["urls"])))
    elif cmd=="test":
        torch.set_num_threads(1)
        with threadpool_limits(1):
            evaluate()
