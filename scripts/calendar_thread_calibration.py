"""Calibrate a small historical ranking model then evaluate it on subsequent real threads."""
import argparse
import calendar
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize

import fgssi_suffix as fg
from public_thread_archive import LISTS,OUT,ROOT,download,prepare_month
from public_thread_transfer import DAY,MONTHS,marked_thread,thread_forest

DEST = OUT/"calendar_followup"
TEST = [(2019,m) for m in (2,3,4)]
VARIANTS = {"prior":2,"relations":4}
PENALTIES = (.01,.1,1.)


def extra():
    tasks = []
    for name in LISTS:
        base = f"https://mail.python.org/pipermail/{name}/"
        page = download(base).decode()
        links = set(re.findall(r'href="([^"]+)"',page,re.I))
        for m in range(2,6):
            key = f"2019-{calendar.month_name[m]}"
            file = next(v for v in (key+".txt",key+".txt.gz") if v in links)
            tasks.append((name,base+file,key))
    result = defaultdict(list)
    audit = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for n,k,rows,info in pool.map(prepare_month,tasks):
            result[n].extend(rows)
            audit[n+"/"+k] = info
    for n,rows in result.items():
        content = "".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows).encode()
        (DEST/(n+".gz")).write_bytes(gzip.compress(content,mtime=0))
    fg.write_json(DEST/"archives.json",audit)


def examples(name,testing):
    paths = [OUT/(name+"-headers.jsonl.gz")]
    if testing:
        paths.append(DEST/(name+".gz"))
    msg = {}
    conflicts = set()
    for p in paths:
        with gzip.open(p,"rt",encoding="utf8") as f:
            for mid,a,t,parent,reply in map(json.loads,f):
                value = (a,t,parent,reply)
                if mid in msg and msg[mid]!=value:
                    conflicts.add(mid)
                msg[mid] = value
    for mid in conflicts:
        del msg[mid]
    threads,edges = thread_forest(msg)
    rows = []
    for y,m in (TEST if testing else MONTHS):
        key = f"{y}-{m:02d}"
        start = int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        end = int(datetime(y+(m==12),m%12+1,1,tzinfo=timezone.utc).timestamp())
        past = [(a,b) for t,a,b in edges if start-98*DAY<=t<start]
        roster = {v for pair in past for v in pair}
        recent = Counter((a,b) for t,a,b in edges if start-28*DAY<=t<start)
        volume = [Counter(a for a,t,_,_ in msg.values() if start-d*DAY<=t<start) for d in (28,98)]
        for mid,thread in sorted(threads.items()):
            if not start<=msg[mid][1]<end:
                continue
            _,mark = marked_thread(mid,thread,msg)
            if mark is None:
                continue
            gold,batches = mark
            visible = {a for b in batches for a in b if a in roster}
            nodes = sorted(roster-visible)
            x = np.log1p([[volume[0][n],volume[1][n],sum(recent[n,v] for v in visible),
                           sum(recent[v,n] for v in visible)] for n in nodes])
            rows.append(dict(id=mid,window=key,list=name,sources=gold,nodes=nodes,x=x,
                             ready=len(visible)>=3 and len(nodes)>=12))
    return rows


def risk(theta,rows,penalty):
    loss = 0.
    grad = np.zeros_like(theta)
    for x,y in rows:
        z = x[:,:len(theta)]@theta
        maximum = z.max()
        w = np.exp(z-maximum)
        other = np.r_[0.,np.cumsum(w[:-1])] + np.r_[np.cumsum(w[:0:-1])[::-1],0.]
        mass = w*other
        normalizer = mass.sum()/2
        loss += 2*maximum+np.log(normalizer)-z[y].sum()
        grad += (mass/normalizer)@x[:,:len(theta)] - x[y,:len(theta)].sum(0)
    return loss/len(rows)+penalty*(theta@theta)/2,grad/len(rows)+penalty*theta


def training_data(rows):
    return [(r["x"],[r["nodes"].index(v) for v in r["sources"]]) for r in rows
            if r["ready"] and set(r["sources"])<=set(r["nodes"])]


def train():
    DEST.mkdir(parents=True,exist_ok=True)
    rows = [r for n in LISTS for r in examples(n,False)]
    # Check the re-extracted earlier cohorts against the frozen primary experiment.
    for n in LISTS:
        prior = {r["id"]:r for r in fg.read_rows(OUT/n/"truth.jsonl")}
        current = [r for r in rows if r["list"]==n]
        assert len(current)==len(prior)
        assert all(r["sources"]==prior[r["id"]]["sources"] and r["ready"]==prior[r["id"]]["scored"] for r in current)
    early = training_data([r for r in rows if r["window"]<"2018-07"])
    val = training_data([r for r in rows if r["window"]>="2018-07"])
    cfg = dict(train=len(early),validation=len(val),months=TEST,script=fg.digest(Path(__file__)),
               origin=fg.digest(OUT/"protocol.json"),inputs={n:fg.digest(OUT/(n+"-headers.jsonl.gz")) for n in LISTS},
               primary="relations versus raw 28-day volume; both in 12, include abstentions; per list",
               features=["log1p 28-day posts","log1p 98-day posts","log1p 28-day replies to visible","log1p 28-day replies from visible"],
               models={})
    for method,dim in VARIANTS.items():
        fits = []
        for reg in PENALTIES:
            opt = minimize(risk,np.zeros(dim),args=(early,reg),jac=True,method="L-BFGS-B",bounds=[(-6,6)]*dim)
            assert opt.success,opt.message
            value = risk(opt.x,val,0.)[0]
            fits.append(dict(penalty=reg,validation_nll=value,theta=opt.x.tolist()))
        cfg["models"][method] = min(fits,key=lambda r:r["validation_nll"])
        cfg["models"][method]["grid_nll"] = [r["validation_nll"] for r in fits]
    cfg = json.loads(json.dumps(cfg))
    path = DEST/"lock.json"
    if path.exists():
        assert fg.load(path)==cfg
    else:
        fg.write_json(path,cfg)


def test():
    cfg = fg.load(DEST/"lock.json")
    assert cfg["script"]==fg.digest(Path(__file__)) and cfg["origin"]==fg.digest(OUT/"protocol.json")
    assert all(fg.digest(OUT/(n+"-headers.jsonl.gz"))==h for n,h in cfg["inputs"].items())
    extra()
    results = {}
    for name in LISTS:
        records = []
        for r in examples(name,True):
            nodes,x = r["nodes"],r["x"]
            tie = fg.tie_order(r["id"],len(nodes))
            scores = {"volume":x[:,0]}
            for key,model in cfg["models"].items():
                theta = np.asarray(model["theta"])
                scores[key] = x[:,:len(theta)]@theta
            lists = {k:([nodes[i] for i in fg.sorted_nodes(v,tie)[:12]] if r["ready"] else []) for k,v in scores.items()}
            records.append(dict(id=r["id"],window=r["window"],sources=r["sources"],ready=r["ready"],lists=lists))
        fg.save_rows(DEST/(name+"-predictions.jsonl"),records)
        aggregate = {}
        for k in ("volume","prior","relations"):
            hits = [set(r["sources"])<=set(r["lists"][k]) for r in records]
            aggregate[k] = dict(hits=sum(hits),month_hits=[sum(h for h,r in zip(hits,records) if r["window"]==f"{y}-{m:02d}") for y,m in TEST])
        diff = Counter()
        for r in records:
            g = set(r["sources"])
            a,b = [g<=set(r["lists"][k]) for k in ("relations","volume")]
            diff["gained"] += a and not b
            diff["lost"] += b and not a
        results[name] = dict(n=len(records),scored=sum(r["ready"] for r in records),methods=aggregate,paired=dict(diff),
                             month_counts=[sum(r["window"]==f"{y}-{m:02d}" for r in records) for y,m in TEST])
    fg.write_json(DEST/"summary.json",results)
    print(results)


if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("step",choices=("train","test"))
    {"train":train,"test":test}[arg.parse_args().step]()
