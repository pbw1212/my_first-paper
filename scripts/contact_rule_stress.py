"""Paired source draws under four transmission rules with frozen source models."""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from build_dept3_pilot import mask, simulate
from distilled_source_pairs import PublicWindow
from evaluate_factorial_history import fitted
from factorial_fgssi import SEEDS,weights
from factorial_history import StaticWindow,folder_for
from five_frame_pairs import frames
from history_components import shortlist
from recent_member_evidence import OUT
from report_history_components import summarize
from score_dept3_candidates import roles

RULES = {"low":(.15,False),"original":(.30,False),"high":(.50,False),"once":(.30,True)}
ATTEMPTS = 200


def replay(stream,sources,seed,prob,once):
    rng = random.Random(seed)
    active = set(sources)
    used = set()
    trace = []
    for t,batch in stream:
        parents = defaultdict(list)
        for u,v in batch:
            if u not in active or v in active or (once and (u,v) in used):
                continue
            if once:
                used.add((u,v))
            if rng.random()<prob:
                parents[v].append(u)
        if parents:
            trace.append(dict(t=t,joins=[[v,sorted(pp)] for v,pp in sorted(parents.items())]))
            active.update(parents)
    return trace


def generate(g,stream,rng):
    events, labels = [], {}
    for i in range(ATTEMPTS):
        src = sorted(rng.sample(g["nodes"],2))
        seed = rng.getrandbits(63)
        base = f"{g['window']}-{i}"
        for rule,(p,once) in RULES.items():
            tr = replay(stream,src,seed,p,once)
            if rule=="original":
                assert tr==simulate(stream,src,seed)
            cut = mask(tr)
            if cut is None:
                continue
            sid = base+"-"+rule
            y = dict(sources=src,trajectory=tr,cut_batches=cut[0])
            labels[sid] = dict(sources=src,emitters=len(roles(y)[0]),rule=rule,base=base,seed=seed)
            events.append(dict(id=sid,window=g["window"],observed=cut[1],k=2))
    return events,labels


def predict(g,raw,events,nets,fg_nets):
    hist = {"full":PublicWindow(g,raw),"static":StaticWindow(g)}
    geometry = fg.Geometry(g)
    for offset in range(0,len(events),8):
        batch = events[offset:offset+8]
        inputs = geometry.inputs(batch)
        strengths = {}
        with torch.no_grad():
            for s,net in fg_nets.items():
                z,_ = net(*inputs)
                strengths[s] = (z[:,:,1]-z[:,:,0]).numpy()
        for i,e in enumerate(batch):
            visible = {v for group in e["observed"] for v in group}
            nodes = sorted(set(g["nodes"])-visible)
            idx = [geometry.index[n] for n in nodes]
            tie = fg.tie_order(e["id"].rsplit("-",1)[0],len(g["nodes"]))[idx]
            lists = {}
            for m,h in hist.items():
                (roster,_,_),(x,_,_) = h.design(frames(e))
                assert roster==nodes
                for s in SEEDS:
                    lists[f"{m}/{s}"] = shortlist(nets[m,s],x,nodes,tie)
            for s in SEEDS:
                lists[f"FGSSI/{s}"] = [nodes[j] for j in fg.sorted_nodes(strengths[s][i,idx],tie)[:12]]
            assert all(len(set(lst))==12 and set(lst)<=set(nodes) for lst in lists.values())
            yield dict(id=e["id"],window=g["window"],lists=lists)


def run(name):
    folder = OUT/"stress"/name
    folder.mkdir(parents=True,exist_ok=True)
    raw = fg.raw_data(name)
    gs,streams = fg.evaluation_windows(name,raw)
    methods = ("full","static","FGSSI")
    nets = {(m,s):fitted(name,s,c) for m,c in (("full","C"),("static","A")) for s in SEEDS}
    fg_nets = {}
    paths = [Path(__file__)]
    for s in SEEDS:
        w = weights(name,s)
        paths.append(w)
        net = fg.FGSSI().eval()
        net.load_state_dict(torch.load(w,weights_only=True))
        fg_nets[s] = net
        for kind in ("temporal","static"):
            f = folder_for(name,kind)
            paths.extend([f/f"add{s}.pt",f/"scale.npz"])
    windows = [g["window"] for g in gs]
    seed0 = 611207+(1000 if name=="company" else 0)
    cfg = dict(rules={k:list(v) for k,v in RULES.items()},attempts=ATTEMPTS,windows=windows,seed=seed0,
               hashes={p.relative_to(OUT.parents[1]).as_posix():fg.digest(p) for p in paths})
    lock = folder/"protocol.json"
    if lock.exists():
        assert fg.load(lock)==cfg
    else:
        fg.write_json(lock,cfg)
    records = []
    public = []
    for i,g in enumerate(gs):
        events,labels = generate(g,streams[g["window"]],random.Random(seed0+i))
        public.extend(events)
        records.extend(dict(r,**labels[r["id"]]) for r in predict(g,raw,events,nets,fg_nets))
    assert len({r["id"] for r in records})==len(records)==len(public)
    fg.save_rows(folder/"observations.jsonl",public)
    fg.save_rows(folder/"predictions.jsonl",records)
    groups = {k:[r for r in records if r["rule"]==k] for k in RULES}
    common = set.intersection(*[{r["base"] for r in grp} for grp in groups.values()])
    contrasts = [("full","static"),("full","FGSSI")]
    stats = {}
    for k,grp in groups.items():
        stats[k] = {}
        for tag,subset in (("eligible",grp),("common",[r for r in grp if r["base"] in common])):
            t = summarize(subset,methods,contrasts,windows)
            t["dual_emitters"] = sum(r["emitters"]==2 for r in subset)
            stats[k][tag] = t
    fg.write_json(folder/"summary.json",stats)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("network",choices=("Dept2","company"))
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run(p.parse_args().network)
