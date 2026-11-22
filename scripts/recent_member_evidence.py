"""Nested additive feature models using the same training pipeline as the full model."""
import argparse

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from distilled_source_pairs import PairNet
from evaluate_factorial_history import dataset, fitted
from factorial_fgssi import OUT as PRIOR, SEEDS
from history_components import collect, fit, shortlist
from report_history_components import summarize
from score_dept3_candidates import roles

OUT = PRIOR.parent/"recent_member_evidence"
PERSON = [*range(6), *range(19,23)]
LINKS = [12,23,24,25]
CONTEXT = [17,18,58,59,60,63]
METHODS = ("person","links","full")
CONTRASTS = [("links","person"),("full","links"),("full","person")]


def examples(name,phase,raw):
    rows = collect(name,phase,raw)
    for _,_,variants,_ in rows:
        full = variants["full"]
        variants.clear()
        variants["full"] = full
        for method,cols in (("person",PERSON),("links",PERSON+LINKS)):
            selected = [*cols,*(c+29 for c in cols),*CONTEXT]
            x = np.zeros_like(full)
            x[:,selected] = full[:,selected]
            variants[method] = x
    return rows


def run(name):
    folder = OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    raw = fg.raw_data(name)
    train,cal = [examples(name,phase,raw) for phase in ("fit","cal")]
    assert (len(train),len(cal))==(1000,300)
    for m in METHODS[:2]:
        fit(train,cal,m,folder)
    del train,cal
    nets = {("full",s):fitted(name,s,"C") for s in SEEDS}
    for m in METHODS[:2]:
        with np.load(folder/(m+".npz")) as scale:
            for s in SEEDS:
                net = PairNet(64,False).eval()
                path = folder/f"{m}-{s}.pt"
                assert fg.digest(path)==fg.load(path.with_suffix(".json"))["sha256"]
                net.load_state_dict(torch.load(path,weights_only=True))
                nets[m,s] = net,scale["mean"],scale["std"]
    _,parent,_,_ = dataset(name)
    labels = {r["id"]:r for r in fg.read_rows(parent/"truth.jsonl")}
    previous = {r["id"]:r for r in fg.read_rows(PRIOR.parent/"history_components"/name/"predictions.jsonl")}
    rows = []
    for o,nodes,design,(a,b) in examples(name,"test",raw):
        tie = fg.tie_order(o["id"],len(nodes))
        lists = {f"{m}/{s}":shortlist(net,design[m],nodes,tie) for (m,s),net in nets.items()}
        assert all(lists[f"full/{s}"]==previous[o["id"]]["lists"][f"full/{s}"] for s in SEEDS)
        rows.append(dict(id=o["id"],window=o["window"],sources=[nodes[a],nodes[b]],emitters=len(roles(labels[o["id"]])[0]),lists=lists))
    assert len(rows)==1200
    fg.save_rows(folder/"predictions.jsonl",rows)
    windows = sorted({r["window"] for r in rows})
    stats = {tag:summarize(group,METHODS,CONTRASTS,windows) for tag,group in
             [("all",rows),("one",[r for r in rows if r["emitters"]==1]),("two",[r for r in rows if r["emitters"]==2])]}
    fg.write_json(folder/"summary.json",stats)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("network",choices=("Dept2","company"))
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run(p.parse_args().network)
