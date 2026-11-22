"""Post-hoc transfer of both Dept2 restricted models without fitting target labels."""
import gzip
import json

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
import locked_dept4 as dept4
from distilled_source_pairs import PairNet,PublicWindow
from evaluate_factorial_history import dataset
from factorial_fgssi import OUT as PRIOR,SEEDS
from five_frame_pairs import frames
from history_components import shortlist
from recent_member_evidence import CONTEXT,CONTRASTS,LINKS,METHODS,OUT,PERSON
from report_history_components import summarize


def run(name):
    fit = OUT/"Dept2"
    if name=="Dept4":
        assert dept4.config()==fg.load(dept4.FOLDER/"protocol_lock.json")
        parent,raw = dept4.FOLDER,dept4.load_events(dept4.RAW)
        graphs = [g for g,_ in dept4.windows(raw)]
        existing = {r["id"]:{s:r["lists"][f"C/{s}"] for s in SEEDS} for r in fg.read_rows(parent/"predictions.jsonl")}
    else:
        _,parent,raw,graphs = dataset(name)
        with gzip.open(PRIOR/name/"cells.jsonl.gz","rt",encoding="utf8") as stream:
            existing = {r["id"]:{s:r["models"][f"C{s}"]["marg12"] for s in SEEDS} for r in map(json.loads,stream)}
    nets = {}
    for m in METHODS[:2]:
        with np.load(fit/(m+".npz")) as scale:
            for s in SEEDS:
                path = fit/f"{m}-{s}.pt"
                assert fg.digest(path)==fg.load(path.with_suffix(".json"))["sha256"]
                model = PairNet(64,False).eval()
                model.load_state_dict(torch.load(path,weights_only=True))
                nets[m,s] = model,scale["mean"],scale["std"]
    public = fg.read_rows(parent/"public.jsonl")
    records = []
    for g in graphs:
        items = [o for o in public if o["window"]==g["window"]]
        if not items:
            continue
        hist = PublicWindow(g,raw)
        for o in items:
            (nodes,_,_),(full,_,_) = hist.design(frames(o))
            tie = fg.tie_order(o["id"],len(nodes))
            lists = {f"full/{s}":v for s,v in existing[o["id"]].items()}
            for m,cols in (("person",PERSON),("links",PERSON+LINKS)):
                keep = [*cols,*(c+29 for c in cols),*CONTEXT]
                x = np.zeros_like(full)
                x[:,keep] = full[:,keep]
                for s in SEEDS:
                    lists[f"{m}/{s}"] = shortlist(nets[m,s],x,nodes,tie)
            records.append(dict(id=o["id"],window=o["window"],lists=lists))
    assert len(records)==len(public)==len(existing)
    folder = OUT/"posthoc"/name
    folder.mkdir(parents=True,exist_ok=True)
    fg.save_rows(folder/"predictions.jsonl",records)
    truth = {r["id"]:r["sources"] for r in fg.read_rows(parent/"truth.jsonl")}
    for r in records:
        r["sources"] = truth[r["id"]]
    fg.write_json(folder/"summary.json",summarize(records,METHODS,CONTRASTS,[g["window"] for g in graphs]))


if __name__=="__main__":
    torch.set_num_threads(1)
    with threadpool_limits(1):
        for network in ("Dept4","Enron"):
            run(network)
