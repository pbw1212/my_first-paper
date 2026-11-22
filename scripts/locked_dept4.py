"""Freeze source models and a calendar-based Dept4 transfer protocol before drawing labels."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from build_dept3_pilot import make_windows, mask, simulate
from distilled_source_pairs import PublicWindow
from evaluate_factorial_history import fitted
from factorial_fgssi import SEEDS, weights
from factorial_history import StaticWindow, folder_for
from five_frame_pairs import frames
from history_components import OUT, shortlist
from probe_cascade_yield import load_events

ROOT = OUT.parents[1]
FOLDER = OUT/"Dept4"
RAW = ROOT/"data/raw/email-Eu-core-temporal-Dept4.txt.gz"
RAW_SHA = "82756f2f2811cf94d4d58b89b95e2fbdb6ce9fa516a3701cec4001e0c6c64b6c"
PLAN = dict(source="Dept2", target="Dept4", url="https://snap.stanford.edu/data/email-Eu-core-temporal.html",
            primary="C minus F; both true sources in 12; equally weighted training seeds",
            secondary="C minus A; static and temporal feature bundles, both additive",
            seeds=list(SEEDS), windows=12, attempts=400, first_seed=202611200,
            calendar="Select 12 evenly spaced complete 14-day windows over 15% to 100% of elapsed time, including zeros",
            sampling="Exactly 400 uniform unordered source draws per window; simulate p=.3 and apply ordinary half-prefix mask",
            uncertainty="10000 paired bootstrap draws over five seeds and circular blocks of three successive selected windows")


def config():
    assert fg.digest(RAW)==RAW_SHA
    paths = [RAW, Path(__file__).resolve()]
    paths.extend(ROOT/"scripts"/(name+".py") for name in (
        "history_components","factorial_history","factorial_fgssi","fgssi_suffix","evaluate_factorial_history",
        "distilled_source_pairs","ordinary_source_roles","partial_source_identification","run_dept3_candidates",
        "temporal_pair_evidence","five_frame_pairs","build_dept3_pilot","evaluate_dept3_pilot","matched_source_baselines",
        "probe_cascade_yield","partial_source_lists"))
    paths.extend(fg.UP/name for name in ("model.py","args.py"))
    for seed in SEEDS:
        paths.append(weights("Dept2",seed))
        for kind in ("static","temporal"):
            folder = folder_for("Dept2",kind)
            paths.extend([folder/"scale.npz",folder/f"add{seed}.json",folder/f"add{seed}.pt"])
    return dict(plan=PLAN, hashes={p.relative_to(ROOT).as_posix():fg.digest(p) for p in paths})


def windows(raw):
    graphs, streams = make_windows(raw,"Dept4",.15,1.)
    indices = np.linspace(0,len(graphs)-1,PLAN["windows"]).round().astype(int)
    assert len(set(indices))==PLAN["windows"]
    return [(graphs[i],streams[graphs[i]["window"]]) for i in indices]


def generate(raw):
    obs, truth, flow = [], [], []
    for i,(g,contacts) in enumerate(windows(raw)):
        rng = random.Random(PLAN["first_seed"]+i)
        before = len(obs)
        for j in range(PLAN["attempts"]):
            sources = sorted(rng.sample(g["nodes"],2))
            seed = rng.getrandbits(63)
            trace = simulate(contacts,sources,seed)
            m = mask(trace)
            if m is None:
                continue
            cut, groups = m
            sid = f"{g['window']}-{j}"
            obs.append(dict(id=sid,window=g["window"],observed=groups,k=2))
            truth.append(dict(id=sid,sources=sources,seed=seed,cut_batches=cut,trajectory=trace))
        flow.append(dict(window=g["window"],start=g["start"],end=g["end"],nodes=len(g["nodes"]),
                         attempts=PLAN["attempts"],eligible=len(obs)-before))
    assert len({e["id"] for e in obs})==len(obs)
    fg.save_rows(FOLDER/"public.jsonl",obs)
    fg.save_rows(FOLDER/"truth.jsonl",truth)
    fg.write_json(FOLDER/"flow.json",flow)


def predict(raw):
    events = fg.read_rows(FOLDER/"public.jsonl")
    models = {(cell,seed):fitted("Dept2",seed,cell) for cell in ("A","C") for seed in SEEDS}
    baseline = {}
    for seed in SEEDS:
        model = fg.FGSSI().eval()
        model.load_state_dict(torch.load(weights("Dept2",seed),weights_only=True))
        baseline[seed] = model
    records = []
    for g,_ in windows(raw):
        items = [e for e in events if e["window"]==g["window"]]
        if not items:
            continue
        hist = {"A":StaticWindow(g),"C":PublicWindow(g,raw)}
        geometry = fg.Geometry(g)
        for offset in range(0,len(items),8):
            batch = items[offset:offset+8]
            inputs = geometry.inputs(batch)
            with torch.no_grad():
                strengths = {}
                for seed,model in baseline.items():
                    logits,_ = model(*inputs)
                    strengths[seed] = (logits[:,:,1]-logits[:,:,0]).numpy()
            for row,event in enumerate(batch):
                visible = {v for group in event["observed"] for v in group}
                nodes = sorted(set(g["nodes"])-visible)
                tie = fg.tie_order(event["id"],len(nodes))
                lists = {}
                for cell,history in hist.items():
                    (order,_,_), (x,_,_) = history.design(frames(event))
                    assert order==nodes
                    for seed in SEEDS:
                        lists[f"{cell}/{seed}"] = shortlist(models[cell,seed],x,nodes,tie)
                idx = [geometry.index[v] for v in nodes]
                for seed in SEEDS:
                    lists[f"F/{seed}"] = [nodes[j] for j in fg.sorted_nodes(strengths[seed][row,idx],tie)[:12]]
                assert all(len(set(s))==12 and set(s)<=set(nodes) for s in lists.values())
                records.append(dict(id=event["id"],window=g["window"],candidates=len(nodes),lists=lists))
    assert len(records)==len(events)
    fg.save_rows(FOLDER/"predictions.jsonl",records)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step",choices=("freeze","generate","predict"))
    step = p.parse_args().step
    FOLDER.mkdir(parents=True,exist_ok=True)
    lock = FOLDER/"protocol_lock.json"
    cfg = config()
    if step=="freeze" and not lock.exists():
        fg.write_json(lock,cfg)
    assert fg.load(lock)==cfg, "Dept4 freeze differs"
    if step!="freeze":
        torch.set_num_threads(1)
        with threadpool_limits(1):
            {"generate":generate,"predict":predict}[step](load_events(RAW))
