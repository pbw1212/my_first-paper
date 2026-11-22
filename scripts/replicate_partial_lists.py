"""Resample propagation in the same held-out windows with frozen models and list budgets."""
import argparse
import hashlib
import math
from collections import Counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from partial_source_identification import (HistoricalFeatures, NEW, OUT, company, draw, load,
    load_events, make_windows, read_rows, save_rows, thresholds, write_json)
from partial_source_lists import lists, relative_ranks, score


def simple_orders(o, data):
    nodes, x, support = data
    # Same tie-breaking rule as the learned list scores.
    rng = np.random.default_rng(1000003 + sum((j+1)*ord(c) for j,c in enumerate(o["id"])))
    tie = rng.permutation(len(nodes))
    values = {"distance": x[:,7], "reach": support.mean(axis=1)}
    return {tag:[nodes[j] for j in np.lexsort((tie,-s))] for tag,s in values.items()}


def budgets(rows, labs, tags):
    outcome = {}
    for tag in tags:
        outcome[tag] = {}
        for k in (5,12,30):
            hits = [len(set(r[tag][:k]).intersection(labs[r["id"]]["sources"])) for r in rows]
            outcome[tag][str(k)] = [sum(h>=1 for h in hits),sum(h==2 for h in hits)]
    return outcome


def run(name):
    folder = OUT/name
    models = joblib.load(folder/"models.joblib")
    split = load(folder/"windows.json")
    ordered = sorted(split.values(),key=lambda w:w["start"])
    assert [w["stage"] for w in ordered] == ["fit"]*5+["cal"]*3+["test"]*4
    assert all(a["end"]<=b["start"] for a,b in zip(ordered,ordered[1:]))
    raw = company() if name=="company" else load_events(NEW)
    graphs,streams = make_windows(raw,name,.15,1.)
    obs = read_rows(folder/"public.jsonl")
    original = {r["id"]:r for r in read_rows(folder/"truth.jsonl")}
    cal_pred = [r for r in read_rows(folder/"predictions.jsonl") if split[r["window"]]["stage"]=="cal"]
    assert thresholds(cal_pred,original)==load(folder/"calibration.json")
    original_lists = {r["id"]:r for r in read_rows(folder/"shortlist_predictions.jsonl")}
    original_summary = load(folder/"shortlists.json")
    cuts = original_summary["calibration"].copy()
    model_hash = hashlib.sha256((folder/"models.joblib").read_bytes()).hexdigest()
    preds, public, truth, cal, initial, flow = [], [], [], [], [], {}
    for g in graphs:
        key = g["window"]
        if key not in split or split[key]["stage"]=="fit":
            continue
        f = HistoricalFeatures(g,raw)
        stage = split[key]["stage"]
        for o in (o for o in obs if o["window"]==key):
            row = dict(original_lists[o["id"]],**simple_orders(o,f.event(o)))
            (cal if stage=="cal" else initial).append(row)
        if stage=="cal":
            continue
        seed = 779103+len(flow)
        items,labs,counts = draw(g,streams[key],seed,80,25)
        assert (counts["dual"],counts["ordinary"])==(80,25)
        flow[key]=dict(counts,seed=seed)
        for o,y in zip(items,labs,strict=True):
            o["id"]=y["id"]="repeat-"+o["id"]
            data = f.event(o)
            orders = dict(lists(o,data,models),**simple_orders(o,data))
            legal = set(g["nodes"])-{v for batch in o["observed"] for v in batch}
            assert set(y["sources"])<=legal
            assert all(len(v)==len(legal) and set(v)==legal for v in orders.values())
            preds.append(dict(id=o["id"],window=key,**orders))
        public.extend(items)
        truth.extend(labs)
    dual_cal = [r for r in cal if original[r["id"]]["dual"]]
    assert len(dual_cal)==240
    idx = math.ceil(.9*(len(dual_cal)+1))-1
    for tag in ("source","pair_cover","prior","distance","reach"):
        fractions = np.array([relative_ranks(r,original[r["id"]],tag) for r in dual_cal])
        rule = np.sort(fractions,axis=0)[idx].tolist()
        if tag in cuts:
            assert rule==cuts[tag]
        cuts[tag]=rule
    repeated = {r["id"]:r for r in truth}
    assert len(repeated)==len(truth) and repeated.keys().isdisjoint(original)
    assert {r["seed"] for r in truth}.isdisjoint(r["seed"] for r in original.values())
    results={}
    for sample,rows,labs in (("initial",initial,original),("repeat",preds,repeated)):
        results[sample]={}
        for group in ("dual","ordinary"):
            group_rows=[r for r in rows if labs[r["id"]][group]]
            metrics=score(group_rows,labs,cuts)
            if sample=="initial":
                assert all(metrics[tag]==original_summary[group][tag] for tag in ("source","pair_cover","prior"))
            results[sample][group]=dict(calibrated=metrics, fixed=budgets(group_rows,labs,cuts))
    single={}
    oracle=load(folder/"oracle.json")
    for r in read_rows(folder/"predictions.jsonl"):
        sid=r["id"]
        if split[r["window"]]["stage"]!="test" or not original[sid]["dual"]:
            continue
        required=oracle[sid]["dual"]["members"]
        if len(required)==1:
            for tag,value in r["outputs"].items():
                counts=single.setdefault(tag,Counter())
                counts["total"]+=1
                counts["named_required"]+=int(value["anchor"] in required)
                counts["named_either"]+=int(value["anchor"] in original[sid]["sources"])
    results.update(calibration=cuts, single_required=single, repeat_draw=flow,model_sha256=model_hash)
    save_rows(folder/"repeat_public.jsonl",public)
    save_rows(folder/"repeat_truth.jsonl",truth)
    save_rows(folder/"repeat_predictions.jsonl",preds)
    write_json(folder/"replication.json",results)
    print(name,"completed validation and repeat draws",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("network",choices=["company","Dept2"])
    with threadpool_limits(limits=1):
        run(parser.parse_args().network)
