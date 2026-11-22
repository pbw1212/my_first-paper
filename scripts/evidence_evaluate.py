"""Score the conditional cohorts and verify both sources by reverse traversal."""

import json
from collections import Counter
from itertools import combinations
from statistics import median

import numpy as np

from build_dept3_pilot import (DAY, OUT as PILOT, SOURCE as DEPT_RAW, SPLITS, build_split,
                               make_windows, mask, read_rows, save_rows, simulate)
from collegemsg_data import CFG, OUT as COLLEGE, SOURCE
from evidence_cohort import OUT
from probe_cascade_yield import load_events
from run_dept3_candidates import write_json


def has_two_lineages(lab):
    traj, cut = lab["trajectory"],lab["cut_batches"]
    ancestors = {v for batch in traj[cut:] for v,_ in batch["joins"]}
    for batch in reversed(traj):
        parents = {u for v,us in batch["joins"] if v in ancestors for u in us}
        ancestors.update(parents)
    return set(lab["sources"])<=ancestors


def describe(rows):
    inside = [r for r in rows if r["direct"] is not None]
    wins = sum(r["prefix128"]<=32<r["direct"] for r in inside)
    losses = sum(r["direct"]<=32<r["prefix128"] for r in inside)
    ans = {"n":len(rows),"in_pool":len(inside),"gained":wins,"lost":losses}
    for method in ("sum","joint","direct","prefix32","prefix128"):
        if rows and method in rows[0]:
            vals = [r[method] for r in inside]
            ans[method] = {"top1":vals.count(1),"top32":sum(i<=32 for i in vals),
                           "median_in_pool":median(vals) if vals else None}
    return ans


def historical():
    result = {}
    _,_,dept_train,_ = build_split(load_events(DEPT_RAW),"train",*SPLITS["train"],target=2000)
    for split in ("rolling","validation"):
        labels = dept_train if split=="rolling" else read_rows(PILOT/"truth/validation.jsonl")
        proves = {r["id"]:has_two_lineages(r) for r in labels}
        orig = read_rows(PILOT.parent/"dept3_forward"/f"{split}_sensitivity_ranks.jsonl")
        result["Dept3_"+split] = {"total":len(orig),**describe([
            {"direct":r["direct_rank"],"prefix128":r["prefix128_rank"]} for r in orig if proves[r["id"]]])}
        prefix = "train" if split=="rolling" else "validation"
        proves = {r["id"]:has_two_lineages(r) for r in read_rows(COLLEGE/"truth"/f"{prefix}.jsonl")}
        orig = [r for r in read_rows(COLLEGE/"event_results.jsonl") if r["id"] in proves]
        assert all(proves[r["id"]]==(r["lineages_visible"]==2) for r in orig)
        result["CollegeMsg_"+split] = {"total":len(orig),**describe([
            {m:r["ranks"][m] for m in ("direct","prefix128")} for r in orig if proves[r["id"]]])}
    return result


def evaluate():
    baseline = historical()
    labels = {r["id"]:r for r in read_rows(OUT/"truth.jsonl")}
    obs = {r["id"]:r for r in read_rows(OUT/"observations.jsonl")}
    pred = {r["id"]:r for r in read_rows(OUT/"candidates.jsonl")}
    assert labels.keys()==obs.keys()==pred.keys()
    raw = load_events(SOURCE)
    _,streams = make_windows(raw,"validation",*CFG["time_splits"]["validation"],width=7*DAY)
    rows = []
    for key in sorted({r["window"] for r in obs.values()}):
        with np.load(OUT/"scores"/f"{key}.npz") as arch:
            ids,data = arch["ids"].tolist(),arch["scores"]
        for sid,x in zip(ids,data,strict=True):
            lab,o,p = labels[sid],obs[sid],pred[sid]
            assert key==lab["window"]==o["window"]==p["window"]
            assert simulate(streams[key],lab["sources"],lab["draw_seed"])==lab["trajectory"]
            assert mask(lab["trajectory"])==(lab["cut_batches"],o["observed"]) and has_two_lineages(lab)
            nodes = p["nodes"]
            assert len(set(nodes))==32 and nodes==p["first128"][:32]
            assert set(nodes).isdisjoint(v for batch in o["observed"] for v in batch)
            truth = set(lab["sources"])
            pairs = list(combinations(nodes,2))
            target = next((i for i,s in enumerate(pairs) if set(s)==truth),None)
            orders = {m:np.array(p[m]) for m in ("sum","joint")}
            tie = np.argsort(orders["sum"])
            for i,m in enumerate(("direct","prefix32","prefix128")):
                orders[m] = np.lexsort((tie,-x[:,i]))
            row = {"id":sid,"window":key,"sources":sorted(truth),
                   "both_by_budget":{str(k):truth<=set(p["first128"][:k]) for k in (12,32,64,128)}}
            for m,order in orders.items():
                assert sorted(order)==list(range(496))
                row[m] = None if target is None else int(np.flatnonzero(order==target)[0])+1
            rows.append(row)
    assert len(rows)==len({r["id"] for r in rows})==len(labels)
    save_rows(OUT/"event_results.jsonl",rows)
    generation=json.loads((OUT/"generation.json").read_text())
    summary = {"historical":baseline,"new":describe(rows),
               "flow":dict(sum((Counter(v) for v in generation.values()),Counter())),
               "windows":{k:describe([r for r in rows if r["window"]==k]) for k in generation},
               "both_by_budget":{k:sum(r["both_by_budget"][k] for r in rows) for k in ("12","32","64","128")}}
    write_json(OUT/"summary.json",summary)
    print(json.dumps(summary))


if __name__=="__main__":
    evaluate()
