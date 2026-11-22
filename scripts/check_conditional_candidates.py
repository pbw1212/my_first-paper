"""Independently replay training labels, compare saved rankings and summarize paired changes."""

import json
from collections import Counter, defaultdict
from itertools import combinations
from statistics import median

from build_dept3_pilot import DAY, make_windows, mask, read_rows, save_rows, simulate
from collegemsg_data import CFG, SOURCE
from conditional_candidates import BUDGETS, MODELS, OUT, VALID
from probe_cascade_yield import load_events
from run_dept3_candidates import write_json
from score_dept3_candidates import roles


def tally(rows):
    ranks = [r["rank"] for r in rows if r["rank"] is not None]
    return dict(events=len(rows),both=len(ranks),top1=ranks.count(1),top32=sum(x<=32 for x in ranks),
                median_ms=round(1000*median(r["seconds"] for r in rows),3),
                total_seconds=round(sum(r["seconds"] for r in rows),3))


def paired(left,right,field):
    def success(r):
        return r["rank"] is not None and (field=="both" or r["rank"]<=32)
    a = {r["id"] for r in left if success(r)}
    b = {r["id"] for r in right if success(r)}
    return {"gained":len(b-a),"lost":len(a-b),"shared":len(a&b)}


def check():
    _, streams = make_windows(load_events(SOURCE),"train",*CFG["time_splits"]["train"],width=7*DAY)
    control_counts = {}
    for arm in MODELS[1:]:
        labels = read_rows(OUT/f"{arm}_truth.jsonl")
        obs = {o["id"]:o for o in read_rows(OUT/f"{arm}_observations.jsonl")}
        assert len(labels)==len(obs)==len({t["id"] for t in labels})
        for lab in labels:
            o = obs[lab["id"]]
            tr = simulate(streams[o["window"]],lab["sources"],lab["draw_seed"])
            assert tr==lab["trajectory"] and mask(tr)==(lab["cut_batches"],o["observed"])
        control_counts[arm]=sum(len(roles(lab)[1])==2 for lab in labels)
        if arm=="conditional":
            assert control_counts[arm]==len(labels)
    lab = {r["id"]:set(r["sources"]) for r in read_rows(VALID/"truth.jsonl")}
    obs = {r["id"]:r for r in read_rows(VALID/"observations.jsonl")}
    previous = {r["id"]:r for r in read_rows(VALID/"event_results.jsonl")}
    previous_nodes = {r["id"]:r["first128"] for r in read_rows(VALID/"candidates.jsonl")}
    events, groups = [], defaultdict(list)
    for r in read_rows(OUT/"predictions.jsonl"):
        sid, arm, n = r["id"],r["model"],r["budget"]
        nodes = r["nodes"]
        assert len(set(nodes))==n and obs[sid]["window"]==r["window"]
        assert set(nodes).isdisjoint(v for batch in obs[sid]["observed"] for v in batch)
        pairs = list(combinations(nodes,2))
        assert sorted(r["order"])==list(range(len(pairs)))
        rank = next((i for i,j in enumerate(r["order"],1) if lab[sid]==set(pairs[j])),None)
        if arm=="original":
            assert nodes==previous_nodes[sid][:n]
            if n==32:
                assert rank==previous[sid]["direct"]
        record=dict(id=sid,window=r["window"],model=arm,budget=n,rank=rank,seconds=r["seconds"])
        events.append(record)
        groups[arm,n].append(record)
    assert set(groups)=={(a,n) for a in MODELS for n in BUDGETS}
    for items in groups.values():
        assert len(items)==len(lab) and {r["id"] for r in items}==set(lab)
    save_rows(OUT/"event_results.jsonl",events)
    rolling=read_rows(OUT/"rolling_coverage.jsonl")
    summary=dict(train_has_two_lineages=control_counts,
        validation={f"{a}_{n}":tally(items) for (a,n),items in groups.items()},
        windows={k:{f"{a}_{n}":tally([r for r in items if r["window"]==k])
                    for (a,n),items in groups.items()} for k in sorted({r["window"] for r in events})},
        rolling={str(c):{"n":sum(r["cut"]==c for r in rolling),
                         **{a:sum(r["cut"]==c and r[a]<=32 for r in rolling) for a in MODELS}}
                 for c in (6,8,10)},paired={})
    contrasts=[((arm,n),("conditional",n)) for arm in MODELS[:2] for n in BUDGETS]
    contrasts.extend(((a,32),(a,64)) for a in MODELS)
    for left,right in contrasts:
        summary["paired"][f"{left[0]}{left[1]}_to_{right[0]}{right[1]}"]={
            f:paired(groups[left],groups[right],f) for f in ("both","top32")}
    write_json(OUT/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="windows"}))


if __name__=="__main__":
    check()
