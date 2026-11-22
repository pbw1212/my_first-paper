"""Join frozen CollegeMsg predictions with held-back source and trajectory labels."""

import math
from collections import Counter, defaultdict
from statistics import mean, median

import numpy as np

from build_dept3_pilot import read_rows, save_rows
from collegemsg_data import OUT
from collegemsg_forward import COLUMNS
from run_dept3_candidates import write_json
from run_dept3_pair_ranker import COORDINATES
from score_dept3_candidates import roles


METHODS = ("sum","joint","direct","prefix32","prefix128","joint_direct","joint_prefix128","half64a","half64b")


def metrics(rows):
    sizes = [r["legal"] for r in rows]
    res = {"n":len(rows),"legal_range":[min(sizes),max(sizes)],"windows":len({r["window"] for r in rows}),
           "candidates":{},"methods":{}}
    for k in (12,32,64,128):
        hits = Counter(r["node_hits"][str(k)] for r in rows)
        res["candidates"][str(k)] = {"hits_0_1_2":[hits[i] for i in range(3)],
                                     "node_recall":(hits[1]+2*hits[2])/(2*len(rows)),
                                     "uniform_expected_both":sum(math.comb(k,2)/math.comb(n,2) for n in sizes)}
    for method in METHODS:
        selected = [r for r in rows if method in r["ranks"]]
        if not selected:
            continue
        ranks = [r["ranks"][method] for r in selected if r["ranks"][method] is not None]
        guesses = [r["top1_hits"][method] for r in selected]
        res["methods"][method] = {"n":len(selected),"in_pool":len(ranks),"top32":sum(k<=32 for k in ranks),
                                  "top1":ranks.count(1),"median_rank_if_in":median(ranks) if ranks else None,
                                  "top1_hit_one":guesses.count(1),"top1_node_recall":sum(guesses)/(2*len(selected))}
    paired = {}
    for method,ref in (("prefix128","direct"),("prefix32","direct"),("joint_prefix128","joint_direct")):
        selected = [r for r in rows if method in r["ranks"] and ref in r["ranks"]]
        a = {r["id"] for r in selected if (r["ranks"][method] or 497)<=32}
        b = {r["id"] for r in selected if (r["ranks"][ref] or 497)<=32}
        paired[method+"_vs_"+ref] = {"n":len(selected),"gained":len(a-b),"lost":len(b-a),"retained":len(a&b)}
    res["paired"] = paired
    res["overlap_halves"] = mean(r["overlap_halves"]/32 for r in rows)
    res["direct_source_count"] = dict(Counter(r["direct_sources"] for r in rows))
    silent = [r for r in rows if r["direct_sources"]==1]
    if silent:
        res["one_silent"] = {"n":len(silent),"active_in32":sum(r["active_in32"] for r in silent),
                             "silent_in32":sum(r["silent_in32"] for r in silent)}
    res["ess128_median_all_pairs_per_event_then_median"] = median(r["ess128_median"] for r in rows)
    return res


def score():
    labels = {r["id"]:r for s in ("train","validation") for r in read_rows(OUT/"truth"/f"{s}.jsonl")}
    pool = {r["id"]:r for r in read_rows(OUT/"candidate_rankings.jsonl")}
    scored = []
    for key in sorted({r["window"] for r in pool.values()}):
        with np.load(OUT/"scores"/f"{key}.npz") as archive:
            ids,nodes,values = archive["ids"].tolist(),archive["nodes"].tolist(),archive["scores"]
            assert archive["columns"].tolist()==list(COLUMNS)
        for sid,ns,x in zip(ids,nodes,values,strict=True):
            r,lab = pool[sid],labels[sid]
            assert ns==r["nodes"]==r["first128"][:32] and len(set(ns))==32 and np.isfinite(x).all()
            truth = set(lab["sources"])
            base = np.asarray(r["sum"])
            orders = {"sum":base}
            for j,m in enumerate(COLUMNS[:-1]):
                orders[m] = base[np.argsort(-x[base,j],kind="stable")]
            if r["joint"] is not None:
                orders["joint"] = np.asarray(r["joint"])
                for m in ("direct","prefix128"):
                    fused = np.argsort(orders["joint"])+np.argsort(orders[m])
                    orders["joint_"+m] = base[np.argsort(fused[base],kind="stable")]
            src = None if not truth<=set(ns) else COORDINATES.index(tuple(sorted(ns.index(v) for v in truth)))
            direct,visible = roles(lab)
            row = {"id":sid,"window":key,"cutoff":r["cutoff"],"legal":r["legal"],"sources":sorted(truth),
                   "node_hits":{str(k):len(truth.intersection(r["first128"][:k])) for k in (12,32,64,128)},
                   "direct_sources":len(direct),"lineages_visible":len(visible),
                   "active_in32":len(direct.intersection(ns)),"silent_in32":len((truth-direct).intersection(ns)),
                   "overlap_halves":len(set(orders["half64a"][:32])&set(orders["half64b"][:32])),
                   "ess128_median":float(np.median(x[:,-1])),"ranks":{},"top1_hits":{}}
            for m,order in orders.items():
                row["ranks"][m] = None if src is None else int(np.flatnonzero(order==src)[0])+1
                a,b = COORDINATES[order[0]]
                row["top1_hits"][m] = len(truth.intersection((ns[a],ns[b])))
            scored.append(row)
    assert len(scored)==len(pool)==len({r["id"] for r in scored})
    save_rows(OUT/"event_results.jsonl",scored)
    summary = {}
    for name in ("rolling","validation"):
        subset = [r for r in scored if (r["cutoff"]==12)==(name=="validation")]
        section = {"all":metrics(subset)}
        for field in ("cutoff","window","direct_sources","lineages_visible"):
            groups = defaultdict(list)
            for r in subset:
                groups[r[field]].append(r)
            section[field] = {k:metrics(v) for k,v in sorted(groups.items())}
        windows = list(section["window"].values())
        totals = np.array([w["n"] for w in windows])
        diff = np.array([w["methods"]["prefix128"]["top32"]-w["methods"]["direct"]["top32"] for w in windows])
        pick = np.random.default_rng(21706).integers(len(windows),size=(10000,len(windows)))
        section["descriptive_window_interval"] = np.quantile(diff[pick].sum(1)/totals[pick].sum(1),[.025,.975]).tolist()
        summary[name] = section
        print(name,[(m,v["n"],v["top32"],v["top1_hit_one"]) for m,v in section["all"]["methods"].items()])
    write_json(OUT/"summary.json",summary)


if __name__ == "__main__":
    score()
