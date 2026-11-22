"""Score frozen pair lists and count event-by-event gains and losses."""

import json
from collections import defaultdict
from statistics import median

import numpy as np

from build_dept3_pilot import OUT as PILOT, SOURCE, SPLITS, build_split, load_events, read_rows, save_rows
from run_dept3_candidates import write_json
from run_dept3_pair_ranker import COORDINATES, OUT
from score_dept3_candidates import roles


METHODS = ("sum", "node_only", "joint")


def summarize(rows):
    n = len(rows)
    in_pool = sum(r["sum"] is not None for r in rows)
    out = {"n": n, "in_pool": in_pool, "methods": {}}
    for method in METHODS:
        ranks = [r[method] for r in rows if r[method] is not None]
        out["methods"][method] = {
            "top1": ranks.count(1), "top32": sum(k <= 32 for k in ranks),
            "conditional_median_rank": median(ranks) if ranks else None,
            "mrr_all": sum(1 / k for k in ranks) / n,
            "conditional_mrr": sum(1 / k for k in ranks) / in_pool if in_pool else None}
    for reference in ("sum", "node_only"):
        gain = loss = earlier = later = 0
        for r in rows:
            if r["joint"] is None:
                continue
            before, after = r[reference], r["joint"]
            gain += after <= 32 < before
            loss += before <= 32 < after
            earlier += after < before
            later += after > before
        out["joint_vs_" + reference] = {"gained": gain, "lost": loss, "rank_improved": earlier, "rank_worsened": later}
    return out


def group(rows, field):
    groups = defaultdict(list)
    for r in rows:
        groups[r[field]].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def score(predictions, labels):
    rows = []
    for pred in predictions:
        label = labels[pred["id"]]
        sources = set(label["sources"])
        direct, visible = roles(label)
        pair_index = None
        if sources.issubset(pred["nodes"]):
            positions = tuple(sorted(pred["nodes"].index(v) for v in sources))
            pair_index = COORDINATES.index(positions)
        row = {k: pred[k] for k in ("id", "window", "fold")}
        row.update({"sources": sorted(sources), "direct": len(direct), "visible_lineage": len(visible)})
        for method in METHODS:
            row[method] = None if pair_index is None else pred[method].index(pair_index) + 1
        rows.append(row)
    return rows


def window_interval(by_window, reference):
    vals = list(by_window.values())
    cases = np.array([v["n"] for v in vals])
    net = np.array([v["methods"]["joint"]["top32"] - v["methods"][reference]["top32"] for v in vals])
    rng = np.random.default_rng(11706)
    selected = rng.integers(len(vals), size=(10000, len(vals)))
    rates = net[selected].sum(axis=1) / cases[selected].sum(axis=1)
    return np.quantile(rates, [0.025, 0.975]).tolist()


def main():
    _, _, labels, _ = build_split(load_events(SOURCE), "train", *SPLITS["train"], target=2000)
    result = {}
    for name in ("rolling", "validation"):
        if name == "validation":
            labels = read_rows(PILOT / "truth/validation.jsonl")
        truth = {r["id"]: r for r in labels}
        rows = score(read_rows(OUT / f"{name}_rankings.jsonl"), truth)
        save_rows(OUT / f"{name}_true_ranks.jsonl", rows)
        section = {"all": summarize(rows), "fold": group(rows, "fold"), "window": group(rows, "window"),
                   "direct_sources": group(rows, "direct"), "visible_lineage": group(rows, "visible_lineage")}
        if name == "rolling":
            section["window_bootstrap_delta_top32"] = {
                ref: window_interval(section["window"], ref) for ref in ("sum", "node_only")}
        result[name] = section
        print(name, json.dumps(section["all"]))
        for field in ("fold", "direct_sources"):
            print(field, [(k, v["n"], v["in_pool"], [v["methods"][m]["top32"] for m in METHODS])
                          for k, v in section[field].items()])
    write_json(OUT / "summary.json", result)


if __name__ == "__main__":
    main()
