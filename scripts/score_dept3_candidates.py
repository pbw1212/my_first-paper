"""Join frozen predictions with labels and separate node and pair screening losses."""

import math
from collections import Counter, defaultdict
from statistics import mean

from build_dept3_pilot import OUT as PILOT, read_rows
from run_dept3_candidates import B, L, OUT, write_json


def counts(rows):
    """Each row is a prediction with its true source set."""
    node_hits = Counter()
    both = Counter()
    exact = top32 = 0
    hit_sum = 0
    for pred, sources in rows:
        nodes = pred["nodes"]
        assert len(nodes) == len(set(nodes)) and sources <= set(nodes)
        for budget in (8, L, 20, 32):
            hits = len(sources.intersection(nodes[:budget]))
            node_hits[budget] += hits
            both[budget] += hits == 2
        pairs = {tuple(pair) for pair in pred["pairs"]}
        assert len(pairs) == B and all(set(pair) <= set(nodes[:L]) for pair in pairs)
        exact += set(pred["pairs"][0]) == sources
        hit_sum += len(sources.intersection(pred["pairs"][0]))
        top32 += tuple(sorted(sources)) in pairs
    return {"n": len(rows), "recall_top1_pair": hit_sum / (2 * len(rows)), "top1_exact": exact,
            "node_hits": dict(node_hits), "both_in_nodes": dict(both), "in_32_pairs": top32,
            "lost_in_node_stage": len(rows) - both[L], "lost_in_pair_stage": both[L] - top32}


def by_group(rows, key):
    groups = defaultdict(list)
    for pred, truth in rows:
        groups[key(pred)].append((pred, truth))
    return {name: counts(items) for name, items in groups.items()}


def roles(label):
    sources = set(label["sources"])
    lineage = {v: {v} for v in sources}
    direct, visible_ancestors = set(), set()
    for i, batch in enumerate(label["trajectory"]):
        for v, parents in batch["joins"]:
            direct.update(sources.intersection(parents))
            lineage[v] = set().union(*(lineage[u] for u in parents))
            if i >= label["cut_batches"]:
                visible_ancestors.update(lineage[v])
    return direct, visible_ancestors


def main():
    val_labels = {item["id"]: item for item in read_rows(PILOT / "truth/validation.jsonl")}
    roles_by_id = {sid: roles(label) for sid, label in val_labels.items()}
    predictions = read_rows(OUT / "validation_predictions.jsonl")
    val = [(p, set(val_labels[p["id"]]["sources"])) for p in predictions]
    main_rows = [(p, truth) for p, truth in val if p["model"] == "full2000"]
    result = {"validation": by_group(val, lambda p: p["model"]),
              "window": by_group(main_rows, lambda p: p["window"]),
              "direct_sources": by_group(main_rows, lambda p: len(roles_by_id[p["id"]][0])),
              "sources_with_visible_descendants": by_group(main_rows, lambda p: len(roles_by_id[p["id"]][1]))}
    picked = Counter()
    for p, sources in main_rows:
        direct, _ = roles_by_id[p["id"]]
        if len(direct) == 1:
            picked["cases"] += 1
            picked["active_in12"] += direct.issubset(p["nodes"][:L])
            picked["silent_in12"] += (sources - direct).issubset(p["nodes"][:L])
    result["one_silent"] = dict(picked)
    result["fixed_pair_budget"] = {
        limit: sum(sorted(s) in p[field] for p, s in main_rows)
        for limit, field in ((12, "pairs"), (20, "pairs_from_20"), (32, "pairs_from_32"))}
    sizes = [len(p["nodes"]) for p, _ in main_rows]
    result["random_expected"] = {"both_in12_rate": mean(math.comb(L, 2) / math.comb(n, 2) for n in sizes),
                                 "in32_rate": mean(B / math.comb(n, 2) for n in sizes)}
    cover = {p["id"]: p for p in read_rows(PILOT / "validation_predictions.jsonl")}
    comparisons = Counter()
    for p, sources in main_rows:
        was = sorted(sources) in cover[p["id"]]["cover_top32"]
        now = sorted(sources) in p["pairs"]
        comparisons[(was, now)] += 1
    result["cover_vs_full_pair32"] = {"both": comparisons[True, True], "new_only": comparisons[False, True],
                                     "cover_only": comparisons[True, False], "neither": comparisons[False, False]}
    draws = read_rows(OUT / "training_draws.jsonl")
    train_labels = {a["accepted_id"]: set(a["sources"]) for a in draws if a["accepted_id"] is not None}
    out_of_fold = [(p, train_labels[p["id"]]) for p in read_rows(OUT / "rolling_predictions.jsonl")]
    result["rolling"] = by_group(out_of_fold, lambda p: p["model"])
    result["rolling_pool"] = counts(out_of_fold)
    result["train_attempts"] = len(draws)
    write_json(OUT / "summary.json", result)
    for key in ("validation", "direct_sources", "sources_with_visible_descendants", "rolling", "window"):
        print(key, [(group, m["n"], m["node_hits"][L], m["both_in_nodes"][L], m["in_32_pairs"],
                     m["top1_exact"]) for group, m in result[key].items()])
    for key in ("random_expected", "cover_vs_full_pair32", "one_silent", "fixed_pair_budget"):
        print(key, result[key])


if __name__ == "__main__":
    main()
