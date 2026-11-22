"""Audit saved rankings, chronological fits, and the joint scoring formula."""

import json
from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np

from build_dept3_pilot import DAY, OUT as PILOT, SOURCE, SPLITS, build_split, load_events, read_rows
from run_dept3_candidates import OUT as NODE_RUN, Features
from run_dept3_pair_ranker import OUT


def score_matrix(obs, pool, features, node_model, paths, recent):
    _, names, matrix = features.event(obs)
    logits = (matrix - node_model["mean"]) / node_model["scale"] @ node_model["weights"] + node_model["intercept"]
    by_id = dict(zip(names, logits))
    observed = [v for group in obs["observed"] for v in group]
    first_size = len(obs["observed"][0])
    prox = {u: np.array([1 / paths[u][v] if v in paths[u] else 0. for v in observed]) for u in pool}
    direct = {u: {v for v in observed if paths[u].get(v) == 1} for u in pool}
    last_month = {u: {v for v in observed if (u, v) in recent} for u in pool}
    n = len(observed)
    rows = []
    for u, v in combinations(pool, 2):
        q = np.maximum(prox[u], prox[v])
        a, b = direct[u], direct[v]
        c, d = last_month[u], last_month[v]
        rows.append([max(by_id[u], by_id[v]), min(by_id[u], by_id[v]), q[:first_size].mean(), q.mean(),
                     len(a & b) / n, min(len(a-b), len(b-a)) / n, len(c | d) / n, len(c & d) / n,
                     np.log1p(recent[u, v] + recent[v, u])])
    return np.array(rows)


def main():
    raw = load_events(SOURCE)
    graphs = {g["window"]: g for g in read_rows(PILOT / "public/graphs.jsonl")}
    models = json.loads((OUT / "models.json").read_text(encoding="utf-8"))["folds"]
    node_models = json.loads((NODE_RUN / "models.json").read_text(encoding="utf-8"))["models"]
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    _, train, train_labels, _ = build_split(raw, "train", *SPLITS["train"], target=2000)
    observations = {o["id"]: o for o in train + read_rows(PILOT / "public/validation.jsonl")}
    truths = {l["id"]: set(l["sources"]) for l in train_labels + read_rows(PILOT / "truth/validation.jsonl")}
    cached, checked_folds = {}, set()
    for split in ("rolling", "validation"):
        previous = {p["id"]: p for p in read_rows(NODE_RUN / f"{split}_predictions.jsonl")
                    if split == "rolling" or p["model"] == "full2000"}
        saved = {r["id"]: r for r in read_rows(OUT / f"{split}_true_ranks.jsonl")}
        rows = read_rows(OUT / f"{split}_rankings.jsonl")
        assert len(rows) == len(previous) == len(saved) and {r["id"] for r in rows} == set(previous)
        totals = Counter()
        for row in rows:
            sid, window, fold, pool = (row[k] for k in ("id", "window", "fold", "nodes"))
            assert pool == previous[sid]["nodes"][:32]
            assert set(pool).isdisjoint(v for b in observations[sid]["observed"] for v in b)
            for m in models[fold].values():
                assert max(graphs[k]["end"] for k in m["windows"]) <= graphs[window]["start"]
            pairs = list(combinations(pool, 2))
            if split == "validation":
                assert [sorted(pairs[i]) for i in row["sum"][:32]] == previous[sid]["pairs_from_32"]
            for key in ("sum", "node_only", "joint"):
                assert sorted(row[key]) == list(range(496))
                pos = next((i for i, j in enumerate(row[key], 1) if set(pairs[j]) == truths[sid]), None)
                assert pos == saved[sid][key]
                totals[key] += pos is not None and pos <= 32
            if split == "rolling" and fold in checked_folds:
                continue
            checked_folds.add(fold)
            if window not in cached:
                start = graphs[window]["start"]
                hist = nx.DiGraph()
                hist.add_nodes_from(graphs[window]["nodes"])
                hist.add_edges_from((u, v) for t, u, v in raw if t < start)
                recent = Counter((u, v) for t, u, v in raw if start - 28 * DAY <= t < start)
                cached[window] = Features(graphs[window], raw), dict(nx.all_pairs_shortest_path_length(hist)), recent
            feature, paths, recent = cached[window]
            x = score_matrix(observations[sid], pool, feature, node_models[fold], paths, recent)
            assert np.max(np.diff(x[:, :2].sum(axis=1)[row["sum"]])) < 1e-9
            for key, model in models[fold].items():
                dim = len(model["features"])
                logits = (x[:, :dim] - model["mean"]) / model["scale"] @ model["weights"]
                assert np.max(np.diff(logits[row[key]])) < 1e-9
        for key, count in totals.items():
            assert count == summary[split]["all"]["methods"][key]["top32"]
        print(split, "verified all pools, permutations and true ranks", len(rows), dict(totals))
    print("Verified joint feature scores for 100 validation events and one event per rolling fold.")


if __name__ == "__main__":
    main()
