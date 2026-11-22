"""Predict from public observations, then join the labels for evaluation."""

import json
import math
import random
from collections import Counter, defaultdict, deque
from statistics import mean

import numpy as np

from build_dept3_pilot import OUT, read_rows, save_rows


UNREACHABLE = 100
METHODS = ("random", "out_degree", "sent_count", "distance_cover")


class History:
    def __init__(self, graph):
        self.nodes = graph["nodes"]
        self.index = {v: i for i, v in enumerate(self.nodes)}
        adjacency = defaultdict(list)
        self.out_degree = Counter()
        self.sent_count = Counter()
        for u, v, count in graph["edges"]:
            adjacency[u].append(v)
            self.out_degree[u] += 1
            self.sent_count[u] += count
        n = len(self.nodes)
        self.distance = np.full((n, n), UNREACHABLE, dtype=int)
        for source in self.nodes:
            row = self.distance[self.index[source]]
            row[self.index[source]] = 0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v in adjacency[u]:
                    if row[self.index[v]] == UNREACHABLE:
                        row[self.index[v]] = row[self.index[u]] + 1
                        queue.append(v)


def predict(item, graph):
    assert item["k"] == 2
    visible = {v for group in item["observed"] for v in group}
    candidates = sorted(set(graph.nodes) - visible)
    result = {"id": item["id"], "window": item["window"], "candidate_count": len(candidates)}
    result["random"] = sorted(random.Random("pilot-random-" + item["id"]).sample(candidates, 2))
    for name in ("out_degree", "sent_count"):
        tie_order = candidates.copy()
        random.Random("pilot-" + name + item["id"]).shuffle(tie_order)
        ranked = sorted(tie_order, key=getattr(graph, name).__getitem__, reverse=True)
        result[name] = sorted(ranked[:2])

    distances = graph.distance[np.ix_([graph.index[v] for v in candidates],
                                     [graph.index[v] for v in sorted(visible)])]
    left, right = np.triu_indices(len(candidates), k=1)
    costs = np.minimum(distances[left], distances[right]).sum(axis=1)
    # Stable sorting after a seeded shuffle gives each tied pair equal priority.
    order = list(range(len(costs)))
    random.Random("pilot-cover-" + item["id"]).shuffle(order)
    order.sort(key=costs.__getitem__)
    shortlist = [[candidates[left[i]], candidates[right[i]]] for i in order[:32]]
    result["distance_cover"] = shortlist[0]
    result["cover_top32"] = shortlist
    result["ties_at_minimum"] = int((costs == costs[order[0]]).sum())
    return result


def aggregate(predictions, truth, graphs, attempts, train_labels):
    scores = {name: [] for name in METHODS}
    middle_errors = Counter()
    by_window = defaultdict(list)
    by_active_sources = defaultdict(list)
    silent_cases = shortlist_hits = 0
    observed_sizes, realized_masks = [], []
    for prediction in predictions:
        label = truth[prediction["id"]]
        sources = set(label["sources"])
        groups = label["trajectory"]
        cut = label["cut_batches"]
        hidden = {v for b in groups[:cut] for v, _ in b["joins"]}
        total_growth = sum(len(b["joins"]) for b in groups)
        observed_sizes.append(total_growth - len(hidden))
        realized_masks.append(len(hidden) / total_growth)
        parents = {u for b in groups for _, p in b["joins"] for u in p}
        n_active = len(sources & parents)
        silent_cases += n_active < 2
        shortlist_hits += sorted(sources) in prediction["cover_top32"]
        for name in METHODS:
            picked = set(prediction[name])
            hits = len(sources & picked)
            scores[name].append(hits)
            middle_errors[name] += len(hidden & picked)
        by_window[prediction["window"]].append(scores["distance_cover"][-1])
        by_active_sources[n_active].append(scores["distance_cover"][-1])

    n = len(predictions)
    summary = {"validation_n": n, "methods": {}}
    for name, series in scores.items():
        summary["methods"][name] = {
            "hits_0_1_2": [series.count(i) for i in range(3)],
            "recall": mean(series) / 2, "exact": series.count(2) / n,
            "incorrect_picks": 2 * n - sum(series), "hidden_middle_picks": middle_errors[name]}
    sizes = [p["candidate_count"] for p in predictions]
    summary.update({
        "random_expected_recall": mean(2 / c for c in sizes),
        "random_expected_exact": mean(1 / math.comb(c, 2) for c in sizes),
        "cover_truth_in_top32": shortlist_hits, "silent_source_cases": silent_cases,
        "tied_minimum_cases": sum(p["ties_at_minimum"] > 1 for p in predictions),
        "mean_visible": mean(observed_sizes), "mean_actual_hidden_ratio": mean(realized_masks),
        "candidate_range": [min(sizes), max(sizes)],
        "cover_by_active_sources": {k: [s.count(i) for i in range(3)]
                                    for k, s in by_active_sources.items()},
    })
    summary["generation"] = {}
    for split in ("train", "validation"):
        totals = []
        for window, graph in graphs.items():
            if not window.startswith(split + "-"):
                continue
            rows = [a for a in attempts if a["window"] == window]
            for_summary = {"window": window, "attempts": len(rows),
                           "zero_growth": sum(a["new_members"] == 0 for a in rows),
                           "accepted": sum(a["accepted_id"] is not None for a in rows)}
            if split == "validation":
                for_summary["cover_hits_0_1_2"] = [by_window[window].count(i) for i in range(3)]
            totals.append(for_summary)
        split_rows = [a for a in attempts if a["window"].startswith(split + "-")]
        accepted = [a for a in split_rows if a["accepted_id"] is not None]
        def avg_degree(rows):
            return mean(graphs[a["window"]].out_degree[v] for a in rows for v in a["sources"])
        labels = train_labels if split == "train" else list(truth.values())
        both_active = 0
        for label in labels:
            senders = {u for b in label["trajectory"] for _, parents in b["joins"] for u in parents}
            both_active += set(label["sources"]).issubset(senders)
        summary["generation"][split] = {
            "attempts": len(split_rows), "accepted": len(accepted),
            "zero_growth": sum(a["new_members"] == 0 for a in split_rows),
            "both_sources_triggered": both_active, "all_mean_source_outdegree": avg_degree(split_rows),
            "accepted_mean_source_outdegree": avg_degree(accepted), "windows": totals}
    return summary


def main():
    public = OUT / "public"
    graphs = {g["window"]: History(g) for g in read_rows(public / "graphs.jsonl")}
    inputs = read_rows(public / "validation.jsonl")
    predictions = [predict(item, graphs[item["window"]]) for item in inputs]
    save_rows(OUT / "validation_predictions.jsonl", predictions)
    truth = OUT / "truth"
    labels = {row["id"]: row for row in read_rows(truth / "validation.jsonl")}
    result = aggregate(predictions, labels, graphs, read_rows(truth / "attempts.jsonl"),
                       read_rows(truth / "train.jsonl"))
    with (OUT / "summary.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["methods"], indent=2))


if __name__ == "__main__":
    main()
