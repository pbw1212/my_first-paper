"""Independent replay and scoring checks against raw contacts and saved outputs."""

import gzip
import json
import random
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx

from build_dept3_pilot import OUT, SOURCE, read_rows


def reproduce(batches, sources, seed):
    rng = random.Random(seed)
    joined = set(sources)
    trajectory = []
    for time, edges in batches:
        successes = [(u, v) for u, v in edges
                     if u in joined and v not in joined and rng.random() < 0.3]
        if not successes:
            continue
        to_parents = defaultdict(list)
        for u, v in successes:
            to_parents[v].append(u)
        trajectory.append({"t": time, "joins": sorted([v, sorted(us)] for v, us in to_parents.items())})
        joined |= set(to_parents)
    return trajectory


def check():
    with gzip.open(SOURCE, "rt") as stream:
        contacts = sorted({tuple(map(int, line.split())) for line in stream})
    public, truth = OUT / "public", OUT / "truth"
    graphs = read_rows(public / "graphs.jsonl")
    checked = {"window_contact_counts": {}, "attempts": 0, "accepted": 0, "scored": 0}
    windows, maps = {}, {}
    bounds = defaultdict(list)
    for g in graphs:
        key, start, end = g["window"], g["start"], g["end"]
        bounds[key.split("-")[0]].append((start, end))
        historical = Counter((u, v) for u, v, t in contacts if t < start and u != v)
        nodes = {x for edge in historical for x in edge}
        assert sorted(nodes) == g["nodes"]
        assert sorted([u, v, w] for (u, v), w in historical.items()) == g["edges"]
        by_time = defaultdict(list)
        for u, v, t in contacts:
            if start <= t < end and u in nodes and v in nodes and u != v:
                by_time[t].append((u, v))
        windows[key] = sorted(by_time.items())
        checked["window_contact_counts"][key] = sum(map(len, by_time.values()))
        if key.startswith("validation-"):
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(historical.keys())
            maps[key] = dict(nx.all_pairs_shortest_path_length(graph))
    for spans in bounds.values():
        assert all(right <= left for (_, right), (left, _) in zip(sorted(spans), sorted(spans)[1:]))
    assert max(e for _, e in bounds["train"]) <= min(s for s, _ in bounds["validation"])

    observations, labels = {}, {}
    for split in ("train", "validation"):
        obs = read_rows(public / f"{split}.jsonl")
        lab = read_rows(truth / f"{split}.jsonl")
        assert len(obs) == len(lab) == 100
        assert len({x["id"] for x in obs}) == 100
        assert [x["id"] for x in obs] == [x["id"] for x in lab]
        observations.update((x["id"], x) for x in obs)
        labels.update((x["id"], x) for x in lab)
    seen = set()
    for attempt in read_rows(truth / "attempts.jsonl"):
        key, sid = attempt["window"], attempt["accepted_id"]
        traj = reproduce(windows[key], attempt["sources"], attempt["draw_seed"])
        groups = [[v for v, _ in b["joins"]] for b in traj]
        total = sum(map(len, groups))
        assert total == attempt["new_members"]
        prefix_sizes = [0]
        for b in groups:
            prefix_sizes.append(prefix_sizes[-1] + len(b))
        cut = max(i for i, size in enumerate(prefix_sizes) if 2 * size <= total)
        assert (sid is not None) == (cut > 0 and total - prefix_sizes[cut] >= 3)
        checked["attempts"] += 1
        if sid is None:
            continue
        assert sid not in seen
        seen.add(sid)
        assert observations[sid] == {"id": sid, "window": key, "k": 2, "observed": groups[cut:]}
        assert labels[sid] == {"id": sid, "sources": attempt["sources"], "draw_seed": attempt["draw_seed"],
                               "trajectory": traj, "cut_batches": cut}
        checked["accepted"] += 1
    assert seen == set(observations)

    histograms = defaultdict(Counter)
    recall_sum = exact_sum = top32_sum = 0.0
    possible32 = certain32 = 0
    predictions = read_rows(OUT / "validation_predictions.jsonl")
    assert {p["id"] for p in predictions} == {i for i in seen if i.startswith("validation-")}
    for prediction in predictions:
        sid = prediction["id"]
        item = observations[sid]
        assert prediction["window"] == item["window"]
        distance = maps[item["window"]]
        visible = {v for b in item["observed"] for v in b}
        cands = sorted(distance.keys() - visible)
        costs = {(a, b): sum(min(distance[a].get(v, 100), distance[b].get(v, 100)) for v in visible)
                 for a, b in combinations(cands, 2)}
        top = [tuple(pair) for pair in prediction["cover_top32"]]
        assert len(set(top)) == 32 and list(top[0]) == prediction["distance_cover"]
        assert [costs[pair] for pair in top] == sorted(costs.values())[:32]
        assert list(costs.values()).count(min(costs.values())) == prediction["ties_at_minimum"]
        sources = tuple(sorted(labels[sid]["sources"]))
        target_cost = costs[sources]
        better = sum(cost < target_cost for cost in costs.values())
        equal = sum(cost == target_cost for cost in costs.values())
        possible32 += better < 32
        certain32 += better + equal <= 32
        top32_sum += min(equal, max(0, 32 - better)) / equal
        exact_sum += (1 / equal) if better == 0 else 0
        best = costs[top[0]]
        winning_pairs = [pair for pair, cost in costs.items() if cost == best]
        recall_sum += sum(len(set(pair).intersection(sources)) for pair in winning_pairs) / (2 * len(winning_pairs))
        for method in ("random", "out_degree", "sent_count", "distance_cover"):
            guess = set(prediction[method])
            assert len(guess) == 2 and guess <= set(cands)
            histograms[method][len(guess & set(labels[sid]["sources"]))] += 1
        checked["scored"] += 1
    assert checked["scored"] == 100
    metrics = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    for method, hist in histograms.items():
        assert metrics["methods"][method]["hits_0_1_2"] == [hist[i] for i in range(3)]
    checked["distance_cover_uniform_ties"] = {
        "expected_recall": recall_sum / len(predictions), "expected_exact": exact_sum / len(predictions),
        "expected_top32_rate": top32_sum / len(predictions),
        "possible_top32_cases": possible32, "certain_top32_cases": certain32}
    (OUT / "audit.json").write_text(json.dumps(checked, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    check()
