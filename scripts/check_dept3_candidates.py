"""Check saved validation rankings with raw prefix contacts and independent features."""

import gzip
import json
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
import numpy as np

from build_dept3_pilot import OUT as PILOT, SOURCE, read_rows
from run_dept3_candidates import OUT


def main():
    fitted = json.loads((OUT / "models.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    graphs = {g["window"]: g for g in read_rows(PILOT / "public/graphs.jsonl")}
    with gzip.open(SOURCE, "rt") as stream:
        raw = {tuple(map(int, line.split())) for line in stream}
    observations = {r["id"]: r for r in read_rows(PILOT / "public/validation.jsonl")}
    truth = {r["id"]: set(r["sources"]) for r in read_rows(PILOT / "truth/validation.jsonl")}
    checked = defaultdict(Counter)
    seen = set()
    for key in sorted({r["window"] for r in observations.values()}):
        start = graphs[key]["start"]
        graph = nx.DiGraph()
        recent = Counter()
        for u, v, t in raw:
            if u == v or t >= start:
                continue
            old = graph.get_edge_data(u, v, {}).get("weight", 0)
            graph.add_edge(u, v, weight=old + 1)
            if start - 28 * 86400 <= t:
                recent[u, v] += 1
        dist = dict(nx.all_pairs_shortest_path_length(graph))
        for r in read_rows(OUT / "validation_predictions.jsonl"):
            if r["window"] != key:
                continue
            sid, tag = r["id"], r["model"]
            assert (sid, tag) not in seen
            seen.add((sid, tag))
            obs, model = observations[sid], fitted["models"][tag]
            assert obs["window"] == key and all(graphs[k]["end"] <= start for k in model["fit_windows"])
            observed = {v for batch in obs["observed"] for v in batch}
            assert set(r["nodes"]) == set(graph) - observed and len(r["nodes"]) == len(set(r["nodes"]))
            scores = {}
            for u in r["nodes"]:
                first = obs["observed"][0]
                rv = [1 / dist[u][v] if v in dist[u] else 0 for v in observed]
                rf = [1 / dist[u][v] if v in dist[u] else 0 for v in first]
                values = list(np.log1p([graph.out_degree(u), graph.in_degree(u),
                                        graph.out_degree(u, weight="weight"), graph.in_degree(u, weight="weight"),
                                        sum(recent[u, v] for v in graph), sum(recent[v, u] for v in graph)]))
                values.extend([sum(x > 0 for x in rv) / len(rv), np.mean(rv), sum(x == 1 for x in rv) / len(rv),
                               sum(x > 0 for x in rf) / len(rf), max(rf), np.mean(rf),
                               np.log1p(sum(recent[u, v] for v in observed) / len(observed))])
                size = len(model["features"])
                scores[u] = np.dot((np.array(values[:size]) - model["mean"]) / model["scale"],
                                   model["weights"]) + model["intercept"]
            vector = np.array([scores[u] for u in r["nodes"]])
            assert np.all(np.diff(vector) <= 1e-9)
            for limit, field in ((12, "pairs"), (20, "pairs_from_20"), (32, "pairs_from_32")):
                if field not in r:
                    continue
                pool = set(r["nodes"][:limit])
                pairs = [tuple(pair) for pair in r[field]]
                assert len(set(pairs)) == 32 and all(a < b and {a, b} <= pool for a, b in pairs)
                expected = sorted((scores[a] + scores[b] for a, b in combinations(pool, 2)), reverse=True)[:32]
                assert np.allclose([scores[a] + scores[b] for a, b in pairs], expected, rtol=0, atol=1e-9)
                checked[tag][limit] += tuple(sorted(truth[sid])) in pairs
            checked[tag]["both_in12"] += truth[sid] <= set(r["nodes"][:12])
    assert len(seen) == 300
    for tag, c in checked.items():
        assert c["both_in12"] == summary["validation"][tag]["both_in_nodes"]["12"]
        assert c[12] == summary["validation"][tag]["in_32_pairs"]
    for limit, n in summary["fixed_pair_budget"].items():
        assert checked["full2000"][int(limit)] == n
    rolling = read_rows(OUT / "rolling_predictions.jsonl")
    assert len(rolling) == len({r["id"] for r in rolling}) == summary["rolling_pool"]["n"]
    for r in rolling:
        assert all(graphs[k]["end"] <= graphs[r["window"]]["start"] for k in fitted["models"][r["model"]]["fit_windows"])
    print(f"Verified {len(seen)} validation rankings and temporal separation of {len(rolling)} rolling predictions.")


if __name__ == "__main__":
    main()
