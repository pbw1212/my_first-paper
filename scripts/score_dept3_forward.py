"""Evaluate archived forward scores. Full held-out histories are used only here."""

import json
from collections import defaultdict
from statistics import mean, median

import numpy as np
from numba import get_num_threads, set_num_threads
from threadpoolctl import threadpool_limits

from build_dept3_pilot import OUT as PILOT, SOURCE, SPLITS, build_split, load_events, read_rows, save_rows
from dept3_forward_model import prefix_graph, transition_rows
from run_dept3_candidates import write_json
from run_dept3_forward import COLUMNS, OUT, design, simulate_pairs
from run_dept3_pair_ranker import COORDINATES, OUT as PAIR_RUN
from score_dept3_candidates import roles


METHODS = ("sum", "joint", "direct", "prefix", "conditional", "joint_direct", "joint_prefix", "half_a", "half_b")
COMPARISONS = (("prefix", "direct"), ("prefix", "joint"), ("prefix", "sum"),
               ("joint_prefix", "joint_direct"), ("joint_prefix", "joint"))


def orders(row, scores):
    base = np.asarray(row["sum"])
    ordered = {"sum": base, "joint": np.asarray(row["joint"])}
    for i, key in enumerate(COLUMNS[:-1]):
        ordered[key] = base[np.argsort(-scores[base, i], kind="stable")]
    for key in ("direct", "prefix"):
        ranksum = np.argsort(ordered[key]) + np.argsort(ordered["joint"])
        ordered["joint_" + key] = base[np.argsort(ranksum[base], kind="stable")]
    return ordered


def event_diagnostics(obs, label, graph, model, true_pair, old_score):
    nll = np.zeros(3)
    count = 0
    for num, den, _ in transition_rows(graph, label["sources"], label["trajectory"]):
        p = (num @ model["weights"]) / (den @ model["weights"])
        nll += [-np.log(p).sum(), -np.log(num[:, 1] / den[1]).sum(), -np.log(num[:, 0] / den[0]).sum()]
        count += len(num)
    diag = {"nll_totals": nll.tolist(), "forward_members": count}
    if true_pair is not None:
        base, edges, *rest = design(obs, graph, model)
        pair = np.array([[graph[1][v] for v in label["sources"]]])
        score, guesses = simulate_pairs(base, edges, pair, *rest, True)
        assert np.allclose(score[0], old_score, rtol=0., atol=1e-10)
        predicted = set(graph[0][guesses[0][guesses[0] >= 0]].tolist())
        hidden = {v for b in label["trajectory"][:label["cut_batches"]] for v, _ in b["joins"]}
        visible = {v for b in obs["observed"] for v in b}
        assert predicted.isdisjoint(set(label["sources"]) | visible)
        n = len(graph[0]) - len(visible) - 2
        denom = len(predicted) + len(hidden)
        diag.update({"hidden_guess": sorted(predicted), "hidden_size": len(hidden),
                     "hidden_f1": 2 * len(predicted & hidden) / denom, "ess": score[0, -1],
                     "random_hidden_f1": 2 * len(hidden) * len(predicted) / (n * denom)})
    return diag


def summarize(rows):
    inside = [r for r in rows if r["sum"] is not None]
    summary = {"n": len(rows), "in_pool": len(inside), "methods": {}, "paired": {}}
    for method in METHODS:
        ranks = [r[method] for r in inside]
        summary["methods"][method] = {"top1": ranks.count(1), "top32": sum(p <= 32 for p in ranks),
                                      "median_rank_in_pool": median(ranks) if ranks else None}
    for current, ref in COMPARISONS:
        gain = sum(r[current] <= 32 < r[ref] for r in inside)
        loss = sum(r[ref] <= 32 < r[current] for r in inside)
        summary["paired"][current + "_vs_" + ref] = {"gained": gain, "lost": loss}
    summary["half_overlap_at32"] = mean(r["overlap"] / 32 for r in rows)
    summary["next_member_nll"] = (np.array([r["nll_totals"] for r in rows]).sum(axis=0)
                                  / sum(r["forward_members"] for r in rows)).tolist()
    if inside:
        summary["true_pair_prefix"] = {"mean_f1": mean(r["hidden_f1"] for r in inside),
                                       "random_f1": mean(r["random_hidden_f1"] for r in inside),
                                       "median_ess": median(r["ess"] for r in inside)}
    return summary


def main():
    set_num_threads(1)
    events = load_events(SOURCE)
    _, train, train_labels, _ = build_split(events, "train", *SPLITS["train"], target=2000)
    obs = {r["id"]: r for r in train + read_rows(PILOT / "public/validation.jsonl")}
    models = json.loads((OUT / "models.json").read_text(encoding="utf-8"))["folds"]
    graphs = {g["window"]: prefix_graph(g, events) for g in read_rows(PILOT / "public/graphs.jsonl")}
    result = {}
    for split in ("rolling", "validation"):
        labels = train_labels if split == "rolling" else read_rows(PILOT / "truth/validation.jsonl")
        truth = {r["id"]: r for r in labels}
        baseline = read_rows(PAIR_RUN / f"{split}_rankings.jsonl")
        with np.load(OUT / f"{split}_scores.npz", allow_pickle=False) as archive:
            scores = archive["scores"]
            assert archive["ids"].tolist() == [r["id"] for r in baseline]
            assert archive["nodes"].tolist() == [r["nodes"] for r in baseline]
        rows = []
        for previous, values in zip(baseline, scores, strict=True):
            row = {k: previous[k] for k in ("id", "window", "fold")}
            sid = row["id"]
            label = truth[sid]
            nodes = previous["nodes"]
            ordered = orders(previous, values)
            sources = label["sources"]
            pair = None
            if set(sources) <= set(nodes):
                pair = COORDINATES.index(tuple(sorted(nodes.index(v) for v in sources)))
            row.update({k: None if pair is None else int(np.where(v == pair)[0][0]) + 1 for k, v in ordered.items()})
            row["overlap"] = len(set(ordered["half_a"][:32]) & set(ordered["half_b"][:32]))
            row["direct_sources"] = len(roles(label)[0])
            row.update(event_diagnostics(obs[sid], label, graphs[row["window"]], models[row["fold"]], pair,
                                         None if pair is None else values[pair]))
            rows.append(row)
        save_rows(OUT / f"{split}_true_ranks.jsonl", rows)
        group = {"all": summarize(rows)}
        for field in ("fold", "window", "direct_sources"):
            buckets = defaultdict(list)
            for r in rows:
                buckets[r[field]].append(r)
            group[field] = {k: summarize(v) for k, v in sorted(buckets.items())}
        if split == "rolling":
            blocks = list(group["window"].values())
            pick = np.random.default_rng(11706).integers(len(blocks), size=(10000, len(blocks)))
            sizes = np.array([v["n"] for v in blocks])
            net = np.array([v["methods"]["prefix"]["top32"] - v["methods"]["direct"]["top32"] for v in blocks])
            group["window_interval_prefix_minus_direct"] = np.quantile(
                net[pick].sum(axis=1) / sizes[pick].sum(axis=1), [0.025, 0.975]).tolist()
        result[split] = group
        print(split, json.dumps(group["all"]), flush=True)
    write_json(OUT / "summary.json", result)


if __name__ == "__main__":
    with threadpool_limits(limits=1, user_api="blas"):
        main()
