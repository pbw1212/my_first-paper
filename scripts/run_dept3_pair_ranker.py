"""Learn pair rankings on forward-in-time, out-of-fold node proposals."""

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from build_dept3_pilot import OUT as PILOT, RAW_SHA, SOURCE, SPLITS, build_split, load_events, read_rows, save_rows
from evaluate_dept3_pilot import UNREACHABLE
from run_dept3_candidates import FEATURES, OUT as NODE_RUN, Features, fit, rank_pool, write_json


OUT = PILOT.parent / "dept3_pair_ranker"
N = 32
LEFT, RIGHT = np.triu_indices(N, 1)
COORDINATES = list(combinations(range(N), 2))
PAIR_ID = {ij: i for i, ij in enumerate(COORDINATES)}
NAMES = ["node_high", "node_low", "earliest_union_proximity", "visible_union_proximity",
         "shared_direct_fraction", "weaker_exclusive_direct_fraction", "recent_union_fraction",
         "recent_shared_fraction", "recent_between_candidates"]
REG = 0.02


@dataclass
class PairEvent:
    sid: str
    window: str
    nodes: list
    x: np.ndarray
    additive: np.ndarray


def pair_event(obs, graph, node_model):
    scores, ranked, _, wide = rank_pool(node_model, graph.event(obs), True)
    pool = ranked[:N]
    position = {v: i for i, v in enumerate(pool)}
    additive = np.array([PAIR_ID[position[a], position[b]] for a, b in wide])
    ids = [graph.history.index[v] for v in pool]
    visible = sorted(v for batch in obs["observed"] for v in batch)
    cols = [graph.history.index[v] for v in visible]
    first = [visible.index(v) for v in obs["observed"][0]]
    distances = graph.history.distance[np.ix_(ids, cols)]
    recip = np.where(distances == UNREACHABLE, 0., 1. / distances)
    direct = distances == 1
    recent = graph.recent[np.ix_(ids, cols)] > 0
    between = graph.recent[np.ix_(ids, ids)]
    best = np.maximum(recip[LEFT], recip[RIGHT])
    exclusive_a = (direct[LEFT] & ~direct[RIGHT]).mean(axis=1)
    exclusive_b = (direct[RIGHT] & ~direct[LEFT]).mean(axis=1)
    node_scores = np.array([scores[v] for v in pool])
    x = np.column_stack((
        node_scores[LEFT], node_scores[RIGHT], best[:, first].mean(axis=1), best.mean(axis=1),
        (direct[LEFT] & direct[RIGHT]).mean(axis=1), np.minimum(exclusive_a, exclusive_b),
        (recent[LEFT] | recent[RIGHT]).mean(axis=1), (recent[LEFT] & recent[RIGHT]).mean(axis=1),
        np.log1p(between[LEFT, RIGHT] + between[RIGHT, LEFT])))
    return PairEvent(obs["id"], obs["window"], pool, x, additive)


def train_ranker(events, labels, dim):
    eligible, target = [], []
    for event in events:
        a, b = labels[event.sid]
        if a in event.nodes and b in event.nodes:
            eligible.append(event.x[:, :dim])
            target.append(PAIR_ID[tuple(sorted((event.nodes.index(a), event.nodes.index(b))))])
    x = np.stack(eligible)
    mean = x.mean(axis=(0, 1))
    scale = x.std(axis=(0, 1))
    scale[scale < 1e-12] = 1
    x = (x - mean) / scale
    examples = np.arange(len(target))
    selected_x = x[examples, target].mean(axis=0)

    def objective(weights):
        scores = x @ weights
        z = logsumexp(scores, axis=1)
        loss = (z - scores[examples, target]).mean() + REG * np.dot(weights, weights) / 2
        prob = np.exp(scores - z[:, None])
        grad = np.einsum("ij,ijk->k", prob, x) / len(target) - selected_x + REG * weights
        return loss, grad

    result = minimize(objective, np.zeros(dim), jac=True, method="L-BFGS-B",
                      options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-7})
    if not result.success:
        raise RuntimeError(result.message)
    return {"features": NAMES[:dim], "mean": mean.tolist(), "scale": scale.tolist(),
            "weights": result.x.tolist(), "eligible_train_events": len(target),
            "all_train_events": len(events), "windows": sorted({e.window for e in events}),
            "iterations": result.nit}


def rank(event, model):
    dim = len(model["features"])
    scores = (event.x[:, :dim] - model["mean"]) / model["scale"] @ model["weights"]
    return event.additive[np.argsort(-scores[event.additive], kind="stable")]


def run():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError("Unexpected Dept3 archive")
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events(SOURCE)
    _, train, truth, draws = build_split(events, "train", *SPLITS["train"], target=2000)
    assert draws == read_rows(NODE_RUN / "training_draws.jsonl")
    labels = {row["id"]: row["sources"] for row in truth}
    public_val = read_rows(PILOT / "public/validation.jsonl")
    graphs = {g["window"]: g for g in read_rows(PILOT / "public/graphs.jsonl")}
    needed = {row["window"] for row in train + public_val}
    features = {k: Features(graphs[k], events) for k in needed}
    models = json.loads((NODE_RUN / "models.json").read_text(encoding="utf-8"))["models"]
    early = [features[o["window"]].event(o) for o in train if o["window"] <= "train-05"]
    models["rolling_5"] = fit(early, labels, len(FEATURES))
    old_rolling = {r["id"]: r for r in read_rows(NODE_RUN / "rolling_predictions.jsonl")}
    old_val = {r["id"]: r for r in read_rows(NODE_RUN / "validation_predictions.jsonl") if r["model"] == "full2000"}
    training, roll_outputs, val_outputs = [], [], []
    saved = {"lambda": REG, "early_node_model": models["rolling_5"], "folds": {}}
    for cutoff in (5, 10, 15, 20, 25):
        tag = "full2000" if cutoff == 25 else f"rolling_{cutoff}"
        obs = public_val if cutoff == 25 else [
            o for o in train if cutoff < int(o["window"].split("-")[1]) <= cutoff + 5]
        batch = [pair_event(o, features[o["window"]], models[tag]) for o in obs]
        if cutoff > 5:
            fitted = {name: train_ranker(training, labels, dim) for name, dim in (("node_only", 2), ("joint", 9))}
            saved["folds"][tag] = fitted
            for e in batch:
                old = (old_val if cutoff == 25 else old_rolling)[e.sid]
                assert e.nodes == old["nodes"][:N]
                if cutoff == 25:
                    replayed = [sorted((e.nodes[LEFT[i]], e.nodes[RIGHT[i]])) for i in e.additive[:32]]
                    assert replayed == old["pairs_from_32"]
                pred = {"id": e.sid, "window": e.window, "fold": tag, "nodes": e.nodes,
                        "sum": e.additive.tolist()}
                pred.update({name: rank(e, model).tolist() for name, model in fitted.items()})
                (val_outputs if cutoff == 25 else roll_outputs).append(pred)
            print(tag, "pair train", len(training), "in pool", fitted["joint"]["eligible_train_events"],
                  "predict", len(batch))
        if cutoff < 25:
            training.extend(batch)
    write_json(OUT / "models.json", saved)
    save_rows(OUT / "rolling_rankings.jsonl", roll_outputs)
    save_rows(OUT / "validation_rankings.jsonl", val_outputs)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
