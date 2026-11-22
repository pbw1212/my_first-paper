"""Fit node-level logistic models and export candidate lists before scoring."""

import hashlib
import json
import random
from collections import Counter
from itertools import combinations

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from build_dept3_pilot import (DAY, OUT as PILOT, RAW_SHA, SOURCE, SPLITS, build_split,
                               load_events, read_rows, save_rows)
from evaluate_dept3_pilot import History, UNREACHABLE


OUT = PILOT.parent / "dept3_candidates"
RECENT_DAYS = 28
L, B = 12, 32
HISTORY_FEATURES = ("out_degree", "in_degree", "sent", "received", "recent_sent", "recent_received")
FEATURES = HISTORY_FEATURES + ("visible_reachable", "visible_mean_reciprocal_distance",
                               "visible_direct_fraction", "earliest_reachable",
                               "earliest_max_reciprocal_distance", "earliest_mean_reciprocal_distance",
                               "recent_visible_contacts_per_member")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Features:
    def __init__(self, graph, events, history_class=History):
        self.history = history_class(graph)
        ids = self.history.nodes
        in_degree, received = Counter(), Counter()
        for _, v, count in graph["edges"]:
            in_degree[v] += 1
            received[v] += count
        self.recent = np.zeros((len(ids), len(ids)), dtype=int)
        t0 = graph["start"]
        for t, u, v in events:
            if t >= t0:
                break
            if t >= t0 - RECENT_DAYS * DAY:
                self.recent[self.history.index[u], self.history.index[v]] += 1
        self.static = np.log1p(np.column_stack((
            [self.history.out_degree[v] for v in ids], [in_degree[v] for v in ids],
            [self.history.sent_count[v] for v in ids], [received[v] for v in ids],
            self.recent.sum(axis=1), self.recent.sum(axis=0))))

    def event(self, obs):
        visible = {v for group in obs["observed"] for v in group}
        nodes = sorted(set(self.history.nodes) - visible)
        index = self.history.index
        rows = [index[v] for v in nodes]
        cols = [index[v] for v in sorted(visible)]
        first = [index[v] for v in obs["observed"][0]]
        d = self.history.distance[np.ix_(rows, cols)]
        f = self.history.distance[np.ix_(rows, first)]
        # Candidates and observations are disjoint, so finite distances are positive.
        rd = np.where(d == UNREACHABLE, 0, 1.0 / d)
        rf = np.where(f == UNREACHABLE, 0, 1.0 / f)
        x = np.column_stack((self.static[rows], (d < UNREACHABLE).mean(axis=1), rd.mean(axis=1),
                             (d == 1).mean(axis=1), (f < UNREACHABLE).mean(axis=1), rf.max(axis=1),
                             rf.mean(axis=1), np.log1p(self.recent[np.ix_(rows, cols)].mean(axis=1))))
        return obs, nodes, x


def fit(rows, truth, feature_count):
    x = np.vstack([features[:, :feature_count] for _, _, features in rows])
    labels, weights = [], []
    for obs, nodes, _ in rows:
        sources = truth[obs["id"]]
        y = np.array([v in sources for v in nodes])
        if y.sum() != 2:
            raise ValueError(f"Missing sources for {obs['id']}")
        labels.extend(y)
        weights.extend(np.where(y, 0.5 / 2, 0.5 / (len(nodes) - 2)))
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-7)
    model.fit(scaler.transform(x), labels, sample_weight=weights)
    return {"features": FEATURES[:feature_count], "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "weights": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]), "iterations": int(model.n_iter_[0]),
            "fit_events": len(rows), "fit_nodes": len(labels),
            "fit_windows": sorted({obs["window"] for obs, _, _ in rows})}


def rank_pool(model, event, wide_pool):
    obs, nodes, features = event
    dim = len(model["features"])
    scores = ((features[:, :dim] - model["mean"]) / model["scale"]) @ model["weights"] + model["intercept"]
    values = dict(zip(nodes, scores.tolist()))
    ranked = nodes.copy()
    rng = random.Random("candidate-order-" + obs["id"])
    rng.shuffle(ranked)
    ranked.sort(key=values.__getitem__, reverse=True)
    pairs = list(combinations(ranked[:L], 2))
    rng.shuffle(pairs)
    pairs.sort(key=lambda pair: values[pair[0]] + values[pair[1]], reverse=True)
    wide = pairs.copy()
    if wide_pool:
        lower = [(a, b) for j, b in enumerate(ranked[L:32], L) for a in ranked[:j]]
        rng.shuffle(lower)
        wide.extend(lower)
        wide.sort(key=lambda pair: values[pair[0]] + values[pair[1]], reverse=True)
    return values, ranked, pairs, wide


def predict(model, rows, tag):
    predictions = []
    for event in rows:
        obs = event[0]
        _, ranked, pairs, wide = rank_pool(model, event, tag == "full2000")
        prediction = {"id": obs["id"], "window": obs["window"], "model": tag,
                      "nodes": ranked, "pairs": [sorted(pair) for pair in pairs[:B]]}
        if tag == "full2000":
            for limit in (20, 32):
                pool = set(ranked[:limit])
                eligible = [pair for pair in wide if pool.issuperset(pair)]
                prediction[f"pairs_from_{limit}"] = [sorted(pair) for pair in eligible[:B]]
        predictions.append(prediction)
    return predictions


def window_index(row):
    return int(row[0]["window"].rsplit("-", 1)[1])


def main():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError("Unexpected archive checksum")
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events(SOURCE)
    train_graphs, train, labels, attempts = build_split(events, "train", *SPLITS["train"], target=2000)
    if train[:100] != read_rows(PILOT / "public/train.jsonl") or labels[:100] != read_rows(PILOT / "truth/train.jsonl"):
        raise ValueError("Expanded training set changed the pilot prefix")
    save_rows(OUT / "training_draws.jsonl", attempts)
    truth = {item["id"]: set(item["sources"]) for item in labels}
    graphs = {g["window"]: g for g in read_rows(PILOT / "public/graphs.jsonl")}
    assert all(g == graphs[g["window"]] for g in train_graphs)
    val = read_rows(PILOT / "public/validation.jsonl")
    used = {item["window"] for item in train + val}
    features = {key: Features(graphs[key], events) for key in used}
    training = [features[item["window"]].event(item) for item in train]
    validation = [features[item["window"]].event(item) for item in val]
    models, predictions, rolling = {}, [], []
    with threadpool_limits(limits=1):
        for tag, subset, dim in (("full100", training[:100], len(FEATURES)),
                                 ("history2000", training, len(HISTORY_FEATURES)),
                                 ("full2000", training, len(FEATURES))):
            models[tag] = fit(subset, truth, dim)
            predictions.extend(predict(models[tag], validation, tag))
        for cutoff in (10, 15, 20):
            tag = f"rolling_{cutoff}"
            earlier = [row for row in training if window_index(row) <= cutoff]
            later = [row for row in training if cutoff < window_index(row) <= cutoff + 5]
            models[tag] = fit(earlier, truth, len(FEATURES))
            rolling.extend(predict(models[tag], later, tag))
    save_rows(OUT / "validation_predictions.jsonl", predictions)
    save_rows(OUT / "rolling_predictions.jsonl", rolling)
    write_json(OUT / "models.json", {"raw_sha256": RAW_SHA, "sklearn_version": sklearn.__version__,
                                    "recent_days": RECENT_DAYS, "L": L, "B": B, "models": models})
    print(f"{len(train)} train events from {len(attempts)} draws; {len(val)} validation events; "
          f"{len(rolling)} out-of-fold predictions")


if __name__ == "__main__":
    main()
