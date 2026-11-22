"""Compare matched CollegeMsg node rankers at two candidate budgets."""

import json
import random
from collections import Counter
from itertools import combinations
from time import perf_counter

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from build_dept3_pilot import DAY, make_windows, mask, read_rows, save_rows, simulate
from collegemsg_candidates import CollegeHistory
from collegemsg_data import CFG, OUT as COLLEGE, SOURCE
from collegemsg_forward import suffix
from dept3_forward_model import kernels, prefix_graph
from evidence_cohort import OUT as VALID
from evidence_evaluate import has_two_lineages
from probe_cascade_yield import load_events
from run_dept3_candidates import FEATURES, Features, fit, rank_pool, write_json

OUT = VALID.parent / "conditional_candidates"
TARGET, CAP = 40, 120000
MODELS = ("original", "matched", "conditional")
BUDGETS = (32, 64)


def generate(raw):
    graphs, streams = make_windows(raw, "train", *CFG["time_splits"]["train"], width=7*DAY)
    val_graphs = read_rows(COLLEGE/"public/graphs.jsonl")
    assert max(g["end"] for g in graphs) <= min(g["start"] for g in val_graphs if g["window"].startswith("val"))
    groups = {"conditional": [], "matched": []}
    flow = {}
    for week, graph in enumerate(graphs, 1):
        key = graph["window"]
        stream = streams[key]
        senders = {u for _, contacts in stream for u, _ in contacts}
        target = TARGET
        for arm in groups:
            conditional = arm == "conditional"
            rng = random.Random((61100 if conditional else 71100) + week)
            count = Counter(accepted=0, attempts=0)
            for attempt in range(1, CAP+1):
                if count["accepted"] == target:
                    break
                count["attempts"] += 1
                sources = sorted(rng.sample(graph["nodes"], 2))
                seed = rng.getrandbits(63)
                # A source with no outgoing contact cannot have descendants.
                # The control only needs one source to be able to grow.
                if (conditional and not senders.issuperset(sources)) or senders.isdisjoint(sources):
                    continue
                tr = simulate(stream, sources, seed)
                masked = mask(tr)
                if masked is None:
                    continue
                cut, observed = masked
                label = dict(sources=sources, cut_batches=cut, trajectory=tr)
                if conditional and not has_two_lineages(label):
                    continue
                count["accepted"] += 1
                sid = f"{arm}-{week:02}-{attempt}"
                obs = dict(id=sid, window=key, k=2, observed=observed)
                groups[arm].append((obs, dict(label, id=sid, window=key, draw_seed=seed)))
            flow.setdefault(key, {})[arm] = dict(count)
            if conditional:
                target = count["accepted"]
            else:
                assert count["accepted"] == target
        print(key, flow[key], flush=True)
    for arm, rows in groups.items():
        save_rows(OUT/f"{arm}_observations.jsonl", (o for o, _ in rows))
        save_rows(OUT/f"{arm}_truth.jsonl", (t for _, t in rows))
    write_json(OUT/"generation.json", flow)


@njit
def direct_scores(base, ptr, cols, weights, pairs, obs, ends):
    scores = np.empty(len(pairs))
    for k, (a, b) in enumerate(pairs):
        rate = base.copy()
        for u in (a, b):
            for p in range(ptr[u], ptr[u+1]):
                rate[cols[p]] += weights[p]
        rate[a] = rate[b] = 0.
        scores[k] = suffix(rate, rate.sum(), obs, ends, ptr, cols, weights)/len(obs)
    return scores


def training(raw, graphs):
    datasets = {arm: read_rows(OUT/f"{arm}_observations.jsonl") for arm in MODELS[1:]}
    labels = {r["id"]:set(r["sources"]) for arm in datasets for r in read_rows(OUT/f"{arm}_truth.jsonl")}
    features = {arm:[] for arm in datasets}
    for k, g in graphs.items():
        if not k.startswith("train"):
            continue
        hist = Features(g, raw, CollegeHistory)
        for arm, items in datasets.items():
            features[arm].extend(hist.event(o) for o in items if o["window"] == k)
    models = {}
    rolling = []
    original = json.loads((COLLEGE/"candidate_models.json").read_text())["node"]
    for cut in (6, 8, 10, 12):
        end = f"train-{cut:02}"
        models[str(cut)] = {"original": original[str(cut)]}
        for arm in datasets:
            earlier = [e for e in features[arm] if e[0]["window"] <= end]
            models[str(cut)][arm] = fit(earlier, labels, len(FEATURES))
        for event in features["conditional"]:
            o = event[0]
            if end < o["window"] <= f"train-{cut+2:02}" and cut < 12:
                row = {"id":o["id"], "window":o["window"], "cut":cut}
                for arm, model in models[str(cut)].items():
                    nodes = rank_pool(model,event,False)[1]
                    row[arm] = max(nodes.index(v)+1 for v in labels[o["id"]])
                rolling.append(row)
    save_rows(OUT/"rolling_coverage.jsonl", rolling)
    write_json(OUT/"models.json", models)
    return models["12"]


def score(raw, graphs, models):
    observations = read_rows(VALID/"observations.jsonl")
    forward = json.loads((COLLEGE/"forward_models.json").read_text())["12"]
    predictions = []
    for key in sorted({o["window"] for o in observations}):
        g = graphs[key]
        features = Features(g, raw, CollegeHistory)
        hist = prefix_graph(g, raw)
        base, edges = kernels(hist, forward)
        csr = csr_matrix(edges)
        for o in (o for o in observations if o["window"] == key):
            event = features.event(o)
            visible = np.array([hist[1][v] for batch in o["observed"] for v in batch])
            ends = np.cumsum([len(batch) for batch in o["observed"]])
            for arm, model in models.items():
                values, nodes, _, wide = rank_pool(model,event,True)
                tie = {tuple(sorted(p)):i for i,p in enumerate(wide)}
                for n in BUDGETS:
                    pairs = list(combinations(nodes[:n], 2))
                    ids = np.array([[hist[1][a],hist[1][b]] for a,b in pairs])
                    args = (base,csr.indptr,csr.indices,csr.data,ids,visible,ends)
                    if not direct_scores.signatures:
                        direct_scores(*args)
                    start = perf_counter()
                    scores = direct_scores(*args)
                    seconds = perf_counter()-start
                    assert np.isfinite(scores).all()
                    order = sorted(range(len(pairs)), key=lambda i: (
                        -scores[i], -(values[pairs[i][0]]+values[pairs[i][1]]),
                        tie.get(tuple(sorted(pairs[i])),len(tie)+i)))
                    predictions.append(dict(id=o["id"],window=key,model=arm,budget=n,nodes=nodes[:n],
                                            order=order,seconds=seconds))
        print(key, "ranked", flush=True)
    save_rows(OUT/"predictions.jsonl", predictions)


def run():
    OUT.mkdir(exist_ok=True)
    write_json(OUT/"protocol.json",dict(target_per_week=TARGET,attempt_cap=CAP,budgets=BUDGETS,
        arms=MODELS,train_seed_base=[61100,71100],rolling_cuts=[6,8,10],final_cut=12,
        control="Independent mask-eligible draws, same number per week as conditional",
        primary="Both true sources among first 32 nodes of frozen validation events",
        secondary="Top 1 and top 32 unordered pairs by frozen direct score. Warm single-thread kernel timing."))
    raw = load_events(SOURCE)
    generate(raw)
    graphs = {g["window"]:g for g in read_rows(COLLEGE/"public/graphs.jsonl")}
    models = training(raw, graphs)
    score(raw, graphs, models)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
