"""Fit an additive, directed contact model for the next first-time participant."""

import hashlib
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from build_dept3_pilot import DAY, OUT as PILOT, RAW_SHA, SOURCE, SPLITS, load_events, make_windows, mask, read_rows, simulate
from run_dept3_candidates import OUT as NODE_RUN


def prefix_graph(graph, events):
    nodes = np.array(graph["nodes"])
    ix = {int(v): i for i, v in enumerate(nodes)}
    n = len(nodes)
    old = np.zeros((n, n))
    recent = np.zeros_like(old)
    for u, v, count in graph["edges"]:
        old[ix[u], ix[v]] = count
    t0 = graph["start"]
    for t, u, v in events:
        if t >= t0:
            break
        if t >= t0 - 28 * DAY:
            recent[ix[u], ix[v]] += 1
    receiver = 1 + recent.sum(axis=0)
    base = np.column_stack((np.full(n, 1 / n), receiver / receiver.sum(), np.zeros((n, 2))))
    old /= np.maximum(1, old.sum(axis=1, keepdims=True))
    recent /= np.maximum(1, recent.sum(axis=1, keepdims=True))
    return nodes, ix, base, old, recent


def transition_rows(graph, seeds, batches):
    _, index, base, old, recent = graph
    rates = base.copy()
    joined = np.zeros(len(rates), dtype=bool)
    def add(group):
        rates[:, 2] += old[group].sum(axis=0)
        rates[:, 3] += recent[group].sum(axis=0)
        joined[group] = True
    add([index[v] for v in seeds])
    for batch in batches:
        group = [index[v] for v, _ in batch["joins"]]
        available = rates[~joined]
        yield rates[group], available.sum(axis=0), available
        add(group)


def history():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError("Unexpected source archive")
    events = load_events(SOURCE)
    _, streams = make_windows(events, "train", *SPLITS["train"][:2])
    graphs = {g["window"]: prefix_graph(g, events) for g in read_rows(PILOT / "public/graphs.jsonl")}
    stats = defaultdict(lambda: {"num": [], "den": [], "lengths": [], "attempts": 0, "zero": 0})
    obs = {}
    for draw in read_rows(NODE_RUN / "training_draws.jsonl"):
        key, sid = draw["window"], draw["accepted_id"]
        traj = simulate(streams[key], draw["sources"], draw["draw_seed"])
        block = stats[key]
        block["attempts"] += 1
        block["zero"] += not traj
        for num, den, _ in transition_rows(graphs[key], draw["sources"], traj):
            block["num"].extend(num)
            block["den"].extend([den] * len(num))
        if sid is not None:
            cut, visible = mask(traj)
            block["lengths"].append((sum(map(len, visible)), sum(len(b["joins"]) for b in traj[:cut])))
            obs[sid] = {"id": sid, "window": key, "observed": visible}
    obs.update((r["id"], r) for r in read_rows(PILOT / "public/validation.jsonl"))
    return graphs, stats, obs


def fit_forward(stats, cutoff):
    windows = sorted(k for k in stats if int(k.split("-")[1]) <= cutoff)
    num = np.vstack([v for k in windows for v in stats[k]["num"]])
    den = np.vstack([v for k in windows for v in stats[k]["den"]])
    def loss(params):
        w = np.exp(np.r_[0., params])
        success, total = num @ w, den @ w
        grad = ((den / total[:, None] - num / success[:, None]).mean(axis=0) * w)[1:]
        return np.log(total / success).mean() + 0.00005 * (params @ params), grad + 0.0001 * params
    res = minimize(loss, np.zeros(3), jac=True, method="L-BFGS-B", bounds=[(-12, 12)] * 3,
                   options={"gtol": 1e-8, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(res.message)
    lengths = defaultdict(list)
    for k in windows:
        for m, h in stats[k]["lengths"]:
            lengths[m].append(h)
    return {"windows": windows, "weights": np.exp(np.r_[0., res.x]).tolist(),
            "attempts": sum(stats[k]["attempts"] for k in windows),
            "zero_growth": sum(stats[k]["zero"] for k in windows), "new_members": len(num),
            "hidden_length_by_visible": {str(m): sorted(h) for m, h in lengths.items()}}


def kernels(graph, model):
    _, _, base, old, recent = graph
    w = model["weights"]
    return base[:, 0] * w[0] + base[:, 1] * w[1], old * w[2] + recent * w[3]
