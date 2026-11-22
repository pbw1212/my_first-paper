"""Refit the pair model on unordered membership with the original supervision and splits."""
import argparse

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from distilled_source_pairs import (CHECKPOINTS, ORDINARY, PLAN, CompleteTrace, Packed, PairNet, PublicWindow,
    evaluation_windows, load, make_windows, pair_rank, raw_data, read_rows, save_rows, shortlist, write_json)
from matched_source_baselines import MAIN, OUT, RUNS, digest, tie_order


def snapshot(event):
    members = sorted(v for group in event["observed"] for v in group)
    assert len(set(members)) == len(members)
    return dict(event, observed=[members])


def examples(name, phase):
    raw = raw_data(name)
    graphs, streams = make_windows(raw, name, .15, 1.)
    public = read_rows(ORDINARY/name/(phase+"_public.jsonl"))
    truth = {r["id"]: r for r in read_rows(ORDINARY/name/(phase+"_truth.jsonl"))}
    for graph in graphs:
        key = graph["window"]
        events = [r for r in public if r["window"] == key]
        if not events:
            continue
        window = PublicWindow(graph, raw)
        oracle = CompleteTrace(graph, streams[key]) if phase == "fit" else None
        for event in events:
            y = truth[event["id"]]
            (nodes, _, _), features = window.design(snapshot(event))
            a, b = y["sources"]
            partners, weights = [b], [1.]
            if oracle is not None:
                a, partners, weights = oracle.evaluate(y, return_distribution=True)
                b = next(v for v in y["sources"] if v != a)
            idx = {v: i for i, v in enumerate(nodes)}
            target = np.zeros(len(idx), dtype=np.float32)
            target[[idx[v] for v in partners]] = weights
            assert target[idx[a]] == 0 and np.isclose(target.sum(), 1)
            yield (*features, idx[a], idx[b], target)


def fit(name):
    folder = OUT/name
    rows = list(examples(name, "fit"))
    x = np.vstack([r[0] for r in rows])
    mean, std = x.mean(0), np.maximum(x.std(0), .025)
    train = Packed(rows, mean, std)
    del rows, x
    val = Packed(list(examples(name, "cal")), mean, std)
    assert (len(train.x), len(val.x)) == (1000, 300)
    np.savez(folder/"snapshot_scale.npz", mean=mean, std=std)
    torch.manual_seed(PLAN["training_seed"])
    rng = np.random.default_rng(PLAN["training_seed"])
    model = PairNet(len(mean), True)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.005)
    best = float("inf")
    curve = {}
    for epoch in range(1, max(CHECKPOINTS)+1):
        model.train()
        for idx in np.array_split(rng.permutation(len(train.x)), int(np.ceil(len(train.x)/24))):
            opt.zero_grad()
            train.loss(model, idx, True).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        if epoch in CHECKPOINTS:
            model.eval()
            with torch.no_grad():
                curve[epoch] = sum(len(i)*val.loss(model, i, False).item()
                                   for i in np.array_split(np.arange(len(val.x)), 15))/len(val.x)
            if curve[epoch] < best:
                best = curve[epoch]
                torch.save(model.state_dict(), folder/"snapshot_pair.pt")
    write_json(folder/"snapshot_selection.json", dict(validation_nll=curve, epoch=min(curve, key=curve.get),
        seed=PLAN["training_seed"], change="sort all visible IDs into one group before feature extraction for fit, cal and evaluation",
        hashes={f: digest(folder/f) for f in ("snapshot_pair.pt", "snapshot_scale.npz")}))


def evaluate(run):
    main = MAIN/run
    plan = load(main/"test_plan.json")
    trained = OUT/plan["source"]
    cfg = load(trained/"snapshot_selection.json")
    assert all(digest(trained/f) == h for f, h in cfg["hashes"].items())
    scale = np.load(trained/"snapshot_scale.npz")
    model = PairNet(len(scale["mean"]), True).eval()
    model.load_state_dict(torch.load(trained/"snapshot_pair.pt", weights_only=True))
    raw = raw_data(plan["network"])
    graphs, _ = (make_windows(raw, plan["network"], .15, 1.) if plan.get("extra_window_check", False)
                 else evaluation_windows(plan["network"], raw))
    public = read_rows(main/"public.jsonl")
    truth = {r["id"]: r for r in read_rows(main/"truth.jsonl")}
    result = []
    for g in graphs:
        if g["window"] not in plan["windows"]:
            continue
        w = PublicWindow(g, raw)
        for e in (r for r in public if r["window"] == g["window"]):
            (nodes, _, _), (x, edge, overlap) = w.design(snapshot(e))
            tx = torch.from_numpy(np.clip((x-scale["mean"])/scale["std"], -6, 6))[None]
            with torch.no_grad():
                scores = model(tx, torch.tensor(edge.astype(np.float16), dtype=torch.float32)[None],
                               torch.tensor(overlap.astype(np.float16), dtype=torch.float32)[None],
                               torch.ones(1, len(nodes), dtype=torch.bool))[0].numpy()
            tie = tie_order(e["id"], len(nodes))
            choice = [nodes[i] for i in shortlist(scores, tie)[0]]
            a, b = (nodes.index(v) for v in truth[e["id"]]["sources"])
            result.append(dict(id=e["id"], chosen=choice, rank=pair_rank(scores, a, b, tie)))
    assert len(result) == len(public)
    save_rows(OUT/run/"snapshot_predictions.jsonl", result)
    print(run, len(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=RUNS)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.phase == "fit":
            assert args.name in ("company", "Dept2")
            fit(args.name)
        else:
            evaluate(args.name)
