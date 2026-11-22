"""Check importance scores by enumeration, rebuild selected histories and audit ranks."""

import json
from itertools import combinations
from statistics import mean

import numpy as np
from numba import set_num_threads
from scipy.special import logsumexp

from build_dept3_pilot import DAY, OUT as PILOT, SOURCE, SPLITS, build_split, load_events, read_rows, save_rows
from dept3_forward_model import prefix_graph
from run_dept3_candidates import write_json
from run_dept3_forward import OUT, design, simulate_pairs
from run_dept3_pair_ranker import OUT as PAIR_RUN


def exact_test():
    base = np.ones(6)
    edges = np.zeros((6, 6))
    edges[2, 4], edges[3, 5] = 2., 5.
    pairs = np.array([[0, 1]])
    visible, ends = np.array([4, 5]), np.array([2])
    guide = np.array([0., 0., 3., 1., 0., 0.])
    draws = np.array([[0.125], [0.375], [0.625], [0.875]])
    scores, _ = simulate_pairs(base, edges, pairs, visible, ends, guide, np.ones(4, dtype=int), draws, False)
    # Each allowable hidden node has forward probability 1/4; q assigns them 3/4 and 1/4.
    joint = 0.25 * ((3/5) * (1/5) + (1/8) * (6/8))
    assert np.allclose(scores[0, :3], [np.log(1/4), np.log(joint)/2, np.log(joint/.5)/2])
    swapped, _ = simulate_pairs(base, edges, pairs, visible[::-1].copy(), ends, guide,
                                np.ones(4, dtype=int), draws, False)
    assert np.allclose(scores, swapped)
    empty, _ = simulate_pairs(base, edges, pairs, visible, ends, guide, np.zeros(4, dtype=int), draws, False)
    assert np.allclose(empty[0, :5], np.log(.25))


def scalar(obs, nodes, pairs, graph, model, raw, start, particles=32):
    base, edges, visible, ends, guide, lengths, draws = design(obs, graph, model, particles)
    ids, ix = graph[:2]
    past = np.zeros_like(edges)
    recent = np.zeros_like(edges)
    for t, u, v in raw:
        if t < start:
            past[ix[u], ix[v]] += 1
            if t >= start - 28 * DAY:
                recent[ix[u], ix[v]] += 1
    w = model["weights"]
    marginal = w[0]/len(ids) + w[1]*(recent.sum(axis=0)+1)/(len(ids)+recent.sum())
    contact = past*w[2]/np.maximum(1, past.sum(axis=1)[:, None]) + recent*w[3]/np.maximum(1, recent.sum(axis=1)[:, None])
    assert np.allclose(marginal, base) and np.allclose(contact, edges)
    look = contact[:, visible[:ends[0]]].mean(axis=1)
    proposal = 1 + 2 * look / max(look.max(), 1e-15)
    proposal[visible] = 0
    assert np.allclose(guide, proposal)
    def probs(active):
        r = marginal + contact[list(active)].sum(axis=0)
        r[list(active)] = 0
        return r / r.sum()
    def suffix(active):
        active = active.copy()
        result = 0.
        for batch in obs["observed"]:
            group = [ix[v] for v in batch]
            result += np.log(probs(active)[group]).sum()
            active.update(group)
        return result
    result = []
    for a, b in pairs:
        seed = {ix[nodes[a]], ix[nodes[b]]}
        ratios, path_scores = [], []
        for h, numbers in zip(lengths, draws):
            active = seed.copy()
            ratio = 0.
            for u in numbers[:h]:
                p = probs(active)
                q = p * proposal
                q /= q.sum()
                node = min(np.searchsorted(np.cumsum(q), u, side="right"), len(q)-1)
                assert q[node] > 0 and node not in visible and node not in active
                ratio += np.log(p[node]/q[node])
                active.add(node)
            ratios.append(ratio)
            path_scores.append(ratio + suffix(active))
        m = len(visible)
        values = np.array(path_scores)
        z = logsumexp(values)
        h = len(values)//2
        result.append([suffix(seed)/m, (z - np.log(len(values)))/m,
                       (z-logsumexp(ratios))/m, (logsumexp(values[:h])-np.log(h))/m,
                       (logsumexp(values[h:])-np.log(h))/m, 1/np.square(np.exp(values-z)).sum()])
    return result


def run():
    set_num_threads(1)
    exact_test()
    raw = load_events(SOURCE)
    _, train, labels, _ = build_split(raw, "train", *SPLITS["train"], target=2000)
    obs = {r["id"]: r for r in train + read_rows(PILOT / "public/validation.jsonl")}
    truth = {r["id"]: set(r["sources"]) for r in labels + read_rows(PILOT / "truth/validation.jsonl")}
    public = {g["window"]: g for g in read_rows(PILOT / "public/graphs.jsonl")}
    models = json.loads((OUT / "models.json").read_text(encoding="utf-8"))["folds"]
    pairs = list(combinations(range(32), 2))
    result = {}
    for split in ("rolling", "validation"):
        baseline = {r["id"]: r for r in read_rows(PAIR_RUN / f"{split}_rankings.jsonl")}
        records = {r["id"]: r for r in read_rows(OUT / f"{split}_true_ranks.jsonl")}
        with np.load(OUT / f"{split}_scores.npz") as arch:
            ids, pools, values = arch["ids"].tolist(), arch["nodes"].tolist(), arch["scores"]
        assert len(ids) == len(set(ids)) == len(baseline) and np.isfinite(values).all()
        folds = set()
        for sid, nodes, scores in zip(ids, pools, values, strict=True):
            meta, o = baseline[sid], obs[sid]
            assert nodes == meta["nodes"]
            g, model = public[o["window"]], models[meta["fold"]]
            assert all(public[k]["end"] <= g["start"] for k in model["windows"])
            m = sum(map(len, o["observed"]))
            assert np.allclose((np.logaddexp(scores[:, 3]*m, scores[:, 4]*m)-np.log(2))/m, scores[:, 1])
            assert np.all((scores[:, -1]>=1-1e-9) & (scores[:, -1]<=32+1e-9))
            sidpair = next((i for i, (a,b) in enumerate(pairs) if {nodes[a], nodes[b]} == truth[sid]), None)
            tiebreak = np.argsort(meta["sum"])
            for i, key in enumerate(("direct", "prefix", "conditional", "half_a", "half_b")):
                rank = None if sidpair is None else 1 + np.sum(scores[:, i] > scores[sidpair, i]) + np.sum(
                    (scores[:, i] == scores[sidpair, i]) & (tiebreak < tiebreak[sidpair]))
                assert rank == records[sid][key]
            if meta["fold"] not in folds:
                choices = [0, 61, 265, 495]
                ref = scalar(o, nodes, [pairs[i] for i in choices], prefix_graph(g, raw), model, raw, g["start"])
                assert np.allclose(ref, scores[choices], atol=1e-8, rtol=0.)
                folds.add(meta["fold"])
        with np.load(OUT / f"{split}_scores_128.npz") as archive:
            hi_ids, hi_nodes, hi = archive["ids"].tolist(), archive["nodes"].tolist(), archive["scores"]
        assert hi_ids == ids
        sample = []
        idx = {sid: i for i, sid in enumerate(ids)}
        high_checks = set()
        per_window = {}
        diagnostics = []
        for sid, nodes, high in zip(hi_ids, hi_nodes, hi, strict=True):
            low, meta = values[idx[sid]], baseline[sid]
            assert nodes == meta["nodes"] and np.isfinite(high).all() and np.allclose(low[:,0], high[:,0], rtol=0, atol=1e-12)
            g, model = public[meta["window"]], models[meta["fold"]]
            if meta["fold"] not in high_checks:
                ref = scalar(obs[sid], nodes, [pairs[61]], prefix_graph(g, raw), model, raw, g["start"], 128)
                assert np.allclose(ref[0], high[61], atol=1e-8, rtol=0.)
                high_checks.add(meta["fold"])
            order = np.argsort(meta["sum"])
            full_order = np.lexsort((order, -high[:,1]))
            hi_top = full_order[:32]
            lo_top = np.lexsort((order, -low[:,1]))[:32]
            hi_a = np.lexsort((order,-high[:,3]))[:32]
            hi_b = np.lexsort((order,-high[:,4]))[:32]
            target = next((i for i,(a,b) in enumerate(pairs) if {nodes[a], nodes[b]} == truth[sid]), None)
            combined = np.lexsort((order, np.argsort(full_order) + np.argsort(meta["joint"])))
            rank = None if target is None else int(np.flatnonzero(full_order == target)[0]) + 1
            diagnostics.append({"id": sid, "window": meta["window"], "direct_rank": records[sid]["direct"],
                                "prefix32_rank": records[sid]["prefix"], "prefix128_rank": rank,
                                "joint_plus_prefix128_rank": None if target is None else int(np.flatnonzero(combined == target)[0])+1})
            sample.append((records[sid]["direct"] is not None and records[sid]["direct"]<=32,
                           target in lo_top.tolist(), target in hi_top.tolist(),
                           len(set(hi_a) & set(hi_b))/32, len(set(hi_top) & set(lo_top))/32,
                           target in hi_a.tolist(), target in hi_b.tolist()))
            block = per_window.setdefault(meta["window"], [0, 0, 0])
            block[0] += 1
            block[1] += sample[-1][0]
            block[2] += sample[-1][2]
        save_rows(OUT / f"{split}_sensitivity_ranks.jsonl", diagnostics)
        result[split] = {"main_events_checked": len(ids), "high_budget_events": len(sample),
                         "direct_hits": sum(r[0] for r in sample), "prefix32_hits": sum(r[1] for r in sample),
                         "prefix128_hits": sum(r[2] for r in sample), "overlap_64_vs_64": mean(r[3] for r in sample),
                         "overlap_32_vs_128": mean(r[4] for r in sample),
                         "128_gained_vs_direct": sum(r[2] and not r[0] for r in sample),
                         "128_lost_vs_direct": sum(r[0] and not r[2] for r in sample),
                         "half64_a_hits": sum(r[5] for r in sample), "half64_b_hits": sum(r[6] for r in sample),
                         "prefix128_top1": sum(r["prefix128_rank"] == 1 for r in diagnostics),
                         "prefix128_median_in_pool": float(np.median([r["prefix128_rank"] for r in diagnostics if r["prefix128_rank"] is not None])),
                         "joint_plus_prefix128_hits": sum(r["joint_plus_prefix128_rank"] is not None and r["joint_plus_prefix128_rank"]<=32 for r in diagnostics),
                         "windows_n_direct_prefix128": per_window}
        if split == "rolling":
            data = np.asarray(list(per_window.values()))
            selections = np.random.default_rng(11706).integers(len(data), size=(10000, len(data)))
            deltas = (data[:,2]-data[:,1])[selections].sum(axis=1) / data[:,0][selections].sum(axis=1)
            result[split]["window_interval_128_minus_direct"] = np.quantile(deltas, [.025, .975]).tolist()
    write_json(OUT / "sampling_check.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    run()
