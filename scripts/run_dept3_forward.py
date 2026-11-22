"""Compare direct and latent-prefix forward scores on frozen dual-source pools."""

import argparse
import hashlib
import json

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from threadpoolctl import threadpool_limits

from build_dept3_pilot import OUT as PILOT, read_rows
from dept3_forward_model import fit_forward, history, kernels
from run_dept3_candidates import write_json
from run_dept3_pair_ranker import LEFT, RIGHT, OUT as PAIR_RUN


OUT = PILOT.parent / "dept3_forward"
PARTICLES = 32
COLUMNS = ["direct", "prefix", "conditional", "half_a", "half_b", "ess"]


@njit
def advance(rate, edges, node):
    rate[node] = 0.
    for v in range(len(rate)):
        if rate[v] > 0:
            rate[v] += edges[node, v]


@njit
def suffix_log(rate, edges, observed, ends):
    start, value = 0, 0.
    for end in ends:
        total = rate.sum()
        for j in range(start, end):
            value += np.log(rate[observed[j]] / total)
        if end != len(observed):
            for j in range(start, end):
                advance(rate, edges, observed[j])
        start = end
    return value


@njit
def logmean(x):
    peak = np.max(x)
    return peak + np.log(np.mean(np.exp(x - peak)))


@njit(parallel=True)
def simulate_pairs(base, edges, pairs, observed, ends, guide, lengths, uniforms, estimate):
    m, steps = uniforms.shape
    scores = np.empty((len(pairs), 6))
    guesses = np.full((len(pairs), steps if estimate else 0), -1, dtype=np.int64)
    for k in prange(len(pairs)):
        a, b = pairs[k]
        initial = base + edges[a] + edges[b]
        initial[a] = initial[b] = 0.
        scores[k, 0] = suffix_log(initial.copy(), edges, observed, ends) / len(observed)
        logw, joint = np.zeros(m), np.empty(m)
        path = np.empty((m, steps), dtype=np.int64)
        for s in range(m):
            rate = initial.copy()
            for t in range(lengths[s]):
                forward_z = rate.sum()
                proposal_z = np.dot(rate, guide)
                draw = uniforms[s, t] * proposal_z
                partial, node = 0., -1
                for v in range(len(rate)):
                    mass = rate[v] * guide[v]
                    if mass > 0:
                        node = v
                        partial += mass
                        if draw < partial:
                            break
                logw[s] += np.log(proposal_z / (forward_z * guide[node]))
                if estimate:
                    path[s, t] = node
                advance(rate, edges, node)
            joint[s] = logw[s] + suffix_log(rate, edges, observed, ends)
        evidence = logmean(joint)
        scores[k, 1] = evidence / len(observed)
        scores[k, 2] = (evidence - logmean(logw)) / len(observed)
        scores[k, 3] = logmean(joint[:m//2]) / len(observed)
        scores[k, 4] = logmean(joint[m//2:]) / len(observed)
        weights = np.exp(joint - np.max(joint))
        weights /= weights.sum()
        scores[k, 5] = 1 / np.sum(weights * weights)
        if estimate:
            inclusion = np.zeros(len(base))
            for s in range(m):
                for t in range(lengths[s]):
                    inclusion[path[s, t]] += weights[s]
            count = int(np.rint(np.dot(weights, lengths.astype(np.float64))))
            for t in range(count):
                best = np.argmax(inclusion)
                guesses[k, t] = best
                inclusion[best] = -1.
    return scores, guesses


def design(obs, graph, model, particles=PARTICLES):
    _, ix, _, _, _ = graph
    base, edges = kernels(graph, model)
    return design_from_kernel(obs, ix, base, edges, model, particles)


def design_from_kernel(obs, ix, base, edges, model, particles):
    groups = obs["observed"]
    visible = np.array([ix[v] for batch in groups for v in batch])
    ends = np.cumsum([len(batch) for batch in groups])
    bridge = edges[:, visible[:ends[0]]].mean(axis=1)
    guide = 1. + 2. * bridge / max(bridge.max(), 1e-15)
    guide[visible] = 0.
    hmap = model["hidden_length_by_visible"]
    m = len(visible)
    nearest = min(map(int, hmap), key=lambda x: (abs(x - m), x))
    empirical = np.array(hmap[str(nearest)]) * m / nearest
    if particles < PARTICLES or particles % PARTICLES:
        raise ValueError("Use complete 32-particle blocks")
    length_blocks, draw_blocks = [], []
    for block in range(particles // PARTICLES):
        key = "dept3-prefix-" + obs["id"] + (f"-block-{block}" if block else "")
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        lengths = np.clip(np.rint(rng.choice(empirical, size=PARTICLES)).astype(np.int64),
                          1, min(m, len(base) - m - 2))
        length_blocks.append(lengths)
        draw_blocks.append(rng.random((PARTICLES, max(lengths))))
    lengths = np.concatenate(length_blocks)
    uniforms = np.vstack([np.pad(x, ((0, 0), (0, max(lengths) - x.shape[1]))) for x in draw_blocks])
    return base, edges, visible, ends, guide, lengths, uniforms


def run(sensitivity=False):
    set_num_threads(min(8, get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    graphs, stats, obs = history()
    if sensitivity:
        models = json.loads((OUT / "models.json").read_text(encoding="utf-8"))["folds"]
    else:
        models = {tag: fit_forward(stats, cutoff) for tag, cutoff in
                  (("rolling_10", 10), ("rolling_15", 15), ("rolling_20", 20), ("full2000", 25))}
        write_json(OUT / "models.json", {"particles": PARTICLES, "columns": COLUMNS, "folds": models})
    for split in ("rolling", "validation"):
        rows = read_rows(PAIR_RUN / f"{split}_rankings.jsonl")
        saved = np.empty((len(rows), len(LEFT), len(COLUMNS)))
        for i, row in enumerate(rows):
            graph = graphs[row["window"]]
            base, edges, *rest = design(obs[row["id"]], graph, models[row["fold"]], 128 if sensitivity else PARTICLES)
            ids = np.array([graph[1][v] for v in row["nodes"]])
            pairs = np.column_stack((ids[LEFT], ids[RIGHT]))
            saved[i], _ = simulate_pairs(base, edges, pairs, *rest, False)
            if (i+1) % 250 == 0:
                print(split, i+1, "events", flush=True)
        assert np.isfinite(saved).all()
        file = f"{split}_scores_128.npz" if sensitivity else f"{split}_scores.npz"
        np.savez_compressed(OUT / file, scores=saved, ids=[r["id"] for r in rows],
                            nodes=np.array([r["nodes"] for r in rows]))
        print(split, "finished", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", action="store_true", help="Use 128 trajectories on both complete cohorts")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run(args.sensitivity)
