"""Shortlist two hidden sources with a relative posterior coverage certificate."""
import argparse
from pathlib import Path

import numpy as np
import torch
from numba import njit
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from blind_source_prior import PROFILES, correlated_prior, law_arrays, representation
from build_dept3_pilot import mask, simulate
from distilled_source_pairs import PublicWindow
from evaluate_factorial_history import fitted
from source_prior_shift import NETWORKS, OUT as BASE, SEEDS, SOURCE, activity, excluding_self, windows

OUT = BASE.parent / "robust_source_list"
BUDGET = 12
DRAWS = 120
RNG = 82041693
METHODS = ("activity", "raw", "fixed_marginal", "fixed_joint", "robust", "both")


def protocol():
    paths = [Path(__file__).resolve(), BASE/"protocol.json", BASE.parent/"blind_source_prior"/"protocol.json"]
    return dict(random_seed=RNG, draws_per_profile_window=DRAWS, budget=BUDGET, profiles=PROFILES,
                seeds=list(SEEDS), source_models=SOURCE, methods=list(METHODS),
                primary="paired robust versus fixed joint coverage, by network and generator; no pooled winner",
                certificate="minimum ratio of within-list pair mass to a component optimum upper bound; model-based",
                solver="same four starts; improving single swaps for both objectives; dominance reduction",
                hashes={p.relative_to(OUT.parents[1]).as_posix():fg.digest(p) for p in paths})


@njit(cache=True)
def frontier(order, y, k):
    # Earlier candidates dominate in x. At equal x, order by descending y then tie.
    kept = []
    largest = np.full(k, -np.inf)
    for i in order:
        if y[i] > largest[0]:
            kept.append(i)
            largest[0] = y[i]
            largest.sort()
    return np.array(kept, dtype=np.int64)


@njit(cache=True)
def objective(masses, coeff, robust):
    if robust:
        return min(masses[0], masses[1], masses[2])
    return masses[0]*coeff[0] + masses[1]*coeff[1] + masses[2]*coeff[2]


@njit(cache=True)
def improve(edges, starts, coeff, robust):
    best = -1.
    solution = starts[0].copy()
    n = len(edges)
    for start in starts:
        chosen = start.copy()
        used = np.zeros(n, np.bool_)
        incident = np.zeros((n,3))
        for i in chosen:
            used[i] = True
            incident += edges[:,i,:]
        mass = incident[chosen].sum(axis=0)*.5
        while True:
            value = objective(mass,coeff,robust)
            high = value + 1e-13
            add = -1
            slot = -1
            for pos, old in enumerate(chosen):
                for new in range(n):
                    if not used[new]:
                        candidate = mass - incident[old] + incident[new] - edges[new,old]
                        trial = objective(candidate,coeff,robust)
                        if trial > high:
                            high, add, slot = trial, new, pos
            if add < 0:
                break
            old = chosen[slot]
            mass = mass - incident[old] + incident[add] - edges[add,old]
            incident += edges[:,add,:] - edges[:,old,:]
            chosen[slot] = add
            used[old], used[add] = False, True
        if value > best:
            best, solution = value, chosen.copy()
    return solution


def decode(h, weights, means, tie, k=BUDGET):
    y = h*weights
    keep = frontier(np.lexsort((tie,-y,-h)), y, k)
    xk, yk = h[keep], y[keep]
    edges = np.stack((np.outer(xk,xk), np.outer(xk,yk)+np.outer(yk,xk), np.outer(yk,yk)),axis=-1)
    edges[np.arange(len(keep)),np.arange(len(keep))] = 0.
    def top(values):
        return fg.sorted_nodes(values, tie[keep])[:k]
    first, last = top(xk), top(yk)
    # Sum over the strongest k-1 neighbors bounds each included row's degree.
    largest_degrees = np.sort(edges[:,:,1],axis=1)[:,-(k-1):].sum(1)
    bound = np.array([edges[first[:,None],first,0].sum()/2,
                      np.sort(largest_degrees)[-k:].sum()/2,
                      edges[last[:,None],last,2].sum()/2])
    scaled = np.ascontiguousarray(edges / bound)
    fixed = bound / means
    fixed /= fixed.sum()
    # Use the whole legal roster for marginal ranking, including those outside the frontier.
    hx, hy = excluding_self(h), excluding_self(y)
    marginal = h*hx/means[0]+(h*hy+y*hx)/means[1]+y*hy/means[2]
    starts = np.stack((first,last,top(marginal[keep]),top(np.sqrt(xk*yk))))
    inside = (top(marginal[keep]), improve(scaled,starts,fixed,False), improve(scaled,starts,fixed,True))
    lists = [keep[v] for v in (first,*inside,last)]
    certs = [float(scaled[idx[:,None],idx].sum(axis=(0,1)).min()/2) for idx in (first,*inside,last)]
    return lists, certs, bound, len(keep)


def draw(g, stream, w, edge, net, j, profile):
    a,b,laws,_ = law_arrays(w)
    mix = PROFILES[profile]
    p = edge[a,b].copy() if mix is None else np.asarray(mix) @ laws
    pi = list(PROFILES).index(profile)
    rng = np.random.default_rng([RNG, NETWORKS.index(net), j, pi])
    for index,pair in enumerate(rng.choice(len(a),DRAWS,replace=True,p=p/p.sum())):
        src = [g["nodes"][q] for q in (a[pair],b[pair])]
        suffix = mask(simulate(stream,src,int(rng.integers(2**62))))
        if suffix is not None:
            event = dict(id=f"rob-{j}-{pi}-{index}",window=g["window"],observed=suffix[1],k=2)
            yield event, src


def run(net):
    assert fg.load(OUT/"protocol.json")==protocol()
    raw,gs = windows(net,False)
    models = {s:fitted(SOURCE[net],s,"C") for s in SEEDS}
    folder = OUT/net
    folder.mkdir(exist_ok=True)
    rows = []
    for j,(g,stream) in enumerate(gs):
        w = activity(g,raw)
        _,_,_,means = law_arrays(w)
        edge = correlated_prior(raw,g)
        hist = PublicWindow(g,raw)
        ix = {n:i for i,n in enumerate(g["nodes"])}
        for p in PROFILES:
            for event,src in draw(g,stream,w,edge,net,j,p):
                nodes, ids, outputs = representation(hist,event,models,w,ix,means)
                local = w[ids]
                tie = fg.tie_order(event["id"], len(nodes))
                record = dict(id=event["id"],window=g["window"],profile=p,sources=src,hits=[],certificates=[],frontier=[])
                volume = fg.sorted_nodes(local,tie)[:BUDGET]
                observed = set(src)
                for _,(_,_,h) in outputs.items():
                    lists, cert, _, count = decode(h,local,means,tie)
                    ranked = [volume,*lists]
                    assert all(len(set(v))==BUDGET for v in ranked)
                    assert all(0 < v <= 1+1e-10 for v in cert)
                    record["hits"].append([int(observed <= {nodes[q] for q in v}) for v in ranked])
                    record["certificates"].append(np.round(cert,8).tolist())
                    record["frontier"].append(count)
                rows.append(record)
    fg.save_rows(folder/"events.jsonl",rows)
    fg.write_json(folder/"audit.json",dict(n=len(rows),sha256=fg.digest(folder/"events.jsonl"),
                                          windows=[g["window"] for g,_ in gs]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze",*NETWORKS))
    cmd=parser.parse_args().action
    OUT.mkdir(exist_ok=True)
    if cmd=="freeze":
        if not (OUT/"protocol.json").exists():
            fg.write_json(OUT/"protocol.json",protocol())
        assert fg.load(OUT/"protocol.json")==protocol()
    else:
        torch.set_num_threads(1)
        with threadpool_limits(1):
            run(cmd)
