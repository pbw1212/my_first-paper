"""Separate present-window eligibility and information about the source identities in prior estimation."""
import argparse

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from blind_source_prior import (NETWORKS,OUT,PROFILES,SEEDS,SOURCE,activity,correlated_prior,draws,estimate,
                               law_arrays,representation,windows)
from build_dept3_pilot import mask,simulate
from distilled_source_pairs import PublicWindow
from evaluate_factorial_history import fitted
from silent_source_oracle import CompleteTrace

REFERENCE = 512
METHODS = ("past_public","current_public","current_trace","current_identities")


def current_acceptance(g,stream,w,name,j):
    a,b,probs,_ = law_arrays(w)
    counts = []
    for k,law in enumerate(probs):
        rng = np.random.default_rng([723831,NETWORKS.index(name),j,k])
        count = 0
        for i in rng.choice(len(a),REFERENCE,p=law):
            src = [g["nodes"][u] for u in (a[i],b[i])]
            count += mask(simulate(stream,src,int(rng.integers(2**62)))) is not None
        counts.append(count)
    return (np.array(counts)+.5)/(REFERENCE+1),counts


def adaptation_traces(g,stream,w,edge,name,j,profile):
    # Replay exactly the frozen adaptation batch, including its original random-number consumption.
    from blind_source_prior import ADAPT_DRAWS
    left,right,classes,_ = law_arrays(w)
    mix = PROFILES[profile]
    prob = edge[left,right] if mix is None else np.asarray(mix)@classes
    prob /= prob.sum()
    k = list(PROFILES).index(profile)
    rng = np.random.default_rng([47361122,NETWORKS.index(name),j,k,1])
    rows = []
    for seq,i in enumerate(rng.choice(len(left),ADAPT_DRAWS,p=prob)):
        src = [g["nodes"][v] for v in (left[i],right[i])]
        tr = simulate(stream,src,int(rng.integers(2**62)))
        masked = mask(tr)
        if masked:
            mid = f"{g['window']}-{k}-0-{seq}"
            rows.append((dict(id=mid,observed=masked[1],window=g["window"]),dict(sources=src,trajectory=tr,cut_batches=masked[0])))
            if len(rows)==32:
                break
    return rows


def run(name):
    folder = OUT/name
    original = {r["id"]:r for r in fg.read_rows(folder/"results.jsonl")}
    fits = {(r["window"],r["profile"]):r for r in fg.read_rows(folder/"adaptation.jsonl")}
    raw,graphs = windows(name,False)
    nets = {s:fitted(SOURCE[name],s,"C") for s in SEEDS}
    hits = {}
    learned = []
    for j,(g,stream) in enumerate(graphs):
        w = activity(g,raw)
        _,_,_,means = law_arrays(w)
        current,counts = current_acceptance(g,stream,w,name,j)
        edge = correlated_prior(raw,g)
        ix = {n:i for i,n in enumerate(g["nodes"])}
        hist = None
        world = None
        for profile in PROFILES:
            reference = fits[g["window"],profile]
            adapt = adaptation_traces(g,stream,w,edge,name,j,profile)
            assert [e["id"] for e,_ in adapt]==reference["observations_used"]
            test,gold = draws(g,stream,w,edge,name,j,profile,1)
            if not test:
                continue
            if hist is None:
                hist = PublicWindow(g,raw)
                world = CompleteTrace(g,stream)
            before = (np.array(reference["reference_accepted"])+.5)/(reference["ref_attempts_each"]+1)
            pub = {s:[] for s in SEEDS}
            complete,identities = [],[]
            for e,label in adapt:
                _,_,output = representation(hist,e,nets,w,ix,means)
                for s in SEEDS:
                    pub[s].append(output[s][1])
                anchor,partners,prob = world.evaluate(label,return_distribution=True)
                wa = w[ix[anchor]]
                wb = prob@w[[ix[v] for v in partners]]
                complete.append(np.array([1.,wa+wb,wa*wb])/means)
                u,v = [w[ix[n]] for n in label["sources"]]
                identities.append(np.array([1.,u+v,u*v])/means)
            trace_mix = estimate(complete,current)
            identity_mix = estimate(identities,current)
            mixing = {}
            for s in SEEDS:
                past = estimate(pub[s],before)
                assert np.allclose(past,reference["estimated"][str(s)])
                mixing[s] = np.array([past,estimate(pub[s],current),trace_mix,identity_mix])
            learned.append(dict(window=g["window"],profile=profile,reference_successes=counts,past_rates=before.tolist(),
                                current_rates=current.tolist(),estimates={s:v.tolist() for s,v in mixing.items()},n=len(adapt)))
            for e in test:
                nodes,_,output = representation(hist,e,nets,w,ix,means)
                tie = fg.tie_order(e["id"],len(nodes))
                labels = np.isin(nodes,gold[e["id"]])
                assert labels.sum()==2
                record = {}
                for s in SEEDS:
                    values = mixing[s]@output[s][0]
                    for m,score in zip(METHODS,values,strict=True):
                        record[f"{m}/{s}"] = int(labels[fg.sorted_nodes(score,tie)[:12]].sum())
                    assert record[f"past_public/{s}"]==original[e["id"]]["hits"][f"adaptive/{s}"]
                hits[e["id"]] = dict(id=e["id"],window=g["window"],profile=profile,hits=record)
    assert set(hits)==set(original)
    dest = folder/"privileged_diagnostic"
    dest.mkdir(exist_ok=True)
    fg.save_rows(dest/"predictions.jsonl",hits.values())
    fg.save_rows(dest/"fit.jsonl",learned)
    summary = {}
    for p in PROFILES:
        subset = [r for r in hits.values() if r["profile"]==p]
        summary[p] = {m:dict(n=len(subset),pct=100*np.mean([r["hits"][f"{m}/{s}"]==2 for r in subset for s in SEEDS])) for m in METHODS}
    fg.write_json(dest/"summary.json",summary)
    print(name,[(p,[round(summary[p][m]["pct"],2) for m in METHODS]) for p in PROFILES])


if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("network",choices=NETWORKS)
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run(arg.parse_args().network)
