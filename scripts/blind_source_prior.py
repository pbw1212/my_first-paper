"""Infer a mixture of source priors from unlabelled eligible suffixes in a new batch."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from build_dept3_pilot import mask,simulate
from distilled_source_pairs import PublicWindow
from evaluate_factorial_history import fitted
from five_frame_pairs import frames
from source_prior_shift import DAYS,NETWORKS,OUT as BASE,SEEDS,SOURCE,activity,excluding_self,windows

OUT = BASE.parent/"blind_source_prior"
PROFILES = {"flat":[1.,0.,0.],"mixed":[1/3,1/3,1/3],"mostly_active":[.05,.15,.8],"linked":None}
EQUAL = np.ones(3)/3
REF_DRAWS,ADAPT_DRAWS,TEST_DRAWS = 100,120,180
WIDTH = 14*86400


def plan():
    return dict(primary="adaptive versus fixed equal draw mixture on all eligible evaluation events, per network and profile",
        secondary="adaptive vs raw and vs activity; oracle sees true draw mixture but not latent per-event component",
        generators=PROFILES,source_models=SOURCE,random_seed=47361122,reference_windows=4,
        reference_draws_per_component=REF_DRAWS,adaptation_draws=ADAPT_DRAWS,max_unlabelled_suffixes=32,test_draws=TEST_DRAWS,
        pseudo_accepted_events=3,hashes={str(p.relative_to(OUT.parents[1])):fg.digest(p) for p in
            [Path(__file__).resolve(),BASE/"protocol.json",OUT.parents[1]/"scripts/source_prior_shift.py"]})


def streams_and_graph(raw,start):
    past = Counter((u,v) for t,u,v in raw if t<start)
    nodes = sorted({n for pair in past for n in pair})
    roster = set(nodes)
    grouped = {}
    for t,u,v in raw:
        if start<=t<start+WIDTH and u in roster and v in roster:
            grouped.setdefault(t,[]).append((u,v))
    return dict(nodes=nodes,start=start),list(sorted(grouped.items()))


def law_arrays(w):
    i,j = np.triu_indices(len(w),1)
    value = np.array([np.ones(len(i)),w[i]+w[j],w[i]*w[j]])
    return i,j,value/value.sum(1,keepdims=True),value.mean(1)


def past_acceptance(raw,start,net,cache):
    """Use only past completed streams, without reading test outcomes or their labels."""
    successes = np.zeros(3,int)
    attempts = 0
    for k in range(1,5):
        origin = start-WIDTH*k
        if origin-DAYS<raw[0][0]:
            continue
        if origin not in cache:
            g,stream = streams_and_graph(raw,origin)
            if len(g["nodes"])<2:
                continue
            w = activity(g,raw)
            a,b,dist,_ = law_arrays(w)
            hit = np.zeros(3,int)
            for c in range(3):
                rng = np.random.default_rng([47361122,NETWORKS.index(net),origin%2**32,c,0])
                for ix in rng.choice(len(a),REF_DRAWS,p=dist[c]):
                    pair = [g["nodes"][v] for v in (a[ix],b[ix])]
                    trace = simulate(stream,pair,int(rng.integers(2**62)))
                    hit[c] += mask(trace) is not None
            cache[origin] = hit
        successes += cache[origin]
        attempts += REF_DRAWS
    return (successes+.5)/(attempts+1),successes.tolist(),attempts


def correlated_prior(raw,g):
    nodes = g["nodes"]
    ix = {v:i for i,v in enumerate(nodes)}
    edge = np.zeros((len(ix),len(ix)),float)
    for t,u,v in raw:
        if g["start"]-DAYS<=t<g["start"]:
            edge[ix[u],ix[v]] += 1
    return 1+8*np.log1p(edge+edge.T)


def draws(g,stream,w,edge,net,j,profile,part):
    left,right,components,_ = law_arrays(w)
    profile_mix = PROFILES[profile]
    prob = edge[left,right] if profile_mix is None else np.asarray(profile_mix)@components
    prob /= prob.sum()
    rng = np.random.default_rng([47361122,NETWORKS.index(net),j,list(PROFILES).index(profile),part+1])
    count = ADAPT_DRAWS if part==0 else TEST_DRAWS
    public,labels = [],{}
    for num,ix in enumerate(rng.choice(len(prob),count,p=prob)):
        pair = [g["nodes"][v] for v in (left[ix],right[ix])]
        trace = simulate(stream,pair,int(rng.integers(2**62)))
        m = mask(trace)
        if m:
            sid = f"{g['window']}-{list(PROFILES).index(profile)}-{part}-{num}"
            public.append(dict(id=sid,window=g["window"],observed=m[1],k=2))
            labels[sid] = pair
    return public,labels


def representation(hist,obs,nets,w,ix,means):
    (nodes,_,_),(x,_,_) = hist.design(frames(obs))
    ids = [ix[v] for v in nodes]
    prior = w[ids]
    out = {}
    for seed,(net,mean,std) in nets.items():
        tx = torch.from_numpy(np.clip((x-mean)/std,-6,6))[None]
        with torch.no_grad():
            logits = net.node(net.encoder(tx))[0,:,0].numpy().astype(float)
        h = np.exp(logits-logits.max())
        u = h*prior
        h_other,u_other = excluding_self(h),excluding_self(u)
        marg = np.stack((h*h_other,h*u_other+u*h_other,u*u_other))/means[:,None]
        evidence = marg.sum(1)/marg[0].sum()
        out[seed] = (marg,evidence,h)
    return nodes,ids,out


def estimate(evidence,acceptance):
    initial = acceptance/acceptance.sum()
    fraction = initial.copy()
    if len(evidence):
        relative = np.asarray(evidence)*acceptance[0]/acceptance
        for _ in range(250):
            posterior = relative*fraction
            updated = ((posterior/posterior.sum(1,keepdims=True)).sum(0)+3*initial)/(len(evidence)+3)
            if np.max(np.abs(updated-fraction))<1e-9:
                fraction = updated
                break
            fraction = updated
    prior = fraction/acceptance
    return prior/prior.sum()


def infer(hist,events,nets,w,ix,means,accept):
    # This function has no argument containing source identities or generating profiles.
    evidence = {s:[] for s in SEEDS}
    for e in events[:32]:
        _,_,outputs = representation(hist,e,nets,w,ix,means)
        for s in SEEDS:
            evidence[s].append(outputs[s][1])
    return {s:estimate(values,accept).tolist() for s,values in evidence.items()}


def predict(hist,events,nets,w,ix,means,estimated,oracle,edge):
    for e in events:
        nodes,ids,outputs = representation(hist,e,nets,w,ix,means)
        tie = fg.tie_order(e["id"],len(nodes))
        def top(score):
            return [nodes[i] for i in fg.sorted_nodes(score,tie)[:12]]
        lists = {"activity":top(w[ids])}
        for s,(marg,_,h) in outputs.items():
            for key,mix in (("raw",[1,0,0]),("fixed",EQUAL),("always_both",[0,0,1]),("adaptive",estimated[s])):
                lists[f"{key}/{s}"] = top(np.asarray(mix)@marg)
            if oracle is None:
                q = edge[np.ix_(ids,ids)].copy()
                np.fill_diagonal(q,0.)
                value = h*(q@h)
            else:
                value = np.asarray(oracle)@marg
            lists[f"oracle/{s}"] = top(value)
        assert all(len(set(v))==12 and set(v)<=set(nodes) for v in lists.values())
        yield dict(id=e["id"],window=e["window"],lists=lists)


def run(net):
    raw,graphs = windows(net,False)
    nets = {s:fitted(SOURCE[net],s,"C") for s in SEEDS}
    folder = OUT/net
    folder.mkdir(parents=True,exist_ok=True)
    fits,records = [],[]
    cache = {}
    for j,(g,stream) in enumerate(graphs):
        w = activity(g,raw)
        _,_,_,means = law_arrays(w)
        edge = correlated_prior(raw,g)
        ix = {n:i for i,n in enumerate(g["nodes"])}
        acceptance,accepted,attempts = past_acceptance(raw,g["start"],net,cache)
        hist = None
        for profile,oracle in PROFILES.items():
            adapt,_ = draws(g,stream,w,edge,net,j,profile,0)
            test,gold = draws(g,stream,w,edge,net,j,profile,1)
            if hist is None and (adapt or test):
                hist = PublicWindow(g,raw)
            estimated = infer(hist,adapt,nets,w,ix,means,acceptance)
            fits.append(dict(window=g["window"],profile=profile,reference_accepted=accepted,ref_attempts_each=attempts,
                             adaptation_eligible=len(adapt),observations_used=[e["id"] for e in adapt[:32]],
                             estimated=estimated,test_eligible=len(test)))
            for record in predict(hist,test,nets,w,ix,means,estimated,oracle,edge):
                lists = record.pop("lists")
                src = gold[record["id"]]
                hits = {k:sum(v in names for v in src) for k,names in lists.items()}
                records.append(dict(record,profile=profile,sources=src,hits=hits))
    fg.save_rows(folder/"adaptation.jsonl",fits)
    fg.save_rows(folder/"results.jsonl",records)
    fg.write_json(folder/"audit.json",dict(ref_windows=len(cache),evaluation_records=len(records),
                                          hashes={p:fg.digest(folder/p) for p in ("results.jsonl","adaptation.jsonl")}))


if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("action",choices=("freeze",*NETWORKS))
    a = arg.parse_args().action
    OUT.mkdir(exist_ok=True)
    lock = OUT/"protocol.json"
    if a=="freeze" and not lock.exists():
        fg.write_json(lock,plan())
    assert fg.load(lock)==plan()
    if a!="freeze":
        torch.set_num_threads(1)
        with threadpool_limits(1):
            run(a)
