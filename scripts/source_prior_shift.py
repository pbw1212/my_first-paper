"""Change only the law of the unordered sources and test frozen likelihood-ratio corrections."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import enron_temporal_transfer as enron
import fgssi_suffix as fg
import locked_dept4 as dept4
from build_dept3_pilot import mask,simulate
from distilled_source_pairs import PublicWindow
from evaluate_factorial_history import fitted
from factorial_fgssi import OUT as PRIOR,SEEDS,weights
from factorial_history import folder_for
from five_frame_pairs import frames
from score_dept3_candidates import roles

OUT = PRIOR.parent/"source_prior_shift"
REGIMES = ("flat","one","both")
OPTIONS = ("flat","one_half","both_half","one","both")
NETWORKS = ("Dept2","company","Dept4","Enron")
SOURCE = {n:("company" if n=="company" else "Dept2") for n in NETWORKS}
DAYS = 28*86400


def protocol():
    paths = [Path(__file__),*(OUT.parents[1]/"scripts"/(x+".py") for x in (
        "build_dept3_pilot","five_frame_pairs","distilled_source_pairs","fgssi_suffix","ordinary_source_roles",
        "evaluate_factorial_history","locked_dept4","enron_temporal_transfer","temporal_pair_evidence","partial_source_identification"))]
    for n in ("company","Dept2"):
        folder = folder_for(n,"temporal")
        paths.append(folder/"scale.npz")
        for s in SEEDS:
            paths.extend([folder/f"add{s}.pt",weights(n,s)])
    return dict(primary="both sources in 12; marginal shortlist for all; independent draws for each source law",
                source_laws="flat, (w_i+w_j), or w_i*w_j over distinct unordered pairs, w=1+outgoing contacts in 28 days",
                source_model=SOURCE,seeds=list(SEEDS),test_attempts=160,cal_attempts=112,
                calibration="32 labelled eligible events across earlier calibration windows per network and law, selected by round-robin; minimize pair NLL over five listed rules averaged across frozen seeds",
                candidates=list(OPTIONS),seed=20261122,
                hashes={p.resolve().relative_to(OUT.parents[1]).as_posix():fg.digest(p) for p in paths})


def windows(net,cal):
    if net=="Enron":
        enron.verify()
        raw = enron.load_events(enron.CONTACTS)
        return raw,list(enron.windows(raw))
    if net=="Dept4":
        assert dept4.config()==fg.load(dept4.FOLDER/"protocol_lock.json")
        raw = dept4.load_events(dept4.RAW)
        return raw,dept4.windows(raw)
    raw = fg.raw_data(net)
    if cal:
        kept,streams = fg.make_windows(raw,net,.15,1.)
        keys = {g["window"] for g,_,_ in fg.cohort(net,"cal")}
        kept = [g for g in kept if g["window"] in keys]
    else:
        kept,streams = fg.evaluation_windows(net,raw)
    return raw,[(g,streams[g["window"]]) for g in kept]


def activity(graph,raw):
    end = graph["start"]
    count = Counter(u for t,u,v in raw if end-DAYS<=t<end)
    return np.array([1.+count[n] for n in graph["nodes"]])


def source_mass(rule,w,left,right):
    if rule=="flat":
        return np.ones(len(left))
    if rule.startswith("one"):
        return w[left]+w[right]
    return w[left]*w[right]


def generate(graph,stream,w,net,window_index,cal):
    a,b = np.triu_indices(len(w),1)
    attempts = 112 if cal else 160
    events,labels = [],{}
    for r,regime in enumerate(REGIMES):
        rng = np.random.default_rng(np.random.SeedSequence([20261122,NETWORKS.index(net),int(cal),window_index,r]))
        mass = source_mass(regime,w,a,b)
        pairs = rng.choice(len(a),size=attempts,replace=True,p=mass/mass.sum())
        for k,index in enumerate(pairs):
            seed = int(rng.integers(1,2**62))
            src = [graph["nodes"][i] for i in (a[index],b[index])]
            tr = simulate(stream,src,seed)
            obs = mask(tr)
            if obs is None:
                continue
            mid = f"{graph['window']}-{regime}-{k}"
            cut,groups = obs
            lab = dict(sources=src,trajectory=tr,cut_batches=cut)
            events.append(dict(id=mid,window=graph["window"],observed=groups,k=2))
            labels[mid] = dict(sources=src,regime=regime,seed=seed,emitters=len(roles(lab)[0]))
    return events,labels


def excluding_self(v):
    return np.r_[0.,np.cumsum(v[:-1])]+np.r_[np.cumsum(v[:0:-1])[::-1],0.]


def pair_posterior(logits,w,rule):
    """Unnormalized marginal and factorized pair mass up to global scale."""
    h = np.exp(logits.astype(float)-np.max(logits))
    weight = np.sqrt(w) if rule.endswith("_half") else w
    if rule.startswith("one"):
        u = h*weight
        return h*excluding_self(u)+u*excluding_self(h),h,u
    u = h*weight if rule.startswith("both") else h
    return u*excluding_self(u),u,None


def prediction(logits,w,rule,tie,nodes):
    marg,h,u = pair_posterior(logits,w,rule)
    order = fg.sorted_nodes(marg,tie)[:12]
    return [nodes[i] for i in order],(marg.sum()/2,h,u)


def assess(record,src,nodes,predictions):
    a,b = [nodes.index(n) for n in src]
    metrics = {}
    for key,(lst,(z,h,u)) in predictions.items():
        prob = (h[a]*h[b] if u is None else h[a]*u[b]+h[b]*u[a])/z
        assert 0<prob<=1 and len(set(lst))==12
        metrics[key] = [int(src[0] in lst)+int(src[1] in lst),round(float(-np.log(prob)),7)]
    record["metrics"] = metrics


def evaluate(g,raw,w,events,labels,nets,fg_models,cal):
    if not events:
        return []
    hist = PublicWindow(g,raw)
    geo = fg.Geometry(g)
    rows = []
    for off in range(0,len(events),8):
        batch = events[off:off+8]
        fg_scores = {}
        if not cal:
            inp = geo.inputs(batch)
            with torch.no_grad():
                for s,model in fg_models.items():
                    out,_ = model(*inp)
                    fg_scores[s] = (out[:,:,1]-out[:,:,0]).numpy()
        for j,e in enumerate(batch):
            (nodes,_,_),(x,_,_) = hist.design(frames(e))
            ix = [geo.index[n] for n in nodes]
            prior = w[ix]
            tie = fg.tie_order(e["id"],len(nodes))
            label = labels[e["id"]]
            pred = {"volume":prediction(np.log(prior),prior,"flat",tie,nodes)}
            for s,(net,mean,std) in nets.items():
                with torch.no_grad():
                    tx = torch.from_numpy(np.clip((x-mean)/std,-6,6))[None]
                    logits = net.node(net.encoder(tx))[0,:,0].numpy()
                for rule in OPTIONS:
                    pred[f"full/{s}/{rule}"] = prediction(logits,prior,rule,tie,nodes)
                if not cal:
                    for rule in dict.fromkeys(["flat",label["regime"]]):
                        pred[f"fg/{s}/{rule}"] = prediction(fg_scores[s][j,ix],prior,rule,tie,nodes)
            row = dict(id=e["id"],window=g["window"],**label)
            assess(row,label["sources"],nodes,pred)
            rows.append(row)
    return rows


def select(rows,keys):
    choice = {}
    for reg in REGIMES:
        by_w = [[r for r in rows if r["regime"]==reg and r["window"]==w] for w in keys]
        ordered = [bucket[i] for i in range(max(map(len,by_w))) for bucket in by_w if len(bucket)>i][:32]
        assert len(ordered)==32
        losses = {key:float(np.mean([r["metrics"][f"full/{s}/{key}"][1] for r in ordered for s in SEEDS])) for key in OPTIONS}
        choice[reg] = dict(selected=min(losses,key=losses.get),losses=losses,ids=[r["id"] for r in ordered])
    return choice


def run(net,cal):
    src = SOURCE[net]
    selection_path = OUT/src/"cal"/"selection.json"
    if not cal:
        selection = fg.load(selection_path)
        assert fg.digest(selection_path)==fg.load(selection_path.parent/"audit.json")["selection"]
    folder = OUT/net/("cal" if cal else "test")
    folder.mkdir(parents=True,exist_ok=True)
    raw,graphs = windows(net,cal)
    nets = {s:fitted(src,s,"C") for s in SEEDS}
    fg_models = {}
    if not cal:
        for s in SEEDS:
            model = fg.FGSSI().eval()
            model.load_state_dict(torch.load(weights(src,s),weights_only=True))
            fg_models[s] = model
    rows,obs = [],[]
    for k,(g,stream) in enumerate(graphs):
        assert g["nodes"]==sorted(g["nodes"])
        w = activity(g,raw)
        events,labels = generate(g,stream,w,net,k,cal)
        obs.extend(events)
        rows.extend(evaluate(g,raw,w,events,labels,nets,fg_models,cal))
    assert len(obs)==len(rows)==len({r["id"] for r in rows})
    fg.save_rows(folder/"observations.jsonl",obs)
    fg.save_rows(folder/"events.jsonl",rows)
    keys = [g["window"] for g,_ in graphs]
    if cal:
        fg.write_json(selection_path,select(rows,keys))
    fg.write_json(folder/"audit.json",dict(rows=len(rows),windows=keys,selection=fg.digest(selection_path),
        events=fg.digest(folder/"events.jsonl"),observations=fg.digest(folder/"observations.jsonl")))


if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("step",choices=("freeze","cal","test"))
    arg.add_argument("network",nargs="?",choices=NETWORKS)
    a = arg.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT/"protocol.json"
    cfg = protocol()
    if a.step=="freeze" and not path.exists():
        fg.write_json(path,cfg)
    assert fg.load(path)==cfg
    if a.step!="freeze":
        assert a.network and (a.step=="test" or a.network in ("Dept2","company"))
        torch.set_num_threads(1)
        with threadpool_limits(1):
            run(a.network,a.step=="cal")
