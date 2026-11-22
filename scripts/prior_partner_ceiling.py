"""Conditional expected coverage and its exact opportunity-cost decomposition under source shift."""
import argparse

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from build_dept3_pilot import mask,simulate
from distilled_source_pairs import PublicWindow
from evaluate_factorial_history import fitted
from five_frame_pairs import frames
from silent_source_oracle import CompleteTrace
from source_prior_shift import NETWORKS,OUT,REGIMES,SEEDS,SOURCE,activity,prediction,windows


def conditional(world,label,regime,w,check):
    a,others,flat = world.evaluate(label,return_distribution=True)
    idx = np.array([world.index[b] for b in others])
    if regime=="flat":
        prior = np.ones(len(w))
    elif regime=="one":
        prior = w[world.index[a]]+w
    else:
        prior = w
    p = flat*prior[idx]
    p /= p.sum()
    cap = np.sort(p)[-11:].sum()
    if check:
        verify = world.evaluate(label,check=True,prior=prior)
        assert np.isclose(verify["mass"],cap)
    return a,dict(zip(others,p)),float(cap)


def expectations(selected,a,partner,cap):
    if a not in selected:
        return 0.,cap,0.
    captured = sum(partner.get(v,0.) for v in selected)
    cost = max(0.,cap-captured)
    assert np.isclose(captured+cost,cap)
    return captured,0.,cost


def run(name):
    folder = OUT/name/"test"
    saved = fg.load(folder/"audit.json")
    for file in ("events","observations"):
        assert saved[file]==fg.digest(folder/(file+".jsonl"))
    obs = fg.read_rows(folder/"observations.jsonl")
    gold = {r["id"]:r for r in fg.read_rows(folder/"events.jsonl")}
    src = SOURCE[name]
    chosen = fg.load(OUT/src/"cal/selection.json")
    networks = {s:fitted(src,s,"C") for s in SEEDS}
    raw,graphs = windows(name,False)
    results = []
    for g,stream in graphs:
        items = [e for e in obs if e["window"]==g["window"]]
        if not items:
            continue
        w = activity(g,raw)
        world = CompleteTrace(g,stream)
        hist = PublicWindow(g,raw)
        for j,e in enumerate(items):
            r = gold[e["id"]]
            law = r["regime"]
            tr = simulate(stream,r["sources"],r["seed"])
            cut,groups = mask(tr)
            assert groups==e["observed"]
            a,partner,cap = conditional(world,dict(sources=r["sources"],trajectory=tr,cut_batches=cut),law,w,j<4)
            (nodes,_,_),(x,_,_) = hist.design(frames(e))
            assert set(partner)<=set(nodes) and a in nodes
            idx = [world.index[n] for n in nodes]
            tie = fg.tie_order(e["id"],len(nodes))
            record = dict(id=e["id"],window=g["window"],regime=law,emitters=r["emitters"],cap=cap,models={})
            for s,(net,mean,std) in networks.items():
                tx = torch.from_numpy(np.clip((x-mean)/std,-6,6))[None]
                with torch.no_grad():
                    logits = net.node(net.encoder(tx))[0,:,0].numpy()
                for label,rule in (("raw","flat"),("calibrated",chosen[law]["selected"])):
                    lst,_ = prediction(logits,w[idx],rule,tie,nodes)
                    hit = set(r["sources"])<=set(lst)
                    assert hit==(r["metrics"][f"full/{s}/{rule}"][0]==2)
                    rb,anchor,ranking = expectations(lst,a,partner,cap)
                    if r["emitters"]==2:
                        # Both identities are fixed by the full trace. Neither is a silent partner.
                        assert np.isclose(rb,hit) and np.isclose(cap,1.)
                        anchor,ranking = 0.,0.
                    record["models"][f"{label}/{s}"] = [int(hit),rb,anchor,ranking]
            results.append(record)
    target = OUT/name/"ceiling"
    target.mkdir(exist_ok=True)
    fg.save_rows(target/"events.jsonl",results)
    summary = {}
    for law in REGIMES:
        summary[law] = {}
        for tag in ("all","one","two"):
            rows = [r for r in results if r["regime"]==law and (tag=="all" or r["emitters"]==(1 if tag=="one" else 2))]
            if not rows:
                continue
            cap = 100*np.mean([r["cap"] for r in rows])
            table = dict(n=len(rows),cap=cap,methods={})
            for m in ("raw","calibrated"):
                value = np.array([[r["models"][f"{m}/{s}"] for s in SEEDS] for r in rows])
                actual,rb,anchor,ranking = 100*value.mean((0,1))
                if tag=="one":
                    assert np.isclose(rb+anchor+ranking,cap)
                table["methods"][m] = dict(actual=actual,conditional=rb,gap=cap-rb,
                    missing_anchor_loss=anchor if tag=="one" else None,partner_ranking_loss=ranking if tag=="one" else None)
            summary[law][tag] = table
    assert len(results)==len(obs)
    fg.write_json(target/"summary.json",summary)


if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("network",choices=NETWORKS)
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run(arg.parse_args().network)
