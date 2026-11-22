"""Frozen 12-person shortlist for capturing both sources, evaluated on fresh simulation draws."""
import argparse
from collections import Counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from partial_source_identification import (OUT, NEW, HistoricalFeatures, company, draw, load,
    load_events, make_windows, read_rows, save_rows, write_json)
from partial_source_lists import pair_distribution, sorted_nodes


PROTOCOL = dict(k=12, seed=383037, dual_per_window=80, ordinary_per_window=25,
    primary="Both sources in the fixed shortlist; paired comparison to individual source scores",
    objective="maximize within-list mass of the frozen symmetric pair score; best single swap until local optimum",
    initializations=["individual scores", "marginal pair score", "greedy growth from highest scoring pair"],
    cohorts="same four test windows as preceding experiments, new simulation draws")


def best_subset(joint, initial, tie, k):
    stock = np.argsort(tie)
    permuted = joint[np.ix_(stock,stock)]
    a,b = np.unravel_index(np.argmax(permuted),permuted.shape)
    seed = [stock[a],stock[b]]
    while len(seed)<k:
        gains = joint[:,seed].sum(axis=1)
        gains[seed] = -np.inf
        seed.append(sorted_nodes(gains,tie)[0])
    starts = [initial[:k], sorted_nodes(joint.sum(axis=1),tie)[:k],seed]
    best, best_score = None, -np.inf
    for start in starts:
        inside = np.zeros(len(joint),bool)
        inside[start] = True
        while True:
            keep = stock[inside[stock]]
            other = stock[~inside[stock]]
            strength = joint[:,keep].sum(axis=1)
            gain = strength[other,None]-joint[np.ix_(other,keep)]-strength[keep]
            i,j = np.unravel_index(np.argmax(gain),gain.shape)
            if gain[i,j]<=1e-12:
                break
            inside[keep[j]] = False
            inside[other[i]] = True
        value = joint[np.ix_(keep,keep)].sum()
        if value>best_score:
            best,best_score = keep,value
    assert best_score+1e-10>=joint[np.ix_(initial[:k],initial[:k])].sum()
    return best


def run(name):
    folder = OUT/name
    write_json(folder/"joint12_protocol.json",PROTOCOL)
    model = joblib.load(folder/"models.joblib")
    split = load(folder/"windows.json")
    raw = company() if name=="company" else load_events(NEW)
    gs,streams = make_windows(raw,name,.15,1.)
    public,truth,pred = [],[],[]
    k = PROTOCOL["k"]
    for i,g in enumerate(g for g in gs if g["window"] in split and split[g["window"]]["stage"]=="test"):
        key = g["window"]
        f = HistoricalFeatures(g,raw)
        items,labs,flow = draw(g,streams[key],PROTOCOL["seed"]+i,80,25)
        assert (flow["dual"],flow["ordinary"])==(80,25)
        for o,y in zip(items,labs,strict=True):
            o["id"]=y["id"]="joint12-"+o["id"]
            data=f.event(o)
            nodes,_,support=data
            initial,tie,joint=pair_distribution(o,data,model)
            orders=dict(source=initial[:k],reach=sorted_nodes(support.mean(axis=1),tie)[:k],
                        marginal=sorted_nodes(joint.sum(axis=1),tie)[:k],joint12=best_subset(joint,initial,tie,k))
            choices={tag:[nodes[j] for j in idx] for tag,idx in orders.items()}
            assert all(len(set(c))==k for c in choices.values())
            pred.append(dict(id=o["id"],window=key,sets=choices))
        public.extend(items)
        truth.extend(labs)
    earlier={l["seed"] for file in ("truth.jsonl","repeat_truth.jsonl") for l in read_rows(folder/file)}
    assert earlier.isdisjoint(y["seed"] for y in truth)
    results={}
    for group in ("dual","ordinary"):
        stats={}
        for row,y in zip(pred,truth,strict=True):
            assert row["id"]==y["id"]
            if not y[group]:
                continue
            for window in ("total",row["window"]):
                metrics=stats.setdefault(window,Counter())
                metrics["n"]+=1
                success={}
                for tag,c in row["sets"].items():
                    hit=len(set(c).intersection(y["sources"]))
                    metrics[tag+"_one"]+=int(hit>=1)
                    metrics[tag+"_both"]+=int(hit==2)
                    success[tag]=hit==2
                metrics["joint_only"]+=int(success["joint12"] and not success["source"])
                metrics["source_only"]+=int(success["source"] and not success["joint12"])
        results[group]=stats
    save_rows(folder/"joint12_public.jsonl",public)
    save_rows(folder/"joint12_truth.jsonl",truth)
    save_rows(folder/"joint12_predictions.jsonl",pred)
    write_json(folder/"joint12_summary.json",results)
    print(name,results["dual"]["total"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("network",choices=["company","Dept2"])
    with threadpool_limits(limits=1):
        run(parser.parse_args().network)
