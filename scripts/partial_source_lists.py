"""Calibrate whole-event source shortlists, and check oracle labels under a second completion."""

import argparse
import math

import joblib
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from partial_source_identification import (HistoricalFeatures, NEW, OUT, company, core, load, load_events,
    make_windows, partner_features, read_rows, save_rows, write_json)

PROTOCOL=dict(target=.90,methods=["source","pair_cover","prior"],endpoint=["at_least_one","both"],
              note="Exploratory after naming benchmark; fixed rules and old fit/cal/test windows.",
              pair_cover="Symmetrized distribution from learned anchor logits and conditional partner for best 12 anchors; independent fallback elsewhere. Greedily maximize probability of touching a pair.",
              sizes="90% calibration quantile with ceil((m+1)*0.90); fixed rank fraction; dual calibration only")


def sorted_nodes(value,tie):
    return np.lexsort((tie,-value))


def pair_distribution(o,features,models):
    nodes,x,support=features
    rng=np.random.default_rng(1000003+sum((j+1)*ord(c) for j,c in enumerate(o["id"])))
    tie=rng.permutation(len(nodes))
    score=models["source"].decision_function(x)
    initial=sorted_nodes(score,tie)
    pi=softmax(score)
    cond=np.tile(score,(len(nodes),1))
    leading=initial[:12]
    rows=np.vstack([partner_features(x,support,a) for a in leading])
    cond[leading]=models["partner"].decision_function(rows).reshape(len(leading),-1)
    np.fill_diagonal(cond,-np.inf)
    joint=pi[:,None]*softmax(cond,axis=1)
    joint=(joint+joint.T)/2
    return initial,tie,joint


def lists(o,features,models):
    nodes=features[0]
    initial,tie,joint=pair_distribution(o,features,models)
    remaining=joint.sum(axis=1)
    selected=[]
    for _ in nodes:
        a=sorted_nodes(remaining,tie)[0]
        selected.append(a)
        remaining-=joint[:,a]
        remaining[a]=-np.inf
    prior=sorted_nodes(np.array([models["frequency"][n] for n in nodes]),tie)
    return {tag:[nodes[j] for j in order] for tag,order in (("source",initial),("pair_cover",selected),("prior",prior))}


def relative_ranks(pred,lab,tag):
    order=pred[tag]
    a,b=sorted(order.index(v)+1 for v in lab["sources"])
    return a/len(order),b/len(order)


def score(rows,labs,cuts):
    result={}
    for tag,qs in cuts.items():
        result[tag]={}
        for j,(endpoint,q) in enumerate(zip(PROTOCOL["endpoint"],qs)):
            sizes=[];covered=[];random_coverage=[]
            for r in rows:
                n=len(r[tag])
                k=min(n,math.ceil(n*q))
                found=len(set(r[tag][:k]).intersection(labs[r["id"]]["sources"]))
                covered.append(found>=j+1)
                sizes.append(k)
                pr=1-(n-k)*(n-k-1)/(n*(n-1)) if j==0 else k*(k-1)/(n*(n-1))
                random_coverage.append(pr)
            result[tag][endpoint]=dict(n=len(rows),covered=int(sum(covered)),mean_size=float(np.mean(sizes)),
                median_size=float(np.median(sizes)),random_same_size_rate=float(np.mean(random_coverage)),
                by_window={k:[sum(ok for r,ok in zip(rows,covered) if r["window"]==k),
                              sum(r["window"]==k for r in rows)] for k in sorted({r["window"] for r in rows})})
        result[tag]["one_in_first_five"]=sum(bool(set(r[tag][:5]).intersection(labs[r["id"]]["sources"])) for r in rows)
        result[tag]["first_two_exact"]=sum(set(r[tag][:2])==set(labs[r["id"]]["sources"]) for r in rows)
    return result


def experiment(name):
    folder=OUT/name
    models=joblib.load(folder/"models.joblib")
    raw=company() if name=="company" else load_events(NEW)
    gs,streams=make_windows(raw,name,.15,1.)
    split=load(folder/"windows.json")
    obs=read_rows(folder/"public.jsonl")
    labs={r["id"]:r for r in read_rows(folder/"truth.jsonl")}
    base_oracle=load(folder/"oracle.json")
    preds=[];stable={}
    for g in gs:
        key=g["window"]
        if key not in split or split[key]["stage"]=="fit":
            continue
        f=HistoricalFeatures(g,raw)
        for o in (o for o in obs if o["window"]==key):
            sid=o["id"]
            preds.append(dict(id=sid,window=key,**lists(o,f.event(o),models)))
            if split[key]["stage"]=="test":
                other=core(g,streams[key],o,labs[sid],1)
                stable[sid]={group:sorted(set(row["members"]).intersection(other[group]["members"]))
                             for group,row in base_oracle[sid].items()}
    cal=[r for r in preds if split[r["window"]]["stage"]=="cal" and labs[r["id"]]["dual"]]
    test=[r for r in preds if split[r["window"]]["stage"]=="test"]
    cuts={}
    for tag in PROTOCOL["methods"]:
        a=np.asarray([relative_ranks(r,labs[r["id"]],tag) for r in cal])
        k=math.ceil(.9*(len(a)+1))
        cuts[tag]=np.sort(a,axis=0)[k-1].tolist()
    save_rows(folder/"shortlist_predictions.jsonl",preds)
    write_json(folder/"stable_oracle.json",stable)
    named={p["id"]:p for p in read_rows(folder/"predictions.jsonl")}
    result={}
    for group in ("dual","ordinary"):
        rr=[r for r in test if labs[r["id"]][group]]
        s=score(rr,labs,cuts)
        s["stable_counts"]=[sum(len(stable[r["id"]][group])==k for r in rr) for k in (0,1,2)]
        s["stable_core_named"]=sum(named[r["id"]]["outputs"]["core"]["anchor"] in stable[r["id"]][group] for r in rr)
        s["mean_roster"]=float(np.mean([len(r["source"]) for r in rr]))
        result[group]=s
    write_json(folder/"shortlists.json",dict(result,calibration=cuts))
    print(name,result["dual"]["source"],result["dual"]["pair_cover"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("network",choices=["company","Dept2"])
    name=p.parse_args().network
    with threadpool_limits(limits=1):
        path=OUT/"shortlist_protocol.json"
        if path.exists():
            assert load(path)==PROTOCOL
        else:
            write_json(path,PROTOCOL)
        experiment(name)
