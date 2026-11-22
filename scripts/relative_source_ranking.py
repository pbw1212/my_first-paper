"""Matched linear controls: absolute versus within-event percentile features."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from partial_source_identification import (OUT,NEW,HistoricalFeatures,company,load,load_events,
    make_windows,read_rows,save_rows,write_json)
from partial_source_lists import sorted_nodes
from transfer_joint_shortlist import FILES


def designs(x):
    # The last two columns contain only event-wide constants.
    values=x[:,:-2]
    return dict(absolute=values,relative=(rankdata(values,axis=0,method="average")-.5)/len(values))


def data(name):
    if name=="company":
        raw=company()
    else:
        path=NEW if name=="Dept2" else OUT.parents[1]/"data/raw"/FILES[name]
        raw=load_events(path)
    return raw,make_windows(raw,name,.15,1.)[0]


def fit():
    folder=OUT/"Dept2"
    raw,gs=data("Dept2")
    split=load(folder/"windows.json")
    obs=read_rows(folder/"public.jsonl")
    truth={y["id"]:y for y in read_rows(folder/"truth.jsonl")}
    train={"absolute":[],"relative":[]}
    ys,weights=[],[]
    for g in gs:
        key=g["window"]
        if key not in split or split[key]["stage"]!="fit":
            continue
        f=HistoricalFeatures(g,raw)
        for o in (o for o in obs if o["window"]==key):
            nodes,x,_=f.event(o)
            y=np.isin(nodes,truth[o["id"]]["sources"])
            assert y.sum()==2
            ys.append(y)
            weights.append(np.where(y,.5/y.sum(),.5/(~y).sum()))
            for tag,features in designs(x).items():
                train[tag].append(features)
    assert len(ys)==700
    models={}
    for tag,arrays in train.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500))
        model.fit(np.vstack(arrays),np.concatenate(ys),logisticregression__sample_weight=np.concatenate(weights))
        models[tag]=model
    joblib.dump(models,OUT/"Dept2_linear.joblib")


def test(name):
    raw,gs=data(name)
    local=name in ("Dept2","company")
    folder=OUT/(name if local else "transfer_"+name)
    prefix="joint12_" if local else ""
    obs=read_rows(folder/(prefix+"public.jsonl"))
    truth={y["id"]:y for y in read_rows(folder/(prefix+"truth.jsonl"))}
    models=joblib.load(OUT/"Dept2_linear.joblib")
    output=[]
    for g in gs:
        items=[o for o in obs if o["window"]==g["window"]]
        if not items:
            continue
        f=HistoricalFeatures(g,raw)
        for o in items:
            nodes,x,_=f.event(o)
            rng=np.random.default_rng(1000003+sum((j+1)*ord(c) for j,c in enumerate(o["id"])))
            tie=rng.permutation(len(nodes))
            sets={}
            for tag,features in designs(x).items():
                idx=sorted_nodes(models[tag].decision_function(features),tie)[:12]
                sets[tag]=[nodes[i] for i in idx]
            output.append(dict(id=o["id"],window=o["window"],sets=sets))
    save_rows(folder/"linear_predictions.jsonl",output)
    result={}
    for group in ("dual","ordinary"):
        group_rows=[r for r in output if truth[r["id"]][group]]
        result[group]={}
        for key in ["total"]+sorted({r["window"] for r in group_rows}):
            rows=[r for r in group_rows if key=="total" or r["window"]==key]
            counts=dict(n=len(rows))
            for tag in models:
                hits=[len(set(r["sets"][tag]).intersection(truth[r["id"]]["sources"])) for r in rows]
                counts[tag+"_one"]=sum(h>=1 for h in hits)
                counts[tag+"_both"]=sum(h==2 for h in hits)
            result[group][key]=counts
    write_json(folder/"linear_summary.json",result)
    print(name,result["dual"]["total"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["fit","company","Dept2","Dept1","CollegeMsg"])
    action=parser.parse_args().action
    with threadpool_limits(limits=1):
        if action=="fit":
            write_json(OUT/"linear_protocol.json",dict(training="Dept2 original 700 events",
                models="raw standardized versus within-candidate percentile standardized, L2 logistic C=1, class-balanced per event",
                endpoint="both sources in 12 people; same frozen evaluation cases; no parameter search"))
            fit()
        else:
            test(action)
