"""Evaluate the same frozen individual scorer under absolute and relative candidate budgets."""
import argparse
import math

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from partial_source_identification import OUT, HistoricalFeatures, read_rows, save_rows, write_json
from relative_source_ranking import data
from partial_source_lists import sorted_nodes


def run(name):
    raw,gs=data(name)
    local=name in ("company","Dept2")
    folder=OUT/(name if local else "transfer_"+name)
    prefix="joint12_" if local else ""
    obs=read_rows(folder/(prefix+"public.jsonl"))
    labels={r["id"]:r for r in read_rows(folder/(prefix+"truth.jsonl"))}
    models=joblib.load(OUT/"Dept2"/"models.joblib")
    preds={r["id"]:r["sets"] for r in read_rows(folder/(prefix+"predictions.jsonl"))}
    audits=[]
    for g in gs:
        items=[o for o in obs if o["window"]==g["window"]]
        if not items:
            continue
        f=HistoricalFeatures(g,raw)
        for o in items:
            nodes,x,support=f.event(o)
            rng=np.random.default_rng(1000003+sum((j+1)*ord(c) for j,c in enumerate(o["id"])))
            tie=rng.permutation(len(nodes))
            orders={"source":sorted_nodes(models["source"].decision_function(x),tie),
                    "reach":sorted_nodes(support.mean(axis=1),tie)}
            sid=o["id"]
            assert [nodes[i] for i in orders["reach"][:12]]==preds[sid]["reach"]
            if name!="company":
                assert [nodes[i] for i in orders["source"][:12]]==preds[sid]["source" if local else "Dept2_source"]
            src=[nodes.index(v) for v in labels[sid]["sources"]]
            ranks={tag:sorted((np.argsort(order)[src]+1).tolist()) for tag,order in orders.items()}
            audits.append(dict(id=sid,window=g["window"],n=len(nodes),ranks=ranks))
    result={}
    for group in ("dual","ordinary"):
        rows=[r for r in audits if labels[r["id"]][group]]
        result[group]={}
        for q in (0,.02,.05,.10):
            sizes=[12 if q==0 else math.ceil(q*r["n"]) for r in rows]
            v=dict(n=len(rows),mean_size=float(np.mean(sizes)))
            for tag in ("source","reach"):
                v[tag+"_both"]=sum(r["ranks"][tag][1]<=k for r,k in zip(rows,sizes))
                v[tag+"_one"]=sum(r["ranks"][tag][0]<=k for r,k in zip(rows,sizes))
            result[group]["fixed12" if q==0 else str(q)]=v
    save_rows(folder/"relative_budget_ranks.jsonl",audits)
    write_json(folder/"relative_budget.json",result)
    print(name,result["dual"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("network",choices=["Dept2","company","Dept1","CollegeMsg"])
    with threadpool_limits(limits=1):
        run(parser.parse_args().network)
