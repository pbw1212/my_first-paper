"""Transfer both frozen models to different networks, without target labels or recalibration."""
import argparse
import hashlib
from collections import Counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from joint_source_shortlist import best_subset
from partial_source_identification import HistoricalFeatures, OUT, company, draw, load_events, make_windows, save_rows, write_json
from partial_source_lists import pair_distribution, sorted_nodes


FILES={"Dept1":"email-Eu-core-temporal-Dept1.txt.gz","CollegeMsg":"CollegeMsg.txt.gz"}
PLAN=dict(primary="both genuine sources among 12 names",training=["Dept2","company"],
          windows="last four complete 14-day windows with >=100 within-roster contacts in 15%-100% frame",
          draw="40 dual and 15 ordinary per window, same actual cases for every predictor; record shortfalls",
          seed=692081,models="all weights fixed, no target calibration, budget 12")


def run(target):
    folder=OUT/("transfer_"+target)
    folder.mkdir(exist_ok=True)
    raw=load_events(OUT.parents[1]/"data/raw"/FILES[target])
    gs,streams=make_windows(raw,target,.15,1.)
    selected=[g for g in gs if sum(len(b) for _,b in streams[g["window"]])>=100][-4:]
    assert len(selected)==4
    paths={s:OUT/s/"models.joblib" for s in PLAN["training"]}
    models={s:joblib.load(p) for s,p in paths.items()}
    write_json(folder/"protocol.json",dict(PLAN,weights={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in paths.items()},
                                       windows=[g["window"] for g in selected]))
    public,truth,pred=[],[],[]
    flow={}
    for i,g in enumerate(selected):
        key=g["window"]
        items,labs,counts=draw(g,streams[key],PLAN["seed"]+i,40,15)
        flow[key]=dict(counts,start=g["start"],end=g["end"],roster=len(g["nodes"]))
        f=HistoricalFeatures(g,raw)
        for o in items:
            o["id"]="transfer-"+o["id"]
            features=f.event(o)
            nodes,_,support=features
            choices={}
            for domain,model in models.items():
                initial,tie,joint=pair_distribution(o,features,model)
                choices[domain+"_source"]=[nodes[j] for j in initial[:12]]
                choices[domain+"_joint"]=[nodes[j] for j in best_subset(joint,initial,tie,12)]
                del joint
            choices["reach"]=[nodes[j] for j in sorted_nodes(support.mean(axis=1),tie)[:12]]
            assert all(len(set(c))==12 for c in choices.values())
            pred.append(dict(id=o["id"],window=key,n=len(nodes),sets=choices))
        for y in labs:
            y["id"]="transfer-"+y["id"]
        public.extend(items)
        truth.extend(labs)
        print(target,key,counts,flush=True)
    save_rows(folder/"public.jsonl",public)
    save_rows(folder/"truth.jsonl",truth)
    save_rows(folder/"predictions.jsonl",pred)
    result={}
    for group in ("dual","ordinary"):
        stats={}
        for r,y in zip(pred,truth,strict=True):
            assert r["id"]==y["id"]
            if not y[group]:
                continue
            for window in ("total",r["window"]):
                row=stats.setdefault(window,Counter())
                row["n"]+=1
                row["sum_legal"]+=r["n"]
                row["expected_random_both"]+=12*11/(r["n"]*(r["n"]-1))
                for method,c in r["sets"].items():
                    h=len(set(c).intersection(y["sources"]))
                    row[method+"_both"]+=int(h==2)
                    row[method+"_one"]+=int(h>=1)
        result[group]=stats
    write_json(folder/"summary.json",dict(result,flow=flow))
    print(target,result["dual"]["total"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("target",choices=FILES)
    with threadpool_limits(limits=1):
        run(parser.parse_args().target)
