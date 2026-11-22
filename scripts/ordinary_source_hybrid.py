"""Test an activity-only control and a validation-selected mixture on another simulation draw."""
import argparse
import hashlib

import joblib
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from joint_source_shortlist import best_subset
from ordinary_source_roles import OUT, PREVIOUS, FILES, NEW, Window, boost, company, draw, load, load_events, make_windows, read_rows, save_rows, write_json
from partial_source_lists import pair_distribution, sorted_nodes

WEIGHTS=(0.,.25,.5,1.)
PROTOCOL=dict(weights=list(WEIGHTS),selection="Maximize ordinary double-hit@12 on the original three calibration windows; ties favor smaller silent weight",
        primary="Fresh ordinary cascades in the same four test windows; paired hybrid vs unchanged dual-trained joint",
        activity="Ordinary pooled tree trained on same early cascades, only past per-person contact activity; public suffix only excludes ineligible people",
        test_seed=41607131,per_window={"local":150,"transfer":60})


def activity(data):
    base,x,_=data
    offset=base[1].shape[1]
    return np.column_stack((x[:,:6],x[:,offset:offset+4]))


def evaluate(o,data,models,old,temps,mix,background):
    base,x,ev=data
    nodes=base[0]
    initial,tie,q=pair_distribution(o,base,old)
    gate=models["gate"].predict_proba(ev[None,:])[0,1]
    active=softmax(models["active"].decision_function(x)/temps)
    silent=softmax(models["silent"].decision_function(x)/temps)
    one=np.outer(active,silent)
    np.fill_diagonal(one,0.)
    one=(one+one.T)/(2*one.sum())
    sets={"activity":sorted_nodes(background.decision_function(activity(data)),tie)[:12],
          "pooled":sorted_nodes(models["pooled"].decision_function(x),tie)[:12],
          "dual_node":initial[:12]}
    for weight in mix:
        fraction=weight*(1-gate)
        joint=q if weight==0 else (1-fraction)*q+fraction*one
        sets[str(weight)]=best_subset(joint,initial,tie,12)
    return dict(id=o["id"],window=o["window"],sets={tag:[nodes[i] for i in order] for tag,order in sets.items()})


def run(name,phase):
    external=name in FILES
    assert not external or phase=="test"
    source="Dept2" if external else name
    training=OUT/source
    output=OUT/name
    role_selection=load(training/"selection.json")
    role=joblib.load(training/"models.joblib")
    old=joblib.load(PREVIOUS/source/"models.joblib")
    raw=company() if name=="company" else load_events(PREVIOUS.parents[1]/"data/raw"/FILES[name] if external else NEW)
    gs,streams=make_windows(raw,name,.15,1.)
    if phase=="cal":
        write_json(training/"hybrid_protocol.json",PROTOCOL)
        obs=read_rows(training/"fit_public.jsonl")
        truth={r["id"]:r for r in read_rows(training/"fit_truth.jsonl")}
        xx=[];yy=[]
        for g in gs:
            items=[o for o in obs if o["window"]==g["window"]]
            if items:
                w=Window(g,raw)
                for o in items:
                    data=w.features(o)
                    xx.append(activity(data))
                    yy.append(np.isin(data[0][0],truth[o["id"]]["sources"]))
        background=boost(xx,yy)
        joblib.dump(background,training/"activity.joblib")
        cal_obs=read_rows(training/"cal_public.jsonl")
        cal_labs=read_rows(training/"cal_truth.jsonl")
        mix=WEIGHTS
        keys={r["window"] for r in cal_obs}
    else:
        selection=load(training/"hybrid_selection.json")
        assert load(training/"hybrid_protocol.json")==PROTOCOL
        background=joblib.load(training/"activity.joblib")
        for file,digest in selection["hashes"].items():
            assert hashlib.sha256((training/file).read_bytes()).hexdigest()==digest
        mix=list(dict.fromkeys([0.,selection["weight"]]))
        keys=set(load(output/"test_flow.json"))
    if phase=="test":
        prevseeds={r["seed"] for r in read_rows(output/"test_truth.jsonl")}
        previous=PREVIOUS/("transfer_"+name if external else name)
        for file in (("truth.jsonl",) if external else ("truth.jsonl","repeat_truth.jsonl","joint12_truth.jsonl")):
            prevseeds.update(r["seed"] for r in read_rows(previous/file))
        for stage in ("fit","cal"):
            prevseeds.update(r["seed"] for r in read_rows(training/(stage+"_truth.jsonl")))
    public=[];labs=[];pred=[];flow={}
    for i,g in enumerate(g for g in gs if g["window"] in keys):
        key=g["window"]
        if phase=="cal":
            items=[o for o in cal_obs if o["window"]==key]
            labels=[y for y in cal_labs if y["window"]==key]
        else:
            n=60 if external else 150
            seed=PROTOCOL["test_seed"]+i+1000*["company","Dept2","Dept1","CollegeMsg"].index(name)
            items,labels,counts=draw(g,streams[key],seed,0,n)
            assert counts["ordinary"]==n
            flow[key]=dict(counts,seed=seed)
            for o,y in zip(items,labels,strict=True):
                assert y["seed"] not in prevseeds
                prevseeds.add(y["seed"])
                o["id"]=y["id"]="hybrid-"+o["id"]
        w=Window(g,raw)
        for o in items:
            pred.append(evaluate(o,w.features(o),role,old,role_selection["temperature"],mix,background))
        labs.extend(labels);public.extend(items)
        print(name,phase,key,flush=True)
    assert [r["id"] for r in labs]==[r["id"] for r in pred]
    save_rows(output/("hybrid_"+phase+"_predictions.jsonl"),pred)
    if phase=="test":
        save_rows(output/"hybrid_test_public.jsonl",public)
        save_rows(output/"hybrid_test_truth.jsonl",labs)
        write_json(output/"hybrid_flow.json",flow)
    else:
        hits={str(v):sum(set(y["sources"])<=set(p["sets"][str(v)]) for y,p in zip(labs,pred)) for v in mix}
        best=max(mix,key=lambda v:hits[str(v)])
        write_json(training/"hybrid_selection.json",dict(weight=best,hits=hits,n=len(labs),hashes={
            f:hashlib.sha256((training/f).read_bytes()).hexdigest() for f in ("models.joblib","activity.joblib","selection.json")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("network",choices=["company","Dept2",*FILES])
    parser.add_argument("phase",choices=["cal","test"])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.network,args.phase)
