"""Exploratory forward-only target calibration after the frozen cross-institution benchmark."""

from math import comb

import numpy as np
from threadpoolctl import threadpool_limits

from masked_pair_benchmark import (Bank, DAY, OUT, PLAN, PairEvidence, company, draw, fit, frozen_models,
    load, make_windows, metrics, read_rows, save_rows, write_json)


def features(bank,w,o):
    nodes,i,j,ds,s,tie,_=w.design(o)
    (_,tail),_,_=bank.design(o,nodes,i,j)
    return nodes,i,j,{"local_static":ds["static"],"local_ordered":ds["ordered"],
                      "local_tail":tail,"local_joint":np.column_stack((ds["ordered"],tail))},s["additive"],tie


def run():
    raw=company()
    f=frozen_models()["Dept3"]
    gs,_=make_windows(raw,"company",*PLAN["span"])
    earlier,streams=make_windows(raw,"adapt",.06,1.)
    obs=read_rows(OUT/"observations.jsonl")
    checkpoint={}
    cohorts=[]
    for g in earlier:
        if g["end"]>gs[-1]["start"]:
            continue
        train,labs,flow=draw(g,streams[g["window"]],348179+g["start"],35,0)
        bank=Bank(raw,g)
        window=PairEvidence(raw,g,f["forward"],f["node"])
        rows=[]
        for o,lab in zip(train,labs):
            nodes,i,j,x,_,_=features(bank,window,o)
            if set(lab["sources"])<=set(nodes):
                a,b=sorted(nodes.index(v) for v in lab["sources"])
                y=int(np.flatnonzero((i==a)&(j==b))[0])
                rows.append(({k:value.astype(np.float32) for k,value in x.items()},y))
        cohorts.append((g,rows))
        print(g["window"],len(rows),"training positives",flush=True)
    predictions=[]
    for g in gs:
        folds=[(h,rows) for h,rows in cohorts if h["end"]<=g["start"]][-4:]
        assert len(folds)>=2
        train=[r for h,rows in folds for r in rows]
        weights={k:fit([(x[k].astype(float),y) for x,y in train],train[0][0][k].shape[1]) for k in train[0][0]}
        checkpoint[g["window"]]=dict(windows=[h["window"] for h,_ in folds],
            last_training_end=folds[-1][0]["end"],inference_start=g["start"],events=len(train),weights=weights)
        bank=Bank(raw,g)
        window=PairEvidence(raw,g,f["forward"],f["node"])
        for o in obs:
            if o["window"]!=g["window"]:
                continue
            nodes,i,j,x,add,tie=features(bank,window,o)
            # MRR requires the complete ordering for retrospective scoring.
            orders={m:np.lexsort((tie,-add,-(x[m]@weights[m]))).tolist() for m in weights}
            predictions.append(dict(id=o["id"],window=g["window"],nodes=nodes,orders=orders))
        print("calibrated",g["window"],flush=True)
    write_json(OUT/"local_models.json",checkpoint)
    save_rows(OUT/"local_predictions.jsonl",predictions)


def evaluate():
    labs={r["id"]:r for r in read_rows(OUT/"truth.jsonl")}
    old={r["id"]:r for r in read_rows(OUT/"event_results.jsonl")}
    rows=[]
    for p in read_rows(OUT/"local_predictions.jsonl"):
        sid=p["id"]
        r=old[sid].copy()
        nodes=p["nodes"]
        target=None
        n=len(nodes)
        i,j=np.triu_indices(n,1)
        if set(labs[sid]["sources"])<=set(nodes):
            a,b=sorted(nodes.index(v) for v in labs[sid]["sources"])
            target=int(np.flatnonzero((i==a)&(j==b))[0])
        assert (target is not None)==r["inside"]
        pairs=comb(n,2)
        for k,order in p["orders"].items():
            assert len(order)==pairs and sorted(order)==list(range(pairs))
            r[k]=None if target is None else order.index(target)+1
        r["rand_mrr"]=sum(1/a for a in range(1,pairs+1))/pairs if r["inside"] else 0
        r["rand_hit32"]=min(1.,32/pairs) if r["inside"] else 0
        rows.append(r)
    assert len(rows)==len(labs)==len({r["id"] for r in rows})
    save_rows(OUT/"local_results.jsonl",rows)
    from check_masked_pair_benchmark import paired
    report={}
    for g in ("ordinary","dual"):
        rr=[r for r in rows if r[g]]
        tags=list(p["orders"])
        comparison=paired(rr,"local_joint")
        # Compare with the same architecture and with its target-domain static ablation.
        def delta(a,b):
            by={}
            for r in rr:
                by.setdefault(r["window"],[]).append((1/r[a] if r[a] else 0)-(1/r[b] if r[b] else 0))
            z=np.array([sum(x) for x in by.values()])
            n=np.array([len(x) for x in by.values()])
            draws=np.random.default_rng(319).integers(len(by),size=(10000,len(by)))
            return dict(delta=z.sum()/n.sum(),windows_positive=int((z>0).sum()),
                        ci=np.quantile(z[draws].sum(axis=1)/n[draws].sum(axis=1),[.025,.975]).tolist())
        report[g]=dict(scores={k:metrics(rr,k) for k in tags},paired=comparison,
                       vs_frozen=delta("local_joint","joint_tail"),vs_local_static=delta("local_joint","local_static"),
                       vs_local_ordered=delta("local_joint","local_ordered"),
                       random_mrr=np.mean([r["rand_mrr"] for r in rr]),
                       random_expected_hits=sum(r["rand_hit32"] for r in rr))
    write_json(OUT/"local_summary.json",report)
    print({k:v["scores"] for k,v in report.items()})


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run()
        evaluate()
