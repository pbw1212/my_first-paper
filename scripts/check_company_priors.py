"""Compare target-calibrated rankings with a suffix-independent, historical source-frequency prior."""

from collections import Counter

import numpy as np

from masked_pair_benchmark import OUT, PLAN, PREV, company, draw, load, make_windows, metrics, read_rows, write_json


def lineage_summary():
    summary={}
    for name,path in (("company",OUT/"truth.jsonl"),("Dept2",PREV/"external_truth.jsonl")):
        stats=[]
        for r in read_rows(path):
            if not r["dual"]:
                continue
            depth=dict.fromkeys(r["sources"],0)
            lineage={v:1<<i for i,v in enumerate(r["sources"])}
            depths,labels=[],[]
            for k,batch in enumerate(r["trajectory"]):
                for v,parents in batch["joins"]:
                    depth[v]=1+min(depth[u] for u in parents)
                    lineage[v]=int(np.bitwise_or.reduce([lineage[u] for u in parents]))
                    if k>=r["cut_batches"]:
                        depths.append(depth[v])
                        labels.append(lineage[v])
            n=len(labels)
            stats.append([n,depths.count(1)/n,np.mean(depths),min(labels.count(1),labels.count(2))/n])
        values=np.asarray(stats)
        summary[name]=dict(n=len(stats),event_medians=dict(zip(
            ("visible","direct_fraction","mean_depth","smaller_exclusive_lineage_fraction"),
            np.median(values,axis=0).tolist())),no_visible_direct=int((values[:,1]==0).sum()))
    write_json(OUT/"lineage_diagnostics.json",summary)


def run():
    raw=company()
    gs,_=make_windows(raw,"company",*PLAN["span"])
    earlier,streams=make_windows(raw,"adapt",.06,1.)
    counts={}
    for g in earlier:
        if g["end"]<=gs[-1]["start"]:
            _,labs,_=draw(g,streams[g["window"]],348179+g["start"],35,0)
            counts[g["window"]]=Counter(v for t in labs for v in t["sources"])
    ck=load(OUT/"local_models.json")
    truth={r["id"]:r for r in read_rows(OUT/"truth.jsonl")}
    refresh={r["id"]:r for r in read_rows(OUT/"local_results.jsonl")}
    predictions={r["id"]:r for r in read_rows(OUT/"local_predictions.jsonl")}
    for g in gs:
        key=g["window"]
        assert ck[key]["last_training_end"]<=g["start"]
        count=sum((counts[w] for w in ck[key]["windows"]),Counter())
        for old in read_rows(OUT/(key+"_predictions.jsonl")):
            sid=old["id"]
            p=predictions[sid]
            assert p["nodes"]==old["nodes"]
            labs=truth[sid]
            nodes=p["nodes"]
            prior=np.log([1.+count[v] for v in nodes])
            i,j=np.triu_indices(len(nodes),1)
            key_seed=348179+sum((k+1)*ord(c) for k,c in enumerate(sid))
            tie=np.random.default_rng(key_seed).permutation(len(i))
            order=np.lexsort((tie,-(prior[i]+prior[j])))
            r=refresh[sid]
            r["frequency"]=None
            if r["inside"]:
                a,b=sorted(nodes.index(v) for v in labs["sources"])
                target=int(np.flatnonzero((i==a)&(j==b))[0])
                r["frequency"]=int(np.flatnonzero(order==target)[0])+1
            assert r["local_joint"]==(None if not r["inside"] else p["orders"]["local_joint"].index(target)+1)
    report={}
    for group in ("dual","ordinary"):
        rows=[r for r in refresh.values() if r[group]]
        by_week={}
        for r in rows:
            by_week.setdefault(r["window"],[]).append(
                (1/r["local_joint"] if r["local_joint"] else 0)-(1/r["frequency"] if r["frequency"] else 0))
        sums=np.array([sum(xs) for xs in by_week.values()])
        sizes=np.array([len(xs) for xs in by_week.values()])
        draws=np.random.default_rng(592).integers(len(sizes),size=(10000,len(sizes)))
        report[group]=dict(scores=metrics(rows,"frequency"),delta=sums.sum()/sizes.sum(),
                           paired_ci=np.quantile(sums[draws].sum(axis=1)/sizes[draws].sum(axis=1),[.025,.975]).tolist())
    write_json(OUT/"frequency_summary.json",report)
    lineage_summary()
    print(report)


if __name__=="__main__":
    run()
