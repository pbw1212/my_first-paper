"""Verify the matched cohorts and compute paired results by real contact window."""
from collections import defaultdict

import numpy as np

import enron_temporal_transfer as enron
from five_frame_pairs import frames
from fgssi_suffix import MAIN, MATCHED, OUT, RUNS, digest, evaluation_windows, load, make_windows, raw_data, read_rows, write_json
from score_dept3_candidates import roles


def index(path):
    rows=read_rows(path)
    idx={r["id"]:r for r in rows}
    assert len(idx)==len(rows)
    return idx


def stats(rows):
    summary={}
    for k in rows[0]["hits"]:
        both=np.array([r["hits"][k] for r in rows])
        s=dict(both=int((both==2).sum()),at_least_one=int((both>0).sum()))
        if k in rows[0]["ranks"]:
            ranks=np.array([r["ranks"][k] for r in rows])
            s.update(top1=int((ranks==1).sum()),top5=int((ranks<=5).sum()),top32=int((ranks<=32).sum()),mrr=float(np.mean(1/ranks)))
        summary[k]=s
    paired={}
    for left,right in (("pair_soft","fgssi"),("five_hard","fgssi"),("pair_soft","jordan"),("five_hard","pair_soft")):
        a=np.array([r["hits"][left]==2 for r in rows])
        b=np.array([r["hits"][right]==2 for r in rows])
        paired[left+"-"+right]=dict(gained=int((a&~b).sum()),lost=int((~a&b).sum()),difference_pp=100*(a.mean()-b.mean()))
    return dict(n=len(rows),methods=summary,paired=paired)


def block_interval(windows,comparison):
    # Circular moving blocks of three neighboring real windows, no resampling of individual cascades.
    rng=np.random.default_rng(711901)
    m=len(windows)
    indices=(rng.integers(0,m,(20000,(m+2)//3,1))+np.arange(3))%m
    indices=indices.reshape(20000,-1)[:,:m]
    counts=np.array([w["n"] for w in windows])
    delta=np.array([w["paired"][comparison]["gained"]-w["paired"][comparison]["lost"] for w in windows])
    return np.quantile(100*delta[indices].sum(1)/counts[indices].sum(1),[.025,.975]).tolist()


def audit(run):
    external=run=="Enron"
    parent=enron.FOLDER if external else MAIN/run
    public=index(parent/"public.jsonl");truth=index(parent/"truth.jsonl")
    if external:
        pred=index(parent/"predictions.jsonl")
        gs=[g for g,_ in enron.windows(enron.load_events(enron.CONTACTS))]
        assert digest(enron.RAW)==enron.RAW_SHA
        assert len(gs)==26 and len({r["seed"] for r in truth.values()})==len(truth)
    else:
        plan=load(parent/"test_plan.json");name=plan["network"]
        raw=raw_data(name)
        gs,_=make_windows(raw,name,.15,1.) if run.endswith("_new_windows") else evaluation_windows(name,raw)
        pred=index(MATCHED/run/"predictions.jsonl")
        fg=index(OUT/run/"fgssi_predictions.jsonl");five=index(OUT/run/"five_predictions.jsonl")
        single=index(MATCHED/run/"snapshot_predictions.jsonl");older=index(parent/"predictions.jsonl")
        assert fg.keys()==five.keys()==single.keys()==older.keys()==public.keys()
        assert all(digest(parent/f)==v for f,v in load(MATCHED/run/"inputs.json")["main"].items())
    assert public.keys()==truth.keys()==pred.keys()
    graphs={g["window"]:g for g in gs}
    grouped=defaultdict(list)
    for i,event in public.items():
        y=truth[i];p=pred[i];key=event["window"]
        seen={v for b in event["observed"] for v in b}
        cuts=np.rint(np.linspace(0,len(event["observed"])-1,5)).astype(int)
        cumulative=[{v for g in event["observed"][:t+1] for v in g} for t in cuts]
        increments=[sorted(a-b) for a,b in zip(cumulative,[set()]+cumulative[:-1]) if a!=b]
        assert increments==frames(event)["observed"]
        legal=set(graphs[key]["nodes"])-seen
        gold=set(y["sources"]);assert len(gold)==2 and gold<=legal and len(legal)==p["n"]
        assert key==y["window"]==p["window"] and y["ordinary"]
        sets=p["sets"].copy();ranks=p["ranks"].copy()
        if not external:
            for k,entry in (("fgssi",fg[i]),("five_hard",five[i]),("snapshot",single[i])):
                sets[k],ranks[k]=entry["chosen"],entry["rank"]
            sets["pair_hard"]=older[i]["sets"]["pair_hard"]
        assert all(len(s)==len(set(s))==12 and set(s)<=legal for s in sets.values())
        assert all(1<=r<=len(legal)*(len(legal)-1)//2 for r in ranks.values())
        hits={k:len(gold.intersection(s)) for k,s in sets.items()}
        if ranks["fgssi"]==1:
            assert hits["fgssi"]==2
        record=dict(hits=hits,ranks=ranks)
        for tag in ("all",key,"emitters_"+str(len(roles(y)[0]))):
            grouped[tag].append(record)
    result={k:stats(records) for k,records in grouped.items()}
    if external:
        windows=[result[k] for k in sorted(graphs) if k in result]
        assert len(windows)==26 and all(w["n"]==60 for w in windows)
        result["calendar_block_intervals"]={c:block_interval(windows,c) for c in result["all"]["paired"]}
    else:
        assert result["all"]["methods"]["pair_soft"]["both"]==load(MATCHED/"summary.json")[run]["groups"]["all"]["methods"]["pair_soft"]["both"]
    return result


if __name__=="__main__":
    enron.verify()
    result={run:audit(run) for run in (*RUNS,"Enron")}
    assert sum(result[r]["all"]["n"] for r in RUNS)==5010 and result["Enron"]["all"]["n"]==1560
    write_json(OUT/"summary.json",result)
    files=[f for f in OUT.rglob("*") if f.is_file() and f.suffix in (".json",".jsonl",".pt",".npz") and f.name!="audit.json"]
    write_json(OUT/"audit.json",dict(n=6570,prior_n=5010,new_n=1560,independent_windows=26,
        checks=["equal event IDs and legal 12-person lists","five observed sets encode the same increments as the hard-label pair model",
                "all ranks within full candidate-pair range","previous main counts reproduced",
                "external model and protocol lock verified","all external hidden trajectories and public observations replayed during evaluation"],
        files={f.relative_to(OUT).as_posix():digest(f) for f in files}))
