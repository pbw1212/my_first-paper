"""Audit event-level comparisons and summarize by seed, window, and source activity."""
import gzip
import json
from collections import defaultdict
import random

import numpy as np

from evaluate_factorial_history import BUDGETS, CELLS, RUNS, dataset
from factorial_fgssi import OUT
from factorial_history import StaticWindow
import fgssi_suffix as fg
from build_dept3_pilot import simulate
from thesis_holdout import masked


def read_gzip(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        return {r["id"]:r for line in f if (r:=json.loads(line))}


def check_static():
    edges=[(0,1,2),(1,0,6),(2,3,1),(3,4,2),(4,5,1),(5,6,1),(5,0,4)]
    g=dict(nodes=list(range(7)),edges=edges)
    h=dict(nodes=g["nodes"],edges=[(v,u,c) for u,v,c in reversed(edges)])
    event=dict(observed=[[0],[2],[4]])
    a=StaticWindow(g).design(event)[1]
    b=StaticWindow(h).design(event)[1]
    assert all(np.array_equal(x,y) for x,y in zip(a,b))


def update(acc,rec):
    acc["n"]+=1
    for key,v in rec.items():
        s=acc[key]
        for j,k in enumerate(BUDGETS):
            s["m"+str(k)]+=v["m"][j]==2
            s["j"+str(k)]+=v["joint"][j]==2
        s["any"]+=v["joint"][1]>0
        s["pool"]+=v["pool"]==2
        s["top1"]+=v["rank"]==1
        s["top32"]+=v["rank"]<=32
        s["reciprocal"]+=1/v["rank"]
    for contrast,left,right,field1,field2 in [
        ("B-A","B","A","m","m"),("D-C","D","C","m","m"),("C-A","C","A","m","m"),
        ("D-B","D","B","m","m"),("Ddecode","D","D","joint","m"),
        ("D-F","D","F","joint","joint"),("C-F","C","F","joint","joint")]:
        a=rec[left][field1][1]==2;b=rec[right][field2][1]==2
        acc[contrast]["gained"]+=a and not b
        acc[contrast]["lost"]+=b and not a


def audit_run(run):
    _,parent,_,gs=dataset(run)
    rosters={g["window"]:set(g["nodes"]) for g in gs}
    public={r["id"]:r for r in fg.read_rows(parent/"public.jsonl")}
    truth={r["id"]:r for r in fg.read_rows(parent/"truth.jsonl")}
    if run.endswith("_masks"):
        name=run.removesuffix("_masks")
        all_graphs,streams=fg.evaluation_windows(name,fg.raw_data(name))
        saved_flow=fg.load(parent/"generation.json")
        recovered=set()
        for j,g in enumerate(all_graphs):
            w=g["window"]
            rng=random.Random(142917+j+(1000 if name=="company" else 0))
            counts={"50":0,"75":0,"both":0,"attempts":400}
            for attempt in range(400):
                sources=sorted(rng.sample(g["nodes"],2));seed=rng.getrandbits(63)
                trajectory=simulate(streams[w],sources,seed)
                hits=[]
                for q in (50,75):
                    obs=masked(trajectory,q)
                    if obs is None:
                        continue
                    sid=f"factorial-{w}-{attempt}-{q}"
                    y=truth[sid]
                    assert (y["sources"],y["seed"],y["trajectory"])==(sources,seed,trajectory)
                    assert obs==(y["cut_batches"],public[sid]["observed"])
                    counts[str(q)]+=1;hits.append(sid);recovered.add(sid)
                counts["both"]+=len(hits)==2
                assert all(truth[sid]["common"]==(len(hits)==2) for sid in hits)
            assert counts==saved_flow[w]
        assert recovered==set(public)
    check={}
    for tag in ("cells","fgssi"):
        q=fg.load(OUT/run/(tag+"_audit.json"));p=OUT/run/(tag+".jsonl.gz")
        assert fg.digest(p)==q["hash"] and all(fg.digest(parent/f)==v for f,v in q["input_hashes"].items())
        assert q["n"]==len(public)
        check[tag]=read_gzip(p)
    assert public.keys()==truth.keys()==check["cells"].keys()==check["fgssi"].keys()
    archive={}
    if run=="Enron":
        archive={r["id"]:(r["sets"]["five_hard"],r["ranks"]["five_hard"],r["sets"]["fgssi"],r["ranks"]["fgssi"])
                 for r in fg.read_rows(parent/"predictions.jsonl")}
    elif not run.endswith("_masks"):
        first={r["id"]:r for r in fg.read_rows(fg.OUT/run/"fgssi_predictions.jsonl")}
        archive={r["id"]:(r["chosen"],r["rank"],first[r["id"]]["chosen"],first[r["id"]]["rank"])
                 for r in fg.read_rows(fg.OUT/run/"five_predictions.jsonl")}
    stats={}
    def group(tag,seed):
        sub=stats.setdefault(tag,{})
        if seed not in sub:
            sub[seed]=defaultdict(lambda:defaultdict(float));sub[seed]["n"]=0
        return sub[seed]
    repeat={"pair_set_mismatches":0,"pair_rank_mismatches":0,"fg_set_mismatches":0,"fg_rank_mismatches":0}
    precision=[];symmetry=[]
    seeds=fg.load(OUT/run/"cells_audit.json")["seeds"]
    for sid,event in public.items():
        lhs,rhs=check["cells"][sid],check["fgssi"][sid]
        gold=set(truth[sid]["sources"]);w=event["window"]
        legal=rosters[w]-{u for b in event["observed"] for u in b}
        assert gold<=legal and len(legal)==lhs["n"]==rhs["n"]
        tags=["all","window/"+w,"emitters/"+str(lhs["emitters"])]
        if "mask" in lhs:
            tags.append("mask/"+str(lhs["mask"]))
            if lhs["common"]:
                tags.append("matched/"+str(lhs["mask"]))
        models={**lhs["models"],**rhs["models"]}
        for seed in seeds:
            rec={k:models[k+str(seed)] for k in (*CELLS,"F")}
            for v in rec.values():
                for key,count in (("list12",v["joint"][1]),("marg12",v["m"][1])):
                    ids=set(v[key])
                    assert len(ids)==len(v[key])==12 and ids<=legal and count==len(gold&ids)
                assert 1<=v["rank"]<=len(legal)*(len(legal)-1)//2
            for cell,v in rec.items():
                if "symmetry_roundoff" in v:
                    symmetry.append(dict(id=sid,cell=cell,seed=seed,error=v["symmetry_roundoff"]))
            if archive and seed==314159:
                a,r,_,_=archive[sid]
                repeat["pair_set_mismatches"]+=set(a)!=set(rec["D"]["list12"])
                repeat["pair_rank_mismatches"]+=r!=rec["D"]["rank"]
                if r!=rec["D"]["rank"]:
                    precision.append(dict(id=sid,method="D",previous=r,current=rec["D"]["rank"]))
            if archive and seed==111:
                _,_,a,r=archive[sid]
                repeat["fg_set_mismatches"]+=set(a)!=set(rec["F"]["list12"])
                repeat["fg_rank_mismatches"]+=r!=rec["F"]["rank"]
                if r!=rec["F"]["rank"]:
                    precision.append(dict(id=sid,method="F",previous=r,current=rec["F"]["rank"]))
            for tag in tags:
                update(group(tag,seed),rec)
    print(run,len(public),repeat,flush=True)
    return dict(groups=stats,archive_recheck=repeat,rank_differences=precision,symmetry_roundoff=symmetry)


def mean_count(groups,cell,metric):
    return np.mean([v[cell][metric] for v in groups.values()])


if __name__=="__main__":
    check_static()
    summary={r:audit_run(r) for r in RUNS}
    fg.write_json(OUT/"summary.json",summary)
    fg.write_json(OUT/"audit.json",dict(summary=fg.digest(OUT/"summary.json"),immutable_archive_rechecks={k:v["archive_recheck"] for k,v in summary.items()},
        evidence_files={p.relative_to(OUT).as_posix():fg.digest(p) for p in OUT.rglob("*_audit.json")},
        static_invariance="reversing all edges and permuting the historical edge order produces identical features"))
