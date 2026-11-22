"""Replay source labels, verify lists and aggregate paired effects by draw and contact window."""
import argparse
import hashlib
import re
from collections import Counter,defaultdict
from itertools import combinations,product

import numpy as np
import torch

from build_dept3_pilot import mask,simulate
from budget_source_training import soft_membership
from distilled_source_pairs import OUT,PREVIOUS,SMS,PairNet,evaluation_windows,load,make_windows,pair_rank,raw_data,read_rows,shortlist,write_json
from score_dept3_candidates import roles
from silent_source_oracle import CompleteTrace


def numerical_checks():
    # Enumerate all seeds and transmission masks that produce one exact observed trace.
    graph={"nodes":list(range(6))}
    stream=[(1,[(0,1),(3,4)]),(2,[(1,2),(4,3)]),(3,[(5,1),(2,5)])]
    trajectory=[{"t":1,"joins":[[1,[0]]]},{"t":2,"joins":[[2,[1]]]}]
    masses=Counter()
    for pair in combinations(graph["nodes"],2):
        for bits in product((0,1),repeat=6):
            active=set(pair);trace=[];index=0
            for t,contacts in stream:
                parents=defaultdict(list)
                for u,v in contacts:
                    if bits[index] and u in active and v not in active:
                        parents[v].append(u)
                    index+=1
                if parents:
                    trace.append(dict(t=t,joins=[[v,parents[v]] for v in sorted(parents)]))
                    active.update(parents)
            if trace==trajectory:
                masses[pair]+=.3**sum(bits)*.7**(6-sum(bits))
    actor,partners,p=CompleteTrace(graph,stream).evaluate(dict(sources=[0,3],trajectory=trajectory),return_distribution=True)
    assert actor==0
    expected=np.array([masses[tuple(sorted((actor,b)))]/sum(masses.values()) for b in partners])
    teacher_error=float(np.abs(expected-p).max())
    assert teacher_error<1e-12
    gen=np.random.default_rng(723)
    scores=torch.tensor(gen.normal(size=(2,23)),dtype=torch.float64,requires_grad=True)
    valid=torch.ones_like(scores,dtype=torch.bool)
    valid[1,-3:]=False
    assert torch.allclose(soft_membership(scores,valid).sum(1),torch.tensor([12.,12.],dtype=scores.dtype),atol=5e-6)
    weights=torch.tensor(gen.normal(size=(2,23)),dtype=scores.dtype)
    grad=torch.autograd.grad((soft_membership(scores,valid)*weights).sum(),scores)[0]
    direction=torch.tensor(gen.normal(size=(2,23)),dtype=scores.dtype)
    eps=.001
    numerical=((soft_membership(scores.detach()+eps*direction,valid)-soft_membership(scores.detach()-eps*direction,valid))*weights).sum()/(2*eps)
    error=float(abs(numerical-(direction*grad).sum()))
    assert error<.002 and not grad[~valid].any()
    for tied in (True,False):
        z=gen.normal(size=(28,28))
        z=z+z.T
        if tied:
            z=np.round(z)
        tie=gen.permutation(len(z))
        pairs=sorted(((i,j) for i in range(len(z)) for j in range(i+1,len(z))),
                     key=lambda ij:(-z[ij],min(tie[list(ij)]),max(tie[list(ij)])))
        assert all(pair_rank(z,i,j,tie)==k for k,(i,j) in enumerate(pairs,1))
        selected,marg,_,logz=shortlist(z.copy(),tie)
        score=lambda c:np.triu(np.exp(z[np.ix_(c,c)]-logz),1).sum()
        assert score(selected)>=score(marg)-1.e-8
    # The pair model must commute with relabelling candidates.
    torch.manual_seed(8321)
    model=PairNet(8,True)
    x=torch.randn(1,26,8);r=torch.randn(1,26,26);r=(r+r.transpose(1,2))/2
    eligible=torch.ones(1,26,dtype=torch.bool);order=torch.randperm(26)
    before=model(x,r,r,eligible)
    after=model(x[:,order],r[:,order][:,:,order],r[:,order][:,:,order],eligible[:,order])
    assert torch.allclose(after,before[:,order][:,:,order],atol=2.e-5) and torch.allclose(before,before.transpose(1,2),atol=2.e-5)
    return dict(soft_membership_directional_derivative_error=error,teacher_probability_error=teacher_error,
                permutation_check=True,pair_rank_check=True)


def audit(run):
    folder=OUT/run
    plan=load(folder/"test_plan.json")
    name=plan["network"];source=plan["source"]
    for lock in ("lock","budget_lock"):
        assert all(hashlib.sha256((OUT/source/f).read_bytes()).hexdigest()==h for f,h in plan[lock].items())
    raw=raw_data(name)
    if run.endswith("_new_windows"):
        graphs,streams=make_windows(raw,name,.15,1.)
        splits=load(PREVIOUS/name/"windows.json")
        assert set(plan["windows"]).isdisjoint(splits)
        assert min(g["start"] for g in graphs if g["window"] in plan["windows"])>=max(v["end"] for v in splits.values() if v["stage"]=="cal")
    else:
        graphs,streams=evaluation_windows(name,raw)
    roster={g["window"]:set(g["nodes"]) for g in graphs}
    obs=read_rows(folder/"public.jsonl");truth=read_rows(folder/"truth.jsonl")
    preds=read_rows(folder/"predictions.jsonl");diags=read_rows(folder/"likelihoods.jsonl")
    assert len({y["seed"] for y in truth})==len(truth)==len({p["id"] for p in preds})
    metrics=defaultdict(lambda:dict(n=0,one=Counter(),both=Counter(),nll=Counter(),pool=Counter(),
                                    gains=Counter(),losses=Counter(),exact=defaultdict(Counter)))
    for o,y,p,d in zip(obs,truth,preds,diags,strict=True):
        key=p["window"]
        assert o["id"]==y["id"]==p["id"]==d["id"] and key==o["window"]==y["window"]
        assert simulate(streams[key],y["sources"],y["seed"])==y["trajectory"]
        assert mask(y["trajectory"])==(y["cut_batches"],o["observed"])
        actual=set(y["sources"])
        legal=roster[key]-{v for b in o["observed"] for v in b}
        assert actual<=legal
        hits={}
        for tag,c in p["sets"].items():
            assert len(c)==len(set(c))==12 and set(c)<=legal
            hits[tag]=len(actual.intersection(c))
        direct,_=roles(y)
        for group in ("total","batch"+str(p["batch"]),key,"silent" if len(direct)==1 else "two_emitters"):
            row=metrics[group];row["n"]+=1
            for tag,h in hits.items():
                row["one"][tag]+=int(h>=1);row["both"][tag]+=int(h==2)
                row["gains"][tag]+=int(h<2 and hits["pair_soft"]==2)
                row["losses"][tag]+=int(h==2 and hits["pair_soft"]<2)
            for tag,m in d["models"].items():
                assert np.isfinite(m["nll"]) and m["nll"]>=-1e-5
                row["nll"][tag]+=m["nll"]
                if m["in_pool"] is not None:
                    assert m["in_pool"] or hits[tag]<2
                    row["pool"][tag]+=int(m["in_pool"])
            for tag,rank in d.get("ranks",{}).items():
                assert 1<=rank<=len(legal)*(len(legal)-1)//2
                rr=row["exact"][tag]
                rr["mrr"]+=1/rank
                for k in (1,5,32):
                    rr[str(k)]+=int(rank<=k)
    for group,row in metrics.items():
        for tag in row["nll"]:
            row["nll"][tag]/=row["n"]
        for tag in row["exact"]:
            row["exact"][tag]["mrr"]/=row["n"]
    for b in sorted({r["batch"] for r in preds}):
        assert metrics["batch"+str(b)]["n"]==len(plan["windows"])*(plan.get("actual_target_per_window") or plan["local_per_window" if name in ("Dept2","company") else "external_per_window"])
    write_json(folder/"summary.json",metrics)
    print(run,metrics["total"]["both"],flush=True)


def paper_tables():
    report=OUT/"双源组合学习与预算实验.md"
    body=report.read_text(encoding="utf-8")
    thesis=(OUT.parents[1]/"论文实验结果与写作底稿.md").read_text(encoding="utf-8")
    assert "distilled_source_pairs/双源组合学习与预算实验.md" in thesis
    for target in re.findall(r"\]\(([^)]+)\)",body):
        assert target.startswith("https://") or (OUT/target).is_file(),target
    runs=["company","Dept2","Dept1","CollegeMsg",SMS]
    display=["制造企业","Dept2","Dept1","CollegeMsg","哥本哈根短信"]
    methods=["pooled","old_joint","node_hard","node_soft","pair_hard","pair_soft","budget_soft"]
    totals={}
    for run,title in zip(runs,display,strict=True):
        folder=OUT/run
        pred=read_rows(folder/"predictions.jsonl");truth=read_rows(folder/"truth.jsonl")
        counts=[sum(set(y["sources"])<=set(p["sets"][m]) for y,p in zip(truth,pred,strict=True)) for m in methods]
        fields=[s.strip().replace("**","").replace(",","") for s in next(line for line in body.splitlines() if line.startswith("| "+title+" |")).strip("|").split("|")]
        assert list(map(int,fields[1:]))==[len(truth),*counts]
        totals[run]=len(truth)
    for run in ("Dept2_new_windows","company_new_windows"):
        folder=OUT/run
        truth={y["id"]:y for y in read_rows(folder/"truth.jsonl")}
        obs={o["id"]:o for o in read_rows(folder/"public.jsonl")}
        diags=read_rows(folder/"likelihoods.jsonl")
        fixed=read_rows(folder/"predictions.jsonl")
        prefix="| Dept2，" if run.startswith("Dept2") else "| 制造企业，"
        lines=body.splitlines()
        first=next(i for i,line in enumerate(lines) if line.startswith(prefix))
        for j,tag in enumerate(("old_joint","node_soft","pair_hard","pair_soft")):
            ranks=[d["ranks"][tag] for d in diags]
            both=sum(set(truth[p["id"]]["sources"])<=set(p["sets"][tag]) for p in fixed)
            cells=[c.strip().replace(",","") for c in lines[first+j].strip("|").split("|")]
            assert list(map(int,cells[2:6]))==[both,*[sum(rank<=k for rank in ranks) for k in (1,5,32)]]
            assert cells[6]==f"{np.mean(1/np.array(ranks)):.4f}"
        graphs,_=make_windows(raw_data(run.split("_")[0]),run.split("_")[0],.15,1.)
        roster={g["window"]:set(g["nodes"]) for g in graphs}
        predicted=read_rows(folder/"small_sample_predictions.jsonl")
        assert {p["id"] for p in predicted}==truth.keys() and len(predicted)==len(truth)
        totals[run]=len(truth)
        summary=load(folder/"small_sample_summary.json")
        for tag,values in summary["models"].items():
            hits=nll=0
            for r in predicted:
                sid=r["id"];c=r["models"][tag]["roster"]
                visible={v for batch in obs[sid]["observed"] for v in batch}
                assert len(set(c))==len(c)==12 and set(c)<=roster[r["window"]]-visible
                hits+=int(set(truth[sid]["sources"])<=set(c))
                nll+=r["models"][tag]["nll"]
            assert values["hits"]==hits and abs(nll/len(predicted)-values["nll"])<1e-9
    assert sum(totals.values())==5010
    return dict(verified_events=totals,main_table_checked_against_predictions=True,exact_rank_table_verified=True,
                small_sample_recounted=True,local_report_links_verified=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",choices=["checks","paper","company","Dept2","Dept1","CollegeMsg",SMS,"company_new_windows","Dept2_new_windows"])
    a=p.parse_args()
    if a.run=="checks":
        write_json(OUT/"numerical_checks.json",numerical_checks())
    elif a.run=="paper":
        write_json(OUT/"delivery_audit.json",paper_tables())
    else:
        audit(a.run)
