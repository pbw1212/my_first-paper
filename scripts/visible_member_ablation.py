"""Paired retention of one additional observed speaker on an identical candidate roster."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import early_only_discussions as task
import fgssi_suffix as fg
from check_early_only_discussions import reference
from five_frame_pairs import frames
from source_prior_shift import excluding_self

OUT=task.OUT/"member_ablation"
METHODS=("volume","simple","anchor","uniform","earliest","balanced","all")


def plan():
    paths=[Path(__file__),task.OUT/"protocol.json",task.r.OUT/"protocol_v2.json",
           Path(reference.__code__.co_filename)]
    paths.extend(task.OUT/(name+".jsonl") for name in task.NAMES)
    return dict(primary="balanced minus exact uniform additional member, among roots with >=3 roster-visible speakers",
                secondary="uniform second member vs latest alone on >=2; full suffix vs anchor; volume and simple full-suffix correction",
                anchor="last roster-visible first arrival, stable lexical order within simultaneous batch",
                balanced="choose earlier visible member whose indicator of a prior outgoing contact from either source has predictive probability closest to one half under anchor posterior averaged over five seeds",
                contacts="28 days strictly before calendar month. Ties favor earlier occurrence then lexical node ID.",
                inference="Dept2 mixed_offset; all observed speakers excluded in every arm, tie order and 12-person budget fixed; conditional evidence attribution on an already observed suffix",
                uncertainty="paired event and model-seed differences, 10000 circular three-month block bootstrap replicates",
                hashes={f.resolve().relative_to(OUT.parents[3]).as_posix():fg.digest(f) for f in paths})


def pair_mass(h,w,means):
    u=h*w
    return (h*excluding_self(h)/means[0]+(h*excluding_self(u)+u*excluding_self(h))/means[1]+
            u*excluding_self(u)/means[2])


def infer(hist,event,groups,legal,models,w,means,tie):
    (nodes,_,_),(x,_,_)=hist.design(frames(dict(event,observed=groups)))
    lookup={n:i for i,n in enumerate(nodes)}
    retained=x[[lookup[v] for v in legal]]
    outputs=[]
    scored=[]
    for _,model,mu,sd in models:
        with torch.no_grad():
            t=torch.from_numpy(np.clip((retained-mu)/sd,-6,6))
            scores=model.node(model.encoder(t))[:,0].double().numpy()
        h=np.exp(scores-scores.max())
        selected=fg.sorted_nodes(pair_mass(h,w,means),tie)[:12]
        outputs.append([legal[i] for i in selected])
        scored.append(h)
    return outputs,scored


def selection(scores,w,means,neighbors):
    # P(at least one source is a neighbor) = 1 - mass of pairs entirely in the complement.
    totals=[pair_mass(h,w,means).sum() for h in scores]
    probs=[]
    for nb in neighbors:
        outside=~nb
        if outside.sum()<2:
            probs.append(1.)
        else:
            probs.append(float(np.clip(np.mean([1-pair_mass(h[outside],w[outside],means).sum()/z
                                          for h,z in zip(scores,totals)]),0,1)))
    return int(np.argmin(np.abs(np.array(probs)-.5))),probs


def measure(name):
    r=task.r
    msg=task.libc.messages() if name==task.libc.NAME else r.messages(name)
    forest,_=task.thread_forest(msg)
    original=fg.read_rows(task.OUT/(name+".jsonl"))
    by_month=defaultdict(list)
    for z in original:
        if z["ready"] and z["visible"]>=2:
            by_month[z["window"]].append(z)
    nets=[m for m in r.fitted_models("Dept2") if m[0]=="mixed_offset"]
    rows=[]
    for g,past,_,_,_,_ in r.calendar_windows(msg,name):
        hist=r.PublicWindow(g,past)
        w=r.activity(g,past)
        means=r.law_arrays(w)[3]
        ix={v:i for i,v in enumerate(g["nodes"])}
        nb=defaultdict(set)
        for t,u,v in past:
            if g["start"]-28*task.DAY<=t<g["start"]:
                nb[v].add(u)
        for z in by_month[g["window"]]:
            root=z["id"]
            gold,batches=reference(root,forest[root],msg)
            obs=[[s for s in b if s in ix] for b in batches]
            obs=[b for b in obs if b]
            visible=[s for b in obs for s in b]
            anchor=visible[-1]
            legal=sorted(ix.keys()-set(visible))
            assert z["sources"]==gold and len(visible)==z["visible"]
            local=w[[ix[s] for s in legal]]
            tie=fg.tie_order(root,len(legal))
            event=dict(id=root,window=g["window"],k=2)
            base,hs=infer(hist,event,[[anchor]],legal,nets,local,means,tie)
            earlier=visible[:-1]
            nbs=[np.array([s in nb[v] for s in legal]) for v in earlier]
            pick,prob=selection(hs,local,means,nbs)
            def hits(lists):
                return [int(set(gold)<=set(lst)) for lst in lists]
            alternatives=[]
            for v in earlier:
                subset=[[s for s in b if s in (v,anchor)] for b in obs]
                ranked,_=infer(hist,event,[b for b in subset if b],legal,nets,local,means,tie)
                alternatives.append(hits(ranked))
            full,_=infer(hist,event,obs,legal,nets,local,means,tie)
            fullhits=hits(full)
            assert fullhits==[z["hits"][f"mixed_offset/{s}"] for s in r.SEEDS]
            rows.append(dict(id=root,window=g["window"],anchor=anchor,earlier=earlier,pick=pick,prob=prob,
                             alternatives=alternatives,hits=dict(volume=[z["hits"]["volume"]]*5,
                             simple=[z["hits"][f"uniform_offset/{s}"] for s in r.SEEDS],
                             anchor=hits(base),uniform=np.mean(alternatives,axis=0).tolist(),earliest=alternatives[0],
                             balanced=alternatives[pick],all=fullhits)))
    assert len(rows)==sum(z["ready"] and z["visible"]>=2 for z in original)
    fg.save_rows(OUT/(name+".jsonl"),rows)


def summarize(rows):
    values={m:np.array([z["hits"][m] for z in rows]) for m in METHODS}
    pairs=(("balanced","uniform"),("earliest","uniform"),("uniform","anchor"),("all","anchor"),("all","simple"),("all","volume"))
    return dict(n=len(rows),both_pct={k:100*v.mean() for k,v in values.items()},
                selector_picks_earliest=sum(z["pick"]==0 for z in rows),
                probability_varies=int(sum(np.ptp(z["prob"])>1e-10 for z in rows)),
                outcome_varies=int(sum(np.any(np.ptp(z["alternatives"],axis=0)>0) for z in rows)),
                contrasts={f"{a}-{b}":task.paired([(z["window"],v) for z,v in zip(rows,values[a]-values[b])],
                                                    task.r.KEYS) for a,b in pairs})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["freeze","report",*task.NAMES])
    cmd=parser.parse_args().action
    OUT.mkdir(exist_ok=True)
    path=OUT/"protocol.json"
    if cmd=="freeze" and not path.exists():
        fg.write_json(path,plan())
    assert fg.load(path)==plan()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if cmd=="report":
            out={}
            for n in task.NAMES:
                rows=fg.read_rows(OUT/(n+".jsonl"))
                out[n]={"two_or_more":summarize(rows),"three_or_more":summarize([z for z in rows if len(z["earlier"])>=2])}
            fg.write_json(OUT/"summary.json",out)
        elif cmd in task.NAMES:
            measure(cmd)
