"""Paired evaluation of the four history/scoring cells and the temporal baseline."""
import argparse
import gzip
import json
import random
import time

import numpy as np
import torch
from numba import njit
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

import enron_temporal_transfer as enron
import fgssi_suffix as fg
from build_dept3_pilot import simulate
from distilled_source_pairs import PairNet, PublicWindow, best_subset, shortlist
from factorial_fgssi import OUT, SEEDS, weights
from factorial_history import StaticWindow, folder_for
from five_frame_pairs import frames
from score_dept3_candidates import roles
from thesis_holdout import masked

RUNS=(*fg.RUNS,"Enron","Dept2_masks","company_masks")
CELLS={"A":("static",False),"B":("static",True),"C":("temporal",False),"D":("temporal",True)}
BUDGETS=(8,12,20)


def generate_masks(name):
    raw=fg.raw_data(name)
    graphs,streams=fg.evaluation_windows(name,raw)
    folder=OUT/(name+"_masks")
    folder.mkdir(parents=True,exist_ok=True)
    public=[];truth=[];counts={}
    for j,g in enumerate(graphs):
        window=g["window"]
        rng=random.Random(142917+j+(1000 if name=="company" else 0))
        accepted={50:0,75:0,"both":0,"attempts":400}
        for r in range(400):
            sources=sorted(rng.sample(g["nodes"],2))
            seed=rng.getrandbits(63)
            trace=simulate(streams[window],sources,seed)
            masks={q:masked(trace,q) for q in (50,75)}
            common=all(masks.values())
            accepted["both"]+=int(common)
            for q,payload in masks.items():
                if payload is None:
                    continue
                cut,groups=payload
                sid=f"factorial-{window}-{r}-{q}"
                public.append(dict(id=sid,window=window,observed=groups,k=2))
                truth.append(dict(id=sid,sources=sources,window=window,seed=seed,trajectory=trace,cut_batches=cut,
                                  mask=q,common=common,base_id=f"{window}-{r}"))
                accepted[q]+=1
        counts[window]=accepted
    fg.save_rows(folder/"public.jsonl",public)
    fg.save_rows(folder/"truth.jsonl",truth)
    fg.write_json(folder/"generation.json",counts)


def dataset(run):
    if run=="Enron":
        enron.verify()
        raw=enron.load_events(enron.CONTACTS)
        return "Dept2",enron.FOLDER,raw,[g for g,_ in enron.windows(raw)]
    if run.endswith("_masks"):
        name=run.removesuffix("_masks");raw=fg.raw_data(name)
        return name,OUT/run,raw,fg.evaluation_windows(name,raw)[0]
    folder=fg.MAIN/run;plan=fg.load(folder/"test_plan.json")
    raw=fg.raw_data(plan["network"])
    gs,_=(fg.make_windows(raw,plan["network"],.15,1.) if run.endswith("_new_windows")
          else fg.evaluation_windows(plan["network"],raw))
    return plan["source"],folder,raw,[g for g in gs if g["window"] in plan["windows"]]


def fitted(source,seed,cell):
    kind,interaction=CELLS[cell];folder=folder_for(source,kind)
    cfg=fg.load(folder/(("pair" if interaction else "add")+str(seed)+".json"))
    from pathlib import Path
    path=Path(cfg["weights"])
    if not path.is_file():
        path=(fg.OUT/source/"five_hard.pt" if kind=="temporal" and interaction and seed==314159 else folder/path.name)
    assert fg.digest(path)==cfg["sha256"]
    with np.load(folder/"scale.npz") as scale:
        mean,std=scale["mean"],scale["std"]
    model=PairNet(len(mean),interaction).eval()
    model.load_state_dict(torch.load(path,weights_only=True))
    return model,mean,std


@njit
def rank_upper(matrix,a,b,tie):
    a,b=min(a,b),max(a,b)
    ref=matrix[a,b]
    left,right=min(tie[a],tie[b]),max(tie[a],tie[b])
    rank=1
    for i in range(len(tie)):
        for j in range(i+1,len(tie)):
            val=matrix[i,j]
            if val>ref:
                rank+=1
            elif val==ref:
                l,r=min(tie[i],tie[j]),max(tie[i],tie[j])
                if l<left or (l==left and r<right):
                    rank+=1
    return rank


def decode_stats(matrix,unary,nodes,tie,gold):
    a,b=[nodes.index(v) for v in gold]
    if unary is not None:
        order=fg.sorted_nodes(unary,tie)
        matrix=np.add.outer(unary,unary) if matrix is None else matrix
        choices={k:order[:k] for k in BUDGETS}
        pool=None
    else:
        picked,marg,pool,logz=shortlist(matrix,tie)
        # With symmetric scores the row sums of the normalized matrix are the endpoint marginals.
        mass=np.exp(matrix.astype(np.float64)-logz)
        sums=mass.sum(1)
        order=fg.sorted_nodes(sums,tie)
        assert np.array_equal(order[:12],marg)
        local=mass[np.ix_(pool,pool)]
        initial=fg.sorted_nodes(sums[pool],tie[pool])
        choices={k:(picked if k==12 else pool[best_subset(local,initial,tie[pool],k)]) for k in BUDGETS}
    def hits(indices):
        return int(a in indices)+int(b in indices)
    for k,idx in choices.items():
        assert len(set(idx))==len(idx)==k
    best12=[nodes[j] for j in choices[12]]
    return dict(rank=int(rank_upper(matrix,a,b,tie)),m=[hits(order[:k]) for k in BUDGETS],
                joint=[hits(choices[k]) for k in BUDGETS],list12=best12,marg12=[nodes[i] for i in order[:12]],
                pool=hits(pool) if pool is not None else None)


def stats_for(matrix,unary,nodes,tie,gold):
    try:
        return decode_stats(matrix,unary,nodes,tie,gold)
    except AssertionError:
        if unary is not None:
            raise
        # The swap gain assumes exact symmetry. Use the same upper triangle
        # used by the unordered-pair rank when floating-point roundoff breaks it.
        upper=np.triu(matrix,1)
        error=np.max(np.abs(upper-np.tril(matrix,-1).T))
        if not (0<error<1e-4 and np.isfinite(upper).all()):
            raise
        result=decode_stats(upper+upper.T,None,nodes,tie,gold)
        result["symmetry_roundoff"]=float(error)
        return result


def evaluate(run,phase):
    source,parent,raw,gs=dataset(run)
    seeds=(314159,) if run=="CollegeMsg" else SEEDS
    public=fg.read_rows(parent/"public.jsonl");truth={r["id"]:r for r in fg.read_rows(parent/"truth.jsonl")}
    if phase=="fgssi":
        nets={}
        for seed in seeds:
            model=fg.FGSSI().eval()
            model.load_state_dict(torch.load(weights(source,seed),weights_only=True))
            nets[seed]=model
    else:
        nets={(seed,cell):fitted(source,seed,cell) for seed in seeds for cell in CELLS}
    folder=OUT/run;folder.mkdir(parents=True,exist_ok=True)
    target=folder/(phase+".jsonl.gz");temp=target.with_suffix(".tmp")
    count=0;timing={};start=time.perf_counter()
    with gzip.open(temp,"wt",encoding="utf-8",compresslevel=2) as stream:
        for g in gs:
            items=[e for e in public if e["window"]==g["window"]]
            if not items:
                continue
            if phase=="fgssi":
                geometry=fg.Geometry(g)
            else:
                hist={"static":StaticWindow(g),"temporal":PublicWindow(g,raw)}
            for offset in range(0,len(items),8):
                batch=items[offset:offset+8]
                if phase=="fgssi":
                    features=geometry.inputs(batch)
                    with torch.no_grad():
                        outputs={}
                        for seed,model in nets.items():
                            logits,_=model(*features)
                            outputs[seed]=(logits[:,:,1]-logits[:,:,0]).numpy().astype(np.float64)
                for row,e in enumerate(batch):
                    label=truth[e["id"]]
                    vis={n for group in e["observed"] for n in group}
                    nodes=sorted(set(g["nodes"])-vis)
                    if "base_id" in label:
                        # Keep the relative ordering of shared candidates across both masks.
                        roster=sorted(g["nodes"])
                        all_ties=dict(zip(roster,fg.tie_order(label["base_id"],len(roster))))
                        tie=np.array([all_ties[v] for v in nodes])
                    else:
                        tie=fg.tie_order(e["id"],len(nodes))
                    results={}
                    if phase=="fgssi":
                        idx=[geometry.index[v] for v in nodes]
                        for seed,v in outputs.items():
                            results[f"F{seed}"]=stats_for(None,v[row,idx],nodes,tie,label["sources"])
                    else:
                        for kind,window in hist.items():
                            base,(x,edge,overlap)=window.design(frames(e))
                            assert base[0]==nodes
                            et=torch.tensor(edge.astype(np.float16),dtype=torch.float32)[None]
                            ov=torch.tensor(overlap.astype(np.float16),dtype=torch.float32)[None]
                            legal=torch.ones(1,len(nodes),dtype=torch.bool)
                            for cell,(key,interaction) in CELLS.items():
                                if key!=kind:
                                    continue
                                for seed in seeds:
                                    model,mean,std=nets[seed,cell]
                                    tick=time.perf_counter()
                                    tx=torch.from_numpy(np.clip((x-mean)/std,-6,6))[None]
                                    with torch.no_grad():
                                        if interaction:
                                            matrix=model(tx,et,ov,legal)[0].numpy()
                                            unary=None
                                        else:
                                            unary=model.node(model.encoder(tx))[0,:,0].numpy().astype(np.float64)
                                            matrix=None
                                    results[f"{cell}{seed}"]=stats_for(matrix,unary,nodes,tie,label["sources"])
                                    timing[cell]=timing.get(cell,0.)+time.perf_counter()-tick
                    record=dict(id=e["id"],window=g["window"],n=len(nodes),emitters=len(roles(label)[0]),models=results)
                    if "mask" in label:
                        record.update(mask=label["mask"],common=label["common"],base_id=label["base_id"])
                    stream.write(json.dumps(record,separators=(",",":"))+"\n")
                    count+=1
            print(run,phase,g["window"],count,flush=True)
    assert count==len(public)
    temp.replace(target)
    fg.write_json(folder/(phase+"_audit.json"),dict(n=count,seeds=seeds,input_hashes={p:fg.digest(parent/p) for p in ["public.jsonl","truth.jsonl"]},
        hash=fg.digest(target),wall_seconds=time.perf_counter()-start,score_decode_seconds=timing))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",choices=RUNS)
    p.add_argument("step",choices=["cells","fgssi","generate"])
    a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if a.step=="generate":
            assert a.run.endswith("_masks")
            generate_masks(a.run.removesuffix("_masks"))
        else:
            evaluate(a.run,a.step)
