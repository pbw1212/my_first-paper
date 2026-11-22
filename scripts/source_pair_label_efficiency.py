"""Matched small-label comparison on the nine reserved development contact windows."""
import argparse

import numpy as np
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from distilled_source_pairs import OUT,Packed,PairNet,PublicWindow,load,make_windows,raw_data,read_rows,samples,save_rows,shortlist,write_json
from ordinary_source_roles import OUT as ORDINARY


SEEDS=(73911,73912,73913)
PLAN=dict(seeds=list(SEEDS),training="50 distinct events per early window; paired subsets of 250 for each seed",
          control="Identical public features, model initialization, batch order and optimization; change only the supervision",
          checkpoints=[6,15,30],selection="minimum intermediate-window true-pair NLL on 300 unchanged validation events",
          endpoints=["reserved-window mean true-pair NLL","both-in-12"],note="Secondary ablation designed after the main tests")


def experiment(name):
    folder=OUT/(name+"_new_windows")
    write_json(folder/"small_sample_plan.json",PLAN)
    fitrows=samples(name,"fit")
    calrows=samples(name,"cal")
    original=read_rows(ORDINARY/name/"fit_public.jsonl")
    groups=[np.array([i for i,o in enumerate(original) if o["window"]==g]) for g in sorted({o["window"] for o in original})]
    assert all(len(g)==200 for g in groups) and len(groups)==5 and len(original)==len(fitrows)
    # samples() walks the windows chronologically, just as the saved training records do.
    assert [r["window"] for r in original]==sorted(r["window"] for r in original)
    fitted={};curves={};subsets={}
    for seed in SEEDS:
        sampling=np.random.default_rng(seed)
        subset=np.concatenate([np.sort(sampling.choice(idx,50,replace=False)) for idx in groups])
        subsets[seed]=[original[i]["id"] for i in subset]
        rows=[fitrows[i] for i in subset]
        concat=np.vstack([r[0] for r in rows])
        mean=concat.mean(0);std=np.maximum(concat.std(0),.025)
        train=Packed(rows,mean,std);cal=Packed(calrows,mean,std)
        for kind in ("hard","soft"):
            tag=f"{seed}_{kind}"
            rng=np.random.default_rng(314159)
            torch.manual_seed(314159)
            model=PairNet(len(mean),True)
            opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005)
            curve={};best=float("inf")
            for epoch in range(1,31):
                for idx in np.array_split(rng.permutation(250),11):
                    opt.zero_grad()
                    loss=train.loss(model,idx,kind=="soft")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                    opt.step()
                if epoch in PLAN["checkpoints"]:
                    with torch.no_grad():
                        value=np.mean([cal.loss(model,idx,False).item() for idx in np.array_split(np.arange(300),15)])
                    curve[epoch]=float(value)
                    if value<best:
                        best=value
                        torch.save(model.state_dict(),folder/(tag+".pt"))
            model.load_state_dict(torch.load(folder/(tag+".pt"),weights_only=True))
            fitted[tag]=(mean,std,model.eval())
            curves[tag]=curve
        np.savez(folder/(str(seed)+"_scale.npz"),mean=mean,std=std)
        print(name,seed,"fit",flush=True)
    write_json(folder/"small_sample_fits.json",dict(subsets=subsets,cal_nll=curves))
    obs=read_rows(folder/"public.jsonl")
    truth={y["id"]:y for y in read_rows(folder/"truth.jsonl")}
    raw=raw_data(name)
    graphs,_=make_windows(raw,name,.15,1.)
    results=[];totals={k:dict(hits=0,nll=0.) for k in fitted}
    for g in graphs:
        items=[o for o in obs if o["window"]==g["window"]]
        if not items:
            continue
        w=PublicWindow(g,raw)
        for o in items:
            base,(x,edge,overlap)=w.design(o)
            nodes=base[0]
            tie=np.random.default_rng(1000003+sum((j+1)*ord(c) for j,c in enumerate(o["id"]))).permutation(len(nodes))
            e=torch.from_numpy(edge.astype(np.float16).astype(np.float32))[None]
            ov=torch.from_numpy(overlap.astype(np.float16).astype(np.float32))[None]
            mask=torch.ones(1,len(nodes),dtype=torch.bool)
            choices={}
            for tag,(mean,std,model) in fitted.items():
                tx=torch.from_numpy(np.clip((x-mean)/std,-6,6))[None]
                with torch.no_grad():
                    scores=model(tx,e,ov,mask)[0].numpy()
                selected,_,_,normalizer=shortlist(scores,tie)
                roster=[nodes[i] for i in selected]
                a,b=[nodes.index(v) for v in truth[o["id"]]["sources"]]
                nll=float(normalizer-scores[a,b])
                totals[tag]["nll"]+=nll
                totals[tag]["hits"]+=int(a in selected and b in selected)
                choices[tag]=dict(roster=roster,nll=nll)
            results.append(dict(id=o["id"],window=o["window"],models=choices))
    for values in totals.values():
        values["nll"]/=len(results)
    save_rows(folder/"small_sample_predictions.jsonl",results)
    write_json(folder/"small_sample_summary.json",dict(n=len(results),models=totals))
    print(name,totals,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("network",choices=["company","Dept2"])
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        experiment(p.parse_args().network)
