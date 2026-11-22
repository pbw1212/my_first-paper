"""Same five observed snapshots and realized source labels as the FGSSI adapter."""
import argparse

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from distilled_source_pairs import CHECKPOINTS, PLAN, Packed, PairNet, PublicWindow, shortlist
from fgssi_suffix import MAIN, OUT, RUNS, cohort, digest, evaluation_windows, load, make_windows, pair_rank, raw_data, read_rows, save_rows, tie_order, write_json


def frames(event):
    batches=event["observed"]
    cuts=np.linspace(0,len(batches)-1,5).round().astype(int)
    first=0;groups=[]
    for last in cuts:
        if last>=first:
            groups.append(sorted(v for b in batches[first:last+1] for v in b))
            first=last+1
    assert first==len(batches)
    return dict(event,observed=groups)


def samples(name,phase):
    rows=[]
    raw=raw_data(name)
    for graph,events,truth in cohort(name,phase):
        window=PublicWindow(graph,raw)
        for event,lab in zip(events,truth,strict=True):
            (nodes,_,_),features=window.design(frames(event))
            a,b=[nodes.index(v) for v in lab["sources"]]
            partner=np.zeros(len(nodes),np.float32)
            partner[b]=1
            rows.append((*features,a,b,partner))
    return rows


def fit(name):
    folder=OUT/name
    rows=samples(name,"fit")
    x=np.vstack([r[0] for r in rows]);mean=x.mean(0);std=np.maximum(x.std(0),.025)
    train=Packed(rows,mean,std)
    del rows,x
    val=Packed(samples(name,"cal"),mean,std)
    assert (len(train.x),len(val.x))==(1000,300)
    np.savez(folder/"five_scale.npz",mean=mean,std=std)
    torch.manual_seed(PLAN["training_seed"])
    rng=np.random.default_rng(PLAN["training_seed"])
    net=PairNet(len(mean),True)
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.005)
    best=float("inf");curve={}
    for epoch in range(1,max(CHECKPOINTS)+1):
        net.train()
        for idx in np.array_split(rng.permutation(len(train.x)),int(np.ceil(len(train.x)/24))):
            opt.zero_grad();train.loss(net,idx,False).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if epoch in CHECKPOINTS:
            net.eval()
            with torch.no_grad():
                value=sum(len(i)*float(val.loss(net,i,False)) for i in np.array_split(np.arange(len(val.x)),15))/len(val.x)
            curve[epoch]=value
            if value<best:
                best=value;torch.save(net.state_dict(),folder/"five_hard.pt")
    write_json(folder/"five_selection.json",dict(curve=curve,selected=min(curve,key=curve.get),
        hashes={f:digest(folder/f) for f in ("five_scale.npz","five_hard.pt")}))


def trained(name):
    folder=OUT/name
    assert all(digest(folder/f)==h for f,h in load(folder/"five_selection.json")["hashes"].items())
    scale=np.load(folder/"five_scale.npz");net=PairNet(len(scale["mean"]),True).eval()
    net.load_state_dict(torch.load(folder/"five_hard.pt",weights_only=True))
    return net,scale["mean"],scale["std"]


def score(window,event,model):
    (nodes,_,_),(x,edge,overlap)=window.design(frames(event))
    net,mean,std=model
    with torch.no_grad():
        matrix=net(torch.from_numpy(np.clip((x-mean)/std,-6,6))[None],
            torch.tensor(edge.astype(np.float16),dtype=torch.float32)[None],torch.tensor(overlap.astype(np.float16),dtype=torch.float32)[None],
            torch.ones(1,len(nodes),dtype=torch.bool))[0].numpy()
    return nodes,matrix


def evaluate(run):
    main=MAIN/run;plan=load(main/"test_plan.json");net=trained(plan["source"])
    raw=raw_data(plan["network"])
    gs,_=make_windows(raw,plan["network"],.15,1.) if run.endswith("_new_windows") else evaluation_windows(plan["network"],raw)
    events=read_rows(main/"public.jsonl");truth={e["id"]:e for e in read_rows(main/"truth.jsonl")}
    results=[]
    for g in (g for g in gs if g["window"] in plan["windows"]):
        window=PublicWindow(g,raw)
        for e in (e for e in events if e["window"]==g["window"]):
            nodes,m=score(window,e,net);tie=tie_order(e["id"],len(nodes))
            chosen=[nodes[i] for i in shortlist(m,tie)[0]]
            a,b=[nodes.index(u) for u in truth[e["id"]]["sources"]]
            results.append(dict(id=e["id"],chosen=chosen,rank=pair_rank(m,a,b,tie)))
    assert len(results)==len(events)
    save_rows(OUT/run/"five_predictions.jsonl",results)
    print(run,len(results),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("name",choices=RUNS);p.add_argument("step",choices=["fit","evaluate"])
    a=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(2):
        if a.step=="fit":
            assert a.name in ("Dept2","company")
            fit(a.name)
        else:
            evaluate(a.name)
