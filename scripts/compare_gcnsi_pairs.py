"""Train the upstream GraphSL GCNSI network on masked Dept3 cascades, then score exact pairs."""

import argparse
import importlib.util
import sys

import numpy as np
import torch
from scipy.sparse.csgraph import laplacian
from threadpoolctl import threadpool_limits
from torch.nn.functional import cross_entropy

from temporal_pair_evidence import (DEPT, EXTERNAL, NEW, OLD, OUT, cohort, draw, load, load_events,
                                    make_windows, metrics, read_rows, save_rows, sha, write_json)


def architecture():
    path=OUT/"graphsl_gcnsi_model.py"
    assert sha(path)==load(OUT/"gcnsi_source.json")["sha256"]
    spec=importlib.util.spec_from_file_location("graphsl_gcnsi_model",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.GCNSI_model()


def encode(g,obs):
    nodes=g["nodes"]
    ix={v:i for i,v in enumerate(nodes)}
    n=len(nodes)
    adj=np.zeros((n,n))
    for u,v,_ in g["edges"]:
        adj[ix[u],ix[v]]=1
    edge=torch.tensor(np.array(adj.nonzero()),dtype=torch.long)
    signal=np.zeros((n,len(obs)))
    for j,o in enumerate(obs):
        signal[[ix[v] for b in o["observed"] for v in b],j]=1
    alpha=.01
    lap=laplacian(adj,normed=True)
    smoothed=(1-alpha)*np.linalg.solve(np.eye(n)-alpha*lap,np.column_stack((signal,np.ones(n))))
    base,constant=smoothed[:,:-1],smoothed[:,-1:]
    x=np.stack((signal,base,(base+constant)*.5,base*.5),axis=-1).transpose(1,0,2)
    return torch.tensor(x,dtype=torch.float32),edge,ix


def batch_predict(model,x,edge):
    b,n,_=x.shape
    edges=edge.unsqueeze(0)+n*torch.arange(b)[:,None,None]
    return model(x.flatten(0,1),edges.transpose(0,1).reshape(2,-1)).reshape(b,n,2)


def fit():
    torch.manual_seed(1938)
    model=architecture()
    optim=torch.optim.Adam(model.parameters(),lr=.001)
    raw=load_events(DEPT)
    gs,streams=make_windows(raw,"train",.15,.6)
    datasets=[]
    for week,g in enumerate(gs,1):
        if week<=10:
            continue
        obs,labs,_=draw(g,streams[g["window"]],90600+week,55,0)
        if not obs:
            continue
        x,edge,ix=encode(g,obs)
        labels=torch.zeros(x.shape[:2],dtype=torch.long)
        for k,r in enumerate(labs):
            labels[k,[ix[v] for v in r["sources"]]]=1
        datasets.append((x,edge,labels))
    rng=np.random.default_rng(1938)
    for _ in range(100):
        for k in rng.permutation(len(datasets)):
            x,e,y=datasets[k]
            for batch in np.array_split(rng.permutation(len(y)),4):
                optim.zero_grad()
                logits=batch_predict(model,x[batch],e)
                legal=x[batch,:,0]==0
                ratio=(legal.sum().item()-2*len(batch))/(2*len(batch))
                loss=cross_entropy(logits[legal],y[batch][legal],weight=torch.tensor([1.,ratio]))
                loss.backward()
                optim.step()
    torch.save(model.state_dict(),OUT/"gcnsi_weights.pt")
    write_json(OUT/"gcnsi_protocol.json",dict(source=load(OUT/"gcnsi_source.json"),seed=1938,epochs=100,
        optimizer="Adam lr 0.001",alpha=.01,training_events=sum(len(v[0]) for v in datasets),
        graph="binary directed historical contacts; node features follow GraphSL LPSI augmentations",
        adaptation="legal unseen nodes in weighted CE; four minibatches per window; sum of two source log-odds; fixed pool",
        weights=sha(OUT/"gcnsi_weights.pt")))


def evaluate(name):
    model=architecture()
    assert sha(OUT/"gcnsi_weights.pt")==load(OUT/"gcnsi_protocol.json")["weights"]
    model.load_state_dict(torch.load(OUT/"gcnsi_weights.pt",weights_only=True))
    model.eval()
    raw=load_events(EXTERNAL if name=="Dept1" else NEW)
    if name=="Dept1":
        gs,obs,labs=cohort(name,raw)
    else:
        obs,labs=read_rows(OUT/"external_public.jsonl"),read_rows(OUT/"external_truth.jsonl")
        gs=make_windows(raw,"external",.15,.8)[0]
    pools={p["id"]:p["nodes"] for p in read_rows(OUT/(name+"_predictions.jsonl"))}
    ranks=[]
    labels={r["id"]:r for r in labs}
    for g in gs:
        items=[o for o in obs if o["window"]==g["window"]]
        if not items:
            continue
        x,edge,ix=encode(g,items)
        with torch.no_grad():
            logits=batch_predict(model,x,edge)
            strengths=(logits[:,:,1]-logits[:,:,0]).numpy()
        for o,s in zip(items,strengths):
            nodes=pools[o["id"]]
            v=s[[ix[k] for k in nodes]]
            i,j=np.triu_indices(len(nodes),1)
            order=np.argsort(-(v[i]+v[j]),kind="stable")
            lab=labels[o["id"]]
            target=set(lab["sources"])
            rank=None
            if target<=set(nodes):
                a,b=sorted(nodes.index(k) for k in target)
                index=np.flatnonzero((i==a)&(j==b))[0]
                rank=int(np.flatnonzero(order==index)[0])+1
            ranks.append(dict(id=o["id"],window=g["window"],dual=lab["dual"],ordinary=lab["ordinary"],gcnsi=rank))
    assert len(ranks)==len(labels)==len({r["id"] for r in ranks})
    save_rows(OUT/(name+"_gcnsi_ranks.jsonl"),ranks)
    summaries={k:metrics([r for r in ranks if r[k]],"gcnsi") for k in ("dual","ordinary")}
    write_json(OUT/(name+"_gcnsi_summary.json"),summaries)
    print(name,summaries)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["fit","Dept1","Dept2"])
    phase=parser.parse_args().phase
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if phase=="fit":
            fit()
        else:
            evaluate(phase)
