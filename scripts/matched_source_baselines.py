"""Matched external source-localization baselines on the frozen ordinary-event cohorts."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from compare_gcnsi_pairs import architecture, batch_predict, encode
from distilled_source_pairs import (ORDINARY, OUT as MAIN, SMS, PairNet, PublicWindow, evaluation_windows,
    load, make_windows, pair_rank, raw_data, read_rows, save_rows, shortlist, sorted_nodes, write_json)
from score_dept3_candidates import roles

OUT=MAIN.parent/"matched_source_baselines"
RUNS=("company","Dept2","Dept1","CollegeMsg",SMS,"company_new_windows","Dept2_new_windows")
SEED=1938
DIRECTIONS=("forward","reverse","undirected")
EPOCHS=(10,25,50)
TEMPS=(.15,.5,1.,3.)
PLAN=dict(training="identical saved ordinary events (1000 fit, 300 chronological validation per development network)",
    gcnsi="Pinned GraphSL GCNSI layers, four LPSI channels, alpha .01, Adam .001, class-balanced CE over legal nodes, 20 events/batch",
    directions=list(DIRECTIONS),epochs=list(EPOCHS),seed=SEED,selection="minimum mean validation hard-label unordered-pair NLL; ties first listed",
    jordan="all legal pairs on historical graph, maximum nearest distance then total nearest distance; no candidate pruning; score divided by calibrated temperature",
    jordan_graphs=["undirected","forward"],temperatures=list(TEMPS),decoding="12 legal nodes; additive GCNSI by top nodes; Jordan uses the same probability-mass list search as the new model",
    evaluation="paired on the 3660 main and 1350 supplemental events, both-in-12 and exact unordered-pair rank; no target-domain tuning")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tie_order(event_id,n):
    return np.random.default_rng(1000003+sum((i+1)*ord(c) for i,c in enumerate(event_id))).permutation(n)


def oriented(g,direction):
    if direction=="forward":
        return g
    edges={(v,u) for u,v,_ in g["edges"]}
    if direction=="undirected":
        edges.update((u,v) for u,v,_ in g["edges"])
    return dict(g,edges=[(u,v,1) for u,v in sorted(edges)])


def adjacency(edge,n):
    a=torch.sparse_coo_tensor(edge.flip(0),torch.ones(edge.shape[1]),size=(n,n)).to_dense()
    a.diagonal().add_(1)
    return a if n<=400 else a.to_sparse().coalesce()


def propagate(adj,x):
    if not adj.is_sparse:
        return torch.matmul(adj,x)
    batch,n,h=x.shape
    return torch.sparse.mm(adj,x.transpose(0,1).reshape(n,-1)).reshape(n,batch,h).transpose(0,1)


def forward(model,x,adj):
    # The pinned PyG module uses the default message(x_j), so aggregation is an unweighted sum.
    h=F.relu(propagate(adj,model.conv1.lin(x)))
    h=F.dropout(h,training=model.training)
    return model.fc(propagate(adj,model.conv2.lin(h)))


def upstream_check():
    torch.manual_seed(SEED)
    model=architecture().eval()
    x=torch.rand(3,7,4)
    edges=torch.tensor([[0,1,4,6,2,0],[1,3,1,4,5,2]])
    ref=batch_predict(model,x,edges)
    probe=torch.randn_like(ref)
    reference=torch.autograd.grad((ref*probe).sum(),list(model.parameters()))
    errors=[]
    for a in (adjacency(edges,7),adjacency(edges,7).to_sparse()):
        ours=forward(model,x,a)
        grads=torch.autograd.grad((ours*probe).sum(),list(model.parameters()))
        errors.append(max(float((ours-ref).abs().max().detach()),*[float((s-t).abs().max()) for s,t in zip(grads,reference)]))
    assert max(errors)<1e-4,errors
    return errors


def cohort(name,phase):
    obs=read_rows(ORDINARY/name/(phase+"_public.jsonl"))
    truth={r["id"]:r for r in read_rows(ORDINARY/name/(phase+"_truth.jsonl"))}
    gs,_=make_windows(raw_data(name),name,.15,1.)
    for g in gs:
        items=[o for o in obs if o["window"]==g["window"]]
        if items:
            yield g,items,[truth[o["id"]] for o in items]


def tensor_set(rows,direction):
    sets=[]
    for g,items,truth in rows:
        x,e,ix=encode(oriented(g,direction),items)
        y=torch.zeros(x.shape[:2],dtype=torch.long)
        for j,r in enumerate(truth):
            y[j,[ix[v] for v in r["sources"]]]=1
        assert (y.sum(1)==2).all() and not ((x[:,:,0]>0)&y.bool()).any()
        sets.append((x,adjacency(e,len(ix)),y))
    return sets


def val_score(model,data):
    n=hits=0
    total=0.
    with torch.no_grad():
        for x,a,y in data:
            for start in range(0,len(x),20):
                inputs=x[start:start+20];labels=y[start:start+20]
                logits=forward(model,inputs,a).double()
                v=logits[:,:,1]-logits[:,:,0]
                legal=inputs[:,:,0]==0
                v=v.masked_fill(~legal,-torch.inf)
                pairs=v[:,:,None]+v[:,None,:]
                pairs=pairs.masked_fill(torch.eye(v.shape[1],dtype=torch.bool),-torch.inf)
                gold=v.masked_fill(~labels.bool(),0).sum(1)
                total+=float((torch.logsumexp(pairs.flatten(1),1)-np.log(2.)-gold).sum())
                hits+=int((labels.gather(1,v.argsort(descending=True,stable=True)[:,:12]).sum(1)==2).sum())
                n+=len(inputs)
    return dict(n=n,nll=total/n,hits=hits)


def gcn_fit(name, *, folder=None, epochs=EPOCHS):
    folder=OUT/name if folder is None else folder
    folder.mkdir(exist_ok=True,parents=True)
    write_json(folder/"protocol.json",dict(PLAN,epochs=list(epochs)))
    errors=upstream_check()
    train=list(cohort(name,"fit"));cal=list(cohort(name,"cal"))
    assert sum(len(o) for _,o,_ in train)==1000 and sum(len(o) for _,o,_ in cal)==300
    curves={};best=float("inf")
    for direction in DIRECTIONS:
        sets=tensor_set(train,direction);validation=tensor_set(cal,direction)
        torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
        model=architecture()
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        for epoch in range(1,max(epochs)+1):
            model.train()
            for j in rng.permutation(len(sets)):
                x,a,y=sets[j]
                for idx in np.array_split(rng.permutation(len(x)),10):
                    opt.zero_grad()
                    logits=forward(model,x[idx],a)
                    legal=x[idx,:,0]==0
                    ratio=(int(legal.sum())-2*len(idx))/(2*len(idx))
                    loss=F.cross_entropy(logits[legal],y[idx][legal],weight=torch.tensor([1.,ratio]))
                    loss.backward();opt.step()
            if epoch in epochs:
                rec=val_score(model.eval(),validation)
                curves[f"{direction}_{epoch}"]=rec
                if rec["nll"]<best:
                    best=rec["nll"]
                    selected=dict(direction=direction,epoch=epoch)
                    torch.save(model.state_dict(),folder/"gcnsi.pt")
        print(name,direction,"finished",flush=True)
    write_json(folder/"gcnsi_selection.json",dict(selected=selected,validation=curves,upstream_max_errors=errors,
        upstream=load(OUT.parent/"temporal_pair_evidence/gcnsi_source.json"),weights_sha256=digest(folder/"gcnsi.pt")))


def distances(g,direction):
    ix={v:j for j,v in enumerate(g["nodes"])}
    a,b=zip(*[(ix[u],ix[v]) for u,v,_ in g["edges"]])
    adj=csr_matrix((np.ones(len(a)),(a,b)),shape=(len(ix),len(ix)))
    d=shortest_path(adj,directed=direction=="forward",unweighted=True)
    penalty=int(np.max(d[np.isfinite(d)]))+1
    return ix,np.where(np.isfinite(d),d,penalty).astype(np.int16),penalty


@njit
def cover_scores(dist,penalty):
    n,m=dist.shape
    score=np.zeros((n,n),np.float64)
    for i in range(n):
        for j in range(i+1,n):
            radius=total=0
            for k in range(m):
                near=min(dist[i,k],dist[j,k])
                radius=max(radius,near)
                total+=near
            score[i,j]=score[j,i]=-radius-.95*total/(m*penalty)
    return score


def jordan_fit(name):
    result={};folder=OUT/name
    folder.mkdir(exist_ok=True,parents=True)
    for direction in PLAN["jordan_graphs"]:
        losses=np.zeros(len(TEMPS))
        n=0
        for g,items,labs in cohort(name,"cal"):
            ix,d,penalty=distances(g,direction)
            for o,y in zip(items,labs,strict=True):
                nodes,vis=event_nodes(g,o)
                rows=[ix[v] for v in nodes]
                cols=[ix[v] for v in vis]
                s=cover_scores(d[np.ix_(rows,cols)],penalty)
                a,b=[nodes.index(v) for v in y["sources"]]
                upper=s[np.triu_indices(len(rows),1)]
                for j,t in enumerate(TEMPS):
                    losses[j]+=logsumexp(upper/t)-s[a,b]/t
                n+=1
        for t,loss in zip(TEMPS,losses):
            result[f"{direction}_{t}"]=float(loss/n)
    best=min(result,key=result.get)
    dr,temp=best.split("_")
    write_json(folder/"jordan_selection.json",dict(direction=dr,temperature=float(temp),validation_nll=result,n=n))


def event_nodes(g,o):
    vis=[v for batch in o["observed"] for v in batch]
    return sorted(set(g["nodes"])-set(vis)),vis


def evaluate(run):
    main=MAIN/run;plan=load(main/"test_plan.json")
    name=plan["network"];source=plan["source"];folder=OUT/run
    folder.mkdir(parents=True,exist_ok=True)
    trained=OUT/source
    cfg=load(trained/"gcnsi_selection.json");jcfg=load(trained/"jordan_selection.json")
    assert digest(trained/"gcnsi.pt")==cfg["weights_sha256"]
    gcn=architecture().eval()
    gcn.load_state_dict(torch.load(trained/"gcnsi.pt",weights_only=True))
    params=np.load(MAIN/source/"scale.npz");mean,std=params["mean"],params["std"]
    pair=PairNet(len(mean),True).eval()
    assert digest(MAIN/source/"pair_soft.pt")==plan["lock"]["pair_soft.pt"]
    pair.load_state_dict(torch.load(MAIN/source/"pair_soft.pt",weights_only=True))
    raw=raw_data(name)
    gs,_=evaluation_windows(name,raw) if not run.endswith("_new_windows") else make_windows(raw,name,.15,1.)
    obs=read_rows(main/"public.jsonl");truth={r["id"]:r for r in read_rows(main/"truth.jsonl")}
    prior={r["id"]:r for r in read_rows(main/"predictions.jsonl")}
    oldrank={r["id"]:r.get("ranks",{}) for r in read_rows(main/"likelihoods.jsonl")}
    pred=[]
    for g in (g for g in gs if g["window"] in plan["windows"]):
        items=[o for o in obs if o["window"]==g["window"]]
        x,edges,ix=encode(oriented(g,cfg["selected"]["direction"]),items)
        adj=adjacency(edges,len(ix))
        with torch.no_grad():
            output=torch.cat([forward(gcn,x[start:start+8],adj) for start in range(0,len(x),8)])
            strengths=(output[:,:,1]-output[:,:,0]).numpy().astype(np.float64)
        jix,d,penalty=distances(g,jcfg["direction"])
        w=PublicWindow(g,raw)
        for o,all_gc in zip(items,strengths,strict=True):
            nodes,vis=event_nodes(g,o);sid=o["id"];tie=tie_order(sid,len(nodes))
            gc=all_gc[[ix[v] for v in nodes]]
            ranks={};sets={"gcnsi":[nodes[i] for i in sorted_nodes(gc,tie)[:12]]}
            jc=cover_scores(d[np.ix_([jix[v] for v in nodes],[jix[v] for v in vis])],penalty)
            jc/=jcfg["temperature"]
            jset,_,_,_=shortlist(jc,tie)
            sets["jordan"]=[nodes[j] for j in jset]
            base,(f,edge,overlap)=w.design(o)
            assert nodes==base[0]
            with torch.no_grad():
                matrix=pair(torch.from_numpy(np.clip((f-mean)/std,-6,6))[None],
                    torch.tensor(edge.astype(np.float16),dtype=torch.float32)[None],
                    torch.tensor(overlap.astype(np.float16),dtype=torch.float32)[None],
                    torch.ones(1,len(nodes),dtype=torch.bool))[0].numpy()
            chosen,_,_,_=shortlist(matrix,tie)
            sets["pair_soft"]=[nodes[j] for j in chosen]
            assert set(sets["pair_soft"])==set(prior[sid]["sets"]["pair_soft"])
            # Labels enter only once each of the three public decisions has been made.
            a,b=[nodes.index(v) for v in truth[sid]["sources"]]
            for tag,s in (("gcnsi",np.add.outer(gc,gc)),("jordan",jc),("pair_soft",matrix)):
                ranks[tag]=pair_rank(s,a,b,tie)
            if "pair_soft" in oldrank[sid]:
                assert ranks["pair_soft"]==oldrank[sid]["pair_soft"]
            pred.append(dict(id=sid,window=g["window"],batch=prior[sid]["batch"],n=len(nodes),sets=sets,ranks=ranks))
        print(run,g["window"],len(items),flush=True)
    save_rows(folder/"predictions.jsonl",pred)
    write_json(folder/"inputs.json",dict(source=source,main={f:digest(main/f) for f in ("public.jsonl","truth.jsonl","predictions.jsonl","test_plan.json")},
        config={f:digest(trained/f) for f in ("gcnsi_selection.json","jordan_selection.json","gcnsi.pt")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run",choices=RUNS)
    parser.add_argument("step",choices=("fit","evaluate"))
    args=parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.step=="fit":
            assert args.run in ("company","Dept2")
            gcn_fit(args.run);jordan_fit(args.run)
        else:
            evaluate(args.run)
