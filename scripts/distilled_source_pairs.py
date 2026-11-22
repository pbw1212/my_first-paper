"""Learn a symmetric distribution over distinct source pairs from public suffixes."""
import argparse
import hashlib
from collections import Counter
from itertools import groupby
from zipfile import ZipFile

import joblib
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import rankdata
from torch import nn
from threadpoolctl import threadpool_limits

from ordinary_source_roles import OUT as ORDINARY, PREVIOUS, FILES, NEW, Window, company, draw, load, load_events, make_windows, read_rows, save_rows, write_json
from partial_source_lists import pair_distribution, sorted_nodes
from joint_source_shortlist import best_subset
from silent_source_oracle import CompleteTrace

OUT=ORDINARY.parent/"distilled_source_pairs"
SMS="CopenhagenSMS"
SMS_SHA="9d51b590273320c42fd4d5d106e74e28e9f8d1254ba67c71131688bbb6435aa7"
VARIANTS=("node_hard","node_soft","pair_hard","pair_soft")
CHECKPOINTS=(3,7,12)
PLAN=dict(primary="pair_soft; both true sources in 12 on ordinary mask-eligible cascades",
    training="same 1000 earlier ordinary events as previous pooled tree; 2x2 matched node/pair and hard/exact conditional labels",
    selection="separate minimum hard-label pair negative log likelihood on 300 intermediate ordinary events at epochs 3,7,12; tie earlier",
    training_seed=314159,optimizer="AdamW lr .002, weight decay .005, batch 24; no candidate negative sampling",
    test_seeds=[87034211,93015411],local_per_window=150,external_per_window=60,
    split="four existing late windows; two new ordinary draws per window. Dept1 and CollegeMsg use frozen Dept2 weights",
    decoder="node top 12; pair local search on union of top 48 posterior marginals and endpoints of top 48 pairs; legacy baseline full original decoder",
    contrasts="pair_soft vs node_soft, pair_hard, ordinary pooled tree and old dual-trained joint; matched marginals decoder ablation",
    interpretation="consistent development improvement requires higher hits than both legacy baselines on both batches of both local networks")


def raw_data(name):
    if name==SMS:
        path=PREVIOUS.parents[1]/"data/raw/copenhagen-sms.csv.zip"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==SMS_SHA
        with ZipFile(path) as archive:
            edges=[tuple(map(int,s.split(b","))) for s in archive.read("edges.csv").splitlines()[1:]]
        return sorted({(t,u,v) for u,v,t in edges if u!=v})
    if name=="company":
        return company()
    return load_events(NEW if name=="Dept2" else PREVIOUS.parents[1]/"data/raw"/FILES[name])


def evaluation_windows(name,raw):
    if name!=SMS:
        gs,streams=make_windows(raw,name,.15,1.)
        keys=load(ORDINARY/name/"test_flow.json")
        return [g for g in gs if g["window"] in keys],streams
    start=raw[0][0]+14*86400
    end=28*86400
    counts=Counter((u,v) for t,u,v in raw if t<start)
    roster={v for pair in counts for v in pair}
    contacts=[(t,u,v) for t,u,v in raw if start<=t<end and u in roster and v in roster]
    key=SMS+"-01"
    graph=dict(window=key,start=start,end=end,nodes=sorted(roster),edges=[[u,v,n] for (u,v),n in sorted(counts.items())])
    stream=[(t,[(u,v) for _,u,v in group]) for t,group in groupby(contacts,key=lambda e:e[0])]
    return [graph],{key:stream}


class PublicWindow:
    def __init__(self,g,raw):
        self.w=Window(g,raw)
        c=self.w.contact
        self.edge=np.minimum(2.,np.log1p(c+c.T)/3.).astype(np.float32)

    def design(self,o):
        base,x,ev=self.w.features(o)
        nodes,_,support=base
        ix=self.w.base.f.history.index
        vis=[ix[v] for batch in o["observed"] for v in batch]
        groups=np.repeat(np.arange(len(o["observed"])),[len(b) for b in o["observed"]])
        reach=self.w.base.reach
        within=reach[np.ix_(vis,vis)]
        rate=np.maximum(.01,reach[:,vis].mean(axis=0))
        evidence=within/(within+rate)
        earlier=groups[:,None]<groups[None,:]
        residual=1-np.where(earlier,evidence,0).max(axis=0)
        extra=np.column_stack((support@residual/residual.sum(),
            (support*residual).max(axis=1),np.mean(support>.5,axis=1)))
        absolute=np.column_stack((x,extra))
        relative=(rankdata(absolute,axis=0,method="average")-.5)/len(nodes)
        features=np.column_stack((absolute,relative,np.broadcast_to(ev,(len(nodes),len(ev))))).astype(np.float32)
        rows=[ix[v] for v in nodes]
        weighted=support.astype(np.float32)*np.sqrt(residual/residual.sum()).astype(np.float32)
        overlap=weighted@weighted.T
        return base,(features,self.edge[np.ix_(rows,rows)],overlap)


class PairNet(nn.Module):
    def __init__(self,dim,interaction):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(dim,40),nn.SiLU(),nn.Linear(40,24),nn.SiLU())
        self.node=nn.Linear(24,1)
        self.interaction=interaction
        if interaction:
            self.project=nn.Linear(24,8)
            self.context=nn.Linear(24,2)
            self.register_buffer("sign",torch.tensor([1.,-1.]*4))

    def forward(self,x,edge,overlap,legal):
        h=self.encoder(x)
        u=self.node(h).squeeze(-1)
        score=u.unsqueeze(1)+u.unsqueeze(2)
        if self.interaction:
            z=self.project(h).tanh()
            weights=self.context((h*legal.unsqueeze(-1)).sum(1)/legal.sum(1,keepdim=True))
            score=score+((z*self.sign)@z.transpose(1,2))+weights[:,0,None,None]*edge+weights[:,1,None,None]*overlap
        return score


def samples(name,phase):
    raw=raw_data(name)
    gs,streams=make_windows(raw,name,.15,1.)
    folder=ORDINARY/name
    obs=read_rows(folder/(phase+"_public.jsonl"))
    labels={y["id"]:y for y in read_rows(folder/(phase+"_truth.jsonl"))}
    result=[]
    for g in gs:
        items=[o for o in obs if o["window"]==g["window"]]
        if not items:
            continue
        w=PublicWindow(g,raw)
        oracle=CompleteTrace(g,streams[g["window"]]) if phase=="fit" else None
        for o in items:
            y=labels[o["id"]]
            base,design=w.design(o)
            ids={v:i for i,v in enumerate(base[0])}
            a,b=y["sources"]
            if oracle is not None:
                a,partners,prob=oracle.evaluate(y,return_distribution=True)
                b=next(v for v in y["sources"] if v!=a)
            else:
                partners,prob=[b],[1.]
            target=np.zeros(len(ids),dtype=np.float32)
            target[[ids[v] for v in partners]]=prob
            assert np.isclose(target.sum(),1.) and target[ids[a]]==0
            result.append((*design,ids[a],ids[b],target))
    return result


class Packed:
    def __init__(self,rows,mean,std):
        n=max(len(r[0]) for r in rows)
        batch=len(rows)
        self.x=torch.zeros(batch,n,len(mean))
        self.edge=torch.zeros(batch,n,n,dtype=torch.float16)
        self.overlap=torch.zeros_like(self.edge)
        self.mask=torch.zeros(batch,n,dtype=torch.bool)
        self.soft=torch.zeros(batch,n)
        self.anchors=torch.empty(batch,dtype=torch.long)
        self.partners=torch.empty_like(self.anchors)
        for i,(x,edge,overlap,a,b,p) in enumerate(rows):
            m=len(x)
            self.x[i,:m]=torch.from_numpy(np.clip((x-mean)/std,-6,6))
            self.edge[i,:m,:m]=torch.from_numpy(edge)
            self.overlap[i,:m,:m]=torch.from_numpy(overlap)
            self.mask[i,:m]=True
            self.anchors[i],self.partners[i]=a,b
            self.soft[i,:m]=torch.from_numpy(p)
        self.diag=torch.eye(n,dtype=torch.bool)

    def loss(self,model,idx,soft):
        mask=self.mask[idx]
        edge=self.edge[idx].float() if model.interaction else None
        overlap=self.overlap[idx].float() if model.interaction else None
        logits=model(self.x[idx],edge,overlap,mask)
        invalid=self.diag | ~mask[:,None,:] | ~mask[:,:,None]
        normalizer=torch.logsumexp(logits.masked_fill(invalid,-torch.inf).flatten(1),1)-np.log(2.)
        anchor_rows=logits[torch.arange(len(idx)),self.anchors[idx]]
        expected=(self.soft[idx]*anchor_rows).sum(1) if soft else anchor_rows.gather(1,self.partners[idx,None]).flatten()
        return (normalizer-expected).mean()


def fit(name):
    folder=OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/"protocol.json",PLAN)
    rows=samples(name,"fit")
    all_x=np.vstack([r[0] for r in rows])
    mean=all_x.mean(0);std=np.maximum(all_x.std(0),.025)
    del all_x
    training=Packed(rows,mean,std)
    del rows
    cal=Packed(samples(name,"cal"),mean,std)
    assert len(training.x)==1000 and len(cal.x)==300
    np.savez(folder/"scale.npz",mean=mean,std=std)
    summary={}
    for variant in VARIANTS:
        torch.manual_seed(PLAN["training_seed"])
        rng=np.random.default_rng(PLAN["training_seed"])
        model=PairNet(len(mean),variant.startswith("pair"))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005)
        curve={};best=float("inf")
        for epoch in range(1,max(CHECKPOINTS)+1):
            model.train()
            for chunk in np.array_split(rng.permutation(len(training.x)),int(np.ceil(len(training.x)/24))):
                optimizer.zero_grad()
                loss=training.loss(model,chunk,variant.endswith("soft"))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5)
                optimizer.step()
            if epoch in CHECKPOINTS:
                model.eval()
                with torch.no_grad():
                    value=sum(len(idx)*cal.loss(model,idx,False).item() for idx in np.array_split(np.arange(len(cal.x)),15))/len(cal.x)
                curve[epoch]=value
                if value<best:
                    best=value
                    torch.save(model.state_dict(),folder/(variant+".pt"))
        summary[variant]=dict(cal_nll=curve,epoch=min(curve,key=curve.get))
        print(name,variant,summary[variant],flush=True)
    write_json(folder/"fit.json",summary)
    files=["scale.npz","protocol.json","fit.json"]+[s+".pt" for s in VARIANTS]
    write_json(folder/"weights_lock.json",{f:hashlib.sha256((folder/f).read_bytes()).hexdigest() for f in files})


def shortlist(scores,tie):
    np.fill_diagonal(scores,-np.inf)
    logits=scores.astype(np.float64)
    logz=logsumexp(logits)-np.log(2.)
    joint=np.exp(logits-logz)
    marginal=sorted_nodes(joint.sum(axis=1),tie)
    # Unordered pair tie-breaking is based on the public event's tie order, including at the cutoff.
    # At large n only the top endpoints are needed; ties at the boundary are resolved explicitly.
    a,b=np.triu_indices(len(tie),1)
    v=scores[a,b]
    threshold=np.partition(v,len(v)-48)[-48]
    candidates=np.flatnonzero(v>=threshold)
    i,j=a[candidates],b[candidates]
    best=candidates[np.lexsort((np.maximum(tie[i],tie[j]),np.minimum(tie[i],tie[j]),-v[candidates]))[:48]]
    pool=np.unique(np.r_[marginal[:48],a[best],b[best]])
    local=joint[np.ix_(pool,pool)]
    order=sorted_nodes(joint.sum(axis=1)[pool],tie[pool])
    picked=pool[best_subset(local,order,tie[pool],12)]
    return picked,marginal[:12],pool,logz


def pair_rank(matrix,a,b,tie):
    i,j=np.triu_indices(len(tie),1)
    value=matrix[i,j]
    left,right=min(a,b),max(a,b)
    target=matrix[left,right]
    first,second=sorted((tie[a],tie[b]))
    earlier=(np.minimum(tie[i],tie[j])<first) | ((np.minimum(tie[i],tie[j])==first) & (np.maximum(tie[i],tie[j])<second))
    return 1+int(np.sum((value>target) | ((value==target) & earlier)))


def run(name,new_windows=False):
    local=name in ("company","Dept2")
    source=name if local else "Dept2"
    folder=OUT/(name+"_new_windows" if new_windows else name)
    folder.mkdir(parents=True,exist_ok=True)
    train=OUT/source
    lock=load(train/"weights_lock.json")
    assert all(hashlib.sha256((train/f).read_bytes()).hexdigest()==h for f,h in lock.items())
    budget_lock=load(train/"budget_lock.json")
    assert all(hashlib.sha256((train/f).read_bytes()).hexdigest()==h for f,h in budget_lock.items())
    scale=np.load(train/"scale.npz");mean,std=scale["mean"],scale["std"]
    nets={}
    for tag in (*VARIANTS,"budget_hard","budget_soft"):
        model=PairNet(len(mean),tag.startswith("pair"))
        model.load_state_dict(torch.load(train/(tag+".pt"),weights_only=True))
        nets[tag]=model.eval()
    old=joblib.load(PREVIOUS/source/"models.joblib")
    pooled=joblib.load(ORDINARY/source/"models.joblib")["pooled"]
    raw=raw_data(name)
    gs,streams=evaluation_windows(name,raw)
    if new_windows:
        assert local
        split=load(PREVIOUS/name/"windows.json")
        end=max(v["end"] for v in split.values() if v["stage"]=="cal")
        gs,streams=make_windows(raw,name,.15,1.)
        gs=[g for g in gs if g["start"]>=end and g["window"] not in split
            and sum(len(edges) for _,edges in streams[g["window"]])>=100]
    keys=[g["window"] for g in gs]
    amount=150 if name==SMS else PLAN["local_per_window" if local else "external_per_window"]
    seeds=[74082699] if new_windows else PLAN["test_seeds"]
    write_json(folder/"test_plan.json",dict(PLAN,network=name,source=source,lock=lock,budget_lock=budget_lock,windows=keys,
        actual_seeds=seeds,extra_window_check=new_windows,actual_target_per_window=amount,raw_sms_sha256=SMS_SHA if name==SMS else None))
    prior=ORDINARY/name
    used=set()
    if new_windows:
        used.update(y["seed"] for y in read_rows(OUT/name/"truth.jsonl"))
    if name!=SMS:
        used.update(y["seed"] for filename in ("test_truth.jsonl","hybrid_test_truth.jsonl") for y in read_rows(prior/filename))
    for phase in ("fit","cal"):
        used.update(y["seed"] for y in read_rows(ORDINARY/source/(phase+"_truth.jsonl")))
    legacy=PREVIOUS/(("transfer_"+name) if not local else name)
    if name!=SMS:
        for filename in (("truth.jsonl","repeat_truth.jsonl","joint12_truth.jsonl") if local else ("truth.jsonl",)):
            used.update(y["seed"] for y in read_rows(legacy/filename))
    preds=[];obs=[];truth=[];diagnostics=[];flow={}
    for i,g in enumerate(g for g in gs if g["window"] in keys):
        key=g["window"]
        w=PublicWindow(g,raw)
        for batch,root in enumerate(seeds,1):
            seed=root+i+1000*["Dept2","company","Dept1","CollegeMsg",SMS].index(name)
            items,labels,counts=draw(g,streams[key],seed,0,amount)
            assert counts["ordinary"]==len(items)==amount
            flow[str(batch)+"_"+key]=dict(counts,seed=seed)
            for o,y in zip(items,labels,strict=True):
                assert y["seed"] not in used
                used.add(y["seed"])
                o["id"]=y["id"]=f"distill-{batch}-"+o["id"]
                base,design=w.design(o)
                nodes=base[0];x,edge,overlap=design
                initial,tie,old_q=pair_distribution(o,base,old)
                pooled_score=pooled.decision_function(x[:,:pooled.n_features_in_])
                sets={"old_joint":best_subset(old_q,initial,tie,12),"pooled":sorted_nodes(pooled_score,tie)[:12]}
                a,b=[nodes.index(v) for v in y["sources"]]
                ranks={}
                if new_windows:
                    ranks["old_joint"]=pair_rank(old_q,a,b,tie)
                    ranks["pooled"]=pair_rank(np.add.outer(pooled_score,pooled_score),a,b,tie)
                del old_q
                tx=torch.from_numpy(np.clip((x-mean)/std,-6,6))[None]
                # Match the precision of relationship matrices used during training.
                e=torch.from_numpy(edge.astype(np.float16).astype(np.float32))[None]
                ov=torch.from_numpy(overlap.astype(np.float16).astype(np.float32))[None]
                legal=torch.ones(1,len(nodes),dtype=torch.bool)
                diagnostic={}
                for tag,model in nets.items():
                    if model.interaction:
                        with torch.no_grad():
                            score=model(tx,e,ov,legal)[0].numpy()
                        picked,marginal,pool,logz=shortlist(score,tie)
                        sets[tag]=picked
                        sets[tag+"_marginal"]=marginal
                        truth_score=float(score[a,b])
                        if new_windows:
                            ranks[tag]=pair_rank(score,a,b,tie)
                    else:
                        # For additive pair energies, ordering individuals is globally optimal.
                        with torch.no_grad():
                            unary=model.node(model.encoder(tx))[0,:,0].numpy().astype(np.float64)
                        order=sorted_nodes(unary,tie)
                        sets[tag]=order[:12]
                        log_single=logsumexp(unary)
                        prob=np.exp(unary-log_single)
                        logz=2*log_single+np.log1p(-(prob@prob))-np.log(2.)
                        truth_score=unary[a]+unary[b]
                        pool=None
                        if new_windows:
                            ranks[tag]=pair_rank(np.add.outer(unary,unary),a,b,tie)
                    diagnostic[tag]=dict(nll=float(logz-truth_score),in_pool=None if pool is None else bool(a in pool and b in pool))
                choices={tag:[nodes[j] for j in idx] for tag,idx in sets.items()}
                preds.append(dict(id=o["id"],batch=batch,window=key,sets=choices))
                diagnostics.append(dict(id=o["id"],models=diagnostic,ranks=ranks))
            obs.extend(items);truth.extend(labels)
            print(name,key,batch,len(items),flush=True)
    save_rows(folder/"public.jsonl",obs);save_rows(folder/"truth.jsonl",truth)
    save_rows(folder/"predictions.jsonl",preds);save_rows(folder/"likelihoods.jsonl",diagnostics)
    write_json(folder/"flow.json",flow)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("network",choices=["Dept2","company",*FILES,SMS])
    parser.add_argument("phase",choices=["fit","test","new_windows"])
    args=parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.phase=="fit":
            assert args.network in ("company","Dept2")
            fit(args.network)
        else:
            run(args.network,args.phase=="new_windows")
