"""Matched retraining under source-prior shift with conditional and marginal pair likelihoods."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from blind_source_prior import PROFILES, correlated_prior, law_arrays
from build_dept3_pilot import mask, simulate
from distilled_source_pairs import CHECKPOINTS, PairNet, PublicWindow
from five_frame_pairs import frames
from source_prior_shift import NETWORKS, OUT as BASE, SEEDS, SOURCE, activity, excluding_self, windows

OUT=BASE.parent/"trained_source_mixtures"
ARMS=("uniform_offset","mixed_plain","mixed_offset","mixed_component")
TRAIN_SEED,TEST_SEED=73041411,40593217
DRAWS=140


def protocol():
    paths=[Path(__file__).resolve(),BASE/"protocol.json",OUT.parents[1]/"scripts/distilled_source_pairs.py"]
    return dict(primary="mixed_offset versus mixed_plain, same mixed training events and 300 mixed validation outcomes, both-in-12",
                secondary="uniform_offset and mixed_component; true pair NLL; no target-label calibration",
                arms=list(ARMS),seeds=list(SEEDS),source_models=SOURCE,train_seed=TRAIN_SEED,test_seed=TEST_SEED,
                test_profiles=PROFILES,test_draws=DRAWS,fit_eligible=1000,validation_eligible=300,
                validation="same draw-level equal mixture for all arms, score with respective deployment rule",
                output="fixed equal draw mixture for offset models; sort posterior node marginals for all arms",
                hashes={p.relative_to(OUT.parents[1]).as_posix():fg.digest(p) for p in paths})


def draw_events(g,stream,w,edge,rng,profile,n,until):
    a,b,laws,means=law_arrays(w)
    prior=PROFILES[profile]
    if prior is None:
        link=edge[a,b]/edge[a,b].sum()
    accepted=0
    attempts=0
    while (accepted if until else attempts)<n:
        size=min(128,n-(accepted if until else attempts))
        components=rng.choice(3,size,p=prior) if prior is not None else np.full(size,3)
        samples=np.zeros(size,dtype=int)
        for r in np.unique(components):
            at=components==r
            samples[at]=rng.choice(len(a),at.sum(),p=link if r==3 else laws[r])
        for r,pair,seed in zip(components,samples,rng.integers(2**62,size=size)):
            attempts+=1
            src=[g["nodes"][j] for j in (a[pair],b[pair])]
            obs=mask(simulate(stream,src,int(seed)))
            if obs is not None:
                accepted+=1
                yield dict(id=f"{g['window']}-{profile}-{attempts}",window=g["window"],observed=obs[1],k=2),src,int(r),means


def designs(g,raw,stream,profile,rng,count,until):
    w=activity(g,raw)
    edge=correlated_prior(raw,g) if profile=="linked" else None
    hist=PublicWindow(g,raw)
    ix={v:j for j,v in enumerate(g["nodes"])}
    for event,src,comp,means in draw_events(g,stream,w,edge,rng,profile,count,until):
        (nodes,_,_),(x,_,_)=hist.design(frames(event))
        pair=np.array([nodes.index(s) for s in src])
        yield x,w[[ix[v] for v in nodes]],pair,comp,means,event


def training(name,phase,profile):
    raw=fg.raw_data(name)
    _,streams=fg.make_windows(raw,name,.15,1.)
    rows=[]
    for j,(g,events,_) in enumerate(fg.cohort(name,phase)):
        rng=np.random.default_rng([TRAIN_SEED,NETWORKS.index(name),int(phase=="cal"),j,int(profile=="mixed")])
        rows.extend(designs(g,raw,streams[g["window"]],profile,rng,len(events),True))
    assert len(rows)==(1000 if phase=="fit" else 300)
    return rows


class Batches:
    def __init__(self,rows,mean,std):
        m=max(len(r[0]) for r in rows)
        n=len(rows)
        self.x=torch.zeros((n,m,len(mean)))
        self.w=torch.zeros((n,m),dtype=torch.float64)
        self.pair=torch.tensor(np.stack([r[2] for r in rows]),dtype=torch.int64)
        self.comp=torch.tensor([r[3] for r in rows],dtype=torch.int64)
        self.means=torch.from_numpy(np.stack([r[4] for r in rows]))
        for i,(x,w,*_) in enumerate(rows):
            self.x[i,:len(x)]=torch.from_numpy(np.clip((x-mean)/std,-6,6))
            self.w[i,:len(w)]=torch.from_numpy(w)

    def loss(self,model,ids,rule):
        w=self.w[ids]
        logits=model.node(model.encoder(self.x[ids]))[:,:,0].double().masked_fill(w==0,-torch.inf)
        return pair_nll(logits,w,self.pair[ids],self.comp[ids],self.means[ids],rule).mean()


def pair_nll(logits,w,pairs,component,means,rule):
    score=logits-logits.max(1,keepdim=True).values
    h=score.exp()
    u=h*w
    # Strict prefixes count each distinct unordered pair once without subtracting large terms.
    h0=torch.nn.functional.pad(h[:,:-1].cumsum(1),(1,0))
    u0=torch.nn.functional.pad(u[:,:-1].cumsum(1),(1,0))
    z=torch.stack(((h*h0).sum(1),(h*u0+u*h0).sum(1),(u*u0).sum(1)),1)
    i,j=pairs.T
    b=torch.arange(len(i))
    wi,wj=w[b,i],w[b,j]
    weight=torch.stack((torch.ones_like(wi),wi+wj,wi*wj),1)
    if rule=="equal":
        denominator,numerator=(z/means).sum(1),(weight/means).sum(1)
    else:
        r=component if rule=="component" else torch.zeros_like(component)
        numerator,denominator=weight[b,r],z[b,r]
    return denominator.log()-score[b,i]-score[b,j]-numerator.log()


def rule_for(arm,fit=False):
    if fit:
        return {"uniform_offset":"flat","mixed_plain":"flat","mixed_offset":"equal","mixed_component":"component"}[arm]
    return "flat" if arm=="mixed_plain" else "equal"


def fit(name):
    val=training(name,"cal","mixed")
    folder=OUT/name/"fit"
    folder.mkdir(parents=True,exist_ok=True)
    for law in ("flat","mixed"):
        rows=training(name,"fit",law)
        assert {r[-1]["window"] for r in rows}.isdisjoint(r[-1]["window"] for r in val)
        values=np.vstack([r[0] for r in rows])
        mean,std=values.mean(0),np.maximum(.025,values.std(0))
        np.savez(folder/(law+".npz"),mean=mean,std=std)
        tr,va=Batches(rows,mean,std),Batches(val,mean,std)
        for arm in (ARMS[:1] if law=="flat" else ARMS[1:]):
            for seed in SEEDS:
                torch.manual_seed(seed)
                rng=np.random.default_rng(seed)
                model=PairNet(len(mean),False)
                opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005)
                best=np.inf
                curve={}
                path=folder/f"{arm}-{seed}.pt"
                for epoch in range(1,max(CHECKPOINTS)+1):
                    model.train()
                    for ids in np.array_split(rng.permutation(len(rows)),int(np.ceil(len(rows)/24))):
                        opt.zero_grad()
                        loss=tr.loss(model,ids,rule_for(arm,True))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                        opt.step()
                    if epoch in CHECKPOINTS:
                        model.eval()
                        with torch.no_grad():
                            loss=sum(len(ids)*va.loss(model,ids,rule_for(arm)).item()
                                     for ids in np.array_split(np.arange(len(val)),15))/len(val)
                        curve[epoch]=loss
                        if loss<best:
                            best=loss
                            torch.save(model.state_dict(),path)
                fg.write_json(path.with_suffix(".json"),dict(curve=curve,epoch=min(curve,key=curve.get),sha256=fg.digest(path),
                              train_component_counts=dict(Counter(str(r[3]) for r in rows)),
                              validation_component_counts=dict(Counter(str(r[3]) for r in val)),scale_sha256=fg.digest(folder/(law+".npz"))))
            print(name,arm,flush=True)


def fitted_models(source):
    folder=OUT/source/"fit"
    out=[]
    for arm in ARMS:
        scale=folder/(("flat" if arm=="uniform_offset" else "mixed")+".npz")
        with np.load(scale) as f:
            mean,std=f["mean"],f["std"]
        for s in SEEDS:
            path=folder/f"{arm}-{s}.pt"
            meta=fg.load(path.with_suffix(".json"))
            assert meta["sha256"]==fg.digest(path) and meta["scale_sha256"]==fg.digest(scale)
            model=PairNet(len(mean),False).eval()
            model.load_state_dict(torch.load(path,weights_only=True))
            out.append((arm,model,mean,std))
    return out


def evaluate(net):
    models=fitted_models(SOURCE[net])
    raw,gs=windows(net,False)
    rows=[]
    for j,(g,stream) in enumerate(gs):
        for pi,profile in enumerate(PROFILES):
            rng=np.random.default_rng([TEST_SEED,NETWORKS.index(net),j,pi])
            for x,w,pair,_,means,event in designs(g,raw,stream,profile,rng,DRAWS,False):
                tie=fg.tie_order(event["id"],len(w))
                hits=[]
                losses=[]
                for arm,model,mean,std in models:
                    with torch.no_grad():
                        logits=model.node(model.encoder(torch.from_numpy(np.clip((x-mean)/std,-6,6))))[:,0].numpy().astype(float)
                    h=np.exp(logits-logits.max())
                    u=h*w
                    hx,ux=excluding_self(h),excluding_self(u)
                    i,k=pair
                    if rule_for(arm)=="equal":
                        marginal=h*hx/means[0]+(h*ux+u*hx)/means[1]+u*ux/means[2]
                        value=h[i]*h[k]*np.sum(np.array([1.,w[i]+w[k],w[i]*w[k]])/means)
                    else:
                        marginal=h*hx
                        value=h[i]*h[k]
                    ranking=fg.sorted_nodes(marginal,tie)[:12]
                    hits.append(int(i in ranking and k in ranking))
                    losses.append(round(float(np.log(marginal.sum()/2/value)),7))
                rows.append(dict(id=event["id"],window=g["window"],profile=profile,hits=hits,nll=losses))
    folder=OUT/net/"test"
    folder.mkdir(parents=True,exist_ok=True)
    fg.save_rows(folder/"events.jsonl",rows)
    fg.write_json(folder/"manifest.json",dict(n=len(rows),sha256=fg.digest(folder/"events.jsonl"),windows=[g["window"] for g,_ in gs]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","train","test"))
    parser.add_argument("network",nargs="?",choices=NETWORKS)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    lock=OUT/"protocol.json"
    if args.action=="freeze" and not lock.exists():
        fg.write_json(lock,protocol())
    assert fg.load(lock)==protocol()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.action=="train":
            assert args.network in ("Dept2","company")
            fit(args.network)
        elif args.action=="test":
            evaluate(args.network)
