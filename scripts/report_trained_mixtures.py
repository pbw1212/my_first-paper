"""Audit source-law training gradients against dense pair enumeration and evaluate paired fits."""
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from train_source_mixtures import (ARMS,DRAWS,NETWORKS,OUT,PROFILES,SEEDS,SOURCE,TEST_SEED,
                                   PublicWindow,activity,correlated_prior,draw_events,fitted_models,
                                   frames,pair_nll,protocol,rule_for,windows)


def loss_check():
    rng=np.random.default_rng(318517)
    worst=0.
    for rule in ("flat","equal","component"):
        for _ in range(10):
            n=14
            z=torch.tensor(rng.normal(size=(3,n)),requires_grad=True)
            pri=torch.from_numpy(1+rng.uniform(0,100,size=(3,n)))
            pri[0,-3:]=0
            pri[2,-1:]=0
            q=z.masked_fill(pri==0,-torch.inf)
            pairs=torch.tensor([[0,1],[1,6],[3,4]])
            comp=torch.arange(3)
            means=torch.tensor([[1,80.,1400.],[1,65.,1000.],[1,54.,800.]])
            approx=pair_nll(q,pri,pairs,comp,means,rule)
            vals=[]
            for t in range(3):
                legal=torch.where(pri[t]>0)[0]
                a,b=torch.combinations(legal).T
                wi,wj=pri[t,a],pri[t,b]
                masses=torch.stack((torch.ones_like(wi),wi+wj,wi*wj),1)
                p=(masses/means[t]).sum(1) if rule=="equal" else masses[:,t if rule=="component" else 0]
                logs=z[t,a]+z[t,b]+p.log()
                i,j=pairs[t]
                vals.append(torch.logsumexp(logs,0)-logs[(a==i)&(b==j)][0])
            exact=torch.stack(vals)
            grad,=torch.autograd.grad(approx.sum(),z)
            ref,=torch.autograd.grad(exact.sum(),z)
            worst=max(worst,float((grad-ref).abs().max()),float((approx-exact).detach().abs().max()))
    assert worst<1e-10
    return worst


def audit(net):
    root=OUT/net/"test"
    m=fg.load(root/"manifest.json")
    assert m["sha256"]==fg.digest(root/"events.jsonl")
    rows=fg.read_rows(root/"events.jsonl")
    lookup={r["id"]:r for r in rows}
    assert len(lookup)==len(rows)==m["n"]
    raw,gs=windows(net,False)
    models=fitted_models(SOURCE[net])
    n=0
    checked=0
    highest=0.
    for j,(g,stream) in enumerate(gs):
        w=activity(g,raw)
        edge=correlated_prior(raw,g)
        hist=PublicWindow(g,raw)
        ix={v:i for i,v in enumerate(g["nodes"])}
        for pi,p in enumerate(PROFILES):
            rng=np.random.default_rng([TEST_SEED,NETWORKS.index(net),j,pi])
            for idx,(e,src,_,means) in enumerate(draw_events(g,stream,w,edge,rng,p,DRAWS,False)):
                row=lookup[e["id"]]
                assert row["window"]==e["window"] and row["profile"]==p
                n+=1
                if idx:
                    continue
                (nodes,_,_),(x,_,_)=hist.design(frames(e))
                local=w[[ix[v] for v in nodes]]
                a,b=[nodes.index(v) for v in src]
                mix=1/means[0]+(local[:,None]+local)/means[1]+np.outer(local,local)/means[2]
                tie=fg.tie_order(e["id"],len(nodes))
                for z,(arm,model,mu,sd) in enumerate(models):
                    with torch.no_grad():
                        s=model.node(model.encoder(torch.from_numpy(np.clip((x-mu)/sd,-6,6))))[:,0].double().numpy()
                    h=np.exp(s-s.max())
                    t=np.outer(h,h)
                    if rule_for(arm)=="equal":
                        t*=mix
                    np.fill_diagonal(t,0)
                    ids=np.lexsort((tie,-t.sum(1)))[:12]
                    assert int(a in ids and b in ids)==row["hits"][z]
                    err=abs(-np.log(2*t[a,b]/t.sum())-row["nll"][z])
                    highest=max(highest,err)
                    checked+=1
    assert n==len(rows) and highest<5.1e-8
    return rows,m["windows"],dict(events=n,explicit_posterior_checks=checked,max_nll_error=highest)


def analyze(rows,keys):
    def get(field):
        return np.array([r[field] for r in rows]).reshape(len(rows),len(ARMS),len(SEEDS))
    hits,nll=get("hits"),get("nll")
    at=np.array([keys.index(r["window"]) for r in rows])
    counts=np.bincount(at,minlength=len(keys))
    rng=np.random.default_rng(431173)
    blocks=((rng.integers(len(keys),size=(10000,(len(keys)+2)//3,1))+np.arange(3))%len(keys)).reshape(10000,-1)[:,:len(keys)]
    seeds=rng.integers(len(SEEDS),size=(10000,1,len(SEEDS)))
    denom=counts[blocks].sum(1)
    ok=denom>0
    out=dict(n=len(rows),window_counts=counts.tolist(),methods={},paired_to_mixed_plain={})
    def contrast(i,j):
        delta=hits[:,i]-hits[:,j]
        nll_delta=nll[:,i]-nll[:,j]
        diff=dict(pp=100*delta.mean(),nll=float(nll_delta.mean()),by_seed_pp=(100*delta.mean(0)).tolist())
        if len(keys)>=10:
            for metric,data,unit in (("ci95_pp",delta,100),("ci95_nll",nll_delta,1)):
                contrib=np.array([data[at==j].sum(0) for j in range(len(keys))])
                boot=contrib[blocks[:,:,None],seeds].sum(1).mean(1)
                diff[metric]=np.quantile(unit*boot[ok]/denom[ok],[.025,.975]).tolist()
        return diff
    for i,key in enumerate(ARMS):
        out["methods"][key]=dict(both_pct=100*hits[:,i].mean(),nll=nll[:,i].mean())
        if key!="mixed_plain":
            out["paired_to_mixed_plain"][key]=contrast(i,1)
    out["mixed_offset_minus_uniform_offset"]=contrast(2,0)
    out["mixed_component_minus_mixed_offset"]=contrast(3,2)
    return out


def main():
    assert fg.load(OUT/"protocol.json")==protocol()
    report={}
    verify=dict(gradient_and_loss_max_error=loss_check(),networks={})
    for n in NETWORKS:
        rows,keys,check=audit(n)
        verify["networks"][n]=check
        report[n]={p:analyze([r for r in rows if r["profile"]==p],keys) for p in PROFILES}
        print(n,[(p,r["n"],round(r["paired_to_mixed_plain"]["mixed_offset"]["pp"],2)) for p,r in report[n].items()],flush=True)
    fg.write_json(OUT/"summary.json",report)
    fg.write_json(OUT/"verification.json",verify)


if __name__=="__main__":
    torch.set_num_threads(1)
    with threadpool_limits(1):
        main()
