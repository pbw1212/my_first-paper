"""Independently enumerate small rosters, replay truth and compare paired shortlist decisions."""
from itertools import combinations
from math import comb

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from robust_source_list import (BUDGET, METHODS, NETWORKS, OUT, PROFILES, SEEDS, SOURCE,
                                PublicWindow, activity, correlated_prior, decode, draw, fitted,
                                law_arrays, protocol, representation, windows)


def matrices(x,w):
    a = np.outer(x,x)
    value = np.stack((a, a*(w[:,None]+w), a*np.outer(w,w)))
    for v in value:
        np.fill_diagonal(v,0.)
    return value


def exhaustive(n=10,k=5):
    rng = np.random.default_rng(15673)
    ratios = []
    for i in range(60):
        x = rng.lognormal(size=n) if i%3 else rng.integers(1,4,n).astype(float)
        w = rng.integers(1,20,n) if i%4 else np.ones(n)
        means = law_arrays(w)[3]
        tie = rng.permutation(n)
        lists,cert,bounds,_ = decode(x,w,means,tie,k)
        value = matrices(x,w)
        sets = np.array(list(combinations(range(n),k)))
        masses = value[:,sets[:,:,None],sets[:,None,:]].sum(axis=(2,3))/2
        exact = masses.max(1)
        assert np.all(exact<=bounds+1e-8) and np.allclose(exact[[0,2]],bounds[[0,2]])
        mixtures = np.r_[np.eye(3),rng.dirichlet([.4]*3,size=40)]
        for idx,c in zip(lists,cert):
            inside = value[:,idx[:,None],idx].sum(axis=(1,2))/2
            assert np.isclose(c,np.min(inside/bounds))
            assert np.all((mixtures@inside)/(mixtures@masses).max(1)>=c-1e-10)
        assert cert[3]>=max(cert[0],cert[1],cert[4])-1e-10
        fixed = (value/means[:,None,None]).sum(0)
        best_marginal = np.sort(fixed.sum(1))[-k:].sum()
        assert np.isclose(fixed.sum(1)[lists[1]].sum(),best_marginal)
        achieved = fixed[np.ix_(lists[2],lists[2])].sum()/2
        optimum = np.max((masses/means[:,None]).sum(0))
        assert achieved<=optimum+1e-8
        ratios.append(cert[3]/(masses/bounds[:,None]).min(0).max())
    return dict(rosters=60,k=k,subsets_per_roster=comb(n,k),mixtures_each=43,
                lowest_solver_to_exact_certificate=float(min(ratios)))


def audit(net,rows,manifest):
    raw,gs = windows(net,False)
    seen = {r["id"]:r for r in rows}
    assert len(seen)==len(rows)==manifest["n"]
    assert manifest["windows"]==[g["window"] for g,_ in gs]
    count = checks = 0
    worst_error = 0.
    model = {SEEDS[0]:fitted(SOURCE[net],SEEDS[0],"C")}
    for j,(g,stream) in enumerate(gs):
        w = activity(g,raw)
        edge = correlated_prior(raw,g)
        means = law_arrays(w)[3]
        hist = PublicWindow(g,raw)
        ix = {n:i for i,n in enumerate(g["nodes"])}
        for profile in PROFILES:
            for idx,(e,src) in enumerate(draw(g,stream,w,edge,net,j,profile)):
                row = seen[e["id"]]
                assert row["sources"]==src and row["window"]==g["window"] and row["profile"]==profile
                count += 1
                if idx:
                    continue
                nodes,ids,res = representation(hist,e,model,w,ix,means)
                marg,_,h = res[SEEDS[0]]
                tie = fg.tie_order(e["id"],len(nodes))
                lists,cert,bound,_ = decode(h,w[ids],means,tie)
                table = matrices(h,w[ids])
                ref = (table/means[:,None,None]).sum(0)
                assert np.allclose(marg.sum(0),ref.sum(0))
                for a,c in zip(lists,cert):
                    ref_cert = (table[:,a[:,None],a].sum(axis=(1,2))/(2*bound)).min()
                    worst_error = max(worst_error,abs(c-ref_cert))
                selected = [fg.sorted_nodes(w[ids],tie)[:BUDGET],*lists]
                expected = [int(set(src)<={nodes[q] for q in s}) for s in selected]
                assert expected==row["hits"][0] and np.allclose(cert,row["certificates"][0],atol=6e-9,rtol=0)
                assert np.isclose(ref.sum(0)[lists[1]].sum(),np.sort(ref.sum(0))[-BUDGET:].sum())
                checks += 1
    assert count==len(seen) and worst_error<1e-9
    return dict(regenerated_eligible=count,dense_pair_checks=checks,max_certificate_error=worst_error)


def summarize(rows,keys):
    data = np.array([r["hits"] for r in rows])
    c = np.array([r["certificates"] for r in rows])
    diff = data[:,:,4]-data[:,:,3]
    result = dict(n=len(rows),both=dict(zip(METHODS,(data.mean(axis=(0,1))*100).tolist())),
                  robust_minus_fixed_joint_pp=float(diff.mean()*100),
                  frontier_median=float(np.median([r["frontier"] for r in rows])),
                  certificate={m:dict(p05=float(np.quantile(c[:,:,i],.05)),median=float(np.median(c[:,:,i])),
                                      min=float(c[:,:,i].min()),mean=float(c[:,:,i].mean()))
                               for i,m in enumerate(METHODS[1:])})
    at = np.array([keys.index(r["window"]) for r in rows])
    counts = np.bincount(at,minlength=len(keys))
    contributions = np.array([diff[at==i].sum(0) for i in range(len(keys))])
    result["window_counts"] = counts.tolist()
    result["delta_pp_by_seed"] = (100*diff.mean(0)).tolist()
    if len(keys)>=10:
        rng = np.random.default_rng(19434)
        b = ((rng.integers(len(keys),size=(10000,(len(keys)+2)//3,1))+np.arange(3))%len(keys)).reshape(10000,-1)[:,:len(keys)]
        s = rng.integers(len(SEEDS),size=(10000,1,len(SEEDS)))
        denom = counts[b].sum(1)
        ok = denom>0
        deltas = contributions[b[:,:,None],s].mean(2).sum(1)
        result["paired_block_ci95_pp"] = np.quantile(100*deltas[ok]/denom[ok],[.025,.975]).tolist()
    return result


def main():
    assert fg.load(OUT/"protocol.json")==protocol()
    verify = dict(exhaustive=exhaustive(),networks={})
    report = {}
    for n in NETWORKS:
        folder = OUT/n
        manifest = fg.load(folder/"audit.json")
        assert fg.digest(folder/"events.jsonl")==manifest["sha256"]
        rows = fg.read_rows(folder/"events.jsonl")
        verify["networks"][n]=audit(n,rows,manifest)
        report[n]={p:summarize([r for r in rows if r["profile"]==p],manifest["windows"]) for p in PROFILES}
        for p,r in report[n].items():
            print(n,p,r["n"],{m:round(r["both"][m],2) for m in METHODS},flush=True)
    fg.write_json(OUT/"verification.json",verify)
    fg.write_json(OUT/"summary.json",report)


if __name__=="__main__":
    torch.set_num_threads(1)
    with threadpool_limits(1):
        main()
