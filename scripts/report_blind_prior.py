"""Audit blind prior adaptation with disjoint unlabelled batches and explicit pair matrices."""
from collections import Counter

import numpy as np
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from blind_source_prior import (EQUAL,NETWORKS,OUT,PROFILES,SEEDS,SOURCE,activity,correlated_prior,draws,estimate,
                               infer,law_arrays,past_acceptance,plan,representation,windows)
from distilled_source_pairs import PublicWindow
from evaluate_factorial_history import fitted

METHODS = ("activity","raw","fixed","always_both","adaptive","oracle")


def check_em():
    evidence = np.array([[1.,.1,.5],[1.,5.,.2],[1.,.01,1.7],[1.,1.,1.3]])
    acceptance = np.array([.3,.54,.77])
    ratio = evidence*acceptance[0]/acceptance
    weight = estimate(evidence,acceptance)*acceptance
    weight /= weight.sum()
    def loss(x):
        eta = np.exp([0.,x[0],x[1]])
        eta /= eta.sum()
        return -np.log(ratio@eta).sum()-3*(acceptance/acceptance.sum())@np.log(eta)
    opt = minimize(loss,np.zeros(2))
    assert abs(opt.fun-loss(np.log(weight[1:]/weight[0])))<1e-8
    assert np.allclose(estimate([],acceptance),EQUAL)


def audit(net):
    folder = OUT/net
    manifest = fg.load(folder/"audit.json")
    for path,h in manifest["hashes"].items():
        assert fg.digest(folder/path)==h
    rows = fg.read_rows(folder/"results.jsonl")
    fits = {(r["window"],r["profile"]):r for r in fg.read_rows(folder/"adaptation.jsonl")}
    by_id = {r["id"]:r for r in rows}
    raw,gs = windows(net,False)
    nets = {s:fitted(SOURCE[net],s,"C") for s in SEEDS}
    cache = {}
    count = 0
    for j,(g,stream) in enumerate(gs):
        w = activity(g,raw)
        edge = correlated_prior(raw,g)
        _,_,_,means = law_arrays(w)
        ix = {v:i for i,v in enumerate(g["nodes"])}
        accept,accepted,num = past_acceptance(raw,g["start"],net,cache)
        hist = None
        for profile,mixture in PROFILES.items():
            fit = fits[g["window"],profile]
            adapt,_ = draws(g,stream,w,edge,net,j,profile,0)
            test,gold = draws(g,stream,w,edge,net,j,profile,1)
            assert fit["observations_used"]==[r["id"] for r in adapt[:32]] and fit["ref_attempts_each"]==num
            assert fit["reference_accepted"]==accepted and fit["test_eligible"]==len(test)
            assert set(fit["observations_used"]).isdisjoint(by_id)
            assert all(by_id[mid]["sources"]==src for mid,src in gold.items())
            count += len(test)
            if not test:
                continue
            if hist is None:
                hist = PublicWindow(g,raw)
            # Refitting from public inputs alone reproduces each recorded mixture.
            res = infer(hist,adapt,nets,w,ix,means,accept)
            assert all(np.allclose(fit["estimated"][str(s)],res[s]) for s in SEEDS)
            e = test[0]
            nodes,indices,outputs = representation(hist,e,nets,w,ix,means)
            local = w[indices]
            priors = np.array([np.ones((len(nodes),len(nodes))),local[:,None]+local,local[:,None]*local])
            tie = fg.tie_order(e["id"],len(nodes))
            ia,ib = [nodes.index(v) for v in gold[e["id"]]]
            for s,(_,_,h) in outputs.items():
                for method,mix in (("adaptive",res[s]),("fixed",EQUAL),("raw",[1,0,0]),("oracle",mixture)):
                    table = edge[np.ix_(indices,indices)] if mix is None else np.einsum("i,ijk->jk",np.array(mix)/means,priors)
                    table = table*np.outer(h,h)
                    np.fill_diagonal(table,0)
                    selected = fg.sorted_nodes(table.sum(1),tie)[:12]
                    assert int(ia in selected)+int(ib in selected)==by_id[e["id"]]["hits"][f"{method}/{s}"]
    assert count==len(rows)==len(by_id)
    return rows,fits,[g["window"] for g,_ in gs]


def summarize(rows,fits,keys,profile):
    group = [r for r in rows if r["profile"]==profile]
    totals = Counter(r["window"] for r in group)
    matrix = np.zeros((len(METHODS),len(SEEDS),len(keys)),int)
    for r in group:
        k = keys.index(r["window"])
        for j,m in enumerate(METHODS):
            for i,s in enumerate(SEEDS):
                matrix[j,i,k] += r["hits"][m if m=="activity" else f"{m}/{s}"]==2
    n = len(group)
    estim = np.array([[fits[k,profile]["estimated"][str(s)] for s in SEEDS] for k in keys])
    counts = np.array([totals[k] for k in keys])
    result = dict(n=n,methods={},differences={},weighted_estimated_draw_mixture=np.average(estim.mean(1),axis=0,weights=counts).tolist(),
                  used_unlabelled=sum(len(fits[k,profile]["observations_used"]) for k in keys),window_eligible=counts.tolist())
    for j,m in enumerate(METHODS):
        result["methods"][m] = dict(both=100*matrix[j].sum(1).mean()/n,window_seed_hits=matrix[j].T.tolist())
    rng = np.random.default_rng(20261122)
    size = len(keys)
    blocks = ((rng.integers(size,size=(10000,(size+2)//3,1))+np.arange(3))%size).reshape(10000,-1)[:,:size]
    seeds = rng.integers(len(SEEDS),size=(10000,len(SEEDS),1))
    for m in ("fixed","raw","activity","always_both","oracle"):
        diff = matrix[METHODS.index("adaptive")]-matrix[METHODS.index(m)]
        res = dict(pp=100*diff.sum(1).mean()/n,by_seed=(100*diff.sum(1)/n).tolist())
        if size>=10:
            den = counts[blocks].sum(1)
            ok = den>0
            values = 100*diff[seeds,blocks[:,None,:]].sum(2).mean(1)[ok]/den[ok]
            res["ci95"] = np.quantile(values,[.025,.975]).tolist()
        result["differences"][m] = res
    return result


def main():
    assert fg.load(OUT/"protocol.json")==plan()
    check_em()
    out = {}
    for n in NETWORKS:
        rows,fits,keys = audit(n)
        out[n] = {}
        for p in PROFILES:
            v = summarize(rows,fits,keys,p)
            out[n][p] = v
            print(n,p,v["n"],[round(v["methods"][m]["both"],2) for m in METHODS],flush=True)
    fg.write_json(OUT/"summary.json",out)


if __name__=="__main__":
    torch.set_num_threads(1)
    with threadpool_limits(1):
        main()
