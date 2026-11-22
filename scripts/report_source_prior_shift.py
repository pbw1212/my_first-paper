"""Recreate source draws, validate the factorizations and report event-paired contrasts."""
import numpy as np

import fgssi_suffix as fg
from source_prior_shift import (NETWORKS,OPTIONS,OUT,REGIMES,SEEDS,SOURCE,activity,generate,
                               pair_posterior,protocol,select,windows)

METHODS = ("volume","raw","matched","calibrated32","always_both","fg","fg_matched")


def verify_algebra():
    rng = np.random.default_rng(111)
    maxdiff = 0.
    for n in (3,12,150):
        s = rng.normal(size=n)
        w = rng.integers(1,800,n).astype(float)
        for rule in OPTIONS:
            q = np.sqrt(w) if rule.endswith("_half") else w
            prior = np.ones((n,n))
            if rule.startswith("one"):
                prior = q[:,None]+q
            if rule.startswith("both"):
                prior = q[:,None]*q
            exact = np.exp(s[:,None]+s)*prior
            np.fill_diagonal(exact,0)
            marg,h,u = pair_posterior(s,w,rule)
            maxdiff = max(maxdiff,float(np.max(np.abs(marg/marg.sum()-exact.sum(1)/exact.sum()))))
            gold = h[0]*h[1] if u is None else h[0]*u[1]+h[1]*u[0]
            assert np.isclose(gold/(marg.sum()/2),exact[0,1]/(exact.sum()/2))
    assert maxdiff<1e-12
    return maxdiff


def key(m,s,law,chosen):
    return dict(volume="volume",raw=f"full/{s}/flat",matched=f"full/{s}/{law}",
                calibrated32=f"full/{s}/{chosen}",always_both=f"full/{s}/both",fg=f"fg/{s}/flat",
                fg_matched=f"fg/{s}/{law}")[m]


def summarize(rows,law,chosen,wins):
    totals = np.zeros(len(wins),int)
    hits = np.zeros((len(METHODS),len(SEEDS),len(wins)),int)
    losses = np.zeros_like(hits,dtype=float)
    gained_lost = np.zeros((len(SEEDS),2),int)
    for r in rows:
        j = wins.index(r["window"])
        totals[j] += 1
        for i,s in enumerate(SEEDS):
            for k,m in enumerate(METHODS):
                h,l = r["metrics"][key(m,s,law,chosen)]
                hits[k,i,j] += h==2
                losses[k,i,j] += l
            a,b = [r["metrics"][key(m,s,law,chosen)][0]==2 for m in ("calibrated32","raw")]
            gained_lost[i] += [a and not b,b and not a]
    n = len(rows)
    assert n>0
    res = dict(n=n,selected=chosen,windows=wins,eligible=totals.tolist(),two_emitters=sum(r["emitters"]==2 for r in rows),
               calibrated_vs_raw_gained_lost=gained_lost.tolist(),methods={},differences={})
    for k,m in enumerate(METHODS):
        res["methods"][m] = dict(both=100*hits[k].sum(1).mean()/n,nll=losses[k].sum(1).mean()/n,
                                 hit_by_seed_window=hits[k].tolist())
    rng = np.random.default_rng(20261122)
    m = len(wins)
    blocks = ((rng.integers(m,size=(10000,(m+2)//3,1))+np.arange(3))%m).reshape(10000,-1)[:,:m]
    seeds = rng.integers(len(SEEDS),size=(10000,len(SEEDS),1))
    for a,b in (("matched","raw"),("calibrated32","raw"),("calibrated32","volume"),("matched","fg_matched"),("raw","volume")):
        delta = hits[METHODS.index(a)]-hits[METHODS.index(b)]
        stat = dict(pp=100*delta.sum(1).mean()/n,seed_diffs=(100*delta.sum(1)/n).tolist())
        if m>=10:
            den = totals[blocks].sum(1)
            ok = den>0
            effect = 100*delta[seeds,blocks[:,None,:]].sum(2).mean(1)[ok]/den[ok]
            stat["ci95"] = np.quantile(effect,[.025,.975]).tolist()
        res["differences"][a+"-"+b] = stat
    return res


def audit(net,cal):
    folder = OUT/net/("cal" if cal else "test")
    meta = fg.load(folder/"audit.json")
    for k in ("events","observations"):
        assert meta[k]==fg.digest(folder/(k+".jsonl"))
    rows = fg.read_rows(folder/"events.jsonl")
    raw,graphs = windows(net,cal)
    obs = {r["id"]:r for r in fg.read_rows(folder/"observations.jsonl")}
    gold = {r["id"]:r for r in rows}
    seen = set()
    for j,(g,stream) in enumerate(graphs):
        events,labs = generate(g,stream,activity(g,raw),net,j,cal)
        for e in events:
            sid = e["id"]
            assert obs[sid]==e and all(gold[sid][k]==v for k,v in labs[sid].items())
            seen.add(sid)
    assert set(obs)==set(gold)==seen and len(seen)==meta["rows"]
    return rows,meta


def main():
    assert fg.load(OUT/"protocol.json")==protocol()
    summary = {}
    error = verify_algebra()
    for source in ("Dept2","company"):
        rows,meta = audit(source,True)
        path = OUT/source/"cal/selection.json"
        assert fg.digest(path)==meta["selection"] and select(rows,meta["windows"])==fg.load(path)
    for net in NETWORKS:
        rows,meta = audit(net,False)
        source = SOURCE[net]
        assert fg.digest(OUT/source/"cal/selection.json")==meta["selection"]
        selection = fg.load(OUT/source/"cal/selection.json")
        summary[net] = {}
        for law in REGIMES:
            group = [r for r in rows if r["regime"]==law]
            chosen = selection[law]["selected"]
            summary[net][law] = {k:summarize(sub,law,chosen,meta["windows"]) for k,sub in
                                 (("all",group),("one_emitter",[r for r in group if r["emitters"]==1]),("two_emitters",[r for r in group if r["emitters"]==2])) if sub}
    fg.write_json(OUT/"results.json",summary)
    fg.write_json(OUT/"verification.json",dict(matrix_probability_max_absolute_error=error,simulations_replayed={
        n:(len(fg.load(OUT/n/"test/audit.json")["windows"])*160*3) for n in NETWORKS},
        calibration_replays=2*3*112*3,summary_sha256=fg.digest(OUT/"results.json")))
    for net,groups in summary.items():
        for law,table in groups.items():
            t = table["all"]
            print(net,law,t["n"],t["selected"],[round(t["methods"][k]["both"],2) for k in METHODS])


if __name__=="__main__":
    main()
