"""Regenerate holdout samples and verify model decisions by enumerating candidate pairs."""
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from five_frame_pairs import frames
from heldout_r_lists import (ATTEMPTS, KEYS, NAMES, OUT, PROFILES, SEED, SEEDS, PublicWindow,
                             activity, calendar_windows, correlated_prior, draw_events, fitted_models,
                             law_arrays, lock, messages)
from train_source_mixtures import rule_for


def audit(name,models,msg=None,folder=None,result=None,seed=None,domain=None):
    msg=messages(name) if msg is None else msg
    folder=OUT/name if folder is None else folder
    seed=SEED if seed is None else seed
    domain=NAMES.index(name) if domain is None else domain
    rows=fg.read_rows(folder/"events.jsonl")
    lookup={r["id"]:r for r in rows}
    assert len(lookup)==len(rows)
    result=fg.load(OUT/"summary.json")[name] if result is None else result
    visited=set()
    explicit=0
    for j,(g,history,stream,vol,real,_) in enumerate(calendar_windows(msg,name)):
        assert all(t<g["start"] for t,_,_ in history)
        assert all(g["start"]<=t<g["start"]+14*86400 for t,_ in stream)
        hist=PublicWindow(g,history)
        w=activity(g,history)
        means=law_arrays(w)[3]
        ix={v:i for i,v in enumerate(g["nodes"])}
        edge=correlated_prior(history,g)
        def verify(e,gold,profile,dense):
            nonlocal explicit
            row=lookup[e["id"]]
            assert e["id"] not in visited
            visited.add(e["id"])
            seen={v for b in e["observed"] for v in b}
            legal=set(ix)-seen
            assert row["sources"]==gold and row["window"]==KEYS[j] and row["profile"]==profile
            assert row["ready"]==(len(seen)>=3 and len(legal)>=12)
            assert row["reachable"]==(set(gold)<=legal)
            if not row["ready"]:
                assert not row["hits"]
            if not dense or not row["ready"]:
                return
            (nodes,_,_),(x,_,_)=hist.design(frames(e))
            assert set(nodes)==legal
            weights=w[[ix[n] for n in nodes]]
            mix=1/means[0]+(weights[:,None]+weights)/means[1]+np.outer(weights,weights)/means[2]
            tie=fg.tie_order(e["id"],len(nodes))
            def hit(scores):
                ids=np.lexsort((tie,-np.asarray(scores)))[:12]
                return int(set(gold)<={nodes[i] for i in ids})
            assert hit([vol[v] for v in nodes])==row["hits"]["volume"]
            for k,(arm,net,mu,sd) in enumerate(models):
                with torch.no_grad():
                    s=net.node(net.encoder(torch.from_numpy(np.clip((x-mu)/sd,-6,6))))[:,0].double().numpy()
                q=np.exp(s-s.max())
                mat=np.outer(q,q)
                if rule_for(arm)=="equal":
                    mat*=mix
                np.fill_diagonal(mat,0)
                assert hit(mat.sum(1))==row["hits"][f"{arm}/{SEEDS[k%len(SEEDS)]}"]
                explicit+=1
        for e,gold in real:
            verify(e,gold,"real",True)
        for pi,p in enumerate(PROFILES):
            rng=np.random.default_rng([seed,domain,j,pi])
            for k,(e,gold,_,_) in enumerate(draw_events(g,stream,w,edge,rng,p,ATTEMPTS,False)):
                verify(e,gold,p,k==0)
    assert visited==set(lookup)
    for p,stats in result.items():
        z=[r for r in rows if r["profile"]==p]
        assert len(z)==stats["n"]
        if z:
            for arm,ct in stats["seed_hits"].items():
                assert [sum(r["hits"].get(f"{arm}/{s}",0) for r in z) for s in SEEDS]==ct
    return dict(regenerated=len(visited),explicit_dense_checks=explicit,sha256=fg.digest(folder/"events.jsonl"),
                unique_messages=len(msg))


if __name__=="__main__":
    assert fg.load(OUT/"protocol_v2.json")==lock()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        fits=fitted_models("Dept2")
        res={n:audit(n,fits) for n in NAMES}
        for n in NAMES:
            res[n]["archives"]=len(fg.load(OUT/n/"archives_v2.json"))
        authors=[{v[0] for v in messages(n).values()} for n in NAMES]
        res["archive_author_overlap"]=dict(shared=len(set.intersection(*authors)),distinct=[len(a) for a in authors])
        fg.write_json(OUT/"verification.json",res)
