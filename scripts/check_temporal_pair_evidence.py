"""Rebuild external truth and summarize paired exact-ranking gains by time window."""

from itertools import combinations
from pathlib import Path

import numpy as np

from temporal_pair_evidence import (NEW, OUT, live_reach, load, load_events, make_windows, mask,
                                    read_rows, roles, sha, simulate, write_json)


def audit():
    for file,digest in load(OUT/"external_lock.json")["inputs"].items():
        assert sha(Path(file))==digest
    assert live_reach([[(0,1),(1,2)]],3)[0,2]==0
    assert live_reach([[(0,1)],[(1,2)]],3)[0,2]==1
    assert live_reach([[(1,2)],[(0,1)]],3)[0,2]==0
    gs,stream=make_windows(load_events(NEW),"external",.15,.8)
    nodes={g["window"]:set(g["nodes"]) for g in gs}
    def byid(filename):
        rows=read_rows(OUT/filename)
        result={r["id"]:r for r in rows}
        assert len(rows)==len(result)
        return result
    obs=byid("external_public.jsonl")
    labs=byid("external_truth.jsonl")
    ranks=byid("Dept2_ranks.jsonl")
    neural=byid("Dept2_gcnsi_ranks.jsonl")
    assert obs.keys()==labs.keys()==ranks.keys()==neural.keys()
    seen=set()
    for p in read_rows(OUT/"Dept2_predictions.jsonl"):
        sid=p["id"]
        r,o,lab=ranks[sid],obs[sid],labs[sid]
        assert sid not in seen
        seen.add(sid)
        g=o["window"]
        tr=simulate(stream[g],lab["sources"],lab["seed"])
        assert tr==lab["trajectory"] and mask(tr)==(lab["cut_batches"],o["observed"])
        if lab["dual"]:
            assert len(roles(lab)[1])==2
        legal=nodes[g]-{v for b in o["observed"] for v in b}
        assert set(p["nodes"])<=legal and len(set(p["nodes"]))==min(64,len(legal))
        pairs=list(combinations(p["nodes"],2))
        target=next((i for i,s in enumerate(pairs) if set(s)==set(lab["sources"])),None)
        for tag,order in p["orders"].items():
            assert sorted(order)==list(range(len(pairs)))
            assert r[tag]==(None if target is None else order.index(target)+1)
        r["gcnsi"]=neural[sid]["gcnsi"]
    assert seen==set(obs)
    report={}
    for group in ("dual","ordinary"):
        rr=[r for r in ranks.values() if r[group]]
        periods=sorted({r["window"] for r in rr})
        rng=np.random.default_rng(907)
        draw=rng.integers(len(periods),size=(10000,len(periods)))
        size=np.array([sum(r["window"]==w for r in rr) for w in periods])
        summary={}
        for tag in ("radius","static","shuffled","fusion","gcnsi"):
            def reciprocal(r,m):
                return 0 if r[m] is None else 1/r[m]
            sums=np.array([sum(reciprocal(r,"ordered")-reciprocal(r,tag) for r in rr if r["window"]==w)
                           for w in periods])
            interval=np.quantile(sums[draw].sum(axis=1)/size[draw].sum(axis=1),[.025,.975])
            summary[tag]=dict(difference=sums.sum()/len(rr),ci=interval.tolist(),
                              positive_windows=int(sum(sums>0)),windows=len(periods))
        report[group]=summary
    write_json(OUT/"external_audit.json",dict(events=len(obs),paired_mrr=report))
    print(len(obs),report)


if __name__=="__main__":
    audit()
