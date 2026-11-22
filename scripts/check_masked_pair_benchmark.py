"""Replay held-out events and score the frozen rankings plus the existing GCNSI checkpoint."""

from itertools import combinations

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from compare_gcnsi_pairs import architecture, batch_predict, encode
from masked_pair_benchmark import (OUT, PLAN, PREV, RAW, arrival, company, joint_bank, load,
    make_windows, mask, metrics, peak_scores, read_rows, roles, save_rows, sha, simulate, write_json)


def check_joint():
    t=np.array([1,1,2,3,4,5,6])
    u=np.array([0,2,1,3,4,5,6])
    v=np.array([1,3,3,4,5,6,7])
    A=arrival(8,t,u,v,np.ones(len(t),bool))
    assert A[0,3]==2 and A[2,3]==1
    a,b=np.array([0]),np.array([2])
    counts,lens,cov,buckets=joint_bank(A[None,:,:],a,b)
    assert list(lens[:,0,0])==[6,3]
    assert np.flatnonzero(counts[1,0]).tolist()==[5,6,7]
    assert cov[0,0,0]==2 and cov[0,0,-1]==8
    score=peak_scores(np.array([0]),a,b,cov,buckets,np.array([5,6,7]))
    assert np.allclose(score,[3/8])
    # No within-timestamp two-hop chain.
    C=arrival(3,np.array([1,1]),np.array([0,1]),np.array([1,2]),np.array([True,True]))
    assert C[0,2]>1_000_000


def paired(rows,primary="joint_tail"):
    weeks=sorted({r["window"] for r in rows})
    denom=np.array([sum(r["window"]==w for r in rows) for w in weeks])
    rng=np.random.default_rng(261119)
    draw=rng.integers(len(weeks),size=(10000,len(weeks)))
    result={}
    tags=("ordered","shuffled","radius","joint_full","snapshot_all","gcnsi","bosoul70")
    for tag in tags:
        def rr(r,m):
            return 1/r[m] if r[m] else 0
        delta=np.array([sum(rr(r,primary)-rr(r,tag) for r in rows if r["window"]==w) for w in weeks])
        ci=np.quantile(delta[draw].sum(axis=1)/denom[draw].sum(axis=1),[.025,.975])
        result[tag]=dict(delta_mrr=delta.sum()/len(rows),ci=ci.tolist(),positive_windows=int((delta>0).sum()),
                         top1_gained=sum(r[primary]==1 and r[tag]!=1 for r in rows),
                         top1_lost=sum(r[primary]!=1 and r[tag]==1 for r in rows))
    return result


def run():
    check_joint()
    raw=company()
    gs,streams=make_windows(raw,"company",*PLAN["span"])
    labs={r["id"]:r for r in read_rows(OUT/"truth.jsonl")}
    obs={r["id"]:r for r in read_rows(OUT/"observations.jsonl")}
    assert labs.keys()==obs.keys()
    assert sha(PREV/"gcnsi_weights.pt")==load(PREV/"gcnsi_protocol.json")["weights"]
    gcn=architecture()
    gcn.load_state_dict(torch.load(PREV/"gcnsi_weights.pt",weights_only=True))
    gcn.eval()
    rows=[]
    for g in gs:
        key=g["window"]
        items=[o for o in obs.values() if o["window"]==key]
        if not items:
            continue
        x,edge,ix=encode(g,items)
        with torch.no_grad():
            logits=torch.cat([batch_predict(gcn,q,edge) for q in x.split(6)])
            strength=(logits[:,:,1]-logits[:,:,0]).numpy()
        pos={o["id"]:k for k,o in enumerate(items)}
        for pred in read_rows(OUT/(key+"_predictions.jsonl")):
            sid=pred["id"]
            lab,o=labs[sid],obs[sid]
            assert pred["window"]==lab["window"]==o["window"]==key
            tr=simulate(streams[key],lab["sources"],lab["seed"])
            assert tr==lab["trajectory"] and mask(tr)==(lab["cut_batches"],o["observed"])
            assert not lab["dual"] or len(roles(lab)[1])==2
            nodes=pred["nodes"]
            legal=set(ix)-{v for b in o["observed"] for v in b}
            assert set(nodes)<=legal and len(set(nodes))==len(nodes)==min(64,len(legal))
            pairs=list(combinations(nodes,2))
            i,j=np.triu_indices(len(nodes),1)
            s=strength[pos[sid],[ix[v] for v in nodes]]
            pred["orders"]["gcnsi"]=np.argsort(-(s[i]+s[j]),kind="stable").tolist()
            target=next((k for k,p in enumerate(pairs) if set(p)==set(lab["sources"])),None)
            row=dict(id=sid,window=key,dual=lab["dual"],ordinary=lab["ordinary"],inside=target is not None)
            for tag,order in pred["orders"].items():
                assert len(order)==len(set(order))==(70 if tag in ("bosoul70","random70") else len(pairs))
                assert min(order)>=0 and max(order)<len(pairs)
                row[tag]=order.index(target)+1 if target in order else None
            rows.append(row)
    assert len(rows)==len(labs)==len({r["id"] for r in rows})
    save_rows(OUT/"event_results.jsonl",rows)
    report={}
    for group in ("dual","ordinary"):
        rr=[r for r in rows if r[group]]
        tags=list(pred["orders"])
        report[group]=dict(n=len(rr),inside=sum(r["inside"] for r in rr),scores={t:metrics(rr,t) for t in tags},
            paired=paired(rr),windows={g["window"]:{t:metrics([r for r in rr if r["window"]==g["window"]],t)
            for t in tags} for g in gs if any(r["window"]==g["window"] for r in rr)})
    report["data"]=dict(unique_events=len(rows),overlap=sum(r["dual"] and r["ordinary"] for r in rows),
                        nodes=len({v for _,a,b in raw for v in (a,b)}),unique_nonself_contacts=len(raw),archive=sha(RAW))
    write_json(OUT/"summary.json",report)
    print({k:v["scores"] for k,v in report.items() if k!="data"})


if __name__=="__main__":
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run()
