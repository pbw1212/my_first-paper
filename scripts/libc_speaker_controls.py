"""Carry the frozen pair and direction hypotheses to the additional real discussion cohort."""
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
import heldout_libc_alpha as corpus
import heldout_r_lists as r
import historical_speaker_pairs as sp
import test_reply_direction as rd
from historical_thread_roles import historical_document
from public_thread_transfer import DAY,thread_forest
from real_trained_mixtures import rank

OUT=corpus.OUT/"early_speaker_controls"


def run():
    cfg=fg.load(sp.OUT/"protocol.json")
    assert cfg["sha256"]==fg.digest(Path(sp.__file__))
    spec=dict(primary=["paired minus roles","reverse contact history minus forward with prior fixed"],
              source_configs=[fg.digest(sp.OUT/"protocol.json"),fg.digest(rd.OUT/"protocol.json"),fg.digest(corpus.OUT/"protocol.json")],
              hypotheses="reuse earlier thresholds and five Dept2 seeds; no new fitting on libc",
              script=fg.digest(Path(__file__)))
    lock=OUT/"protocol.json"
    if not lock.exists():
        fg.write_json(lock,spec)
    assert fg.load(lock)==spec
    msg=corpus.messages()
    forest,_=thread_forest(msg)
    docs=[historical_document(k,b,msg[k][1],msg[k][0])[:3] for k,b in forest.items()]
    models=[m for m in r.fitted_models("Dept2") if m[0]=="mixed_offset"]
    rows=[]
    for g,h,_,vol,real,_ in r.calendar_windows(msg,corpus.NAME):
        ix={v:i for i,v in enumerate(g["nodes"])}
        c=np.zeros((len(ix),len(ix)))
        for t,a,b in docs:
            if g["start"]-98*DAY<=t<g["start"]-14*DAY and a in ix and b in ix:
                c[ix[a],ix[b]]+=1
        v=np.array([vol[n]+.5 for n in ix])
        w=r.activity(g,h)
        means=r.law_arrays(w)[3]
        past_reverse=sorted((t,b,a) for t,a,b in h)
        rev=dict(g,edges=sorted((b,a,x) for a,b,x in g["edges"]))
        histories=[r.PublicWindow(g,h),r.PublicWindow(rev,past_reverse)]
        independent=sp.post(c,v,cfg["selected_tau"])
        together=sp.post(c,v,cfg["selected_tau"],cfg["selected_kappa"])
        for e,gold in real:
            ids,ready=sp.legal(e,ix)
            row=dict(id=e["id"],window=e["window"],ready=ready,hits={})
            if ready:
                tie=fg.tie_order(e["id"],len(ids))
                order=fg.sorted_nodes(v[ids],tie)
                pair=together[np.ix_(ids,ids)]
                pair/=pair.sum()
                inside=independent[np.ix_(ids,ids)]
                truth={ix[n] for n in gold if n in ix}
                def hit(selected):
                    return int(len(truth)==2 and truth<=set(ids[selected]))
                row["hits"]["volume"]=hit(order[:12])
                for m,mat in zip(sp.METHODS[1:],(inside,pair,sp.margins(pair))):
                    joint=(mat+mat.T)/mat.sum()
                    selected=sp.best_subset(joint,order,tie,12) if len(ids)>12 else order
                    row["hits"][m]=hit(selected)
                for label,history in zip(("forward","reverse"),histories):
                    lists=rank(history,e,models,w,ix,means,vol)
                    assert int(set(gold)<=set(lists["volume"]))==row["hits"]["volume"]
                    for seed in r.SEEDS:
                        row["hits"][f"{label}/{seed}"]=int(set(gold)<=set(lists[f"mixed_offset/{seed}"]))
            rows.append(row)
    fg.save_rows(OUT/"events.jsonl",rows)
    sums={m:sum(row["hits"].get(m,0) for row in rows) for m in [*sp.METHODS,*(f"{v}/{s}" for v in ("forward","reverse") for s in r.SEEDS)]}
    out=dict(n=len(rows),scored=sum(z["ready"] for z in rows),hits=sums,contrasts={})
    a=np.array([r.KEYS.index(z["window"]) for z in rows])
    c=np.bincount(a,minlength=24)
    rng=np.random.default_rng(83652)
    blocks=((rng.integers(24,size=(10000,8,1))+np.arange(3))%24).reshape(-1,24)
    seeds=rng.integers(5,size=(10000,1,5))
    den=c[blocks].sum(1)
    ok=den>0
    for target,base in (("paired","roles"),("paired","matched_margins"),("paired","volume"),("reverse","forward")):
        def score(z,m,s):
            return z["hits"].get(f"{m}/{s}" if m in ("reverse","forward") else m,0)
        delta=np.array([[score(z,target,s)-score(z,base,s) for s in r.SEEDS] for z in rows])
        table=np.array([delta[a==i].sum(0) for i in range(24)])
        vals=table[blocks[:,:,None],seeds].sum(1).mean(1)
        out["contrasts"][target+"-"+base]=dict(pp=100*delta.mean(),ci95_pp=np.quantile(100*vals[ok]/den[ok],[.025,.975]).tolist(),
                                                gained_per_seed=(delta>0).sum(0).tolist(),lost_per_seed=(delta<0).sum(0).tolist())
    fg.write_json(OUT/"summary.json",out)


if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run()
