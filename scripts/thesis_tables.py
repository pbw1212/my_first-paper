"""Replay test cascades, audit rankings, and rebuild the thesis tables and figure."""

import json
from collections import defaultdict
from itertools import combinations
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evidence_evaluate import has_two_lineages
from thesis_holdout import (EXTERNAL, OUT, PROTOCOL, SOURCES, frozen_models, load,
    load_events, masked, read_rows, save_rows, sha, simulate, windows, write_json)


def audit(name,source):
    folder=OUT/name
    raw=load_events(source)
    gs,streams=windows(name,raw)
    assert gs==read_rows(folder/"graphs.jsonl")
    contacts={g["window"]:sum(len(batch) for _,batch in streams[g["window"]]) for g in gs}
    gs={g["window"]:g for g in gs}
    labs=read_rows(folder/"truth.jsonl")
    lookup={r["id"]:r for r in labs}
    inputs=read_rows(folder/"public.jsonl")
    obs={(o["id"],o["mask"]):o for o in inputs}
    assert len(labs)==len(lookup) and len(inputs)==len(obs)
    results=[]
    expected=set()
    common=set()
    for r in labs:
        tr=simulate(streams[r["window"]],r["sources"],r["seed"])
        assert tr==r["trajectory"] and masked(tr,50)[0]==r["cut_batches"]
        if r["dual"]:
            assert has_two_lineages(r)
        eligible=0
        for f in PROTOCOL["fraction"]:
            v=masked(tr,f)
            if v and (f==50 or r["dual"]):
                expected.add((r["id"],f))
                assert obs[r["id"],f]["observed"]==v[1]
                eligible+=has_two_lineages(dict(r,cut_batches=v[0]))
        if r["dual"] and eligible==3:
            common.add(r["id"])
    assert expected==obs.keys()
    seen=set()
    for k,g in gs.items():
        for p in read_rows(folder/(k+"_rankings.jsonl")):
            key=(p["id"],p["mask"])
            assert key not in seen
            seen.add(key)
            r=lookup[p["id"]]
            o=obs[key]
            assert o["window"]==r["window"]==p["window"]==k
            visible={v for b in o["observed"] for v in b}
            legal=set(g["nodes"])-visible
            nodes=p["nodes"]
            assert set(nodes)<=legal and len(set(nodes))==len(nodes)==min(64,len(legal))
            pairs=list(combinations(nodes,2))
            true=set(r["sources"])
            target=next((i for i,pair in enumerate(pairs) if set(pair)==true),None)
            narrow={i for i,pair in enumerate(pairs) if set(pair)<=set(nodes[:32])}
            row=dict(id=r["id"],window=k,mask=o["mask"],ordinary=r["ordinary"],dual=r["dual"],
                     common=r["id"] in common,legal=len(legal),visible=len(visible),
                     pool32=true<=set(nodes[:32]),pool64=target is not None)
            for method,order in p["orders"].items():
                domain=narrow if method.endswith("32") else set(range(len(pairs)))
                assert len(order)==len(domain) and set(order)==domain
                row[method]=order.index(target)+1 if target in domain else None
            for method in ("radius","add"):
                restricted=[k for k in p["orders"][method+"64"] if k in narrow]
                row[method+"32"]=restricted.index(target)+1 if target in narrow else None
            results.append(row)
    assert seen==expected
    save_rows(folder/"event_results.jsonl",results)
    return results,contacts


METHODS=("direct32","prefix32","add64","radius64","node64","direct64","fusion64","radius32","add32")


def counts(rows):
    if not rows:
        return {"n":0}
    result={"n":len(rows),"pool32":sum(r["pool32"] for r in rows),"pool64":sum(r["pool64"] for r in rows),
        "visible_median":float(np.median([r["visible"] for r in rows])),
        "random_full_rate":float(np.mean([min(1.,32/comb(r["legal"],2)) for r in rows]))}
    for m in METHODS:
        if m in rows[0]:
            ranks=[r[m] for r in rows if r[m] is not None]
            result[m]=dict(top1=ranks.count(1),hit32=sum(v<=32 for v in ranks),
                           mrr=sum(1/v for v in ranks)/len(rows))
    return result


def hit(r,m):
    return r[m] is not None and r[m]<=32


def contrast(rows,old,new):
    parts=defaultdict(list)
    for r in rows:
        parts[r["window"]].append(int(hit(r,new))-int(hit(r,old)))
    delta=np.array([sum(a) for a in parts.values()])
    size=np.array([len(a) for a in parts.values()])
    rng=np.random.default_rng(4327)
    draw=rng.integers(len(parts),size=(10000,len(parts)))
    ci=np.quantile(100*delta[draw].sum(axis=1)/size[draw].sum(axis=1),[.025,.975])
    return dict(gain=sum(hit(r,new) and not hit(r,old) for r in rows),
                loss=sum(hit(r,old) and not hit(r,new) for r in rows),
                difference_pp=100*delta.sum()/len(rows),interval_pp=ci.tolist())


def summarize(rows,contacts):
    result=dict(contacts=contacts)
    for group in ("dual","ordinary"):
        half=[r for r in rows if r["mask"]==50 and r[group]]
        result[group]=counts(half)
        result[group+"_windows"]={k:counts([r for r in half if r["window"]==k]) for k in contacts}
        if half:
            result[group+"_paired"]={b+"_vs_"+a:contrast(half,a,b) for a,b in (
                ("direct32","prefix32"),("direct64","fusion64"),("node64","fusion64"),
                ("add64","fusion64"),("direct32","fusion64"),("radius32","prefix32"))}
    result["masks"]={str(f):counts([r for r in rows if r["mask"]==f and r["common"]])
                     for f in PROTOCOL["fraction"]}
    result["overlap"]=sum(r["mask"]==50 and r["dual"] and r["ordinary"] for r in rows)
    return result


def figure(summary):
    names=("Dept1","CollegeMsg")
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.9),layout="constrained")
    for ax,name in zip(axes,names):
        s=summary[name]["dual"]
        methods=("direct32","prefix32","radius64","add64","node64","direct64","fusion64")
        labels=("Direct / 32","Prefix / 32","Radius / 64","Node sum / 64","Node only / 64","Direct / 64","Fusion / 64")
        bars=ax.barh(labels,[100*s[m]["hit32"]/s["n"] for m in methods],
                     color=["#8193a3","#387a86","#8193a3","#8193a3","#8193a3","#8193a3","#387a86"])
        ax.bar_label(bars,labels=[f'{s[m]["hit32"]}/{s["n"]}' for m in methods],padding=3,fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0,40)
        ax.set_xlabel("True source pair in top 32 (%)")
        ax.set_title(name+(" (department transfer)" if name=="Dept1" else " (time holdout)"))
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"holdout_comparison.png",dpi=180)
    plt.close(fig)


def main():
    frozen_models()
    summary={}
    for name,source in {**SOURCES,"Dept1":EXTERNAL}.items():
        summary[name]=summarize(*audit(name,source))
    write_json(OUT/"summary.json",summary)
    figure(summary)
    for k,s in summary.items():
        print(k,"dual",s["dual"],"ordinary",s["ordinary"],"mask cases",s["masks"]["25"]["n"])


if __name__=="__main__":
    main()
