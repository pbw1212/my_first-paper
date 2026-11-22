"""Oracle diagnostic: count exact-suffix alternatives in one completed live-contact world per case."""

import random

import numpy as np
from numba import njit

from masked_pair_benchmark import (OUT, PLAN, PREV, arrival, company, load_events, make_windows,
                                   mask, read_rows, roles, save_rows, write_json)
from temporal_pair_evidence import NEW


def complete_world(stream,ix,lab,completion=0):
    # Match the simulator's random draws on exposed contacts. Supply fresh iid
    # Bernoulli outcomes on unused contacts so counterfactual pairs can traverse them.
    rng=random.Random(lab["seed"])
    suffix="" if completion==0 else "-"+str(completion)
    spare=random.Random("unused-contacts-"+lab["id"]+suffix)
    active=set(lab["sources"])
    times,uu,vv=[],[],[]
    replay=[]
    for k,(timestamp,contacts) in enumerate(stream,1):
        parents={}
        for u,v in contacts:
            exposed=u in active and v not in active
            if (rng if exposed else spare).random()<.3:
                times.append(k)
                uu.append(ix[u])
                vv.append(ix[v])
                if exposed:
                    parents.setdefault(v,[]).append(u)
        if parents:
            replay.append(dict(t=timestamp,joins=[[v,sorted(parents[v])] for v in sorted(parents)]))
            active.update(parents)
    assert replay==lab["trajectory"]
    return tuple(np.asarray(x,dtype=np.int32) for x in (times,uu,vv))


@njit
def dual_visible(times,u,v,at,a,b,visible):
    label=np.zeros(len(at),np.int8)
    label[a],label[b]=1,2
    for k,t in enumerate(times):
        if at[v[k]]==t and at[u[k]]<t:
            label[v[k]]|=label[u[k]]
    evidence=0
    for j in visible:
        evidence|=label[j]
    return evidence==3


@njit
def compatible(A,t,u,v,visible,groups,legal,true_a,true_b):
    any_pairs=dual_pairs=0
    example_a=example_b=-1
    original_seen=False
    original_dual=False
    first_batch_size=np.sum(groups==0)
    for j,a in enumerate(legal):
        for b in legal[j+1:]:
            previous=-1
            matches=True
            for k,node in enumerate(visible):
                when=min(A[a,node],A[b,node])
                if when>=2_000_000_000 or (k and ((groups[k]==groups[k-1] and when!=previous) or
                                                   (groups[k]!=groups[k-1] and when<=previous))):
                    matches=False
                    break
                previous=when
            if not matches:
                continue
            at=np.minimum(A[a],A[b])
            first=at[visible[0]]
            hidden=np.sum((at>0)&(at<first))
            total=np.sum((at>0)&(at<2_000_000_000))
            if total-hidden!=len(visible) or hidden==0 or not hidden<=total//2<hidden+first_batch_size:
                continue
            any_pairs+=1
            both=dual_visible(t,u,v,at,a,b,visible)
            dual_pairs+=int(both)
            if a==true_a and b==true_b:
                original_seen=True
                original_dual=both
            elif both and example_a<0:
                example_a,example_b=a,b
    return any_pairs,dual_pairs,example_a,example_b,original_seen,original_dual


def run(name,completion=0):
    primary=name=="Manufacturing"
    raw=company() if primary else load_events(NEW)
    graphs,streams=make_windows(raw,"company" if primary else "external",*(PLAN["span"] if primary else [.15,.8]))
    loc=OUT if primary else PREV
    obs={o["id"]:o for o in read_rows(loc/("observations.jsonl" if primary else "external_public.jsonl"))}
    labs=read_rows(loc/("truth.jsonl" if primary else "external_truth.jsonl"))
    results=[]
    for g in graphs:
        nodes=g["nodes"]
        ix={v:i for i,v in enumerate(nodes)}
        for lab in labs:
            if lab["window"]!=g["window"]:
                continue
            o=obs[lab["id"]]
            t,u,v=complete_world(streams[lab["window"]],ix,lab,completion)
            A=arrival(len(ix),t,u,v,np.ones(len(t),np.bool_))
            visible=np.array([ix[v] for batch in o["observed"] for v in batch])
            group=np.repeat(np.arange(len(o["observed"])),[len(b) for b in o["observed"]])
            legal=np.setdiff1d(np.arange(len(nodes)),visible)
            a,b=sorted(ix[v] for v in lab["sources"])
            count,dual,c,d,found,found_dual=compatible(A,t,u,v,visible,group,legal,a,b)
            assert found and (not lab["dual"] or found_dual)
            if c>=0:
                # Independent forward replay of the witness, without the arrival-matrix shortcut.
                from itertools import groupby
                active={c,d}
                trace=[]
                for stamp,edges in groupby(zip(t,u,v),key=lambda edge:edge[0]):
                    parents={}
                    for _,s,target in edges:
                        if s in active and target not in active:
                            parents.setdefault(target,[]).append(s)
                    if parents:
                        trace.append(dict(t=int(stamp),joins=[[nodes[q],[nodes[z] for z in parents[q]]]
                                                             for q in sorted(parents)]))
                        active.update(parents)
                cut,observed=mask(trace)
                assert observed==o["observed"]
                assert len(roles(dict(sources=[nodes[c],nodes[d]],trajectory=trace,cut_batches=cut))[1])==2
            results.append(dict(id=o["id"],window=g["window"],dual=lab["dual"],ordinary=lab["ordinary"],
                all_pairs=count,dual_pairs=dual,alternative_dual=None if c<0 else [nodes[c],nodes[d]]))
    assert len(results)==len(labs)
    if completion==0:
        save_rows(OUT/(name+"_oracle.jsonl"),results)
    summaries={}
    for group,key in (("dual","dual_pairs"),("ordinary","all_pairs")):
        counts=np.array([r[key] for r in results if r[group]])
        summaries[group]=dict(n=len(counts),ambiguous=int(np.sum(counts>1)),median=float(np.median(counts)),
                             p90=float(np.quantile(counts,.9)),maximum=int(max(counts)),
                             mean_reciprocal_class_size=float(np.mean(1/counts)))
    return summaries


if __name__=="__main__":
    result={name:run(name) for name in ("Manufacturing","Dept2")}
    write_json(OUT/"equivalence_summary.json",result)
    sensitivity={name:[run(name,k) for k in (1,2)] for name in result}
    write_json(OUT/"equivalence_sensitivity.json",sensitivity)
    print(result)
