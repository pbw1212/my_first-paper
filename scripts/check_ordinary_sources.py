"""Replay the held-out events, audit all public predictions, and check the oracle on a finite toy model."""
import argparse
from collections import Counter,defaultdict
from itertools import combinations,product

import numpy as np

from build_dept3_pilot import P,mask,simulate
from ordinary_source_roles import OUT,PREVIOUS,FILES,NEW,company,load_events,load,make_windows,read_rows,write_json
from score_dept3_candidates import roles
from silent_source_oracle import CompleteTrace


def toy_check():
    graph={"nodes":list(range(16))}
    stream=[(1,[(0,2),(1,2),(3,0)]),(2,[(2,3),(1,4)]),(3,[(5,1),(3,4)])]
    oracle=CompleteTrace(graph,stream)
    mass=defaultdict(Counter)
    for seeds in combinations(graph["nodes"],2):
        for bits in product([0,1],repeat=7):
            active=set(seeds)
            trace=[]
            k=0
            for t,edges in stream:
                joins=defaultdict(list)
                for u,v in edges:
                    if bits[k] and u in active and v not in active:
                        joins[v].append(u)
                    k+=1
                if joins:
                    trace.append((t,tuple((v,tuple(joins[v])) for v in sorted(joins))))
                    active.update(joins)
            if trace:
                mass[tuple(trace)][seeds]+=P**sum(bits)*(1-P)**(len(bits)-sum(bits))
    max_error=0.;single=0
    for trace,distribution in mass.items():
        lab=dict(sources=next(iter(distribution)),trajectory=[dict(t=t,joins=joins) for t,joins in trace])
        row=oracle.evaluate(lab,check=True)
        if len(row["mandatory"])==2:
            assert len(distribution)==1
            continue
        single+=1
        a=row["mandatory"][0]
        probs={next(v for v in pair if v!=a):w/sum(distribution.values()) for pair,w in distribution.items()}
        expected=sum(sorted(probs.values(),reverse=True)[:11])
        chosen=sum(probs[v] for v in row["partners"])
        max_error=max(max_error,abs(row["mass"]-expected),abs(chosen-expected))
        prior=np.sqrt(1+np.arange(len(graph["nodes"])))
        weights={v:p*prior[v] for v,p in probs.items()}
        weighted=oracle.evaluate(lab,prior=prior)
        exact_weighted=sum(sorted(weights.values(),reverse=True)[:11])/sum(weights.values())
        max_error=max(max_error,abs(weighted["mass"]-exact_weighted))
    assert max_error<1e-10
    return dict(seed_pairs=120,live_masks=128,nonempty_traces=len(mass),single_emitter_traces=single,max_error=max_error)


def contact_counts(edges,ix,n):
    matrix=np.zeros((n,n),dtype=np.int32)
    for u,v in edges:
        matrix[ix[u],ix[v]]+=1
    return matrix


def audit(name,hybrid):
    local=name not in FILES
    folder=OUT/name
    selection=load((folder if local else OUT/"Dept2")/("hybrid_selection.json" if hybrid else "selection.json"))
    tag=str(selection["weight"]) if hybrid else "role_"+str(selection["temperature"])
    baseline="0.0" if hybrid else "dual_joint"
    prefix="hybrid_test" if hybrid else "test"
    public=read_rows(folder/(prefix+"_public.jsonl"))
    labels=read_rows(folder/(prefix+"_truth.jsonl"))
    pred=read_rows(folder/(prefix+"_predictions.jsonl"))
    diagnostics={} if hybrid else {r["id"]:r for r in read_rows(folder/"test_diagnostics.jsonl")}
    raw=company() if name=="company" else load_events(NEW if local else PREVIOUS.parents[1]/"data/raw"/FILES[name])
    gs,streams=make_windows(raw,name,.15,1.)
    graphs={g["window"]:g for g in gs}
    result={};contacts={};oracles={};budgets=[]
    assert len({o["id"] for o in public})==len(public)
    for o,y,r in zip(public,labels,pred,strict=True):
        key=o["window"];sid=o["id"]
        assert sid==y["id"]==r["id"] and key==y["window"]==r["window"]
        sources=set(y["sources"])
        assert simulate(streams[key],y["sources"],y["seed"])==y["trajectory"]
        assert mask(y["trajectory"])==(y["cut_batches"],o["observed"])
        direct,visible=roles(y)
        assert len(direct) in (1,2)
        legal=set(graphs[key]["nodes"])-{v for batch in o["observed"] for v in batch}
        assert sources<=legal
        hits={}
        for method,c in r["sets"].items():
            assert len(set(c))==len(c)==12 and set(c)<=legal
            hits[method]=sources<=set(c)
        groups=["all","silent" if len(direct)==1 else "two_emitters","dual_visible" if len(visible)==2 else "one_visible"]
        for group in groups:
            subset=result.setdefault(group,{})
            for view in ("total",key):
                counts=subset.setdefault(view,Counter())
                counts["n"]+=1
                for method,hit in hits.items():
                    counts[method]+=int(hit)
                counts["selected_gained"]+=int(hits[tag] and not hits[baseline])
                counts["selected_lost"]+=int(hits[baseline] and not hits[tag])
                counts["selected"]+=int(hits[tag])
        if hybrid:
            continue
        d=diagnostics[sid]
        oc=d["oracle"]
        assert set(oc["mandatory"])==direct
        result["all"]["total"]["oracle_expected"]+=oc["mass"]
        if len(direct)==1:
            s=next(iter(direct));b=next(iter(sources-direct))
            assert (b in oc["partners"])==oc["hit"]
            if key not in oracles:
                oracles[key]=CompleteTrace(graphs[key],streams[key])
            rich=oracles[key].evaluate(y)
            assert rich["mass"]==oc["mass"] and rich["partners"]==oc["partners"]
            budgets.append(dict(id=sid,k50=rich["k50"],k90=rich["k90"],n=rich["legal_partners"]))
            if key not in contacts:
                g=graphs[key];ix={v:j for j,v in enumerate(g["nodes"])}
                before=[(u,v) for t,u,v in raw if g["start"]-14*86400<=t<g["start"]]
                now=[e for _,edges in streams[key] for e in edges]
                contacts[key]=(ix,{nm:contact_counts(e,ix,len(ix)) for nm,e in (("history",before),("current",now))})
            ix,matrices=contacts[key]
            people=sorted(legal-{s})
            ids=np.array([ix[v] for v in people])
            tie=np.random.default_rng(sum((i+1)*ord(c) for i,c in enumerate(sid))).permutation(len(ids))
            simple={}
            for nm,m in matrices.items():
                # Approximate absence of emission using contacts outside the supplied anchor.
                cost=m.sum(axis=1)[ids]-m[ids,ix[s]]-m[ix[s],ids]
                indices=np.lexsort((tie,cost))[:11]
                simple[nm]=b in [people[i] for i in indices]
            for view in ("total",key):
                c=result["silent"][view]
                c["oracle_expected"]+=oc["mass"]
                c["oracle_hits"]+=int(oc["hit"])
                c["uniform_public"]+=min(1.,11/len(people))
                c["uniform_trace"]+=min(1.,11/oc["legal_partners"])
                for model,ok in {**d["known_anchor"],**simple}.items():
                    c["given_anchor_"+model]+=int(ok)
                for model,selected in r["sets"].items():
                    c["emitter_"+model]+=int(s in selected)
                    c["partner_"+model]+=int(b in selected)
    if budgets:
        result["oracle_partner_budget"]={tag:{"mean":float(np.mean([r[tag] for r in budgets])),
                 "median":float(np.median([r[tag] for r in budgets]))} for tag in ("k50","k90","n")}
        from build_dept3_pilot import save_rows
        save_rows(folder/"oracle_partner_budgets.jsonl",budgets)
    write_json(folder/(prefix+"_summary.json"),result)
    print(name,prefix,result["all"]["total"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("network",choices=["company","Dept2",*FILES,"toy"])
    parser.add_argument("--hybrid",action="store_true")
    a=parser.parse_args()
    if a.network=="toy":
        result=toy_check()
        write_json(OUT/"oracle_finite_check.json",result)
        print(result)
    else:
        audit(a.network,a.hybrid)
