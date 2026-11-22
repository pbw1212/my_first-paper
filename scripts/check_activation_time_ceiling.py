"""Independent contact replay and subset enumeration for the complete activation-time oracle."""
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.special import logsumexp

import fgssi_suffix as fg
from activation_time_ceiling import (ActivationTimes, BASE, LOG_Q, NETWORKS, OUT, REGIMES,
                                     activity,exact_list,protocol,simulate,times_only,windows)


def direct(stream,joins,pair,w,law,index):
    active=set(pair)
    by_time=defaultdict(set)
    for t,v in joins:
        by_time[t].add(v)
    l=0.
    for t,batch in stream:
        counts=Counter(v for u,v in batch if u in active and v not in active)
        now=by_time.get(t,())
        if any(counts[v]==0 for v in now):
            return -np.inf
        for target,num in counts.items():
            l+=np.log(-np.expm1(LOG_Q*num)) if target in now else LOG_Q*num
        active.update(now)
    wa,wb=(w[index[v]] for v in pair)
    return l+{"flat":0.,"one":np.log(wa+wb),"both":np.log(wa*wb)}[law]


def toy_cases():
    rng=np.random.default_rng(823243)
    checked=0
    error=0.
    multigroup=0
    for case in range(55):
        n=int(rng.integers(7,13))
        nodes=list(range(n))
        true=sorted(rng.choice(n,2,replace=False).tolist())
        stream=[]
        for t in range(8):
            contacts=[tuple(map(int,rng.choice(n,2,replace=False))) for _ in range(11)]
            contacts.extend([contacts[0]]*2)
            stream.append((t,contacts))
        tr=simulate(stream,true,case)
        joins=times_only(tr)
        if not joins:
            continue
        w=1+rng.integers(25,size=n)
        world=ActivationTimes(dict(nodes=nodes),stream)
        for law in REGIMES:
            a,b,l,cover,unseen=world.distribution(joins,w,law)
            possible=list(combinations(unseen,2))
            ll=np.array([direct(stream,joins,pair,w,law,world.index) for pair in possible])
            ll-=logsumexp(ll)
            table=dict(zip(zip(a,b),l))
            assert set(table)=={pair for pair,s in zip(possible,ll) if np.isfinite(s)}
            for pair,s in zip(possible,ll):
                if pair in table:
                    error=max(error,abs(table[pair]-s))
            selected,high=exact_list(a,b,np.exp(l),cover,unseen,n,3)
            best=max(sum(np.exp(s) for pair,s in zip(possible,ll) if set(pair)<=set(c))
                     for c in combinations(unseen,min(len(unseen),3)))
            assert np.isclose(high,best)
            checked+=1
            multigroup+=len(cover)>1
    assert error<1e-9 and multigroup>0
    n,k=19,4
    a,b=np.triu_indices(n,1)
    p=rng.uniform(size=len(a))
    p/=p.sum()
    picked,value=exact_list(a,b,p,np.arange(n-1),np.arange(n),n,k)
    matrix=np.zeros((n,n))
    matrix[a,b]=p
    sets=np.array(list(combinations(range(n),k)))
    optimum=matrix[sets[:,:,None],sets[:,None,:]].sum((1,2)).max()
    assert np.isclose(value,optimum,atol=1e-10)
    return dict(exhaustive_distributions=checked,multi_member_covers=multigroup,
                max_log_posterior_error=error,milp_subsets_checked=len(sets))


def audit(net):
    raw,gs=windows(net,False)
    results=fg.read_rows(OUT/net/"events.jsonl")
    table={r["id"]:r for r in results}
    records=fg.read_rows(BASE/net/"test/events.jsonl")
    assert len(results)==len(table)==len(records)
    rng=np.random.default_rng(78934)
    dense_checks=pairs_checked=0
    err=0.
    for g,stream in gs:
        events=[r for r in records if r["window"]==g["window"]]
        if not events:
            continue
        world=ActivationTimes(g,stream)
        w=activity(g,raw)
        for law in REGIMES:
            cases=[r for r in events if r["regime"]==law]
            if not cases:
                continue
            r=cases[0]
            joins=times_only(simulate(stream,r["sources"],r["seed"]))
            a,b,l,_,unseen=world.distribution(joins,w,law)
            values=dict(zip(zip(a,b),l))
            true=tuple(sorted(world.index[v] for v in r["sources"]))
            reference=direct(stream,joins,r["sources"],w,law,world.index)
            assert np.isclose(l[np.flatnonzero((a==true[0])&(b==true[1]))[0]],table[r["id"]]["true_log_prob"])
            some=rng.choice(len(a),size=min(len(a),3),replace=False)
            pairs=[(a[i],b[i]) for i in some]+[tuple(sorted(rng.choice(unseen,2,replace=False))) for _ in range(2)]
            for pair in pairs:
                name=[world.nodes[i] for i in pair]
                ll=direct(stream,joins,name,w,law,world.index)
                assert np.isfinite(ll)==(pair in values)
                if np.isfinite(ll):
                    err=max(err,abs((ll-reference)-(values[pair]-values[true])))
                pairs_checked+=1
            dense_checks+=1
    assert err<1e-7
    return dict(events=len(table),window_law_direct_checks=dense_checks,direct_pair_checks=pairs_checked,
                max_log_odds_error=err,events_sha256=fg.digest(OUT/net/"events.jsonl"))


if __name__=="__main__":
    assert fg.load(OUT/"protocol.json")==protocol()
    report=dict(synthetic=toy_cases(),networks={n:audit(n) for n in NETWORKS})
    fg.write_json(OUT/"verification.json",report)
    print(report,flush=True)
