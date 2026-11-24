"""Shared public scorer and paired draws for the frozen propagation and k-source experiments."""
import random
from collections import defaultdict

from closure_common import *
from build_dept3_pilot import mask,simulate
from source_prior_shift import windows
from factorial_history import StaticWindow


def test_windows(name):
    raw,windows0=windows(name,False)
    if name=="Enron":
        windows0=[windows0[i] for i in np.linspace(0,len(windows0)-1,8).round().astype(int)]
    return raw,windows0


def selection(g,raw,k,rng,biased):
    if not biased:
        return rng.sample(g["nodes"],k)
    w=activity(g,raw)
    return pick(g["nodes"],w,k,rng)


def pick(nodes,w,k,rng):
    # Product law on distinct sets by iid sampling conditional on all draws being distinct.
    while True:
        ids=rng.choices(nodes,weights=w,k=k)
        if len(set(ids))==k:
            return ids


def replay(g,stream,sources,seed,rule):
    if rule=="SI":
        return simulate(stream,sources,seed)
    life={"SIR48":2*DAY,"SIR72":3*DAY}.get(rule,10**15)
    starts={v:g["start"] for v in sources}
    trace=[]
    rng=random.Random(seed)
    exo=random.Random(seed^983165)
    # Independent Bernoulli 0.02 per non-source per 14-day window;
    # conditional arrival uniform over actual contact timestamps.
    spontaneous=defaultdict(list)
    if rule=="noise" and stream:
        for v in sorted(set(g["nodes"])-set(sources)):
            if exo.random()<.02:
                spontaneous[exo.randrange(len(stream))].append(v)
    for j,(t,batch) in enumerate(stream):
        parents=defaultdict(list)
        for u,v in batch:
            if u in starts and t-starts[u]<life and v not in starts and rng.random()<.30:
                parents[v].append(u)
        for v in spontaneous.get(j,[]):
            if v not in starts:
                parents.setdefault(v,[])
        if parents:
            trace.append(dict(t=t,joins=[[v,sorted(pp)] for v,pp in sorted(parents.items())]))
            starts.update({v:t for v in parents})
    return trace


def silent_count(gold,tr):
    parents={u for batch in tr for _,pp in batch["joins"] for u in pp}
    return len(set(gold)-parents)


class Scorer:
    def __init__(self,g,raw,models):
        self.models=models
        self.hist=PublicWindow(g,raw)
        self.static=StaticWindow(g) if "static" in models else None
        self.vol=Counter(a for t,a,b in raw if g["start"]-28*DAY<=t<g["start"])

    def predict(self,mid,window,groups):
        # The predictor is never passed gold or k. The additive encoder consumes only the public groups.
        event=frames(dict(id=mid,window=window,observed=groups))
        (nodes,_,_),(x,_,_)=self.hist.design(event)
        tie=fg.tie_order(mid,len(nodes))
        scores={}
        for m,models in self.models.items():
            if m=="static":
                (ns,_,_),(xx,_,_)=self.static.design(event)
                assert ns==nodes
            else:
                xx=features(x,m)
            scores[m]=logits(models,xx)
        scores["volume"]=[np.array([self.vol[n] for n in nodes])]*len(SEEDS)
        return nodes,tie,scores


def score_lists(nodes,tie,scores):
    return {m:[[nodes[i] for i in fg.sorted_nodes(v,tie)[:12]] for v in vec] for m,vec in scores.items()}


def hits(lists,gold,recall=False):
    return {m:[len(set(ls)&set(gold))/len(gold) if recall else int(set(gold)<=set(ls)) for ls in values] for m,values in lists.items()}


def check_lists(row):
    for sets in row["lists"].values():
        for ls in sets:
            assert len(ls)==len(set(ls))==12
            assert set(ls).isdisjoint(v for b in row["observed"] for v in b)

