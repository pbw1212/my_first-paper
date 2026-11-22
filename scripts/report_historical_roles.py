"""Verify head-of-thread identities and paired differences across unseen communities."""
from collections import defaultdict
from datetime import datetime,timezone

import numpy as np

import fgssi_suffix as fg
from historical_thread_roles import DAY,OUT,TARGETS,TARGET_MONTHS,choose,coverage,read_headers,signature
from public_thread_transfer import marked_thread

METHODS = ("volume","role_history","direct","retrieval","retrieval_marginal")


def verify(name,rows):
    msg = read_headers(name)
    found = defaultdict(list)
    starts = {r["id"] for r in rows}
    for mid,(a,t,p,_) in msg.items():
        cur = mid
        seen = {mid}
        while p is not None and p in msg and p not in seen:
            if msg[p][1]>msg[cur][1]:
                break
            seen.add(p)
            cur = p
            p = msg[cur][2]
        if cur in starts:
            found[cur].append((t,mid,a))
    contacts = [(t,a,msg[p][0]) for a,t,p,_ in msg.values() if p in msg and msg[p][1]<=t and msg[p][0]!=a]
    for y,m in TARGET_MONTHS:
        start = int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        key = f"{y}-{m:02d}"
        roster = {v for t,a,b in contacts if start-98*DAY<=t<start for v in (a,b)}
        volume = defaultdict(int)
        for a,t,_,_ in msg.values():
            if start-28*DAY<=t<start:
                volume[a] += 1
        for r in (item for item in rows if item["window"]==key):
            reason,(sources,groups) = marked_thread(r["id"],found[r["id"]],msg)
            assert reason=="eligible" and sources==r["sources"]
            visible = {v for group in groups for v in group}&roster
            legal = sorted(roster-visible)
            assert (len(visible)>=3 and len(legal)>=12)==r["ready"]==bool(r["lists"])
            if r["ready"]:
                assert all(len(set(v))==len(v)==12 and set(v)<=set(legal) for v in r["lists"].values())
                tie = fg.tie_order(r["id"],len(legal))
                order = sorted(range(len(legal)),key=lambda i:(-volume[legal[i]],tie[i]))
                assert [legal[i] for i in order[:12]]==r["lists"]["volume"]


def audit_decoder():
    rng = np.random.default_rng(8144)
    for n in (12,13,30,200):
        for _ in range(10):
            p,q = (rng.dirichlet(np.ones(n)*.4) for _ in range(2))
            sel,margin = choose(p,q,rng.permutation(n))
            mat = np.outer(p,q)
            np.fill_diagonal(mat,0.)
            val = mat[np.ix_(sel,sel)].sum()
            assert np.isclose(val,coverage(p,q,sel)) and val+1e-12>=coverage(p,q,margin)
            for i in range(12):
                for j in set(range(n))-set(sel):
                    replace = sel.copy()
                    replace[i] = j
                    assert val+1e-12>=coverage(p,q,replace)


def main():
    assert fg.load(OUT/"protocol.json")["hashes"]==signature()
    audit_decoder()
    out = {}
    for name in TARGETS:
        path = OUT/name/"predictions.jsonl"
        rows = fg.read_rows(path)
        verify(name,rows)
        flow = fg.load(OUT/name/"flow.json")
        n = len(rows)
        assert sum(d["eligible"] if "eligible" in d else 0 for d in flow)==n==len({r["id"] for r in rows})
        months = [v["window"] for v in flow]
        totals = np.array([v.get("eligible",0) for v in flow])
        table = dict(n=n,roots=sum(v.get("roots",0) for v in flow),scored=sum(r["ready"] for r in rows),
                     reachable=sum(v.get("two_reachable",0) for v in flow),methods={},contrasts={},prediction_sha256=fg.digest(path))
        results = {}
        for m in METHODS:
            both,first,second = [],[],[]
            for r in rows:
                names = r["lists"].get(m,[])
                a,b = [x in names for x in r["sources"]]
                first.append(a)
                second.append(b)
                both.append(a and b)
            results[m] = np.array(both,dtype=int)
            table["methods"][m] = dict(both=sum(both),first=sum(first),second=sum(second),pct=100*sum(both)/n)
        rng = np.random.default_rng(20261122)
        blocks = ((rng.integers(24,size=(10000,8,1))+np.arange(3))%24).reshape(10000,24)
        for a,b in (("retrieval","volume"),("retrieval","direct"),("direct","role_history"),("retrieval","retrieval_marginal")):
            delta = results[a]-results[b]
            diffs = np.array([sum(x for r,x in zip(rows,delta,strict=True) if r["window"]==w) for w in months])
            den = totals[blocks].sum(1)
            keep = den>0
            interval = np.quantile(100*diffs[blocks].sum(1)[keep]/den[keep],[.025,.975])
            table["contrasts"][a+"-"+b] = dict(pp=100*delta.mean(),gains=int((delta>0).sum()),losses=int((delta<0).sum()),
                                               monthly=diffs.tolist(),ci95=interval.tolist())
        out[name] = table
        print(name,n,table["methods"],table["contrasts"]["retrieval-volume"])
    fg.write_json(OUT/"summary.json",out)


if __name__=="__main__":
    main()
