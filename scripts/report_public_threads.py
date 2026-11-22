"""Independently check real early-speaker labels and aggregate lists by calendar month."""
from collections import Counter,defaultdict
from datetime import datetime,timezone

import numpy as np

import fgssi_suffix as fg
from factorial_fgssi import SEEDS
from public_thread_archive import LISTS,OUT
from public_thread_transfer import DAY,MONTHS,records,lock

METHODS = ("volume","contacts","FGSSI","person","links","full")


def validate(name,truth,public):
    messages,_ = records(name)
    # Walk every archived message to its root without the extraction pipeline's memoized forest.
    earliest = defaultdict(dict)
    for mid,(author,t,_,_) in messages.items():
        ancestor = mid
        visited = set()
        while ancestor in messages and ancestor not in visited:
            visited.add(ancestor)
            _,at,parent,reply = messages[ancestor]
            if parent is None:
                if not reply and ancestor in truth and 0<=t-at<14*DAY:
                    first = earliest[ancestor]
                    first[author] = min(t,first.get(author,t))
                break
            if parent not in messages or messages[parent][1]>at:
                break
            ancestor = parent
    contacts = set()
    for author,t,parent,_ in messages.values():
        if parent in messages and messages[parent][1]<=t and messages[parent][0]!=author:
            contacts.add((t,author,messages[parent][0]))
    rosters = {}
    for y,m in MONTHS:
        t0 = int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        rosters[f"{y}-{m:02d}"] = {v for t,a,b in contacts if t0-98*DAY<=t<t0 for v in (a,b)}
    for mid,r in truth.items():
        ordered = sorted(earliest[mid],key=lambda n:(earliest[mid][n],n))
        assert ordered[:2]==r["sources"] and messages[mid][0]==ordered[0]
        roster = rosters[r["window"]]
        assert [n in roster for n in ordered[:2]]==r["in_roster"]
        others = ordered[2:]
        groups = defaultdict(list)
        for v in others:
            groups[earliest[mid][v]].append(v)
        consumed = 0
        visible = []
        for _,group in sorted(groups.items()):
            consumed += len(group)
            if consumed>len(others)//2:
                known = sorted(set(group)&roster)
                if known:
                    visible.append(known)
        assert visible==public[mid]["observed"]
    return rosters


def table(subset,months):
    n = len(subset)
    if n==0:
        return dict(n=0)
    counts = np.zeros((len(METHODS),len(SEEDS),len(months),3),int)
    draws = Counter()
    chance = 0.
    for y,p in subset:
        key = y["window"]
        j = months.index(key)
        draws[key] += 1
        if not y["scored"]:
            continue
        if all(y["in_roster"]):
            chance += 12*11/(p["candidates"]*(p["candidates"]-1))
        for i,m in enumerate(METHODS):
            for k,s in enumerate(SEEDS):
                chosen = p["lists"][m if m in ("volume","contacts") else f"{m}/{s}"]
                a,b = [v in chosen for v in y["sources"]]
                counts[i,k,j] += [a and b,a,b]
    aggregate = {}
    for i,m in enumerate(METHODS):
        among = counts[i].sum(1)
        both,first,second = 100*among.mean(0)/n
        aggregate[m] = dict(both=both,first=first,second=second,both_counts=among[:,0].tolist())
    totals = np.array([draws[w] for w in months])
    rng = np.random.default_rng(20261121)
    W = len(months)
    blocks = ((rng.integers(W,size=(10000,(W+2)//3,1))+np.arange(3))%W).reshape(10000,-1)[:,:W]
    seeds = rng.integers(len(SEEDS),size=(10000,len(SEEDS),1))
    contrasts = {}
    for a,b in (("links","volume"),("links","person"),("links","contacts"),("full","links"),("full","FGSSI")):
        delta = counts[METHODS.index(a),:,:,0]-counts[METHODS.index(b),:,:,0]
        den = totals[blocks].sum(1)
        valid = den>0
        boot = (100*delta[seeds,blocks[:,None,:]].sum(2).mean(1))[valid]/den[valid]
        contrasts[a+"-"+b] = dict(pp=100*delta.sum(1).mean()/n,ci95=np.quantile(boot,[.025,.975]).tolist())
    return dict(n=n,expected_random_both=100*chance/n,methods=aggregate,contrasts=contrasts,months=months,
                eligible_per_month=totals.tolist(),both_by_method_seed_month={m:counts[i,:,:,0].tolist() for i,m in enumerate(METHODS)})


def main():
    assert fg.load(OUT/"protocol.json")==lock()
    results = {}
    identifiers = []
    for name in LISTS:
        folder = OUT/name
        files = [folder/(f+".jsonl") for f in ("truth","public","predictions")]
        data = [fg.read_rows(p) for p in files]
        truth,public,pred = [{r["id"]:r for r in rows} for rows in data]
        assert all(len(rows)==len(truth) for rows in data) and set(truth)==set(pred)==set(public)
        roster = validate(name,truth,public)
        for mid,y in truth.items():
            seen = {v for b in public[mid]["observed"] for v in b}
            legal = roster[y["window"]]-seen
            p = pred[mid]
            assert p["window"]==y["window"] and p["candidates"]==len(legal)
            assert bool(p["lists"])==y["scored"]==(len(seen)>=3 and len(legal)>=12)
            assert all(len(g)==len(set(g))==12 and set(g)<=legal for g in p["lists"].values())
        flow = fg.load(folder/"flow.json")
        months = [w["window"] for w in flow["months"]]
        all_rows = [(y,pred[mid]) for mid,y in truth.items()]
        assert len(all_rows)==sum(w.get("eligible",0) for w in flow["months"])
        groups = dict(all=all_rows,scored=[r for r in all_rows if r[0]["scored"]],
                      reachable=[r for r in all_rows if r[0]["scored"] and all(r[0]["in_roster"])])
        results[name] = dict(tables={tag:table(group,months) for tag,group in groups.items()},
                             roots=sum(w["roots"] for w in flow["months"]),both_in_history=sum(all(y["in_roster"]) for y in truth.values()),
                             hashes={p.name:fg.digest(p) for p in files},metadata_hash=fg.digest(OUT/(name+"-headers.jsonl.gz")))
        identifiers.append(set(truth))
    results["shared_root_message_ids"] = len(set.intersection(*identifiers))
    results["archive_manifest_sha256"] = fg.digest(OUT/"archive.json")
    fg.write_json(OUT/"summary.json",results)
    for name in LISTS:
        r = results[name]
        t = r["tables"]["all"]
        print(name,r["roots"],[(k,v["n"]) for k,v in r["tables"].items()],{m:round(v["both"],2) for m,v in t["methods"].items()},t["contrasts"]["links-volume"])


if __name__=="__main__":
    main()
