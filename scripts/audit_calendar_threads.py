"""Replay later lists and independently check the pair objective and the activity control."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import fgssi_suffix as fg
from calendar_thread_calibration import DEST,TEST,examples,risk
from public_thread_archive import LISTS,OUT
from public_thread_transfer import DAY,MONTHS,records


def volume_check(name):
    old,_ = records(name)
    truth = fg.read_rows(OUT/name/"truth.jsonl")
    obs = {r["id"]:r for r in fg.read_rows(OUT/name/"public.jsonl")}
    previous = {r["id"]:r for r in fg.read_rows(OUT/name/"predictions.jsonl")}
    unrestricted_hits = 0
    for year,month in MONTHS:
        start = int(datetime(year,month,1,tzinfo=timezone.utc).timestamp())
        recent = Counter(a for a,t,_,_ in old.values() if start-28*DAY<=t<start)
        roster = set()
        for a,t,p,_ in old.values():
            if start-98*DAY<=t<start and p in old and a!=old[p][0] and old[p][1]<=t:
                roster.update([a,old[p][0]])
        for r in truth:
            if r["window"]!=f"{year}-{month:02d}":
                continue
            hidden = {n for b in obs[r["id"]]["observed"] for n in b}
            legal = sorted(roster-hidden)
            order = fg.tie_order(r["id"],len(legal))
            chosen = sorted(range(len(legal)),key=lambda i:(-recent[legal[i]],order[i]))[:12]
            names = [legal[i] for i in chosen]
            if r["scored"]:
                assert names==previous[r["id"]]["lists"]["volume"]
            unrestricted_hits += set(r["sources"])<=set(names)
    return unrestricted_hits


def audit():
    lock = fg.load(DEST/"lock.json")
    assert lock["script"]==fg.digest(Path(__file__).with_name("calendar_thread_calibration.py"))
    results = fg.load(DEST/"summary.json")
    output = {}
    for name in LISTS:
        archived = {r["id"]:r for r in fg.read_rows(DEST/(name+"-predictions.jsonl"))}
        old = {r["id"] for r in fg.read_rows(OUT/name/"truth.jsonl")}
        assert not old.intersection(archived)
        totals = Counter()
        for r in examples(name,True):
            saved = archived[r["id"]]
            assert saved["sources"]==r["sources"] and saved["ready"]==r["ready"]
            tie = fg.tie_order(r["id"],len(r["nodes"]))
            for m,lst in saved["lists"].items():
                theta = np.array([1.]) if m=="volume" else np.asarray(lock["models"][m]["theta"])
                scores = r["x"][:,:len(theta)]@theta
                order = sorted(range(len(scores)),key=lambda i:(-scores[i],tie[i]))
                assert lst==([r["nodes"][i] for i in order[:12]] if r["ready"] else [])
                totals[m] += set(r["sources"])<=set(lst)
        assert all(results[name]["methods"][k]["hits"]==v for k,v in totals.items())
        output[name] = dict(months=TEST,verified_followup=len(archived),distinct_from_earlier=True,
                            historical_volume_even_when_suffix_short=volume_check(name))
    x = np.array([[.3,.9],[0.,1.1],[.1,1.5],[2.,.7],[.2,.1]])
    theta = np.array([.2,-.8])
    z = x@theta
    i,j = np.triu_indices(len(z),1)
    expected = logsumexp(z[i]+z[j])-z[0]-z[2]
    val,g = risk(theta,[(x,[0,2])],0)
    num = [(risk(theta+a,[(x,[0,2])],0)[0]-risk(theta-a,[(x,[0,2])],0)[0])/2e-6 for a in np.eye(2)*1e-6]
    assert np.isclose(expected,val) and np.allclose(num,g)
    fg.write_json(DEST/"audit.json",output)


if __name__=="__main__":
    audit()
