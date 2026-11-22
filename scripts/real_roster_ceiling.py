"""Separate early-participant identifiability from shortlist accuracy in real archives."""
from bisect import bisect_left
from collections import Counter,defaultdict
from datetime import datetime,timezone

import numpy as np

import fgssi_suffix as fg
from historical_thread_roles import OUT as ROLES,read_headers
from public_thread_archive import LISTS,OUT as PYTHON
from public_thread_transfer import DAY,marked_thread,thread_forest

OUT = PYTHON.parent/"real_roster_ceiling"
STAGES = ("edge98_gated","edge98","post98","archive_at_month","archive_safe_clock","archive_at_root")


def run(name):
    msg = read_headers(name)
    trees,edges = thread_forest(msg)
    if name in LISTS:
        parent = PYTHON/name
        base = fg.read_rows(parent/"predictions.jsonl")
    else:
        parent = ROLES/name
        base = fg.read_rows(parent/"predictions.jsonl")
    times_by_author = defaultdict(list)
    for a,t,_,_ in msg.values():
        times_by_author[a].append(t)
    for times in times_by_author.values():
        times.sort()
    first = {n:t[0] for n,t in times_by_author.items()}
    summary = {}
    rows = []
    for month in sorted({r["window"] for r in base}):
        start = int(datetime.strptime(month,"%Y-%m").replace(tzinfo=timezone.utc).timestamp())
        cutoff = start-98*DAY
        edge98 = {v for t,u,v in edges if cutoff<=t<start for v in (u,v)}
        posters = {a for a,t in times_by_author.items() if bisect_left(t,start)>bisect_left(t,cutoff)}
        archive = {n for n,t in first.items() if t<start}
        assert edge98<=archive and posters<=archive
        # A recipient may have last posted before the 98-day cutoff; keep all existing graph members.
        extended = edge98|posters
        for old in (r for r in base if r["window"]==month):
            mid = old["id"]
            _,(sources,groups) = marked_thread(mid,trees[mid],msg)
            seen = {n for b in groups for n in b}
            t0 = msg[mid][1]
            arrivals = {}
            for t,_,v in trees[mid]:
                if v in seen and t0<=t<t0+14*DAY:
                    arrivals[v] = min(t,arrivals.get(v,t))
            # An additional observation setting with public suffix timestamps and a known 14-day horizon.
            safe_time = max(start,max(arrivals.values())-14*DAY)
            assert start<=safe_time<=t0
            safe_archive = {n for n,t in first.items() if t<safe_time}
            all_before_root = {n for n,t in first.items() if t<t0}
            rosters = [edge98,edge98,extended,archive,safe_archive,all_before_root]
            gates = [len(seen&edge98)>=3,True,True,True,True,True]
            scores = []
            for m,roster,gate in zip(STAGES,rosters,gates,strict=True):
                nodes = sorted(roster-seen)
                can_issue = gate and len(nodes)>=12
                available = [n in roster for n in sources]
                end = {"archive_at_root":t0,"archive_safe_clock":safe_time}.get(m,start)
                if can_issue:
                    volume = np.array([bisect_left(times_by_author[v],end)-bisect_left(times_by_author[v],end-28*DAY) for v in nodes])
                    tie = fg.tie_order(mid,len(nodes))
                    lst = [nodes[i] for i in fg.sorted_nodes(volume,tie)[:12]]
                else:
                    lst = []
                if m=="edge98_gated":
                    assert lst==old["lists"].get("volume",[])
                role_hits = [n in lst for n in sources]
                scores.append(dict(method=m,candidates=len(nodes),issued=can_issue,available=available,
                                   reachable=all(available) and can_issue,hit=all(role_hits),role_hits=role_hits))
            assert all(int(b["reachable"])>=int(a["reachable"]) for a,b in zip(scores,scores[1:]))
            rows.append(dict(id=mid,month=month,scores=scores))
    for j,stage in enumerate(STAGES):
        records = [r["scores"][j] for r in rows]
        summary[stage] = dict(n=len(rows),both_reachable=sum(r["reachable"] for r in records),both_hit=sum(r["hit"] for r in records),
            first_seen=sum(r["available"][0] for r in records),second_seen=sum(r["available"][1] for r in records),
            issued=sum(r["issued"] for r in records),median_candidates=float(np.median([r["candidates"] for r in records])))
    assert len(rows)==len(base)
    fg.save_rows(OUT/(name+".jsonl"),rows)
    return summary


if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    result = {n:run(n) for n in (*LISTS,"gcc","ffmpeg-devel")}
    fg.write_json(OUT/"summary.json",result)
    for n,data in result.items():
        print(n,[(m,v["both_reachable"],v["both_hit"]) for m,v in data.items()])
