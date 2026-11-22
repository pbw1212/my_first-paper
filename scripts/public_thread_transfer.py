"""Frozen transfer to the first two distinct archived speakers in real discussion trees."""
import argparse
import gzip
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from build_dept3_pilot import mask
from distilled_source_pairs import PairNet,PublicWindow
from evaluate_factorial_history import fitted
from factorial_fgssi import SEEDS,weights
from factorial_history import folder_for
from five_frame_pairs import frames
from history_components import shortlist
from public_thread_archive import LISTS,OUT,ROOT
from recent_member_evidence import CONTEXT,LINKS,PERSON,OUT as FIT

DAY = 86400
MONTHS = [(y,m) for y in (2017,2018) for m in range(1,13)]
PLAN = dict(lists=list(LISTS),months=MONTHS,history_days=98,followup_days=14,
            candidates="endpoints of distinct nonself, resolved replies in 98 days before calendar month",
            edges="reply author to immediate parent author; distinct (second,sender,recipient)",
            population="verified roots in calendar period and 14-day thread; first two distinct addresses, unique second arrival, ordinary mask on later first arrivals",
            observations="filter suffix to historical roster after masking; abstain if fewer than 3 retained; out-of-roster gold counted as miss",
            primary="both in 12, all mask-eligible threads, frozen Dept2 links minus 28-day message volume; separate by list",
            secondary=["full","person","FGSSI","28-day outgoing contact count to visible members","uniform expected","root","second"],
            seeds=list(SEEDS),bootstrap="10000 paired draws over training seeds and circular three-month blocks")


def lock():
    prefix = FIT/"Dept2"
    paths = [Path(__file__), ROOT/"scripts/public_thread_archive.py", ROOT/"scripts/fgssi_suffix.py",ROOT/"scripts/ordinary_source_roles.py",
             ROOT/"scripts/distilled_source_pairs.py",ROOT/"scripts/temporal_pair_evidence.py",ROOT/"scripts/run_dept3_candidates.py",
             ROOT/"scripts/build_dept3_pilot.py",ROOT/"scripts/partial_source_identification.py",ROOT/"scripts/five_frame_pairs.py",
             ROOT/"scripts/history_components.py",ROOT/"scripts/recent_member_evidence.py",ROOT/"scripts/evaluate_factorial_history.py",
             fg.UP/"model.py",fg.UP/"args.py"]
    for s in SEEDS:
        paths.extend([folder_for("Dept2","temporal")/f"add{s}.pt",folder_for("Dept2","temporal")/"scale.npz",weights("Dept2",s)])
        for m in ("links","person"):
            paths.extend([prefix/f"{m}-{s}.pt",prefix/(m+".npz")])
    cfg = dict(plan=PLAN,hashes={p.resolve().relative_to(ROOT).as_posix():fg.digest(p) for p in paths})
    return json.loads(json.dumps(cfg))


def records(name):
    with gzip.open(OUT/(name+"-headers.jsonl.gz"),"rt",encoding="utf8") as f:
        raw = [tuple(json.loads(line)) for line in f]
    unique = {}
    conflicts = set()
    for mid,a,t,p,reply in raw:
        value = (a,t,p,reply)
        if mid in unique and unique[mid]!=value:
            conflicts.add(mid)
        unique[mid] = value
    for mid in conflicts:
        del unique[mid]
    return unique,dict(parsed=len(raw),unique=len(unique),conflicting_ids=len(conflicts))


def thread_forest(messages):
    roots = {}
    for mid in messages:
        path = []
        cur = mid
        root = None
        while cur not in roots:
            path.append(cur)
            _,t,p,reply = messages[cur]
            if not p:
                if not reply:
                    root = cur
                break
            if p not in messages or p in path or messages[p][1]>t:
                break
            cur = p
        else:
            root = roots[cur]
        for member in path:
            roots[member] = root
    threads = defaultdict(list)
    contacts = set()
    for mid,(a,t,p,_) in messages.items():
        if roots[mid]:
            threads[roots[mid]].append((t,mid,a))
        if p in messages:
            parent_a,parent_t,_,_ = messages[p]
            if parent_t<=t and a!=parent_a:
                contacts.add((t,a,parent_a))
    return threads,sorted(contacts)


def marked_thread(root,thread,messages):
    author,t0,_,_ = messages[root]
    first = {}
    for t,_,a in thread:
        if t0<=t<t0+14*DAY:
            first[a] = min(t,first.get(a,t))
    others = sorted((t,a) for a,t in first.items() if a!=author)
    if not others:
        return "no_second",None
    if others[0][0]==t0 or (len(others)>1 and others[0][0]==others[1][0]):
        return "ambiguous_early_time",None
    # Only first appearances after the first two enter the original masking rule.
    tr = [dict(joins=[[a,[]] for _,a in grp]) for _,grp in groupby(others[1:],key=lambda item:item[0])]
    masked = mask(tr)
    if not masked:
        return "insufficient_suffix",None
    return "eligible",([author,others[0][1]],masked[1])


def models():
    nets = {("full",s):fitted("Dept2",s,"C") for s in SEEDS}
    for m in ("person","links"):
        with np.load(FIT/"Dept2"/(m+".npz")) as scale:
            for s in SEEDS:
                net = PairNet(64,False).eval()
                net.load_state_dict(torch.load(FIT/"Dept2"/f"{m}-{s}.pt",weights_only=True))
                nets[m,s] = net,scale["mean"],scale["std"]
    fgs = {}
    for s in SEEDS:
        fgnet = fg.FGSSI().eval()
        fgnet.load_state_dict(torch.load(weights("Dept2",s),weights_only=True))
        fgs[s] = fgnet
    return nets,fgs


def score_month(graph,history,events,messages,nets,fgs):
    hist = PublicWindow(graph,history)
    geo = fg.Geometry(graph)
    start = graph["start"]
    volume = Counter(a for a,t,_,_ in messages.values() if start-28*DAY<=t<start)
    past = Counter((a,b) for t,a,b in history if t>=start-28*DAY)
    for offset in range(0,len(events),4):
        batch = events[offset:offset+4]
        inputs = geo.inputs(batch)
        scores = {}
        with torch.no_grad():
            for s,net in fgs.items():
                logits,_ = net(*inputs)
                scores[s] = (logits[:,:,1]-logits[:,:,0]).numpy()
        for row,e in enumerate(batch):
            (nodes,_,_),(full,_,_) = hist.design(frames(e))
            tie = fg.tie_order(e["id"],len(nodes))
            idx = [geo.index[n] for n in nodes]
            vis = {v for b in e["observed"] for v in b}
            lists = {}
            def top(val):
                return [nodes[j] for j in fg.sorted_nodes(np.asarray(val),tie)[:12]]
            lists["volume"] = top([volume[n] for n in nodes])
            lists["contacts"] = top([sum(past[n,v] for v in vis) for n in nodes])
            for m,cols in (("full",None),("links",PERSON+LINKS),("person",PERSON)):
                x = full
                if cols is not None:
                    selected = [*cols,*(c+29 for c in cols),*CONTEXT]
                    x = np.zeros_like(full)
                    x[:,selected] = full[:,selected]
                for s in SEEDS:
                    lists[f"{m}/{s}"] = shortlist(nets[m,s],x,nodes,tie)
            for s in SEEDS:
                lists[f"FGSSI/{s}"] = top(scores[s][row,idx])
            assert all(len(set(v))==12 and set(v)<=set(nodes) for v in lists.values())
            yield dict(id=e["id"],window=e["window"],lists=lists,candidates=len(nodes))


def run(name):
    assert fg.load(OUT/"protocol.json")==lock()
    msg,info = records(name)
    threads,contacts = thread_forest(msg)
    info.update(resolved_messages=sum(map(len,threads.values())),distinct_contacts=len(contacts))
    nets,fgs = models()
    flow,public,truth,pred = [],[],[],[]
    for y,m in MONTHS:
        start = int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        end = int(datetime(y+(m==12),m%12+1,1,tzinfo=timezone.utc).timestamp())
        key = f"{y}-{m:02d}"
        history = [(t,a,b) for t,a,b in contacts if start-98*DAY<=t<start]
        edges = Counter((a,b) for _,a,b in history)
        roster = {n for pair in edges for n in pair}
        graph = dict(start=start,window=key,nodes=sorted(roster),edges=[(u,v,c) for (u,v),c in sorted(edges.items())])
        ct = Counter()
        todo = []
        for root,thread in sorted(threads.items()):
            if not start<=msg[root][1]<end:
                continue
            ct["roots"] += 1
            reason,data = marked_thread(root,thread,msg)
            ct[reason] += 1
            if data is None:
                continue
            early,late = data
            observed = [[a for a in b if a in roster] for b in late]
            observed = [b for b in observed if b]
            seen = {v for b in observed for v in b}
            assert not seen.intersection(early) and len(set(early))==2
            e = dict(id=root,window=key,observed=observed)
            can_score = len(seen)>=3 and len(roster-seen)>=12
            ct["scored"] += can_score
            ct["both_in_history"] += set(early)<=roster
            public.append(e)
            truth.append(dict(id=root,window=key,sources=early,in_roster=[a in roster for a in early],scored=can_score))
            if can_score:
                todo.append(e)
            else:
                pred.append(dict(id=root,window=key,lists={},candidates=len(roster-seen)))
        if todo:
            pred.extend(score_month(graph,history,todo,msg,nets,fgs))
        flow.append(dict(window=key,roster=len(roster),**ct))
    assert len(public)==len(truth)==len(pred)==len({r["id"] for r in pred})
    folder = OUT/name
    folder.mkdir(exist_ok=True)
    for file,rows in (("public",public),("truth",truth),("predictions",pred)):
        fg.save_rows(folder/(file+".jsonl"),rows)
    fg.write_json(folder/"flow.json",dict(headers=info,months=flow))


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze",*LISTS))
    action = p.parse_args().action
    if action=="freeze":
        OUT.mkdir(parents=True,exist_ok=True)
        path = OUT/"protocol.json"
        if not path.exists():
            fg.write_json(path,lock())
        assert fg.load(path)==lock()
    else:
        torch.set_num_threads(1)
        with threadpool_limits(1):
            run(action)
