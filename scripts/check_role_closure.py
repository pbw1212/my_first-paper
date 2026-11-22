"""Check the final cohort counts, replay the independent community, and audit neutralization."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

import finish_role_experiments as exp


def check(name,summary):
    msg,months,_=exp.cohort(name)
    forest,contacts=exp.re.thread_forest(msg)
    early={}
    monthly=defaultdict(list)
    docs=[]
    for root,thread in forest.items():
        pair=exp.reference(root,thread,msg)
        if pair:
            gold,batches=pair
            t=msg[root][1]
            docs.append((t,gold,{n for grp in batches for n in grp}))
            early[root]=pair
            monthly[datetime.fromtimestamp(t,timezone.utc).strftime("%Y-%m")].append(root)
    cfg=exp.fg.load(exp.re.OUT/"protocol.json")["models"]
    rows=exp.fg.read_rows(exp.OUT/(name+".jsonl"))
    data={z["id"]:z for z in rows}
    assert len(rows)==len(data)==summary["n"]
    visited=set()
    lists=0
    for y,m in months:
        key=f"{y}-{m:02d}"
        start=int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        roster={v for t,a,b in contacts if start-98*exp.re.DAY<=t<start for v in (a,b)}
        cnt=Counter((a,b) for t,a,b in contacts if start-28*exp.re.DAY<=t<start)
        posts=[Counter(a for a,t,_,_ in msg.values() if start-d*exp.re.DAY<=t<start) for d in (28,98)]
        previous=[d for d in docs if start-98*exp.re.DAY<=d[0]<start-14*exp.re.DAY]
        roles=[Counter(d[1][j] for d in previous) for j in (0,1)]
        freq=Counter(v for _,_,s in previous for v in s)
        occurrence=[Counter((gold[j],v) for _,gold,s in previous for v in s) for j in (0,1)]
        sample=True
        for root in sorted(monthly[key]):
            z=data[root]
            visited.add(root)
            gold,batches=early[root]
            visible=sorted({v for b in batches for v in b}&roster)
            nodes=sorted(roster-set(visible))
            ready=bool(visible) and len(nodes)>=12
            assert z["ready"]==ready and z["reachable"]==(set(gold)<=set(nodes)) and z["window"]==key
            if not ready:
                assert z["roles"]=={}
                continue
            # Check every held-out decision and at least one previously evaluated case per month.
            if name!=exp.SAMBA and not sample:
                continue
            sample=False
            tie=exp.fg.tie_order(root,len(nodes))
            x=np.array([[np.log1p(v[n]) for v in [*posts,*roles]]+
                [np.log1p(np.mean([cnt[n,v] for v in visible])),np.log1p(np.mean([cnt[v,n] for v in visible]))] for n in nodes])
            ext=[]
            for j in (0,1):
                mtx=np.array([[np.log((occurrence[j][n,v]+8*(freq[v]+.5)/(len(previous)+1))/(roles[j][n]+8))
                               -np.log((freq[v]+.5)/(len(previous)+1)) for v in visible] for n in nodes])
                ext.append(np.column_stack((x,mtx.mean(1),mtx.max(1))))
            for arm,dim in [*exp.re.WIDTH.items(),("neutral",8)]:
                probs=[]
                for j in (0,1):
                    feat=ext[j][:,:dim].copy()
                    if arm=="neutral":
                        feat[:,6:]=0.
                    coeff=cfg["role_suffix" if arm=="neutral" else arm][j]["theta"]
                    probs.append(softmax(feat@coeff))
                idx=exp.re.choose(*probs,tie)[0] if len(nodes)>12 else np.arange(12)
                selected={nodes[i] for i in idx}
                assert int(gold[0] in selected)+2*int(gold[1] in selected)==z["roles"][arm]
                lists+=1
            vol={nodes[i] for i in np.lexsort((tie,-np.array([posts[0][n] for n in nodes])))[:12]}
            assert int(gold[0] in vol)+2*int(gold[1] in vol)==z["roles"]["volume"]
    assert visited==set(data)
    for arm in ("volume","roles","reply","neutral","role_suffix"):
        count=sum(z["roles"].get(arm,0)==3 for z in rows)
        assert summary["hit_counts"][arm]==[count]*5
    return dict(eligible=len(data),role_lists_recomputed=lists,event_sha256=exp.fg.digest(exp.OUT/(name+".jsonl")))


if __name__=="__main__":
    assert exp.fg.load(exp.OUT/"protocol.json")==exp.spec()
    with threadpool_limits(1):
        results=exp.fg.load(exp.OUT/"summary.json")
        audit={n:check(n,results[n]) for n in exp.COHORTS}
        entries=exp.fg.load(exp.OUT/"archives.json")
        exp.fg.write_json(exp.OUT/"verification.json",dict(cohorts=audit,downloaded_archives=len(entries),
                          parsed_headers=sum(v["parsed"] for v in entries),new_unique_messages=len(exp.samba_messages()),
                          script=exp.fg.digest(Path(__file__))))
