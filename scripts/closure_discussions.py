"""Reuse cached thread forests, projections and the two archived endpoint definitions."""
from collections import defaultdict
from datetime import datetime, timezone

from closure_common import *
import early_only_discussions as early
import finish_role_experiments as finish
import member_role_evidence as role
from public_thread_transfer import MONTHS, marked_thread, thread_forest
from real_trained_mixtures import messages, TEST, LISTS
from historical_thread_roles import historical_document

REAL=[*LISTS,*finish.COHORTS]


def calendar(name,late=False):
    if name in LISTS:
        msg=messages(name,late)
        months=TEST if late else MONTHS
    else:
        msg,months,_=finish.cohort(name)
    forest,contacts=thread_forest(msg)
    roots=defaultdict(list)
    docs=[]
    for mid,thread in sorted(forest.items()):
        first,t=msg[mid][:2]
        key=datetime.fromtimestamp(t,timezone.utc).strftime("%Y-%m")
        payload=early.prefix(mid,thread,msg)
        if payload:
            roots[key].append((mid,*payload))
        docs.append(historical_document(mid,thread,t,first))
    for y,m in months:
        key=f"{y}-{m:02}"
        start=int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        hist=[z for z in contacts if start-98*DAY<=z[0]<start]
        edges=Counter((a,b) for _,a,b in hist)
        g=dict(window=key,start=start,nodes=sorted({v for edge in edges for v in edge}),edges=[(*e,c) for e,c in sorted(edges.items())])
        volume=Counter(a for a,t,_,_ in msg.values() if start-28*DAY<=t<start)
        yield g,hist,roots[key],volume,(msg,contacts,docs)


def cut(groups,depth):
    if not depth:
        return groups
    n=sum(map(len,groups))
    target=n*depth//100
    count=i=0
    while i<len(groups) and count+len(groups[i])<=target:
        count+=len(groups[i])
        i+=1
    return groups[i:] if i and n-count>=3 else None


def project(g,mid,groups):
    roster=set(g["nodes"])
    obs=[[v for v in b if v in roster] for b in groups]
    return dict(id=mid,window=g["window"],observed=[b for b in obs if b],k=2)

