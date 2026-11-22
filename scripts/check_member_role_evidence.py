"""Reconstruct role evidence with counters, check source-role labels and posterior decoding."""
from collections import Counter,defaultdict
from datetime import datetime,timezone

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

import member_role_evidence as model
import role_evidence_time_holdout as future
from check_early_only_discussions import reference

fg=model.fg


def verify(name,is_future):
    task=model.task
    msg=(future.load_messages() if is_future else task.libc.messages() if name==task.libc.NAME else task.r.messages(name))
    records=fg.read_rows(future.OUT/"events.jsonl" if is_future else model.OUT/(name+".jsonl"))
    old={z["id"]:z for z in records}
    forest,contacts=model.thread_forest(msg)
    roots=defaultdict(list)
    docs=[]
    for root,thread in forest.items():
        payload=reference(root,thread,msg)
        if payload is None:
            continue
        first_later={v for batch in payload[1] for v in batch}
        t=msg[root][1]
        docs.append((t,*payload[0],first_later))
        roots[datetime.fromtimestamp(t,timezone.utc).strftime("%Y-%m")].append((root,*payload))
    models=fg.load(model.OUT/"protocol.json")["models"]
    checked=set()
    lists=0
    swap_checks=0
    for y,m in ([(2024,i) for i in range(1,13)] if is_future else task.r.TARGET_MONTHS):
        t=int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        k=f"{y}-{m:02d}"
        past=[(u,v) for when,u,v in contacts if t-98*model.DAY<=when<t]
        roster={s for uv in past for s in uv}
        replies=Counter((u,v) for when,u,v in contacts if t-28*model.DAY<=when<t)
        volume=[Counter(a for a,stamp,_,_ in msg.values() if t-d*model.DAY<=stamp<t) for d in (28,98)]
        complete=[d for d in docs if t-98*model.DAY<=d[0]<t-14*model.DAY]
        freq=Counter(v for *_,late in complete for v in late)
        roles=[Counter(d[j] for d in complete) for j in (1,2)]
        pairs=[Counter((d[j],v) for d in complete for v in d[3]) for j in (1,2)]
        first_check=True
        for root,gold,batches in roots[k]:
            checked.add(root)
            z=old[root]
            seen=sorted(roster & {s for b in batches for s in b})
            nodes=sorted(roster-set(seen))
            ready=bool(seen) and len(nodes)>=12
            if is_future:
                assert z["ready"]==ready and z["reachable"]==(set(gold)<=set(nodes))
            truth=z["source_roles"] if is_future else z["roles"]
            if not ready:
                assert not truth
                continue
            tie=fg.tie_order(root,len(nodes))
            common=np.array([[np.log1p(c[n]) for c in [*volume,*roles]]+
                [np.log1p(np.mean([replies[n,v] for v in seen])),np.log1p(np.mean([replies[v,n] for v in seen]))] for n in nodes])
            coords=[]
            for j in (0,1):
                # Scalar predictive prevalence with a beta prior centered on the community rate.
                e=np.array([[np.log((pairs[j][n,v]+model.SMOOTH*(freq[v]+.5)/(len(complete)+1))/
                               (roles[j][n]+model.SMOOTH))-np.log((freq[v]+.5)/(len(complete)+1)) for v in seen] for n in nodes])
                coords.append(np.column_stack((common,e.mean(1),e.max(1))))
            for arm,dim in model.WIDTH.items():
                p,q=[softmax(coords[j][:,:dim]@models[arm][j]["theta"]) for j in (0,1)]
                idx=model.choose(p,q,tie)[0] if len(nodes)>12 else np.arange(len(nodes))
                sel={nodes[i] for i in idx}
                assert int(gold[0] in sel)+2*int(gold[1] in sel)==truth[arm]
                lists+=1
                if first_check and len(nodes)>12:
                    matrix=np.outer(p,q)+np.outer(q,p)
                    np.fill_diagonal(matrix,0.)
                    base=matrix[np.ix_(idx,idx)].sum()
                    for slot in range(12):
                        keep=np.delete(idx,slot)
                        gains=matrix[:,keep].sum(1)
                        other=np.setdiff1d(np.arange(len(nodes)),idx)
                        assert matrix[np.ix_(keep,keep)].sum()+2*gains[other].max()<=base+1e-11
                    swap_checks+=1
            first_check=False
    assert checked==set(old) and len(records)==len(checked)
    return dict(eligible=len(records),role_lists_verified=lists,one_swap_objectives_checked=swap_checks)


if __name__=="__main__":
    with threadpool_limits(1):
        report={n:verify(n,False) for n in model.task.NAMES}
        report["libc-alpha-2024"]=verify("libc-alpha",True)
        fg.write_json(model.OUT/"verification.json",report)
