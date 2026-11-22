"""Check early speakers independently and decompose the gain on equally scoreable roots."""
from datetime import datetime,timezone

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import early_only_discussions as task
import fgssi_suffix as fg
from five_frame_pairs import frames


def reference(root,thread,msg):
    t0=msg[root][1]
    first=msg[root][0]
    times={}
    for when,_,who in sorted(thread):
        if t0<=when<t0+14*task.DAY:
            times.setdefault(who,when)
    del times[first]
    if len(times)<2:
        return None
    second_time=min(times.values())
    first_group=[n for n,t in times.items() if t==second_time]
    if second_time==t0 or len(first_group)!=1:
        return None
    second=first_group[0]
    del times[second]
    return [first,second],[[n for n in sorted(times) if times[n]==t] for t in sorted(set(times.values()))]


def check(name):
    r=task.r
    folder=task.libc.OUT if name==task.libc.NAME else r.OUT/name
    msg=task.libc.messages() if name==task.libc.NAME else r.messages(name)
    forest,_=task.thread_forest(msg)
    old={z["id"]:z for z in fg.read_rows(folder/"events.jsonl") if z["profile"]=="real"}
    rows=fg.read_rows(task.OUT/(name+".jsonl"))
    by_id={z["id"]:z for z in rows}
    models=[m for m in r.fitted_models("Dept2") if m[0]=="mixed_offset"]
    found=set()
    effects=[]
    explicit=0
    for g,h,_,vol,ordinary,_ in r.calendar_windows(msg,name):
        previous={e["id"]:e for e,_ in ordinary}
        ix={v:i for i,v in enumerate(g["nodes"])}
        hist=r.PublicWindow(g,h)
        w=r.activity(g,h)
        means=r.law_arrays(w)[3]
        sampled=False
        for root,thread in sorted(forest.items()):
            if datetime.fromtimestamp(msg[root][1],timezone.utc).strftime("%Y-%m")!=g["window"]:
                continue
            payload=reference(root,thread,msg)
            if payload is None:
                assert root not in by_id
                continue
            src,batches=payload
            z=by_id[root]
            found.add(root)
            obs=[[v for v in b if v in ix] for b in batches]
            obs=[b for b in obs if b]
            vis={v for b in obs for v in b}
            nodes=sorted(ix.keys()-vis)
            ready=len(vis)>0 and len(nodes)>=12
            assert z["sources"]==src and z["ready"]==ready and z["reachable"]==(set(src)<=set(nodes))
            assert z["visible"]==len(vis) and z["later"]==sum(map(len,batches))
            if not ready:
                assert not z["hits"]
                continue
            tie=fg.tie_order(root,len(nodes))
            ranked=np.lexsort((tie,-np.array([vol[n] for n in nodes])))
            assert int(set(src)<={nodes[i] for i in ranked[:12]})==z["hits"]["volume"]
            common=root in old and old[root]["ready"]
            if sampled and not common:
                continue
            sampled=True
            e=dict(id=root,window=g["window"],k=2,observed=obs)
            (nd,_,_),(feat,_,_)=hist.design(frames(e))
            assert nd==nodes
            loc=w[[ix[n] for n in nodes]]
            weights=1/means[0]+(loc[:,None]+loc)/means[1]+np.outer(loc,loc)/means[2]
            np.fill_diagonal(weights,0)
            if common:
                (oldnodes,_,_),(oldfeat,_,_)=hist.design(frames(previous[root]))
                assert set(nodes)<=set(oldnodes)
                index={n:i for i,n in enumerate(oldnodes)}
                retain=[index[n] for n in nodes]
                removed=len(oldnodes)-len(nodes)
            codes=[]
            for i,(_,net,mu,sd) in enumerate(models):
                def dense(x):
                    with torch.no_grad():
                        s=net.node(net.encoder(torch.from_numpy(np.clip((x-mu)/sd,-6,6))))[:,0].double().numpy()
                    q=np.exp(s-s.max())
                    values=(np.outer(q,q)*weights).sum(1)
                    order=np.lexsort((tie,-values))[:12]
                    return int(set(src)<={nodes[j] for j in order})
                new=dense(feat)
                key=f"mixed_offset/{r.SEEDS[i]}"
                assert z["hits"][key]==new
                explicit+=1
                if common:
                    codes.append([old[root]["hits"][key],dense(oldfeat[retain]),new])
            if common:
                effects.append(dict(id=root,window=g["window"],removed=removed,hits=codes))
    assert len(rows)==len(by_id)==len(found)
    assert len(effects)==sum(by_id[k]["ready"] and v["ready"] for k,v in old.items())
    fg.save_rows(task.OUT/(name+"-common.jsonl"),effects)
    result=dict(roots=len(rows),checked_dense_lists=explicit,sha256=fg.digest(task.OUT/(name+".jsonl")))
    def contrast(i,j):
        return task.paired([(e["window"],np.array(e["hits"])[:,i]-np.array(e["hits"])[:,j]) for e in effects],r.KEYS)
    result["same_scoreable"]=dict(n=len(effects),old_filter_updated_pct=(100*np.mean([v["hits"] for v in effects],axis=(0,1))).tolist() if effects else [],
                                   filter_minus_old=contrast(1,0),update_minus_filter=contrast(2,1),update_minus_old=contrast(2,0))
    return result


if __name__=="__main__":
    assert fg.load(task.OUT/"protocol.json")==task.plan()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        fg.write_json(task.OUT/"verification.json",{n:check(n) for n in task.NAMES})
