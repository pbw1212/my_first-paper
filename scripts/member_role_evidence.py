"""Predict distinct early roles from historical co-occurrences with later speakers."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from threadpoolctl import threadpool_limits

import early_only_discussions as task
import fgssi_suffix as fg
from historical_thread_roles import choose,historical_document,objective,REGULARIZERS
from public_thread_transfer import MONTHS,DAY,thread_forest
from real_trained_mixtures import LISTS,PYTHON,messages as python_messages

OUT=task.OUT/"role_evidence"
WIDTH={"roles":4,"reply":6,"role_suffix":8}
SMOOTH=8.


class History:
    def __init__(self,start,msg,contacts,docs):
        past=[(t,a,b) for t,a,b in contacts if start-98*DAY<=t<start]
        self.nodes=sorted({v for _,a,b in past for v in (a,b)})
        self.ix={v:i for i,v in enumerate(self.nodes)}
        n=len(self.nodes)
        volume=[Counter(a for a,t,_,_ in msg.values() if start-days*DAY<=t<start) for days in (28,98)]
        self.replies=np.zeros((n,n))
        for t,a,b in past:
            if t>=start-28*DAY:
                self.replies[self.ix[a],self.ix[b]]+=1
        complete=[d for d in docs if start-98*DAY<=d[0]<start-14*DAY and d[2] is not None and d[3]]
        counts=np.zeros((2,n))
        pairs=np.zeros((2,n,n))
        freq=np.zeros(n)
        for _,first,second,late in complete:
            seen=[self.ix[v] for v in late if v in self.ix]
            freq[seen]+=1
            for k,author in enumerate((first,second)):
                if author in self.ix:
                    i=self.ix[author]
                    counts[k,i]+=1
                    pairs[k,i,seen]+=1
        background=(freq+.5)/(len(complete)+1)
        self.enrichment=np.log((pairs+SMOOTH*background)/((counts[:,:,None]+SMOOTH)*background))
        self.base=np.log1p(np.column_stack(([volume[0][v] for v in self.nodes],
            [volume[1][v] for v in self.nodes],counts.T)))

    def design(self,batches):
        seen=sorted({self.ix[v] for b in batches for v in b if v in self.ix})
        ids=np.setdiff1d(np.arange(len(self.nodes)),seen)
        nodes=[self.nodes[i] for i in ids]
        if len(ids)<12 or not seen:
            return nodes,None
        outgoing=self.replies[np.ix_(ids,seen)].mean(1)
        incoming=self.replies[np.ix_(seen,ids)].mean(0)
        base=np.column_stack((self.base[ids],np.log1p(outgoing),np.log1p(incoming)))
        values=[]
        for mat in self.enrichment:
            e=mat[np.ix_(ids,seen)]
            values.append(np.column_stack((base,e.mean(1),e.max(1))))
        return nodes,values


def examples(name):
    if name in LISTS:
        msg=python_messages(name,False)
    else:
        msg=task.libc.messages() if name==task.libc.NAME else task.r.messages(name)
    forest,contacts=thread_forest(msg)
    docs=[]
    roots=defaultdict(list)
    for root,thread in sorted(forest.items()):
        first,start=msg[root][:2]
        docs.append(historical_document(root,thread,start,first))
        roots[datetime.fromtimestamp(start,timezone.utc).strftime("%Y-%m")].append(root)
    months=MONTHS if name in LISTS else task.r.TARGET_MONTHS
    for y,m in months:
        start=int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        history=History(start,msg,contacts,docs)
        key=f"{y}-{m:02d}"
        for root in roots[key]:
            payload=task.prefix(root,forest[root],msg)
            if payload:
                gold,batches=payload
                nodes,values=history.design(batches)
                yield root,key,gold,nodes,values


def train():
    split=[[[],[]],[[],[]]]
    for name in LISTS:
        for _,key,gold,nodes,x in examples(name):
            if x is None or key=="2018-06":
                continue
            for k,v in enumerate(gold):
                if v in nodes:
                    split[int(key>"2018-06")][k].append((x[k],nodes.index(v)))
    selected={}
    for arm,dim in WIDTH.items():
        selected[arm]=[]
        for k in range(2):
            tr=[(x[:,:dim],i) for x,i in split[0][k]]
            val=[(x[:,:dim],i) for x,i in split[1][k]]
            fits=[]
            for reg in REGULARIZERS:
                opt=minimize(objective,np.zeros(dim),args=(tr,reg),jac=True,method="L-BFGS-B",bounds=[(-8,8)]*dim)
                assert opt.success,opt.message
                fits.append(dict(reg=reg,fit_n=len(tr),val_n=len(val),val_nll=objective(opt.x,val,0.)[0],theta=opt.x.tolist()))
            selected[arm].append(min(fits,key=lambda c:c["val_nll"]))
    cfg=dict(primary="role_suffix minus reply, both documented early authors in 12; all early-only eligible roots in each test list",
             controls="roles only, historical direct replies, month-start 28-day post count, frozen Dept2 mixed_offset",
             historical_documents="root starts within previous 98 days; 14-day follow-up finished strictly before cutoff; unique second and at least one later distinct author",
             evidence="per role, smoothed probability ratio of each observed later speaker conditional on candidate role to corpus prevalence; mean and maximum log ratio",
             smoothing=SMOOTH,training="Python-dev and Python-ideas Jan 2017-May 2018; June embargo; Jul-Dec 2018 select regularization by role NLL",
             known_targets="previously evaluated R lists and GNU libc; no target-event fit or target-based parameter selection",
             models=selected,hashes={p.name:fg.digest(p) for p in
                [Path(__file__),task.OUT/"protocol.json",*(PYTHON/(n+"-headers.jsonl.gz") for n in LISTS),
                 Path(__file__).with_name("historical_thread_roles.py")]})
    path=OUT/"protocol.json"
    if path.exists():
        assert fg.load(path)==cfg
    else:
        fg.write_json(path,cfg)


def test(name):
    cfg=fg.load(OUT/"protocol.json")
    assert cfg["hashes"][Path(__file__).name]==fg.digest(Path(__file__))
    old={z["id"]:z for z in fg.read_rows(task.OUT/(name+".jsonl"))}
    rows=[]
    for root,key,gold,nodes,x in examples(name):
        row=dict(id=root,window=key,roles={})
        z=old[root]
        assert z["sources"]==gold and z["ready"]==(x is not None)
        if x is not None:
            tie=fg.tie_order(root,len(nodes))
            for m,dim in WIDTH.items():
                p,q=[softmax(x[k][:,:dim]@cfg["models"][m][k]["theta"]) for k in (0,1)]
                ids=choose(p,q,tie)[0] if len(nodes)>12 else np.arange(len(nodes))
                sel={nodes[i] for i in ids}
                assert len(sel)==12
                row["roles"][m]=int(gold[0] in sel)+2*int(gold[1] in sel)
            ids=fg.sorted_nodes(x[0][:,0],tie)[:12]
            sel={nodes[i] for i in ids}
            row["roles"]["volume"]=int(gold[0] in sel)+2*int(gold[1] in sel)
            assert (row["roles"]["volume"]==3)==z["hits"]["volume"]
        rows.append(row)
    assert set(old)=={z["id"] for z in rows} and len(rows)==len(old)
    fg.save_rows(OUT/(name+".jsonl"),rows)


def report(name):
    rows=fg.read_rows(OUT/(name+".jsonl"))
    old={z["id"]:z for z in fg.read_rows(task.OUT/(name+".jsonl"))}
    result=dict(n=len(rows),methods={},contrasts={})
    hit={}
    for m in ["volume",*WIDTH]:
        code=np.array([z["roles"].get(m,0) for z in rows])
        hit[m]=np.repeat((code==3).astype(int)[:,None],5,axis=1)
        result["methods"][m]=dict(both=int(sum(code==3)),both_pct=100*(code==3).mean(),
                                   root=int(sum(code%2)),second=int(sum(code//2)))
    for tag in ("mixed_offset","uniform_offset"):
        hit[tag]=np.array([[old[z["id"]]["hits"].get(f"{tag}/{s}",0) for s in task.r.SEEDS] for z in rows])
    for m in ("reply","roles","volume","mixed_offset","uniform_offset"):
        diff=hit["role_suffix"]-hit[m]
        result["contrasts"][m]=task.paired([(z["window"],d) for z,d in zip(rows,diff)],task.r.KEYS)
        result["contrasts"][m].update(gained_per_seed=(diff>0).sum(0).tolist(),lost_per_seed=(diff<0).sum(0).tolist())
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["fit","report",*task.NAMES])
    cmd=parser.parse_args().action
    OUT.mkdir(exist_ok=True)
    with threadpool_limits(1):
        if cmd=="fit":
            train()
        elif cmd=="report":
            fg.write_json(OUT/"summary.json",{n:report(n) for n in task.NAMES})
        else:
            test(cmd)
