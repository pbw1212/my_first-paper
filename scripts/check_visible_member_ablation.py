"""Verify the selector against explicit source-pair probabilities and replay the shortlist hits."""
import ast
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import visible_member_ablation as vb


def check_revision():
    earlier=vb.OUT/"measurement_v1.py"
    old=vb.fg.load(vb.OUT/"protocol_v1.json")
    key="scripts/visible_member_ablation.py"
    assert vb.fg.digest(earlier)==old["hashes"][key]
    current=vb.fg.load(vb.OUT/"protocol.json")
    assert current==vb.plan()
    old["hashes"][key]=current["hashes"][key]
    assert current==old
    def functions(p):
        tree=ast.parse(p.read_text(encoding="utf8"))
        return {x.name:ast.dump(x) for x in tree.body if isinstance(x,ast.FunctionDef)}
    a,b=functions(earlier),functions(Path(vb.__file__))
    assert set(a)==set(b) and [k for k in a if a[k]!=b[k]]==["summarize"]


def audit(name):
    task=vb.task
    r=task.r
    msg=task.libc.messages() if name==task.libc.NAME else r.messages(name)
    forest,_=task.thread_forest(msg)
    ro=vb.fg.read_rows(vb.OUT/(name+".jsonl"))
    expected={z["id"]:z for z in vb.fg.read_rows(task.OUT/(name+".jsonl")) if z["ready"] and z["visible"]>=2}
    assert set(expected)=={z["id"] for z in ro}
    models=[row for row in r.fitted_models("Dept2") if row[0]=="mixed_offset"]
    worst=0.
    alternatives=0
    for g,past,_,_,_,_ in r.calendar_windows(msg,name):
        ix={n:i for i,n in enumerate(g["nodes"])}
        w=r.activity(g,past)
        means=r.law_arrays(w)[3]
        hist=r.PublicWindow(g,past)
        for row in (z for z in ro if z["window"]==g["window"]):
            root=row["id"]
            gold,batches=task.prefix(root,forest[root],msg)
            obs=[[v for v in b if v in ix] for b in batches]
            seen=[v for b in obs for v in b]
            legal=sorted(set(ix)-set(seen))
            local=w[[ix[v] for v in legal]]
            prior=1/means[0]+(local[:,None]+local)/means[1]+np.outer(local,local)/means[2]
            np.fill_diagonal(prior,0.)
            tie=vb.fg.tie_order(root,len(legal))
            def matrices(allowed):
                sub=[[v for v in b if v in allowed] for b in obs]
                e=dict(id=root,window=g["window"],observed=[b for b in sub if b],k=2)
                (nodes,_,_),(x,_,_)=hist.design(vb.frames(e))
                kept=[nodes.index(v) for v in legal]
                for _,net,mu,sd in models:
                    with torch.no_grad():
                        s=net.node(net.encoder(torch.from_numpy(np.clip((x[kept]-mu)/sd,-6,6))))[:,0].double().numpy()
                    h=np.exp(s-s.max())
                    mass=prior*np.outer(h,h)
                    yield mass/mass.sum()
            def hit(mat):
                sel=np.lexsort((tie,-mat.sum(1)))[:12]
                return int(set(gold)<={legal[i] for i in sel})
            anchor=seen[-1]
            assert row["anchor"]==anchor and seen[:-1]==row["earlier"]
            tables=list(matrices({anchor}))
            mean=np.mean(tables,axis=0)
            probs=[]
            assert list(map(hit,tables))==row["hits"]["anchor"]
            for v,results in zip(seen[:-1],row["alternatives"],strict=True):
                active={u for t,u,b in past if t>=g["start"]-28*task.DAY and b==v}
                nb=np.array([s in active for s in legal])
                probs.append(float(mean[nb[:,None]|nb].sum()))
                assert [hit(t) for t in matrices({anchor,v})]==results
                alternatives+=1
            err=max(abs(np.array(probs)-row["prob"]))
            worst=max(worst,err)
            dist=np.abs(np.array(probs)-.5)
            assert dist[row["pick"]]<=dist.min()+1e-12
            a=np.array(row["alternatives"])
            assert np.allclose(a.mean(0),row["hits"]["uniform"])
            assert a[row["pick"]].tolist()==row["hits"]["balanced"]
            assert row["hits"]["all"]==[expected[root]["hits"][f"mixed_offset/{s}"] for s in r.SEEDS]
    assert worst<1e-10
    return dict(events=len(ro),additional_member_sets=alternatives,explicit_models_per_set=len(models),
                max_probability_error=worst,sha256=vb.fg.digest(vb.OUT/(name+".jsonl")))


if __name__=="__main__":
    check_revision()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        vb.fg.write_json(vb.OUT/"verification.json",{n:audit(n) for n in vb.task.NAMES})
