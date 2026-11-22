"""Reproduce representative close ranks with the archived and current CPU settings."""
import torch
from threadpoolctl import threadpool_limits

import five_frame_pairs as five
import fgssi_suffix as fg
from evaluate_factorial_history import dataset, rank_upper
from factorial_fgssi import OUT


def compare(run,sid,method):
    paths=dataset(run)
    source,folder,raw,graphs=paths
    public=fg.read_rows(folder/"public.jsonl")
    e=next(r for r in public if r["id"]==sid)
    y=next(r for r in fg.read_rows(folder/"truth.jsonl") if r["id"]==sid)
    g=next(g for g in graphs if g["window"]==e["window"])
    items=[r for r in public if r["window"]==e["window"]]
    pos=items.index(e)
    result={}
    for threads,batch in ((1,8),(2,16)):
        torch.set_num_threads(threads)
        with threadpool_limits(threads):
            if method=="D":
                n,m=five.score(five.PublicWindow(g,raw),e,five.trained(source))
            else:
                import numpy as np
                geo=fg.Geometry(g)
                start=pos//batch*batch
                with torch.no_grad():
                    logits,_=fg.trained(source)(*geo.inputs(items[start:start+batch]))
                x=logits[pos-start]
                n=sorted(set(geo.nodes)-{v for b in e["observed"] for v in b})
                odds=(x[:,1]-x[:,0]).numpy().astype(np.float64)[[geo.index[v] for v in n]]
                m=np.add.outer(odds,odds)
            a,b=[n.index(v) for v in y["sources"]]
            tie=fg.tie_order(sid,len(n))
            r=fg.pair_rank(m,a,b,tie)
            assert r==rank_upper(m,a,b,tie)
            result[threads]=dict(rank=r,gold_logit=float(m[min(a,b),max(a,b)]))
    return dict(id=sid,run=run,method=method,thread_counts=result)


if __name__=="__main__":
    cases=[("company","distill-2-company-13-51","D"),("CollegeMsg","distill-2-CollegeMsg-08-21","D"),
           ("Dept2_new_windows","distill-1-Dept2-28-135","F"),("Enron","Enron-09-39","F")]
    fg.write_json(OUT/"rank_precision_check.json",[compare(*c) for c in cases])
