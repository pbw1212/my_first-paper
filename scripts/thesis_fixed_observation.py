"""Exploratory matched early/late views of three visible members from one trajectory."""

from collections import Counter
from itertools import combinations

import numpy as np
from threadpoolctl import threadpool_limits

from score_dept3_candidates import roles
from thesis_holdout import (EXTERNAL, OUT, SOURCE, Window, frozen_models, load_events,
                             read_rows, save_rows, write_json)
from thesis_tables import contrast, counts


def views(trajectory):
    groups=[[v for v,_ in b["joins"]] for b in trajectory]
    spans=[]
    for left in range(1,len(groups)):
        n=0
        for end in range(left+1,len(groups)+1):
            n+=len(groups[end-1])
            if n>=3:
                if n==3:
                    spans.append((left,end))
                break
    if spans and spans[0][1]<=spans[-1][0]:
        return [(left,end,groups[left:end]) for left,end in (spans[0],spans[-1])]
    return []


def main():
    f=frozen_models()
    summary={}
    for name,source in (("Dept1",EXTERNAL),("CollegeMsg",SOURCE)):
        folder=OUT/name
        raw=load_events(source)
        model=f["Dept3" if name=="Dept1" else name]
        labs=[r for r in read_rows(folder/"truth.jsonl") if r["dual"]]
        rows=[]
        for g in read_rows(folder/"graphs.jsonl"):
            w=Window(g,raw,model["forward"],model["node"])
            for r in labs:
                if r["window"]!=g["window"]:
                    continue
                for tag,(start,end,batches) in zip(("early","late"),views(r["trajectory"])):
                    o=dict(id=r["id"],window=r["window"],observed=batches)
                    nodes,i,j,_,x,d,add,tie=w.design(o)
                    rank=np.lexsort((tie,-add,-(x@model["fusion"])))
                    true=set(r["sources"])
                    target=next((k for k,(a,b) in enumerate(zip(i,j)) if {nodes[a],nodes[b]}==true),None)
                    pos=None if target is None else rank.tolist().index(target)+1
                    label=dict(r,cut_batches=start,trajectory=r["trajectory"][:end])
                    rows.append(dict(id=r["id"],window=r["window"],tag=tag,batch_start=start,batch_end=end,
                        fusion=pos,both_visible=len(roles(label)[1])==2))
        save_rows(folder/"fixed_three.jsonl",rows)
        pairs=[dict(id=a["id"],window=a["window"],early=a["fusion"],late=b["fusion"],
                    dual_in_both=a["both_visible"] and b["both_visible"]) for a,b in zip(rows[::2],rows[1::2])
               if a["id"]==b["id"] and a["batch_end"]<=b["batch_start"]]
        assert len(pairs)*2==len(rows)
        summary[name]={}
        for tag,sample in (("all",pairs),("both_lineages_in_both",[p for p in pairs if p["dual_in_both"]])):
            summary[name][tag]=dict(n=len(sample),
                hits={k:sum(r[k] is not None and r[k]<=32 for r in sample) for k in ("early","late")},
                paired=contrast(sample,"early","late") if sample else {})
    write_json(OUT/"fixed_three_summary.json",summary)
    print(summary)


if __name__=="__main__":
    with threadpool_limits(limits=1,user_api="blas"):
        main()
