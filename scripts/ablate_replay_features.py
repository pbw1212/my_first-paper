"""Post-test mechanism diagnosis on Dept3 training and the previously used Dept1 development cohort."""

import numpy as np
from threadpoolctl import threadpool_limits

from temporal_pair_evidence import (DEPT, EXTERNAL, OUT, PairEvidence, cohort, draw, fit,
                                    frozen_models, load_events, make_windows, metrics, models, write_json)


def components(x):
    # Columns 0:14 form the static design. The last two are reach-volume priors.
    return dict(with_source_volume=np.column_stack((x[:,:14],x[:,-2:])),with_member_support=x[:,:-2])


def run():
    raw=load_events(DEPT)
    gs,streams=make_windows(raw,"train",.15,.6)
    nm,fm,_=models("Dept3")
    data={k:[] for k in components(np.zeros((1,21)))}
    for week,g in enumerate(gs,1):
        if week<=10:
            continue
        cut="rolling_"+str((week-1)//5*5)
        obs,labs,_=draw(g,streams[g["window"]],90600+week,55,0)
        if not obs:
            continue
        w=PairEvidence(raw,g,fm[cut],nm[cut])
        for o,lab in zip(obs,labs):
            nodes,i,j,xx,*_=w.design(o)
            if not set(lab["sources"])<=set(nodes):
                continue
            a,b=sorted(nodes.index(v) for v in lab["sources"])
            y=int(np.flatnonzero((i==a)&(j==b))[0])
            for k,x in components(xx["ordered"]).items():
                data[k].append((x,y))
    weights={k:fit(r,r[0][0].shape[1]) for k,r in data.items()}
    del data
    raw=load_events(EXTERNAL)
    gs,obs,labs=cohort("Dept1",raw)
    labels={r["id"]:r for r in labs}
    rr=[]
    f=frozen_models()["Dept3"]
    for g in gs:
        items=[o for o in obs if o["window"]==g["window"]]
        if not items:
            continue
        w=PairEvidence(raw,g,f["forward"],f["node"])
        for o in items:
            nodes,i,j,xx,ss,tie,_=w.design(o)
            lab=labels[o["id"]]
            row=dict(id=o["id"],dual=lab["dual"],ordinary=lab["ordinary"])
            for k,x in components(xx["ordered"]).items():
                row[k]=None
                if set(lab["sources"])<=set(nodes):
                    a,b=sorted(nodes.index(v) for v in lab["sources"])
                    y=np.flatnonzero((i==a)&(j==b))[0]
                    order=np.lexsort((tie,-ss["additive"],-(x@weights[k])))
                    row[k]=int(np.flatnonzero(order==y)[0])+1
            rr.append(row)
    report={c:{k:metrics([r for r in rr if r[c]],k) for k in weights} for c in ("dual","ordinary")}
    write_json(OUT/"ablation_dev.json",dict(report,weights=weights))
    print(report)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run()
