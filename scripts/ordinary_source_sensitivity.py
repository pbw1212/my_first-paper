"""Check a historical activity prior and the public gate for one versus two emitting sources."""
import numpy as np
from sklearn.metrics import brier_score_loss,roc_auc_score

from ordinary_source_roles import OUT,PREVIOUS,FILES,NEW,company,load_events,make_windows,read_rows,write_json
from score_dept3_candidates import roles
from silent_source_oracle import CompleteTrace


def run(name):
    folder=OUT/name
    local=name not in FILES
    train=folder if local else OUT/"Dept2"
    old=read_rows(train/"fit_truth.jsonl")
    prior_two=np.mean([len(roles(y)[0])==2 for y in old])
    raw=company() if name=="company" else load_events(NEW if local else PREVIOUS.parents[1]/"data/raw"/FILES[name])
    gs,streams=make_windows(raw,name,.15,1.)
    truth=read_rows(folder/"test_truth.jsonl")
    pred=read_rows(folder/"test_predictions.jsonl")
    assert [y["id"] for y in truth]==[r["id"] for r in pred]
    yy=np.array([len(roles(y)[0])==2 for y in truth],dtype=int)
    prob=np.array([r["prob_two"] for r in pred])
    gate=dict(auc=roc_auc_score(yy,prob),brier=brier_score_loss(yy,prob),
              training_constant_brier=brier_score_loss(yy,np.full(len(yy),prior_two)),actual=float(yy.mean()),
              mean_forecast=float(prob.mean()))
    rows={}
    for g in gs:
        labs=[y for y in truth if y["window"]==g["window"] and len(roles(y)[0])==1]
        if not labs:
            continue
        oracle=CompleteTrace(g,streams[g["window"]])
        degree=np.zeros(len(oracle.nodes))
        for u,v,count in g["edges"]:
            degree[oracle.index[u]]+=1
        w=np.sqrt(1+degree)
        importance=[];masses=[];actual=[];uniform=[]
        for lab in labs:
            weighted=oracle.evaluate(lab,prior=w)
            uniform.append(oracle.evaluate(lab)["mass"])
            masses.append(weighted["mass"]);actual.append(weighted["hit"])
            a,b=(oracle.index[v] for v in lab["sources"])
            importance.append(w[a]*w[b])
        imp=np.array(importance)
        rows[g["window"]]=dict(n=len(labs),effective_n=float(imp.sum()**2/(imp@imp)),
            expected=float(np.average(masses,weights=imp)),empirical=float(np.average(actual,weights=imp)),
            uniform=float(np.mean(uniform)))
    # Give each of the four observed windows the same weight under both source priors.
    avg={tag:float(np.mean([r[tag] for r in rows.values()])) for tag in ("expected","uniform","empirical")}
    return dict(gate=gate,alternative_prior="P({a,b}) proportional to sqrt((1+historical outdegree[a])*(1+historical outdegree[b]))",
                windows=rows,equal_window=avg)


if __name__=="__main__":
    result={name:run(name) for name in ("company","Dept2","Dept1","CollegeMsg")}
    write_json(OUT/"prior_and_gate.json",result)
    for name,r in result.items():
        print(name,r["equal_window"],r["gate"]["auc"])
