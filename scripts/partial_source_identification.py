"""Selective one-source prediction and calibrated partner sets from public suffix observations."""

import argparse
import hashlib
import math
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from diagnose_source_equivalence import complete_world, compatible
from masked_pair_benchmark import ROOT, arrival, company, draw, load_events, make_windows, save_rows, read_rows, write_json, load
from run_dept3_candidates import Features
from temporal_pair_evidence import NEW, history_kernels

OUT=ROOT/"runs/partial_source_identification"
PLAN=dict(primary="Precision and response rate for naming one true source; target empirical calibration precision 80%, minimum 15 answers",
          split="12 evenly spaced activity-eligible complete 14-day windows over 15%-100%; 5 fit, 3 calibration, 4 test",
          activity="at least 100 directed within-roster contacts per window; retrospective cohort frame",
          per_window="fit 140 dual; calibration and test 80 dual plus 25 ordinary, cap 60000; independently seeded",
          anchor="Boosting supervised on mandatory identities in one realized world; compare ordinary source supervision and frequency",
          confidence="maximum raw decision score; descriptive 100%, 25%, 10% calibration-quantile policies plus target80",
          partner="conditional on correctly named anchor; 90% calibration rank fraction; evaluate conditional AND end-to-end coverage",
          oracle="intersection of all compatible pairs; dual-compatible for dual subgroup, any compatible pair for ordinary",
          transfer_gate="Leave Dept4 unexamined unless target80 names at least 15 test cases at >=80% precision on either development network.")


def core(g,stream,o,lab,completion=0):
    ix={v:i for i,v in enumerate(g["nodes"])}
    t,u,v=complete_world(stream,ix,lab,completion)
    A=arrival(len(ix),t,u,v,np.ones(len(t),bool))
    visible=np.array([ix[z] for b in o["observed"] for z in b])
    groups=np.repeat(np.arange(len(o["observed"])),[len(b) for b in o["observed"]])
    legal=np.setdiff1d(np.arange(len(ix)),visible)
    a,b=sorted(ix[z] for z in lab["sources"])
    args=(A,t,u,v,visible,groups)
    ans=compatible(*args,legal,a,b)
    assert ans[-2] and (not lab["dual"] or ans[-1])
    common={}
    for tag,col in (("ordinary",0),("dual",1)):
        if not lab[tag]:
            continue
        ids=list(lab["sources"]) if ans[col]==1 else [
            g["nodes"][z] for z in (a,b) if compatible(*args,legal[legal!=z],a,b)[col]==0]
        common[tag]=dict(members=sorted(ids),pairs=ans[col])
    return common


class HistoricalFeatures:
    def __init__(self,g,raw):
        self.f=Features(g,raw)
        self.reach=history_kernels(raw,g)[0]

    def event(self,o):
        _,nodes,static=self.f.event(o)
        ix=self.f.history.index
        candidates=np.array([ix[v] for v in nodes])
        obs=np.array([ix[z] for b in o["observed"] for z in b])
        r=self.reach[np.ix_(candidates,obs)]
        rate=self.reach[:,obs].mean(axis=0)
        support=r/(r+np.maximum(.01,rate))
        earliest=support[:,:len(o["observed"][0])]
        n=len(nodes)
        x=np.column_stack((static, support.mean(axis=1),support.max(axis=1),earliest.mean(axis=1),
            self.reach[candidates].sum(axis=1),np.full(n,len(obs)),np.full(n,len(obs)/(n+len(obs)))))
        return nodes,x,support


def partner_features(x,support,a):
    explained=support[a]
    residual=1-explained
    return np.column_stack((x,support@residual/max(residual.sum(),1e-8),
        np.maximum(support-explained,0).mean(axis=1),np.maximum(explained-support,0).mean(axis=1)))


def boost(xs,ys):
    X=np.vstack(xs)
    y=np.concatenate(ys)
    # Give each event the same total weight and use a constant positive class upweight.
    weights=np.concatenate([np.full(len(t),1/len(t)) for t in ys])
    pos=weights[y].sum()
    weights[y]*=weights[~y].sum()/pos
    weights*=len(y)/weights.sum()
    estimator=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=12,min_samples_leaf=30,
           l2_regularization=3,learning_rate=.06,early_stopping=False,random_state=712)
    return estimator.fit(X,y,sample_weight=weights)


def fit_models(data,labels,oracle):
    xs=[];actual=[];required=[];px=[];py=[]
    freq=Counter()
    for sid,(nodes,x,support) in data.items():
        lab=labels[sid]
        truth=lab["sources"]
        freq.update(truth)
        xs.append(x)
        actual.append(np.isin(nodes,truth))
        required.append(np.isin(nodes,oracle[sid]["dual"]["members"]))
        a,b=(nodes.index(v) for v in truth)
        for first,second in ((a,b),(b,a)):
            allowed=np.arange(len(nodes))!=first
            px.append(partner_features(x,support,first)[allowed])
            py.append((np.arange(len(nodes))==second)[allowed])
    return dict(source=boost(xs,actual),core=boost(xs,required),partner=boost(px,py),frequency=freq)


def predict(o,data,models):
    nodes,x,support=data
    rng=np.random.default_rng(int(hashlib.sha256(o["id"].encode()).hexdigest()[:12],16))
    tie=rng.permutation(len(nodes))
    result={}
    raw_source=models["source"].decision_function(x)
    for method,score in (("core",models["core"].decision_function(x)),("source",raw_source),
                         ("prior",np.log1p([models["frequency"][n] for n in nodes]))):
        order=np.lexsort((tie,-score))
        a=order[0]
        strength=models["partner"].decision_function(partner_features(x,support,a))
        allowed=np.delete(np.arange(len(nodes)),a)
        partners=allowed[np.lexsort((tie[allowed],-strength[allowed]))]
        independent=allowed[np.lexsort((tie[allowed],-raw_source[allowed]))]
        result[method]=dict(anchor=nodes[a],confidence=float(score[a]),
             partners=[nodes[j] for j in partners],independent=[nodes[j] for j in independent])
    return dict(id=o["id"],window=o["window"],n=len(nodes),outputs=result)


def rank(row,truth,field):
    if row["anchor"] not in truth:
        return None
    other=next(a for a in truth if a!=row["anchor"])
    return row[field].index(other)+1


def thresholds(cal,labels):
    rules={}
    for method in ("core","source","prior"):
        rows=[(p["outputs"][method],labels[p["id"]]["sources"]) for p in cal if labels[p["id"]]["dual"]]
        scores=np.array([r["confidence"] for r,_ in rows])
        cuts={"always":float(scores.min()-1e6)}
        cuts.update({str(q):float(np.quantile(scores,1-q)) for q in (.25,.10)})
        candidates=[]
        for q in (1,.8,.6,.4,.25,.15,.1,.05):
            cut=np.quantile(scores,1-q)
            selected=[r["anchor"] in y for r,y in rows if r["confidence"]>=cut]
            if len(selected)>=15 and np.mean(selected)>=.80:
                candidates.append(float(cut))
        cuts["target80"]=min(candidates) if candidates else None
        rules[method]={}
        for tag,cut in cuts.items():
            chosen=[(r,y) for r,y in rows if cut is not None and r["confidence"]>=cut]
            correct=[(r,y) for r,y in chosen if r["anchor"] in y]
            entry=dict(cut=cut,selected=len(chosen),correct=len(correct))
            for field in ("partners","independent"):
                fractions=sorted(rank(r,y,field)/len(r[field]) for r,y in correct)
                k=math.ceil(.90*(len(fractions)+1))
                entry[field]=fractions[k-1] if 1<=k<=len(fractions) else 1.
            rules[method][tag]=entry
    return rules


def summarize(pred,labels,oracle,rules,group):
    cohort=[p for p in pred if labels[p["id"]][group]]
    result={}
    for method,m_rules in rules.items():
        result[method]={}
        for tag,rule in m_rules.items():
            selected=[p for p in cohort if rule["cut"] is not None and p["outputs"][method]["confidence"]>=rule["cut"]]
            correct=required=partner_hit=indep_hit=top_pair=0
            sizes=[];indep_sizes=[];all_sizes=[]
            for p in selected:
                r=p["outputs"][method]
                truth=labels[p["id"]]["sources"]
                ok=r["anchor"] in truth
                correct+=ok
                required+=r["anchor"] in oracle[p["id"]][group]["members"]
                nr=len(r["partners"])
                k=min(nr,math.ceil(rule["partners"]*nr))
                ki=min(nr,math.ceil(rule["independent"]*nr))
                sizes.append(k);indep_sizes.append(ki);all_sizes.append(nr)
                if ok:
                    pos=rank(r,truth,"partners")
                    partner_hit+=pos<=k
                    top_pair+=pos==1
                    indep_hit+=rank(r,truth,"independent")<=ki
            result[method][tag]=dict(n=len(cohort),named=len(selected),correct=int(correct),required=int(required),
                partner_hit=int(partner_hit),independent_hit=int(indep_hit),exact_first=int(top_pair),
                mean_set=float(np.mean(sizes)) if sizes else None,
                mean_independent_set=float(np.mean(indep_sizes)) if sizes else None,
                mean_legal_partners=float(np.mean(all_sizes)) if sizes else None)
    return result


def experiment(name):
    raw=company() if name=="company" else load_events(NEW if name=="Dept2" else ROOT/"data/raw/email-Eu-core-temporal-Dept4.txt.gz")
    gs,streams=make_windows(raw,name,.15,1.)
    usable=[g for g in gs if sum(len(b) for _,b in streams[g["window"]])>=100]
    assert len(usable)>=12
    selected=[usable[i] for i in np.linspace(0,len(usable)-1,12).astype(int)]
    folder=OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    split={}
    obs=[];labs=[];oracle={};fitdata={};flow={};pred=[];checks={}
    for k,g in enumerate(selected):
        stage="fit" if k<5 else ("cal" if k<8 else "test")
        key=g["window"];split[key]=stage
        stream=streams[key]
        items,truth,c=draw(g,stream,104003+k,140 if k<5 else 80,0 if k<5 else 25)
        flow[key]=dict(c,start=g["start"],end=g["end"],stage=stage)
        f=HistoricalFeatures(g,raw)
        for o,t in zip(items,truth,strict=True):
            feature=f.event(o)
            if stage=="fit":
                fitdata[o["id"]]=feature
            else:
                pred.append(predict(o,feature,models))
            oracle[o["id"]]=core(g,stream,o,t)
        obs.extend(items);labs.extend(truth)
        if k==4:
            labels={r["id"]:r for r in labs}
            models=fit_models(fitdata,labels,oracle)
            joblib.dump(models,folder/"models.joblib")
            fitdata.clear()
        if k==7:
            rules=thresholds(pred,{r["id"]:r for r in labs})
            write_json(folder/"calibration.json",rules)
        print(name,key,stage,len(items),flush=True)
    labels={r["id"]:r for r in labs}
    save_rows(folder/"public.jsonl",obs)
    save_rows(folder/"truth.jsonl",labs)
    write_json(folder/"oracle.json",oracle)
    save_rows(folder/"predictions.jsonl",pred)
    write_json(folder/"windows.json",flow)
    test=[p for p in pred if split[p["window"]]=="test"]
    summary={group:summarize(test,labels,oracle,rules,group) for group in ("dual","ordinary")}
    for group in summary:
        rr=[p for p in test if labels[p["id"]][group]]
        hist=Counter(len(oracle[p["id"]][group]["members"]) for p in rr)
        summary[group]["core_counts"]={str(i):hist[i] for i in (0,1,2)}
        summary[group]["random_anchor"]=float(np.mean([2/p["n"] for p in rr]))
    summary["test_windows"]={k:summarize([p for p in test if p["window"]==k],labels,oracle,rules,"dual")
                              for k,s in split.items() if s=="test"}
    write_json(folder/"summary.json",summary)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("network",choices=["company","Dept2","Dept4"])
    name=p.parse_args().network
    OUT.mkdir(exist_ok=True)
    path=OUT/"protocol.json"
    if path.exists():
        assert load(path)==PLAN
    else:
        write_json(path,PLAN)
    if name=="Dept4":
        passes=[load(OUT/n/"summary.json")["dual"]["core"]["target80"] for n in ("company","Dept2")]
        assert any(r["named"]>=15 and r["correct"]/r["named"]>=.8 for r in passes)
    with threadpool_limits(limits=1):
        experiment(name)
