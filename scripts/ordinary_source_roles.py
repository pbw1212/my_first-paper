"""Fit on ordinary cascades and evaluate active and silent sources under a shared shortlist budget."""
import argparse
import hashlib

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from partial_source_identification import (OUT as PREVIOUS, HistoricalFeatures, NEW, boost, company,
    draw, load, load_events, make_windows, partner_features, read_rows, save_rows, write_json)
from partial_source_lists import pair_distribution, sorted_nodes
from joint_source_shortlist import best_subset
from score_dept3_candidates import roles
from silent_source_oracle import CompleteTrace
from transfer_joint_shortlist import FILES

OUT=PREVIOUS.parent/"ordinary_source_roles"
PLAN=dict(budget=12, per_window={"fit":200,"cal":100,"test":150},seeds={"fit":9051700,"cal":9052100,"test":9052500},
    cohort="First mask-eligible events; uniform two-source draws; no requirement on emission or observable ancestry",
    split="Original five fit, three validation and four test windows per network. Fresh propagation seeds.",
    primary="Both true sources in 12 names on all ordinary events; compare role mixture against ordinary pooled learner and frozen dual-trained joint",
    selection="For role mixture only, choose score temperature in [1, 0.65, 1.5] by aggregate calibration both-hit count; break ties in listed order",
    oracle="Expected Bayes-optimal coverage with full within-window contacts and entire timed parent-labelled trace, uniform source prior, known simulation p",
    diagnosis="With a single direct emitter, give its true identity and evaluate the best 11 partners; report separately from public performance")
TEMPERATURES=(1.,.65,1.5)


class Window:
    def __init__(self,graph,raw):
        self.base=HistoricalFeatures(graph,raw)
        n=len(graph["nodes"])
        ix=self.base.f.history.index
        width=14*86400
        end=graph["start"]
        matrices=[]
        for i in range(6):
            left=end-(i+1)*width
            if left<raw[0][0]:
                break
            edges=[(ix[u],ix[v]) for t,u,v in raw if left<=t<left+width]
            counts=np.zeros((n,n))
            if edges:
                a,b=np.array(edges).T
                np.add.at(counts,(a,b),1.)
            matrices.append(counts)
        stack=np.stack(matrices)
        outgoing=stack.sum(axis=2)
        self.quiet=np.column_stack(((outgoing==0).mean(axis=0),(.7**outgoing).mean(axis=0),
                                    np.log1p(outgoing.std(axis=0)),np.log1p(outgoing[0])))
        self.contact=np.average(stack,axis=0,weights=.75**np.arange(len(stack)))

    def features(self,o):
        base=self.base.event(o)
        nodes,x,support=base
        ix=self.base.f.history.index
        rows=[ix[v] for v in nodes]
        cols=[ix[v] for b in o["observed"] for v in b]
        outgoing=self.contact[np.ix_(rows,cols)]
        incoming=self.contact[np.ix_(cols,rows)]
        extra=np.column_stack((self.quiet[rows],np.log1p(outgoing.mean(axis=1)),np.log1p(incoming.mean(axis=0)),
                               np.log1p(incoming.max(axis=0))))
        ev=np.array([np.log1p(len(cols)),np.log1p(len(o["observed"])),len(o["observed"][0])/len(cols),
                     support.mean(),support.mean(axis=1).max(),np.log1p(len(nodes))])
        return base,np.column_stack((x,extra)),ev


def fit(training):
    xs=[]; targets={tag:[] for tag in ("pooled","active","silent")}
    events=[];two=[]
    for data,y in training:
        (nodes,_,_),x,ev=data
        direct,_=roles(y)
        current=dict(pooled=y["sources"],active=list(direct),silent=list(set(y["sources"])-direct))
        xs.append(x)
        for tag,values in current.items():
            targets[tag].append(np.isin(nodes,values))
        two.append(len(direct)==2)
        events.append(ev)
    models={tag:boost(xs,ys) for tag,ys in targets.items()}
    models["gate"]=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500)).fit(events,two)
    return models


def weighted_pairs(a,s,p,t):
    active=softmax(a/t)
    silent=softmax(s/t)
    def pair(left,right):
        q=np.outer(left,right)
        np.fill_diagonal(q,0.)
        return (q+q.T)/(2*q.sum())
    return p*pair(active,active)+(1-p)*pair(active,silent)


def predictions(o,data,models,old,temps):
    base,x,ev=data
    nodes=base[0]
    initial,tie,joint=pair_distribution(o,base,old)
    plain=sorted_nodes(models["pooled"].decision_function(x),tie)
    active=models["active"].decision_function(x)
    silent=models["silent"].decision_function(x)
    gate=float(models["gate"].predict_proba(ev[None,:])[0,1])
    sets=dict(dual_node=initial[:12],dual_joint=best_subset(joint,initial,tie,12),
              pooled=plain[:12],reach=sorted_nodes(base[2].mean(axis=1),tie)[:12])
    for t in temps:
        joint=weighted_pairs(active,silent,gate,t)
        sets["role_"+str(t)]=best_subset(joint,plain,tie,12)
    result={tag:[nodes[i] for i in order] for tag,order in sets.items()}
    assert all(len(c)==len(set(c))==12 for c in result.values())
    return dict(id=o["id"],window=o["window"],sets=result,prob_two=gate)


def diagnose(data,y,oracle,models,old):
    base,x,_=data
    nodes,features,support=base
    direct=oracle["mandatory"]
    if len(direct)!=1:
        return {}
    anchor=nodes.index(direct[0])
    other=next(v for v in y["sources"] if v!=direct[0])
    tie=np.random.default_rng(1000003+sum((j+1)*ord(c) for j,c in enumerate(y["id"]))).permutation(len(nodes))
    choices=dict(old_partner=old["partner"].decision_function(partner_features(features,support,anchor)),
                 silent=models["silent"].decision_function(x),pooled=models["pooled"].decision_function(x))
    result={}
    for tag,logits in choices.items():
        logits[anchor]=-np.inf
        shortlist=[nodes[i] for i in sorted_nodes(logits,tie)[:11]]
        result[tag]=other in shortlist
    return result


def experiment(name,phase):
    folder=OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    external=name in FILES
    assert not external or phase=="test"
    source="Dept2" if external else name
    train_dir=OUT/source
    previous=PREVIOUS/("transfer_"+name if external else name)
    plan=dict(PLAN,training=source,test_count_per_window=60 if external else 150)
    proto=folder/"protocol.json"
    if proto.exists():
        # Original development protocols already record their local sizes and training split.
        assert load(proto)==plan or (not external and load(proto)==PLAN)
    else:
        write_json(proto,plan)
    raw=company() if name=="company" else load_events(PREVIOUS.parents[1]/"data/raw"/FILES[name] if external else NEW)
    gs,streams=make_windows(raw,name,.15,1.)
    if external:
        splits={key:{"stage":"test"} for key in load(previous/"protocol.json")["windows"]}
    else:
        splits=load(previous/"windows.json")
    old=joblib.load(PREVIOUS/source/"models.joblib")
    files=("truth.jsonl",) if external else ("truth.jsonl","repeat_truth.jsonl","joint12_truth.jsonl")
    prevseeds={r["seed"] for file in files for r in read_rows(previous/file)}
    if phase!="fit":
        models=joblib.load(train_dir/"models.joblib")
        for earlier in (["fit"] if phase=="cal" else ["fit","cal"]):
            prevseeds.update(r["seed"] for r in read_rows(train_dir/(earlier+"_truth.jsonl")))
    if phase=="test":
        selected=load(train_dir/"selection.json")
        assert selected["weights"]==hashlib.sha256((train_dir/"models.joblib").read_bytes()).hexdigest()
        temps=[selected["temperature"]]
    else:
        temps=TEMPERATURES
    public=[];truth=[];pred=[];diag=[];training=[];flow={}
    chosen=[g for g in gs if g["window"] in splits and splits[g["window"]]["stage"]==phase]
    for i,g in enumerate(chosen):
        key=g["window"]
        w=Window(g,raw)
        world=CompleteTrace(g,streams[key])
        amount=60 if external else PLAN["per_window"][phase]
        seed=PLAN["seeds"][phase]+i+(4000 if name=="Dept1" else 8000 if name=="CollegeMsg" else 0)
        obs,labs,counts=draw(g,streams[key],seed,0,amount)
        assert counts["ordinary"]==amount
        flow[key]=dict(counts,seed=seed,start=g["start"],end=g["end"])
        checked=0
        for o,y in zip(obs,labs,strict=True):
            o["id"]=y["id"]=phase+"-"+o["id"]
            assert y["seed"] not in prevseeds
            prevseeds.add(y["seed"])
            data=w.features(o)
            if phase=="fit":
                training.append((data,y))
            else:
                pred.append(predictions(o,data,models,old,temps))
                if phase=="test":
                    oc=world.evaluate(y,check=checked<2)
                    checked+=int(len(oc["mandatory"])==1)
                    diag.append(dict(id=o["id"],oracle=oc,known_anchor=diagnose(data,y,oc,models,old)))
        public.extend(obs);truth.extend(labs)
        print(name,phase,key,len(obs),flush=True)
    save_rows(folder/(phase+"_public.jsonl"),public)
    save_rows(folder/(phase+"_truth.jsonl"),truth)
    write_json(folder/(phase+"_flow.json"),flow)
    if phase=="fit":
        joblib.dump(fit(training),folder/"models.joblib")
        return
    save_rows(folder/(phase+"_predictions.jsonl"),pred)
    if phase=="cal":
        hits={str(t):sum(set(y["sources"])<=set(r["sets"]["role_"+str(t)]) for y,r in zip(truth,pred,strict=True)) for t in temps}
        choice=max(temps,key=lambda t:hits[str(t)])
        write_json(folder/"selection.json",dict(temperature=choice,hits=hits,n=len(truth),
                   weights=hashlib.sha256((folder/"models.joblib").read_bytes()).hexdigest()))
    else:
        save_rows(folder/"test_diagnostics.jsonl",diag)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("network",choices=["company","Dept2",*FILES])
    parser.add_argument("phase",choices=["fit","cal","test"])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        experiment(args.network,args.phase)
