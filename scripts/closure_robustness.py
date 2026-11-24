"""WP3. Capacity-matched inputs and frozen non-SI stress tests."""
import copy
import random
import time
import joblib
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from closure_common import *
from closure_cascades import Scorer,check_lists,hits,mask,replay,score_lists,test_windows
from factorial_fgssi import weights
from factorial_history import folder_for

F=BASE/"wp3_robustness"
TARGETS=["Dept2","company","Dept4","Enron","GNU libc"]
ARMS=["person","links","full"]
RULES=["SI","SIR48","SIR72","noise"]
DRAWS=100


def phase(name,part):
    raw=fg.raw_data(name)
    xs=[]; labels=[]; ids=[]; starts=[]
    for g,events,truth in fg.cohort(name,part):
        h=PublicWindow(g,raw)
        for e,y in zip(events,truth,strict=True):
            (nodes,_,_),(x,_,_)=h.design(frames(e))
            assert e["id"]==y["id"] and x.shape==(len(nodes),64)
            yy=np.isin(nodes,y["sources"]).astype(int)
            assert yy.sum()==2
            xs.append(x); labels.append(yy); ids.append(e["id"]); starts.append(g["start"])
    return np.vstack(xs),np.concatenate(labels),np.cumsum([0,*map(len,xs)]),ids,starts


def loss(probs,y,offsets):
    z=np.log(probs)-np.log1p(-probs)
    u=np.exp(z)
    sums=np.add.reduceat(u,offsets[:-1])
    pairs=(sums*sums-np.add.reduceat(u*u,offsets[:-1]))/2
    return float((np.log(pairs)-np.add.reduceat(z*y,offsets[:-1])).mean())


def train(net):
    x,y,_,ids,ts=phase(net,"fit")
    v,vy,offset,vid,vs=phase(net,"cal")
    assert len(ids)==1000 and len(vid)==300 and not set(ids)&set(vid) and max(ts)<min(vs)
    records={}
    models={}
    for m in ARMS:
        train_x=features(x,m)
        val_x=features(v,m)
        models[m]=[]
        for s in SEEDS:
            tree=HistGradientBoostingClassifier(max_leaf_nodes=15,min_samples_leaf=32,max_bins=63,learning_rate=.1,
                max_features=.8,l2_regularization=1,early_stopping=False,warm_start=True,random_state=s)
            best=None
            curve={}
            for rounds in (24,48):
                tree.set_params(max_iter=rounds).fit(train_x,y)
                val=loss(tree.predict_proba(val_x)[:,1],vy,offset)
                if not curve or val<min(curve.values()):
                    best=copy.deepcopy(tree)
                curve[rounds]=val
            records[f"{m}/{s}"]=dict(validation_pair_nll=curve,iterations=best.n_iter_)
            models[m].append(best)
    joblib.dump(models,F/(net+"-trees.joblib"),compress=3)
    return dict(selection=records,train_ids=ids,validation_ids=vid,last_fit=max(ts),first_validation=min(vs),last_validation=max(vs))


def extra(rows,stat,a,b):
    key=a+" minus "+b
    for r in rows:
        r.setdefault("delta",{})[key]=np.subtract(r["scores"][a],r["scores"][b]).tolist()
    stat["differences"][key]=interval([r["delta"][key] for r in rows],[r["window"] for r in rows],stat["keys"])


def tree_test(name,training):
    src=source(name)
    models=joblib.load(F/(src+"-trees.joblib"))
    nns=nets(src,["full"])["full"]
    fgs=[]
    for s in SEEDS:
        model=fg.FGSSI().eval()
        model.load_state_dict(torch.load(weights(src,s),weights_only=True))
        fgs.append(model)
    rows,keys=[],[]
    for g,raw,items,_ in cohort(name):
        keys.append(g["window"])
        if name in ("Dept2","company"):
            assert g["start"]>training[name]["last_validation"]
        if not items:
            continue
        h=PublicWindow(g,raw)
        geo=fg.Geometry(g)
        for off in range(0,len(items),24):
            batch=items[off:off+24]
            vv=[h.design(frames(e)) for e,_ in batch]
            stack=np.vstack([design[0] for _,design in vv])
            offsets=np.cumsum([0,*[len(base[0]) for base,_ in vv]])
            out={m:[t.decision_function(features(stack,m)) for t in models[m]] for m in ARMS}
            with torch.inference_mode():
                inp=geo.inputs([e for e,_ in batch])
                fgout=[net(*inp)[0].numpy() for net in fgs]
            for j,((e,srcs),(base,design)) in enumerate(zip(batch,vv)):
                nodes=base[0]
                ix=[geo.index[n] for n in nodes]
                tie=fg.tie_order(e["id"],len(nodes))
                scores={m:[z[offsets[j]:offsets[j+1]] for z in arr] for m,arr in out.items()}
                scores["MLP"]=logits(nns,design[0])
                scores["FGSSI"]=[z[j,ix,1]-z[j,ix,0] for z in fgout]
                rr={m:[ranks(z,tie,nodes,srcs) for z in arr] for m,arr in scores.items()}
                for budget,k in [("12",12),("5%",math.ceil(.05*len(nodes)))]:
                    rows.append(dict(id=e["id"],window=e["window"],budget=budget,n=len(nodes),ranks=rr,
                        scores={m:[int(max(p)<=k) for p in pairs] for m,pairs in rr.items()}))
    res={}
    for budget in ["12","5%"]:
        rr=[r for r in rows if r["budget"]==budget]
        assert not set(r["id"] for r in rr)&set(training[src]["train_ids"]+training[src]["validation_ids"]) if name==src else True
        res[budget]=agg(rr,keys,"MLP")
        for a,b in [("full","FGSSI"),("links","person"),("full","links")]:
            extra(rr,res[budget],a,b)
    save_rows(F/("tree-"+name+".jsonl.gz"),rows)
    return res


def stress(name):
    raw,gs=test_windows(name)
    models=nets("Dept2",["full","person","links","static"])
    rows=[]
    for j,(g,stream) in enumerate(gs):
        scorer=Scorer(g,raw,models)
        rng=random.Random(719312+100*j+int(name=="Enron"))
        for i in range(DRAWS):
            gold=sorted(rng.sample(g["nodes"],2))
            seed=rng.getrandbits(63)
            mid=g["window"]+"-stress-"+str(i)
            for rule in RULES:
                tr=replay(g,stream,gold,seed,rule)
                obs=mask(tr)
                if obs is None:
                    continue
                lists=score_lists(*scorer.predict(mid,g["window"],obs[1]))
                rows.append(dict(id=mid,window=g["window"],rule=rule,seed=seed,sources=gold,observed=obs[1],lists=lists,scores=hits(lists,gold)))
    keys=[g["window"] for g,_ in gs]
    common={k for k,v in Counter(r["id"] for r in rows).items() if v==len(RULES)}
    ans={}
    for rule in RULES:
        rr=[r for r in rows if r["rule"]==rule]
        ans[rule]={}
        for tag,subset in [("all",rr),("common",[r for r in rr if r["id"] in common])]:
            s=agg(subset,keys,"static")
            if subset:
                extra(subset,s,"links","person")
            ans[rule][tag]=s
    save_rows(F/("stress-"+name+".jsonl.gz"),rows)
    return ans


def run():
    paths=[Path(__file__),Path(__file__).with_name("closure_common.py"),Path(__file__).with_name("closure_cascades.py"),*cohort_inputs(TARGETS)]
    for n in ("Dept2","company"):
        paths.extend(RUN/"ordinary_source_roles"/n/(p+"_"+q+".jsonl") for p in ("fit","cal") for q in ("public","truth"))
    lock(F,dict(estimator="sklearn HistGradientBoostingClassifier",version=sklearn.__version__,objective="candidate binary log loss; both positives/all negatives; 64 columns with original masks",
        validation="choose 24 or 48 rounds by event pair NLL; 15 leaves, 63 bins, 32 min leaf, lr .1, feature fraction .8, L2=1; no target selection",
        budgets=["12","ceil(5%)"],sources="company local, otherwise Dept2",targets=TARGETS,stress_seed=719312,
        stress="100 paired uniform draws in each of 4 Dept2 / 8 evenly spaced Enron windows, .30 synchronous contacts, original half-mask",
        non_SI="48h or 72h infectious then permanent recovery; .02 spontaneous probability per non-source per window, one random observed contact time; simultaneous internal success takes precedence",
        populations="all eligible within rule plus intersection of the four rules"),paths)
    training={n:train(n) for n in ("Dept2","company")}
    dump(F/"training.json",training)
    trees={}
    for name in TARGETS:
        trees[name]=tree_test(name,training)
        print("trees",name,trees[name]["12"]["methods"]["full"]["mean"],flush=True)
    tick=time.perf_counter()
    res={n:stress(n) for n in ("Dept2","Enron")}
    dump(F/"summary.json",dict(tree=trees,stress=res,stress_wall_seconds=time.perf_counter()-tick))
    lines=["|Network|Budget|Person tree|Links tree|Full tree|Full MLP|FGSSI|","|---|---|---:|---:|---:|---:|---:|"]
    for n,d in trees.items():
        for b,s in d.items():
            lines.append("|"+n+"|"+b+"|"+"|".join(f'{100*s["methods"][m]["mean"]:.2f}%' for m in [*ARMS,"MLP","FGSSI"])+"|")
    (F/"table.md").write_text("\n".join(lines)+"\n",encoding="utf8")


def verify():
    s=load(F/"summary.json")
    old=load(BASE/"wp1_budgets/summary.json")
    for name,d in s["tree"].items():
        rows=read_rows(F/("tree-"+name+".jsonl.gz"))
        for b,stat in d.items():
            part=[r for r in rows if r["budget"]==b]
            audit_agg(part,stat)
            for r in part:
                for m,rs in r["ranks"].items():
                    assert r["scores"][m]==[int(max(v)<=(12 if b=="12" else (r["n"]+19)//20)) for v in rs]
            assert np.isclose(stat["methods"]["MLP"]["mean"],old[name]["methods"]["Both-in-"+b]["mean"])
    for name,d in s["stress"].items():
        rows=read_rows(F/("stress-"+name+".jsonl.gz"))
        raw,gs=test_windows(name)
        common={k for k,n in Counter(r["id"] for r in rows).items() if n==4}
        saved={(r["id"],r["rule"]):r for r in rows}
        for j,(g,stream) in enumerate(gs):
            rng=random.Random(719312+100*j+int(name=="Enron"))
            for i in range(DRAWS):
                gold=sorted(rng.sample(g["nodes"],2))
                seed=rng.getrandbits(63)
                for rule in RULES:
                    obs=mask(replay(g,stream,gold,seed,rule))
                    key=(g["window"]+"-stress-"+str(i),rule)
                    assert (obs is not None)==(key in saved)
                    if obs:
                        r=saved[key]
                        assert r["observed"]==obs[1] and r["sources"]==gold
                        check_lists(r)
                        assert hits(r["lists"],gold)==r["scores"]
        for rule,ss in d.items():
            rr=[r for r in rows if r["rule"]==rule]
            audit_agg(rr,ss["all"])
            audit_agg([r for r in rr if r["id"] in common],ss["common"])
    evidence(F,dict(fits=30,mlp_anchor_budgets=10,regenerated_stress_draw_rules=4*12*DRAWS,all_ranks_and_list_hits=True))


if __name__=="__main__":
    if "--verify" not in sys.argv:
        run()
    verify()

