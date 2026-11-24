"""WP5. Common real-depth scan and controlled simulations with asymmetric early roles."""
import random
from collections import defaultdict
from itertools import groupby
from scipy.optimize import minimize
from scipy.special import softmax
from closure_common import *
from closure_discussions import REAL,calendar,cut,project,role
from closure_cascades import Scorer,simulate,test_windows
from historical_thread_roles import choose,objective
from source_prior_shift import pair_posterior

F=BASE/"wp5_sim2real"
DEPTHS=[0,25,50]
REAL_NAMES=[n for n in REAL if "samba" not in n]
CELLS=["uniform_symmetric","active_symmetric","uniform_asymmetric","active_asymmetric"]
DRAWS=160


def role_list(x,theta,tie):
    p,q=[softmax(x[:, :len(t)]@t) for t in theta]
    return choose(p,q,tie)[0] if len(x)>12 else np.arange(12)


def real(name,cfg):
    neural=nets("Dept2",["full"])["full"]
    rows=[];keys=[]
    for g,raw,roots,vol,args in calendar(name,late=name.startswith("python")):
        keys.append(g["window"])
        if not roots:
            continue
        hist=role.History(g["start"],*args)
        assert hist.nodes==g["nodes"]
        public=None
        for mid,gold,original in roots:
            for depth in DEPTHS:
                masked=cut(original,depth)
                if masked is None:
                    continue
                e=project(g,mid,masked)
                seen={v for b in e["observed"] for v in b}
                nodes,x=hist.design(masked)
                ready=len(seen)>=3 and len(nodes)>=12
                lists={}
                if ready:
                    if public is None:
                        public=PublicWindow(g,raw)
                    tie=fg.tie_order(mid,len(nodes))
                    lists["volume"]=[[nodes[i] for i in fg.sorted_nodes(np.array([vol[n] for n in nodes]),tie)[:12]]]
                    for arm in ("roles","reply"):
                        order=role_list(x[0],[v["theta"] for v in cfg[arm]],tie)
                        lists[arm]=[[nodes[i] for i in order]]
                    (ns,_,_),(design,_,_)=public.design(frames(e))
                    assert ns==nodes
                    lists["full"]=[[nodes[i] for i in fg.sorted_nodes(z,tie)[:12]] for z in logits(neural,design)]
                scores={}
                for arm in ("volume","roles","reply","full"):
                    rates=[int(set(gold)<=set(ls)) for ls in lists.get(arm,[[]])]
                    scores[arm]=rates*len(SEEDS) if len(rates)==1 else rates
                rows.append(dict(id=mid,window=g["window"],depth=depth,visible=len(seen),n=len(nodes),ready=ready,
                    reachable=set(gold)<=set(nodes),sources=gold,lists=lists,scores=scores))
    allids={mid for mid,n in Counter(r["id"] for r in rows).items() if n==3}
    commonready={mid for mid,n in Counter(r["id"] for r in rows if r["ready"]).items() if n==3}
    result={}
    for depth in DEPTHS:
        group=[r for r in rows if r["depth"]==depth]
        result[str(depth)]={}
        for label,subset in [("all",group),("common",[r for r in group if r["id"] in allids]),("common_ready",[r for r in group if r["id"] in commonready])]:
            stat=agg(subset,keys,"volume")
            stat.update(scored=sum(r["ready"] for r in subset),scored_reachable=sum(r["ready"] and r["reachable"] for r in subset))
            result[str(depth)][label]=stat
    scans={}
    matched=[]
    all_rows={d:{r["id"]:r for r in rows if r["depth"]==d} for d in DEPTHS}
    for hi,lo in [(25,0),(50,25),(50,0)]:
        ids=sorted(all_rows[hi].keys() & all_rows[lo].keys())
        entry={}
        for arm in ("volume","roles","reply","full"):
            arr=[]
            for mid in ids:
                a,b=all_rows[hi][mid],all_rows[lo][mid]
                diff=np.subtract(a["scores"][arm],b["scores"][arm]).tolist()
                matched.append(dict(id=mid,window=a["window"],method=arm,deeper=hi,shallower=lo,delta=diff))
                arr.append(diff)
            if ids:
                entry[arm]=interval(arr,[all_rows[hi][mid]["window"] for mid in ids],keys)
        scans[f"D{hi}-D{lo}"]=dict(n=len(ids),methods=entry)
    from itertools import combinations
    changes={}
    for a,b in combinations(("volume","roles","reply","full"),2):
        gaps=[result[str(d)]["all"]["methods"][a]["mean"]-result[str(d)]["all"]["methods"][b]["mean"] if result[str(d)]["all"]["n"] else None for d in DEPTHS]
        changes[a+" minus "+b]=dict(gaps=gaps,strict_reversals=[[DEPTHS[i],DEPTHS[i+1]] for i in range(2) if gaps[i] is not None and gaps[i+1] is not None and gaps[i]*gaps[i+1]<0])
    save_rows(F/("real-depth-pairs-"+name+".jsonl.gz"),matched)
    dump(F/("depth-comparisons-"+name+".json"),dict(paired=scans,ordering=changes))
    save_rows(F/("real-"+name+".jsonl.gz"),rows)
    return result


def bounded(raw,g,stream):
    t=g["start"]
    past=[(a,u,v) for a,u,v in raw if t-98*DAY<=a<t]
    counts=Counter((u,v) for _,u,v in past)
    nodes=sorted({u for e in counts for u in e})
    roster=set(nodes)
    public=dict(g,nodes=nodes,edges=[(*e,c) for e,c in sorted(counts.items())])
    clipped=[(t0,[(u,v) for u,v in batch if u in roster and v in roster]) for t0,batch in stream]
    return public,past,[(t0,b) for t0,b in clipped if b]


def prior_roles(raw,g):
    # Completed pseudo-discussions on six preceding two-week slices. Every synthetic
    # role label and every edge used for these counts predates the prediction cutoff.
    end=g["start"]
    out=[Counter(),Counter()]
    for j in range(1,7):
        start=end-(j+1)*14*DAY
        history=Counter(u for t,u,v in raw if start-28*DAY<=t<start)
        roster=sorted({u for t,a,b in raw if start-98*DAY<=t<start for u in (a,b)})
        if len(roster)<3:
            continue
        nodes=set(roster)
        contacts=[z for z in raw if start<=z[0]<start+14*DAY and z[1] in nodes and z[2] in nodes]
        stream=[(t,[(u,v) for _,u,v in batch]) for t,batch in groupby(contacts,key=lambda z:z[0])]
        weights=[1+history[v] for v in roster]
        rng=random.Random(193411+start)
        for _ in range(80):
            root=rng.choices(roster,weights=weights)[0]
            tr=simulate(stream,[root],rng.getrandbits(63))
            if tr and len(tr[0]["joins"])==1 and len(tr)>1:
                out[0][root]+=1
                out[1][tr[0]["joins"][0][0]]+=1
    return out


class SimRoles:
    def __init__(self,g,raw,past):
        self.roster=g["nodes"]
        self.ix={v:i for i,v in enumerate(self.roster)}
        rc=prior_roles(raw,g)
        old=Counter(a for _,a,b in past)
        recent=Counter(a for t,a,b in past if t>=g["start"]-28*DAY)
        self.base=np.log1p([[recent[v],old[v],rc[0][v],rc[1][v]] for v in self.roster])
        self.contacts=np.zeros((len(self.roster),len(self.roster)),int)
        for t,a,b in past:
            if t>=g["start"]-28*DAY:
                self.contacts[self.ix[a],self.ix[b]]+=1

    def design(self,groups):
        vis={v for b in groups for v in b}
        nodes=sorted(set(self.roster)-vis)
        ii=[self.ix[v] for v in nodes]
        jj=[self.ix[v] for v in sorted(vis)]
        x=np.column_stack([self.base[ii],np.log1p(self.contacts[np.ix_(ii,jj)].mean(1)),np.log1p(self.contacts[np.ix_(jj,ii)].mean(0))])
        return nodes,x


def events(g,stream,past,seed,count,cells,depths):
    rng=random.Random(seed)
    w=activity(g,past)
    for i in range(count):
        uniform=rng.choice(g["nodes"])
        biased=rng.choices(g["nodes"],weights=w)[0]
        perm=rng.sample(g["nodes"],2)
        simseed=rng.getrandbits(63)
        for cell in cells:
            root=biased if cell.startswith("active") else uniform
            partner=next(v for v in perm if v!=root)
            asym=cell.endswith("_asymmetric")
            tr=simulate(stream,[root] if asym else [root,partner],simseed)
            if asym:
                if not tr or len(tr[0]["joins"])!=1:
                    continue
                partner=tr[0]["joins"][0][0]
                tr=tr[1:]
            groups=[[v for v,_ in b["joins"]] for b in tr]
            for depth in depths:
                obs=cut(groups,depth)
                if obs and sum(map(len,obs))>=3 and len(g["nodes"])-sum(map(len,obs))>=12:
                    yield i,cell,depth,[root,partner],obs


def fit_roles():
    raw=fg.raw_data("Dept2")
    graphs,streams=fg.make_windows(raw,"Dept2",.15,1.)
    split={}
    dates={}
    for pi,phase in enumerate(("fit","cal")):
        xs=[[],[]]
        dates[phase]=[]
        keys={g["window"] for g,_,_ in fg.cohort("Dept2",phase)}
        for j,g0 in enumerate(graphs):
            if g0["window"] not in keys:
                continue
            g,past,stream=bounded(raw,g0,streams[g0["window"]])
            dates[phase].append(g["start"])
            hist=SimRoles(g,raw,past)
            for i,cell,d,gold,obs in events(g,stream,past,395123+j,320,["active_asymmetric"],[0,50]):
                nodes,x=hist.design(obs)
                for k in (0,1):
                    xs[k].append((x,nodes.index(gold[k])))
        split[phase]=xs
    assert max(dates["fit"])+14*DAY<=min(dates["cal"])
    params=[]
    for k in (0,1):
        trials=[]
        for reg in (.01,.1):
            opt=minimize(objective,np.zeros(6),args=(split["fit"][k],reg),jac=True,method="L-BFGS-B",bounds=[(-8,8)]*6)
            assert opt.success
            trials.append(dict(theta=opt.x.tolist(),reg=reg,nll=objective(opt.x,split["cal"][k],0.)[0]))
        params.append(min(trials,key=lambda z:z["nll"]))
    return dict(params=params,dates=dates,n_fit_views=len(split["fit"][0]),n_validation_views=len(split["cal"][0]))


def simulated(name,cfg):
    raw,gs=test_windows(name)
    models=nets("Dept2",["full"])
    rows=[]
    for j,(g0,stream0) in enumerate(gs):
        if name=="Dept2":
            assert max(cfg["dates"]["cal"])+14*DAY<=g0["start"]
        g,past,stream=bounded(raw,g0,stream0)
        scorer=Scorer(g,past,models)
        rolehist=SimRoles(g,raw,past)
        for i,cell,d,gold,obs in events(g,stream,past,581193+int(name=="Enron")*100+j,DRAWS,CELLS,[0,50]):
            mid=f'{g["window"]}-bridge-{i}'
            nodes,tie,scores=scorer.predict(mid,g["window"],obs)
            ns,x=rolehist.design(obs)
            assert ns==nodes
            loc=1+scores["volume"][0]
            values={"uniform_model":scores["full"],"prior_corrected":[pair_posterior(z,loc,"one")[0] for z in scores["full"]]}
            lst={m:[[nodes[q] for q in fg.sorted_nodes(z,tie)[:12]] for z in zz] for m,zz in values.items()}
            ll=role_list(x,[v["theta"] for v in cfg["params"]],tie)
            lst["role"]=[[nodes[q] for q in ll]]
            rates={m:[int(set(gold)<=set(ls)) for ls in l] for m,l in lst.items()}
            rates["role"]*=len(SEEDS)
            rows.append(dict(id=mid,window=g["window"],cell=cell,depth=d,sources=gold,lists=lst,scores=rates))
    keys=[g["window"] for g,_ in gs]
    tables={}; splits={}
    for d in [0,50]:
        subset=[r for r in rows if r["depth"]==d]
        for cell in CELLS:
            rr=[r for r in subset if r["cell"]==cell]
            tables[cell+"/"+str(d)]=agg(rr,keys,"uniform_model")
        sets=defaultdict(dict)
        for r in subset:
            sets[r["id"]][r["cell"]]=r
        both={i:v for i,v in sets.items() if len(v)==4}
        records=[]
        for i,v in both.items():
            z={m:np.array([v[c]["scores"][m] for c in CELLS]) for m in next(iter(v.values()))["scores"]}
            # Symmetric allocation of the interaction (average both intervention orders).
            for m,(a,b,c,h) in z.items():
                prior=((b-a)+(h-c))/2
                role_effect=((c-a)+(h-b))/2
                records.append(dict(id=i,window=v[CELLS[0]]["window"],model=m,prior=prior.tolist(),role=role_effect.tolist(),total=(h-a).tolist()))
        splits[str(d)]={}
        for m in ("uniform_model","prior_corrected","role"):
            rs=[r for r in records if r["model"]==m]
            splits[str(d)][m]=dict(n=len(rs),**{k:interval([r[k] for r in rs],[r["window"] for r in rs],keys) for k in ["prior","role","total"]}) if rs else dict(n=0)
        save_rows(F/(name+f"-contributions-D{d}.jsonl.gz"),records)
    save_rows(F/("sim-"+name+".jsonl.gz"),rows)
    return dict(cells=tables,common_draw_decomposition=splits)


def run():
    cfgpath=role.OUT/"protocol.json"
    lock(F,dict(real=REAL_NAMES,python_test="2019 Feb-Apr only, following frozen 2017-18 fit and validation",
        depths=DEPTHS,eligibility="D0 unique second + >=1 later; D25/D50 hide maximal complete prefix up to floor(q*N), require hidden batch and >=3 later. All depths abstain if <3 roster-visible or <12 legal. Missing author or abstention scores zero.",
        comparison="all eligible, common roots, common ready. D50 matches historical mask; D0 uses stricter shared abstention.",
        sim="98-day closed roster, Dept2 4 / Enron 8 windows. 160 coupled attempts per cell/window. 2x2 uniform vs active first sender, symmetric second uniform vs unique first activation. D0 and D50.",
        sim_roles="28/98-day sends, counts of root/first activation in 80 auxiliary historical cascades per completed fortnight in 98 days, mean incoming/outgoing 28-day contacts. Role-wise conditional likelihood on earlier Dept2 only, validation selects penalty; frozen transfer.",
        prior="one-active unordered correction (wi+wj); first sender actually weighted; imperfect for asymmetric targets",
        decomposition="common eligible attempts only, equal allocation of interaction across both orders. These are controlled simulated contrasts; real causal shares are unidentified.",
        seeds=dict(history=193411,train=395123,test=581193)),[Path(__file__),Path(__file__).with_name("closure_discussions.py"),Path(__file__).with_name("closure_cascades.py"),cfgpath])
    cfg=load(cfgpath)["models"]
    reals={name:real(name,cfg) for name in REAL_NAMES}
    fit=fit_roles()
    dump(F/"role_fit.json",fit)
    sims={name:simulated(name,fit) for name in ("Dept2","Enron")}
    dump(F/"summary.json",dict(real=reals,sim=sims))
    fig,axes=plt.subplots(2,3,figsize=(11,6.2),sharex=True)
    for ax,(name,d) in zip(axes.flat,reals.items()):
        for m in ("volume","roles","reply","full"):
            yy=[d[str(k)]["all"]["methods"].get(m,{}).get("mean",np.nan)*100 for k in DEPTHS]
            ax.plot(DEPTHS,yy,"o-",label=m)
        ax.set_title(name);ax.set_xticks(DEPTHS,["D0","D25","D50"])
    axes[0,0].set_ylabel("Both-in-12 (%)");axes[1,0].set_ylabel("Both-in-12 (%)")
    axes[0,0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(F/"depth_scan.png",dpi=170);plt.close(fig)
    lines=["|Network|Depth|Model|Common draws|Prior (pp)|Role (pp)|Total (pp)|","|---|---|---|---:|---:|---:|---:|"]
    for n,s in sims.items():
        for d,stats in s["common_draw_decomposition"].items():
            for m,z in stats.items():
                if z["n"]:
                    lines.append(f'|{n}|D{d}|{m}|{z["n"]}|'+ "|".join(f'{100*z[k]["mean"]:+.2f}' for k in ("prior","role","total"))+"|")
    (F/"mechanisms.md").write_text("\n".join(lines)+"\n",encoding="utf8")


def verify():
    from real_trained_mixtures import reference
    s=load(F/"summary.json")
    count=0;anchored=0
    for name,d in s["real"].items():
        rows=read_rows(F/("real-"+name+".jsonl.gz"))
        if name.startswith("python"):
            old=reference(name,True)
            rr={r["id"]:r for r in rows if r["depth"]==50}
            assert rr.keys()==old.keys()
            for i,(src,ok,ls) in old.items():
                r=rr[i]
                assert src==r["sources"] and ok==r["ready"] and ls==r["lists"].get("volume",[[]])[0]
            anchored+=len(old)
        else:
            old={r["id"]:r for r in fg.read_rows(role.OUT/"closure"/(name+".jsonl"))}
            shallow={r["id"]:r for r in rows if r["depth"]==0}
            assert old.keys()==shallow.keys()
            for i,r in shallow.items():
                if r["ready"]:
                    for m in ("volume","roles","reply"):
                        assert (old[i]["roles"][m]==3)==bool(r["scores"][m][0])
                    anchored+=1
        count+=len(rows)
        common={i for i,n in Counter(r["id"] for r in rows).items() if n==3}
        ready={i for i,n in Counter(r["id"] for r in rows if r["ready"]).items() if n==3}
        for dep,stats in d.items():
            rr=[r for r in rows if r["depth"]==int(dep)]
            for label,part in [("all",rr),("common",[r for r in rr if r["id"] in common]),("common_ready",[r for r in rr if r["id"] in ready])]:
                audit_agg(part,stats[label])
        for r in rows:
            for m,values in r["scores"].items():
                z=[int(set(r["sources"])<=set(ls)) for ls in r["lists"].get(m,[[]])]
                assert z*(len(SEEDS) if len(z)==1 else 1)==values
    for n,d in s["sim"].items():
        rows=read_rows(F/("sim-"+n+".jsonl.gz"))
        for key,t in d["cells"].items():
            c,depth=key.split("/")
            audit_agg([r for r in rows if r["cell"]==c and r["depth"]==int(depth)],t)
        for depth,t in d["common_draw_decomposition"].items():
            cov=read_rows(F/(n+f"-contributions-D{depth}.jsonl.gz"))
            for m,result in t.items():
                rs=[r for r in cov if r["model"]==m]
                assert len(rs)==result["n"]
                for r in rs:
                    assert np.allclose(np.add(r["prior"],r["role"]),r["total"])
                for k in ("prior","role","total"):
                    if rs:
                        assert np.isclose(np.mean([r[k] for r in rs]),result[k]["mean"])
    evidence(F,dict(real_event_depths=count,archive_anchor_checks=anchored,sim_factorial_identity=True))


if __name__=="__main__":
    if "--verify" not in sys.argv:
        run()
    verify()

