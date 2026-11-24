"""WP2. Tie-aware per-event historical contact discrimination and frozen feature gains."""
from scipy.stats import rankdata,pearsonr,spearmanr
from closure_common import *
from closure_discussions import REAL,calendar,project,role

F=BASE/"wp2_predictability"
SIM=["Dept1",*NAMES]
VARS=["auc","contact_gap","outdegree","density","median_candidates","repeat28"]


def real_windows(name):
    for g,h,roots,vol,_ in calendar(name):
        yield g,h,[(project(g,mid,b),gold) for mid,gold,b in roots],vol


def prior_lists(name):
    path=RECENT/name/"predictions.jsonl" if name in ("company","Dept2") else RECENT/"posthoc"/name/"predictions.jsonl"
    return {r["id"]:r["lists"] for r in fg.read_rows(path)} if path.exists() else {}


def network(name,real):
    archive={} if real else prior_lists(name)
    models={} if archive else nets(source(name),["person","links"])
    rows,keys,struct=[],[],[]
    for g,raw,items,vol in (real_windows(name) if real else cohort(name)):
        keys.append(g["window"])
        ix={v:i for i,v in enumerate(g["nodes"])}
        n=len(ix)
        counts=np.zeros((n,n),dtype=np.int32)
        for t,u,v in raw:
            if g["start"]-28*DAY<=t<g["start"]:
                counts[ix[u],ix[v]]+=1
        edges=len({(a,b) for a,b,c in g["edges"] if a!=b})
        struct.append(dict(window=g["window"],nodes=n,edges=edges,outdegree=edges/n if n else 0,
                           density=edges/(n*(n-1)) if n>1 else 0,contacts28=int(counts.sum()),unique28=int(np.count_nonzero(counts))))
        hist=None
        for e,gold in items:
            seen={v for b in e["observed"] for v in b}
            nodes=sorted(ix.keys()-seen)
            ready=len(seen)>0 and len(nodes)>=12
            y=set(gold)
            hits={m:[0]*len(SEEDS) for m in (["person","links","volume"] if real else ["person","links"])}
            r=dict(id=e["id"],window=g["window"],n=len(nodes),ready=ready)
            if ready and y<=set(nodes) and len(nodes)>2:
                scores=counts[np.ix_([ix[v] for v in nodes],[ix[v] for v in seen])].sum(1)
                pos=np.array([v in y for v in nodes])
                a=scores[pos]
                b=scores[~pos]
                r["diagnostic"]=dict(auc=float((rankdata(scores)[pos].sum()-3)/(2*len(b))),
                    true_contact=float((a>0).mean()),other_contact=float((b>0).mean()),
                    true_counts=a.tolist(),negative_hist=[[int(v),int(c)] for v,c in zip(*np.unique(b,return_counts=True))])
            if archive:
                for m in hits:
                    hits[m]=[int(y<=set(archive[e["id"]][f"{m}/{s}"])) for s in SEEDS]
            elif ready:
                if hist is None:
                    hist=PublicWindow(g,raw)
                (order,_,_),(x,_,_)=hist.design(frames(e))
                assert order==nodes
                tie=fg.tie_order(e["id"],len(nodes))
                for m,ms in models.items():
                    hits[m]=[int(max(ranks(z,tie,nodes,gold))<=12) for z in logits(ms,features(x,m))]
                if real:
                    hit=int(max(ranks([vol[v] for v in nodes],tie,nodes,gold))<=12)
                    hits["volume"]=[hit]*len(SEEDS)
            r["scores"]=hits
            rows.append(r)
    base="volume" if real else "person"
    stat=agg(rows,keys,base)
    if real:
        for r in rows:
            r["delta"]["links minus person"]=np.subtract(r["scores"]["links"],r["scores"]["person"]).tolist()
        stat["differences"]["links minus person"]=interval([r["delta"]["links minus person"] for r in rows],[r["window"] for r in rows],keys)
    d=[r["diagnostic"] for r in rows if "diagnostic" in r]
    pct=lambda k:float(np.mean([z[k] for z in d])) if d else None
    diff=stat["differences"]["links minus "+base]["mean"]
    base_rate=stat["methods"][base]["mean"]
    stat.update(auc=pct("auc"),true_contact=pct("true_contact"),other_contact=pct("other_contact"),
                contact_gap=pct("true_contact")-pct("other_contact") if d else None,auc_n=len(d),ready=sum(r["ready"] for r in rows),
                outdegree=np.mean([z["outdegree"] for z in struct]),density=np.mean([z["density"] for z in struct]),
                median_candidates=np.median([r["n"] for r in rows]),repeat28=1-sum(z["unique28"] for z in struct)/sum(z["contacts28"] for z in struct),
                windows=struct,gain=diff,relative_gain=diff/base_rate if base_rate else None,baseline=base)
    return rows,stat


def run():
    paths=cohort_inputs(SIM)+[role.OUT/"protocol.json",Path(__file__),Path(__file__).with_name("closure_discussions.py"),Path(__file__).with_name("closure_common.py")]
    lock(F,dict(sim=SIM,real=REAL,real_mask="D0; >=1 projected visible, >=12 candidates; Python 2017-18 only frozen Dept2 scores; both targets are early speakers",
        diagnostic="only scoreable events with two roster targets; event AUC, ties 0.5, directed outgoing sum to all visible; unweighted events",
        gain="all eligible Both-in-12 including abstentions; links-person for simulated and links-volume for real; common links-person also provided",
        structure="window-equal unique directed outdegree and density, median legal pool over eligible events; pooled repeated directed contacts beyond first / all 28-day contacts",
        correlation="unweighted descriptive network/period points; split real/sim, pooled and common contrast. Dependence across periods and related networks."),paths)
    result={}
    for typ,names in [("sim",SIM),("real",REAL)]:
        for name in names:
            key=typ+" "+name
            rows,s=network(name,typ=="real")
            save_rows(F/(key+".jsonl.gz"),rows)
            result[key]=s
            print(key,s["n"],s["auc_n"],s["auc"],s["gain"],flush=True)
    corr={}
    for group in ["sim","real","pooled","pooled_common"]:
        cohort0=[v for k,v in result.items() if (group.startswith("pooled") or k.startswith(group)) and v["auc"] is not None]
        out={}
        for var in VARS:
            a=[z[var] for z in cohort0]
            y=[z["differences"]["links minus person"]["mean"] if group=="pooled_common" else z["gain"] for z in cohort0]
            out[var]=dict(n=len(y),pearson=float(pearsonr(a,y).statistic),spearman=float(spearmanr(a,y).statistic))
        corr[group]=out
    dump(F/"summary.json",dict(networks=result,correlations=corr))
    fig,ax=plt.subplots(figsize=(8,5))
    for typ in ["sim","real"]:
        group=[(k,v) for k,v in result.items() if k.startswith(typ) and v["auc"] is not None]
        xx=np.array([v["auc"] for _,v in group])
        yy=np.array([v["gain"]*100 for _,v in group])
        ax.scatter(xx,yy,label=typ)
        grid=np.linspace(xx.min(),xx.max())
        ax.plot(grid,np.polyval(np.polyfit(xx,yy,1),grid),alpha=.7)
        for (k,v),x,y in zip(group,xx,yy):
            ax.annotate(k.split(" ",1)[1],(x,y),xytext=(3,4),textcoords="offset points",fontsize=7)
    ax.axhline(0,color=".7",lw=1)
    ax.set(xlabel="Mean within-event contact AUC",ylabel="Both-in-12 gain (percentage points)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(F/"auc_gain.png",dpi=160)
    plt.close(fig)


def verify():
    summary=load(F/"summary.json")
    old=load(RECENT/"contact_diagnostic_and_audit.json")
    for key,s in summary["networks"].items():
        rows=read_rows(F/(key+".jsonl.gz"))
        audit_agg(rows,s)
        for r in rows:
            if "diagnostic" not in r:
                continue
            d=r["diagnostic"]
            neg=d["negative_hist"]
            ref=sum(c*((a>v)+0.5*(a==v)) for a in d["true_counts"] for v,c in neg)/(2*sum(c for _,c in neg))
            assert abs(ref-d["auc"])<1e-12
        if key in ("sim Dept2","sim company"):
            anchor=old[key.split()[1]]
            assert np.allclose([s["auc"],s["true_contact"],s["other_contact"]],[anchor[k] for k in ("mean_within_event_auc","true_source_has_contact","other_candidate_has_contact")],rtol=0,atol=1e-12)
    evidence(F,dict(network_periods=len(summary["networks"]),independent_auc="pair enumeration from negative count histograms",archive_anchors=["Dept2","company"]))


if __name__=="__main__":
    if "--verify" not in sys.argv:
        run()
    verify()

