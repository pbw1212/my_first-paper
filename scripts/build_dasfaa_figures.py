"""Build the seven LNCS vector figures from sealed summaries and archived Enron hits."""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs"
BASE = RUN / "paper_closure"
PAPER = ROOT / "paper_dasfaa"
OUT = PAPER / "figures"
WIDTH = 4.8
COL = ["#0072B2", "#D55E00", "#00836B", "#805891", "#545965", "#A46716"]
MARK = ["o","s","^","D","v","P"]
NAMES = {"company":"Company","GNU libc":"GNU libc","python-dev":"Python-Dev",
         "python-ideas":"Python-Ideas","r-devel":"R-devel","r-package-devel":"R-package",
         "libc-alpha":"GNU libc 21–22","libc-alpha-2024":"GNU libc 2024","samba-technical-2020":"Samba 2020"}
PR = ["flat","mixed","mostly_active","linked"]
ARMS = ["uniform_offset","mixed_plain","mixed_offset","mixed_component"]


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def wp(s):
    return read(BASE/s/"summary.json")


def save(fig, name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+".pdf"), metadata={"Author":"", "CreationDate":None,"ModDate":None})
    plt.close(fig)


def axes(n,height,**kw):
    return plt.subplots(1,n,figsize=(WIDTH,height),**kw)


def legend(fig, handles, cols, top=.995):
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,top),ncol=cols,
               frameon=False,columnspacing=.95,handlelength=1.3,handletextpad=.4,borderpad=.1)


def style(ax, title):
    ax.set_title(title,loc="left",fontweight="bold",fontsize=8.5,pad=4)
    ax.grid(axis="y",color="#dfE3E5",lw=.55)
    ax.set_axisbelow(True)


def overview():
    fig, ax = axes(1,2.52)
    fig.subplots_adjust(left=.01,right=.99,bottom=.025,top=.98)
    ax.set(xlim=(0,10),ylim=(0,6)); ax.axis("off")
    def label(x,y,t,**kw):
        ax.text(x,y,t,ha="center",va="center",**kw)
    def arrow(a,b, color="#374859",ls="-"):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=7,linewidth=.9,
                                    color=color,linestyle=ls,shrinkA=0,shrinkB=0))
    label(2.45,5.7,"(a) Temporal contact record",weight="bold")
    label(7.58,5.7,"(b) Candidate ranking",weight="bold")
    ax.add_patch(Rectangle((.12,2.2),2.82,2.1,fc="#f4e9dd",ec="none"))
    ax.add_patch(Rectangle((2.97,2.2),1.91,2.1,fc="#e2f1ed",ec="none"))
    label(1.55,4.77,"Hidden prefix");label(3.92,4.77,"Suffix  O")
    for (a,b) in [((.73,3.71),(1.38,3.71)),((1.80,3.71),(2.36,3.71)),
                  ((2.75,3.71),(3.42,3.71)),((3.87,3.71),(4.38,3.16)),((3.87,3.75),(4.34,4.06))]:
        arrow(a,b,COL[0])
    for x,y,t in [(.52,3.71,"A"),(1.60,3.71,"u"),(2.57,3.71,"v"),
                   (3.65,3.71,"w"),(4.52,3.02,"x"),(4.54,4.10,"y"),(.52,2.76,"B")]:
        label(x,y,t,bbox=dict(boxstyle="circle,pad=.15",ec="#58636a",fc="white",lw=.7))
    label(1.86,2.71,"no transmissions",fontsize=8)
    arrow((.25,1.89),(4.76,1.89))
    label(2.5,1.52,"Time within the cascade")
    label(2.49,.75,"Visible batches: {w}, {x, y}\nPre-window ties: A → w, B → y")
    for y,t in [(4.86,"Personal / contact / full history"),(3.60,"Two-layer MLP  →  score  $s_i$"),
                (2.34,"Unordered-pair loss\nand optional prior offset"),(1.06,"Top-12 candidate list")]:
        ax.add_patch(FancyBboxPatch((5.26,y-.43),4.58,.86,boxstyle="round,pad=.03,rounding_size=.09",
                                   fc="#edf2f6" if y>1.1 else "#dbece7",ec="#68818d",lw=.7))
        label(7.55,y,t)
    for y in [4.86,3.6,2.34]:arrow((7.55,y-.47),(7.55,y-.79))
    save(fig,"fig1_overview")


def budgets():
    data=wp("wp1_budgets")
    fig, aa=axes(2,2.54,sharey=True,gridspec_kw={"width_ratios":[3,2]})
    fig.subplots_adjust(left=.11,right=.98,bottom=.19,top=.73,wspace=.12)
    handles=[]
    for i,(n,v) in enumerate(data.items()):
        handles.append(Line2D([],[],marker=MARK[i],ls="",color=COL[i],label=NAMES.get(n,n)))
        for ax,ks in zip(aa,[["6","12","24"],["5%","10%"]]):
            for j,k in enumerate(ks):
                s=v["methods"]["Both-in-"+k]; x=j+(i-2.5)*.105
                ax.vlines(x,*np.array(s["ci95"])*100,color=COL[i],lw=.85)
                ax.plot(x,100*s["mean"],MARK[i],color=COL[i],ms=3.2,clip_on=False)
            ax.set(xticks=range(len(ks)),xticklabels=ks,xlim=(-.44,len(ks)-.56),ylim=(-1,53))
    aa[0].set_ylabel("Joint coverage (%)")
    for ax,t in zip(aa,["(a) Fixed list size","(b) Fraction of pool"]):style(ax,t)
    legend(fig,handles,3)
    save(fig,"fig2_budgets")


def selectivity():
    data=wp("wp2_predictability")["networks"]
    fig,aa=axes(2,2.64,sharey=True)
    fig.subplots_adjust(left=.115,right=.985,top=.88,bottom=.19,wspace=.13)
    # Axes-relative anchors keep the departmental labels off their markers and leaders.
    simulated_anchors = {
        "Dept1": (.78, .56), "Dept2": (.81, .80), "company": (.03, .35),
        "Dept4": (.48, .95), "Enron": (.40, .68), "CollegeMsg": (.11, .56),
        "GNU libc": (.51, .39),
    }
    real_anchors = [(.63,.42),(.60,.61),(.04,.65),(.03,.39),(.60,.04),(.37,.12),(.27,.27)]
    for j,(prefix,ax,title) in enumerate(zip(["sim","real"],aa,["(a) vs. personal history","(b) vs. post volume"])):
        rows=[(n.split(" ",1)[1],d["auc"],100*d["gain"]) for n,d in data.items() if n.startswith(prefix+" ")]
        xx,yy=np.array([(a,b) for _,a,b in rows]).T
        ax.set(xlim=(.48,.785),ylim=(-27,27),xticks=[.5,.6,.7],yticks=[-20,-10,0,10,20])
        ax.axhline(0,lw=.6,color="#798387")
        xs=np.linspace(xx.min(),xx.max(),50);ax.plot(xs,np.polyval(np.polyfit(xx,yy,1),xs),color=COL[j],lw=.8)
        ax.scatter(xx,yy,color=COL[j],s=12,marker=MARK[j])
        anchors = [simulated_anchors[n] for n, _, _ in rows] if prefix == "sim" else real_anchors
        for (n,x,y),loc in zip(rows,anchors,strict=True):
            short={"libc-alpha":"libc 21–22","libc-alpha-2024":"libc 2024","samba-technical-2020":"Samba",
                   "python-dev":"Py-dev","python-ideas":"Py-ideas","r-package-devel":"R-pkg"}.get(n,NAMES.get(n,n))
            ax.annotate(short,(x,y),xytext=loc,textcoords="axes fraction",fontsize=8,va="center",
                        arrowprops=dict(arrowstyle="-",lw=.45,color="#738089",shrinkA=1,shrinkB=2),
                        bbox=dict(fc="white",ec="none",pad=.15))
        style(ax,title);ax.set_xlabel("Contact AUC")
    aa[0].set_ylabel("Both-in-12 gain (pp)")
    save(fig,"fig3_selectivity")


def prior_intervals(summary):
    """Replay the archived paired bootstrap design, retaining marginal as well as paired rates."""
    arc=Path(os.environ.get("CLOSURE_ARCHIVE",ROOT.parent))
    path=arc/"runs/trained_source_mixtures/Enron/test/events.jsonl"
    manifest=read(RUN/"trained_source_mixtures/Enron/test/manifest.json")
    stats_path=PAPER/"evidence/prior_window_hits.json"
    if path.is_file():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest["sha256"]
        rows=[json.loads(l) for l in path.read_text(encoding="utf8").splitlines()]
        keys=manifest["windows"]; ix={k:i for i,k in enumerate(keys)}
        sufficient={}
        for law in PR:
            counts=np.zeros(len(keys),int);hits=np.zeros((len(keys),4,5),int)
            for row in rows:
                if row["profile"]==law:
                    k=ix[row["window"]];counts[k]+=1;hits[k]+=np.reshape(row["hits"],(4,5))
            sufficient[law]=dict(n=counts.tolist(),hits=hits.tolist())
        packed=dict(source_sha256=manifest["sha256"],profiles=sufficient)
        if stats_path.exists(): assert packed==read(stats_path)
        else: stats_path.write_text(json.dumps(packed,separators=(",",":"))+"\n",encoding="utf8")
    data=read(stats_path);assert data["source_sha256"]==manifest["sha256"]
    result={}
    for law,v in data["profiles"].items():
        c=np.array(v["n"]);h=np.array(v["hits"]); k=len(c)
        rng=np.random.default_rng(431173)
        blocks=((rng.integers(k,size=(10000,(k+2)//3,1))+np.arange(3))%k).reshape(10000,-1)[:,:k]
        seeds=rng.integers(5,size=(10000,1,5)); denom=c[blocks].sum(1);ok=denom>0
        boot={}
        for i,arm in enumerate(ARMS):
            z=h[:,i]; boot[arm]=100*z[blocks[:,:,None],seeds].sum(1).mean(1)[ok]/denom[ok]
            assert np.isclose(z.sum()/c.sum()*20,summary[law]["methods"][arm]["both_pct"])
        delta=boot["mixed_offset"]-boot["mixed_plain"]
        assert np.allclose(np.quantile(delta,[.025,.975]),summary[law]["paired_to_mixed_plain"]["mixed_offset"]["ci95_pp"])
        result[law]={a:np.quantile(b,[.025,.975]) for a,b in boot.items()}
    return result


def priors():
    summary=read(RUN/"trained_source_mixtures/summary.json")["Enron"]
    ci=prior_intervals(summary)
    fig,aa=axes(2,2.82)
    fig.subplots_adjust(left=.115,right=.985,top=.72,bottom=.19,wspace=.34)
    xs=np.arange(4)
    for i,(arm,n) in enumerate(zip(["mixed_plain","mixed_offset","uniform_offset"],["Direct","Explicit","Deploy"])):
        vals=[summary[k]["methods"][arm]["both_pct"] for k in PR];xx=xs+(i-1)*.24
        aa[0].bar(xx,vals,.23,color=COL[i],label=n)
        aa[0].vlines(xx,*np.array([ci[k][arm] for k in PR]).T,color="#1d2629",lw=.85)
    aa[0].set(xticks=xs,xticklabels=["Unif.","Equal","80%","Tie"],ylabel="Joint coverage (%)",ylim=(0,55))
    bottom=np.zeros(4);nets=["Dept2","company","Dept4","Enron"]
    for j,(key,lab) in enumerate(zip(["conditional","missing_anchor_loss","partner_ranking_loss"],["Captured","Emitter missed","Partner ranking"])):
        vals=[read(RUN/"source_prior_shift"/n/"ceiling/summary.json")["both"]["one"]["methods"]["calibrated"][key] for n in nets]
        aa[1].bar(xs,vals,.68,bottom=bottom,color=["#216481","#ddb681","#cbd5da"][j],hatch=["","///",""][j],edgecolor="#50616a",lw=.45,label=lab);bottom+=vals
    aa[1].set(xticks=xs,xticklabels=["D2","Co.","D4","Enr."],ylim=(0,100))
    for ax,t in zip(aa,["(a) Enron source prior","(b) Oracle ceiling (%)"]):style(ax,t)
    for ax in aa:
        ax.legend(loc="lower left",bbox_to_anchor=(-.03,1.16),ncol=1,frameon=False,fontsize=8,
                  handlelength=1.2,labelspacing=.12,borderpad=0)
    save(fig,"fig4_prior_and_ceiling")


def multisource():
    data=wp("wp4_ksources")
    fig,aa=axes(2,2.4,sharey=True)
    fig.subplots_adjust(left=.11,right=.98,top=.76,bottom=.20,wspace=.16)
    xs=np.arange(1,4)
    for i,(n,by) in enumerate(data.items()):
        z=[by[f"hidden_uniform/{k}"] for k in xs]
        aa[0].plot(xs,[100*r["silent_event"]["mean"] for r in z],marker=MARK[i],color=COL[i])
        for key,ls in [("All-in-12","-"),("Recall@12","--")]:
            aa[1].plot(xs,[100*r[key]["methods"]["full"]["mean"] for r in z],ls,marker=MARK[i],color=COL[i])
    for ax,t in zip(aa,["(a) At least one silent source","(b) Full model at 12"]):
        style(ax,t);ax.set(xticks=xs,xlim=(.9,3.1),ylim=(-3,101),xlabel="Actual source count")
    aa[0].set_ylabel("Events / recall (%)")
    handles=[Line2D([],[],color=COL[i],marker=MARK[i],ls="",label=n) for i,n in enumerate(data)]
    handles += [Line2D([],[],color=".3",ls=l,label=n) for l,n in [("-","All in list"),("--","Per-source recall")]]
    legend(fig,handles,4);save(fig,"fig5_multisource")


def depths():
    data=wp("wp5_sim2real")["real"]
    fig,aa=plt.subplots(2,3,figsize=(WIDTH,3.25))
    fig.subplots_adjust(left=.09,right=.985,top=.76,bottom=.10,wspace=.34,hspace=.64)
    methods=["volume","roles","reply","full"]; names=["Post volume","Role history","Roles + replies","Full temporal"]
    styles=["-","--",":","-."]
    for ax,(n,by) in zip(aa.flat,data.items()):
        for i,m in enumerate(methods):
            rates=[100*by[d]["common"]["methods"][m]["mean"] for d in ["0","25","50"]]
            ax.plot(range(3),rates,ls=styles[i],marker=["o","s","^","x"][i],ms=5.4-i*.65,
                    mfc="none",mew=.8,color=COL[i],label=names[i])
        style(ax,NAMES[n]);ax.set(xticks=range(3),xticklabels=["D0","D25","D50"],xlim=(-.12,2.12))
        top=max(1,ax.get_ylim()[1]);ax.set_ylim(-.05*top,top)
    legend(fig,aa.flat[0].get_legend_handles_labels()[0],2)
    save(fig,"fig6_real_depth")


def decomposition():
    data=wp("wp5_sim2real")["sim"];xx=np.arange(4)
    fig,ax=axes(1,2.32);fig.subplots_adjust(left=.13,right=.98,top=.81,bottom=.24)
    pairs=[(n,d) for n in data for d in ["0","50"]]
    for j,(key,label) in enumerate([("prior","Sender activity prior"),("role","First-activation role")]):
        stats=[data[n]["common_draw_decomposition"][d]["uniform_model"][key] for n,d in pairs]
        x=xx+(j-.5)*.30
        ax.bar(x,[100*r["mean"] for r in stats],.29,color=COL[j],label=label)
        ax.errorbar(x,[100*r["mean"] for r in stats],yerr=np.array([[100*(r["mean"]-r["ci95"][0]),100*(r["ci95"][1]-r["mean"])] for r in stats]).T,fmt="none",ecolor="#202b31",lw=.9,capsize=2)
    ax.axhline(0,color="#535c60",lw=.65);ax.set(xticks=xx,xticklabels=[n+"\nD"+d for n,d in pairs], ylabel="Coverage change (pp)",ylim=(-12,67))
    ax.grid(axis="y",color=".88",lw=.5);ax.set_axisbelow(True)
    legend(fig,ax.get_legend_handles_labels()[0],2)
    save(fig,"fig7_sim2real_decomp")


if __name__=="__main__":
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8.5,
        "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.65,"lines.linewidth":1.15,"lines.markersize":4,
        "ytick.major.size":2,"xtick.major.size":2,"xtick.major.pad":2,"ytick.major.pad":2})
    for path,d in read(BASE/"wp6_tables/protocol.json")["inputs"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==d,path
    for build in [overview,budgets,selectivity,priors,multisource,depths,decomposition]:
        build()
