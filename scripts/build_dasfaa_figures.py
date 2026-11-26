"""Build the six LNCS vector figures from sealed summaries and archived Enron hits."""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.path import Path as PlotPath
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.legend_handler import HandlerTuple
import numpy as np
from scipy.stats import spearmanr

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
    # Retain the common LNCS width and a small margin above and below all artists.
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    crop = Bbox.from_extents(0, bounds.y0 - .025, WIDTH, bounds.y1 + .025)
    fig.savefig(OUT/(name+".pdf"), bbox_inches=crop, metadata={"Author":"", "CreationDate":None,"ModDate":None})
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
    label(1.55,4.77,"Hidden prefix");label(3.94,4.77,"Visible suffix")
    for a,b in [((.73,3.71),(1.38,3.71)),((1.80,3.71),(2.36,3.71)),
                ((2.75,3.71),(3.42,3.71)),((3.78,3.54),(4.04,3.18)),
                ((3.87,3.75),(4.31,4.08))]:
        arrow(a,b,COL[0])
    tie_color = "#666971"
    tie_style = (0, (2.8, 2))
    # Route historical contacts around the chain of successful transmissions.
    for vertices in [
        [(.61, 3.91), (1.35, 4.59), (2.95, 4.59), (3.55, 3.92)],
        [(.64, 2.64), (1.36, 2.25), (2.70, 2.28), (3.44, 2.39),
         (4.39, 2.52), (4.76, 3.06), (4.52, 3.94)],
    ]:
        path = PlotPath(vertices, [PlotPath.MOVETO] + [PlotPath.CURVE4] * (len(vertices) - 1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=7,
                                     color=tie_color, lw=.8, linestyle=tie_style))
    for x,y,t in [(.52,3.71,"a"),(1.60,3.71,"u"),(2.57,3.71,"v"),
                   (3.65,3.71,"w"),(4.12,3.02,"x"),(4.49,4.13,"y"),(.52,2.76,"b")]:
        label(x,y,f"${t}$",fontsize=9,bbox=dict(boxstyle="circle,pad=.15",ec="#58636a",fc="white",lw=.7))
    label(1.86,2.71,"no transmissions",fontsize=8)
    arrow((.25,1.89),(4.76,1.89))
    label(2.5,1.52,"Time within the cascade")
    label(2.49,.95,"Visible batches: {w}, {x, y}")
    for y,color,ls,name in [(.54,COL[0],"-","Unobserved transmission"),(.15,tie_color,tie_style,"Pre-window tie")]:
        arrow((.48,y),(1.17,y),color,ls)
        ax.text(1.3,y,name,va="center",fontsize=8)
    ax.add_patch(FancyBboxPatch((5.26,4.28),4.58,1.0,boxstyle="round,pad=.03,rounding_size=.09",
                               fc="#edf2f6",ec="#68818d",lw=.7))
    label(7.55,5.06,"Nested features",fontsize=8.5)
    tier_boxes = [
        (5.40, 1.01, "Personal\n26", "#fff"),
        (6.54, 1.41, "+ Recent\ncontacts 34", "#eee9f1"),
        (8.08, 1.62, "+ Full temporal\nhistory 64", "#e2f1ed"),
    ]
    for x, width, name, fill in tier_boxes:
        ax.add_patch(FancyBboxPatch((x, 4.39), width, .47,
                                   boxstyle="round,pad=0,rounding_size=.04",
                                   fc=fill, ec="none"))
        label(x + width / 2, 4.625, name, fontsize=6.45, linespacing=.93)
    for y,t in [(3.58,"Shared MLP → score $s_i$"),
                (2.32,r"$q(i,j)=\exp(s_i+s_j)/Z$" "\n" r"$O(|V|)$ normalization"),
                (1.06,"Top-$K$ candidate list")]:
        ax.add_patch(FancyBboxPatch((5.26,y-.43),4.58,.86,boxstyle="round,pad=.03,rounding_size=.09",
                                   fc="#edf2f6" if y>1.1 else "#dbece7",ec="#68818d",lw=.7))
        label(7.55,y,t,fontsize=9)
    for top,bottom in [(4.25,4.04),(3.12,2.78),(1.86,1.52)]:
        arrow((7.55,top),(7.55,bottom))
    save(fig,"fig1_overview")


def budgets():
    data=wp("wp1_budgets")
    fig, aa=axes(2,2.19,sharey=True,gridspec_kw={"width_ratios":[3,2]})
    fig.subplots_adjust(left=.115,right=.98,bottom=.21,top=.75,wspace=.48)
    handles=[]
    for i,(n,v) in enumerate(data.items()):
        handles.append(Line2D([],[],marker=MARK[i],ls="",color=COL[i],label=NAMES.get(n,n)))
        for ax,ks in zip(aa,[["6","12","24"],["5%","10%"]]):
            for j,k in enumerate(ks):
                s=v["methods"]["Both-in-"+k]; x=j+(i-2.5)*.12
                lo, hi = np.array(s["ci95"]) * 100
                ax.vlines(x,lo,hi,color=COL[i],lw=.95)
                ax.hlines([lo, hi],x-.025,x+.025,color=COL[i],lw=.8)
                ax.plot(x,100*s["mean"],MARK[i],color=COL[i],ms=3.2)
            ax.set(xticks=range(len(ks)),xticklabels=ks,xlim=(-.48,len(ks)-.52),ylim=(-3,53))
    aa[0].set_xlabel("List size $K$",fontsize=8.5)
    aa[1].set_xlabel("Budget (% of legal pool)")
    aa[1].tick_params(axis="y",labelleft=True)
    for ax in aa:
        ax.set_ylabel("Joint coverage (%)",labelpad=3)
    for ax,t in zip(aa,["(a) Fixed list size","(b) Fraction of pool"]):style(ax,t)
    legend(fig,handles,3)
    save(fig,"fig2_budgets")


def selectivity():
    data = wp("wp2_predictability")["networks"]
    fig, aa = axes(2, 2.42, sharey=True)
    fig.subplots_adjust(left=.115, right=.975, top=.88, bottom=.19, wspace=.15)
    sim_anchors = {
        "Dept1": (.79, .59), "Dept2": (.75, .80), "Dept4": (.50, .94),
        "Enron": (.52, .577), "GNU libc": (.67, .515),
    }
    real_labels = {
        "python-dev": ("Python-Dev", (.64, .46), "left"),
        "python-ideas": ("Python-Ideas", (.56, .70), "left"),
        "r-devel": ("R-devel", (.04, .68), "left"),
        "r-package-devel": ("R-package-devel", (.03, .34), "left"),
        "libc-alpha": ("GNU libc\n2021–22", (.92, .105), "right"),
        "libc-alpha-2024": ("GNU libc 2024", (.22, .12), "left"),
        "samba-technical-2020": ("Samba 2020", (.42, .57), "center"),
    }
    leaders = dict(arrowstyle="-", lw=.45, color="#738089", shrinkA=1, shrinkB=2)
    titles = ["(a) Simulated: vs. personal history", "(b) Mailing lists: vs. post volume"]
    for j, (prefix, ax, title) in enumerate(zip(["sim", "real"], aa, titles)):
        rows = [(n.split(" ", 1)[1], d["auc"], 100 * d["gain"])
                for n, d in data.items() if n.startswith(prefix + " ")]
        xx, yy = np.array([(a, b) for _, a, b in rows]).T
        ax.set(xlim=(.48, .78), ylim=(-32, 27), xticks=[.5, .6, .7],
               yticks=[-20, -10, 0, 10, 20])
        ax.axhline(0, lw=.6, color="#798387")
        xs = np.linspace(xx.min(), xx.max(), 50)
        ax.plot(xs, np.polyval(np.polyfit(xx, yy, 1), xs), color=COL[j], lw=.8)
        for name, x, y in rows:
            ax.plot(x, y, marker=MARK[j], color=COL[j], ms=3.5, lw=0)
            if prefix == "sim":
                if name not in sim_anchors:
                    continue
                label, pos, align = name, sim_anchors[name], "left"
            else:
                label, pos, align = real_labels[name]
            ax.annotate(label, (x, y), xytext=pos, textcoords="axes fraction",
                        fontsize=7.4, ha=align, va="center", linespacing=1.02,
                        arrowprops=leaders, bbox=dict(fc="white", ec="none", pad=.12))
        r = np.corrcoef(xx, yy)[0, 1]
        rho = spearmanr(xx, yy).statistic
        style(ax, "")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7.4, pad=15)
        ax.text(0, 1.025, rf"$r={r:.2f},\ \rho={rho:.2f}$  (n=7)",
                transform=ax.transAxes, fontsize=8, va="bottom")
        ax.set_xlabel("Contact AUC")
    left = aa[0]
    left.set_ylabel("Both@12 gain (pp)")
    detail = left.inset_axes([.17, .14, .66, .28])
    detail.set_facecolor("#f2f5f6")
    detail.set(xlim=(.5554, .5587), ylim=(-.025, .123),
               xticks=[.556, .558], yticks=[0, .1], yticklabels=["0.00", "0.10"])
    detail.tick_params(labelsize=6.2, pad=1, length=2)
    for spine in detail.spines.values():
        spine.set(visible=True, linewidth=.5, color="#738089")
    for key, label, offset, align in [("company", "Company", (4, -1), "left"),
                                      ("CollegeMsg", "CollegeMsg", (-4, 1), "right")]:
        item = data["sim " + key]
        point = item["auc"], 100 * item["gain"]
        detail.plot(*point, marker="o", color=COL[0], ms=3.6, lw=0)
        detail.annotate(label, point, xytext=offset, textcoords="offset points",
                        fontsize=6.5, ha=align, va="center")
    for corner, x in [(0, detail.get_xlim()[0]), (1, detail.get_xlim()[1])]:
        connector = ConnectionPatch((corner, 1), (x, detail.get_ylim()[0]),
                                    coordsA=detail.transAxes, coordsB=left.transData,
                                    lw=.5, color="#738089", zorder=1.5)
        left.add_artist(connector)
    save(fig, "fig3_selectivity")


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
    fig,aa=axes(2,2.37,gridspec_kw={"width_ratios":[.88,1.12]})
    fig.subplots_adjust(left=.115,right=.985,top=.66,bottom=.20,wspace=.60)
    xs=np.arange(4)
    for i,(arm,n) in enumerate(zip(["mixed_plain","mixed_offset","uniform_offset"],["Direct","Explicit Prior","Deployment-Only Prior"])):
        vals=[summary[k]["methods"][arm]["both_pct"] for k in PR];xx=xs+(i-1)*.24
        aa[0].bar(xx,vals,.23,color=COL[i],label=n)
        aa[0].vlines(xx,*np.array([ci[k][arm] for k in PR]).T,color="#1d2629",lw=.85)
    aa[0].set(xticks=xs,xticklabels=["Unif.","Equal","80%\nprod.","Tie"],ylabel="Joint coverage (%)",ylim=(0,55))
    bottom=np.zeros(4);nets=["Dept2","company","Dept4","Enron"]
    for j,(key,lab) in enumerate(zip(["conditional","missing_anchor_loss","partner_ranking_loss"],["Captured mass","Emitter omission","Partner-misranking loss"])):
        vals=[read(RUN/"source_prior_shift"/n/"ceiling/summary.json")["both"]["one"]["methods"]["calibrated"][key] for n in nets]
        aa[1].bar(xs,vals,.68,bottom=bottom,color=["#216481","#ddb681","#cbd5da"][j],hatch=["","///",""][j],edgecolor="#50616a",lw=.45,label=lab);bottom+=vals
    for x,y in zip(xs,bottom):
        aa[1].text(x,y+1.8,f"{y:.2f}",ha="center",fontsize=7.8)
    aa[1].set(xticks=xs,xticklabels=[NAMES.get(n,n) for n in nets],ylabel="Oracle ceiling (%)",ylim=(0,100))
    aa[1].yaxis.labelpad = 2
    aa[1].tick_params(axis="x",labelsize=7.15)
    for ax,t in zip(aa,["(a) Enron source prior","(b) One-emitter ceiling"]):style(ax,t)
    for ax in aa:
        ax.legend(loc="lower left",bbox_to_anchor=(-.03,1.19),ncol=1,frameon=False,fontsize=8,
                  handlelength=1.2,labelspacing=.12,borderpad=0)
    save(fig,"fig4_prior_and_ceiling")


def multisource():
    data=wp("wp4_ksources")
    fig,aa=axes(2,2.14)
    fig.subplots_adjust(left=.111,right=.98,top=.78,bottom=.20,wspace=.44)
    xs=np.arange(1,4)
    third_label_height = 2 + 100 * max(by["hidden_uniform/3"]["silent_event"]["mean"] for by in data.values())
    swatches = []
    for i,(n,by) in enumerate(data.items()):
        z=[by[f"hidden_uniform/{k}"] for k in xs]
        xbars = xs+(i-.5)*.56
        silent = np.array([r["silent_event"]["mean"]*100 for r in z])
        fill = dict(facecolor=COL[i], hatch="" if i == 0 else "//",
                    edgecolor=COL[i] if i == 0 else "#813b12", linewidth=.45)
        aa[0].bar(xbars, silent, .36, **fill)
        marker = Line2D([], [], color=COL[i], marker=MARK[i], ls="")
        swatches.append((Patch(**fill), marker))
        label_heights = silent + 2 + 3.5 * i * (silent > 0)
        label_heights[-1] = third_label_height
        for x, v, height in zip(xbars, silent, label_heights):
            aa[0].text(x, height, f"{v:.2f}",
                       fontsize=6.5, ha="center", va="bottom")
        xcurves = xs+(i-.5)*.14
        for key,ls in [("All-in-12","-"),("Recall@12","--")]:
            stats = [r[key]["methods"]["full"] for r in z]
            rates = 100*np.array([s["mean"] for s in stats])
            lo,hi = 100*np.array([s["ci95"] for s in stats]).T
            aa[1].errorbar(xcurves,rates,yerr=[rates-lo,hi-rates],linestyle=ls,marker=MARK[i],
                           color=COL[i],mfc=COL[i] if ls=="-" else "white",lw=1,capsize=2,
                           errorevery=1 if ls=="-" else (1,1))
    for ax,t in zip(aa,["(a) At least one silent source","(b) Retrieval at 12 names"]):
        style(ax,t);ax.set(xticks=xs,xlim=(.50,3.50),ylim=(-3,105),xlabel="Actual source count")
    aa[0].set_xlim(.44, 3.56)
    aa[0].set_ylabel("Silent-source events (%)")
    aa[1].set_ylabel("Coverage / recall (%)",labelpad=1)
    lines = [Line2D([], [], color=".3", ls=ls, marker="o", mfc=fill, markersize=3.6,
                    label=name) for ls, fill, name in
             [("-", ".3", "All sources"), ("--", "white", "Per-source recall")]]
    aa[1].legend(handles=lines,loc="upper right",frameon=False,handlelength=2.0,fontsize=7.5)
    fig.legend(swatches, list(data), loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=2, frameon=False, handlelength=2.9, borderpad=.1,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.2)})
    save(fig,"fig5_multisource")


def decomposition():
    data=wp("wp5_sim2real")["sim"];xx=np.arange(4)
    fig, ax = axes(1, 1.1)
    fig.subplots_adjust(left=.08, right=.98, top=.76, bottom=.20)
    pairs=[(n,d) for n in data for d in ["0","50"]]
    top = 0
    for j,(key,label) in enumerate([("prior","Sender activity prior"),("role","First-recipient target")]):
        stats=[data[n]["common_draw_decomposition"][d]["uniform_model"][key] for n,d in pairs]
        x=xx+(j-.5)*.30
        rates = [100*r["mean"] for r in stats]
        ax.bar(x,rates,.29,color=COL[j],label=label)
        ax.errorbar(x,[100*r["mean"] for r in stats],yerr=np.array([[100*(r["mean"]-r["ci95"][0]),100*(r["ci95"][1]-r["mean"])] for r in stats]).T,fmt="none",ecolor="#202b31",lw=.9,capsize=2)
        if key == "role":
            for pos,rate,rec in zip(x,rates,stats):
                upper = 100*rec["ci95"][1]
                ax.text(pos,upper+2,f"{rate:+.2f}",ha="center",va="bottom",fontsize=8)
                top = max(top,upper+12)
    ax.axhline(0, color="#535c60", lw=.65)
    ax.set(xticks=xx, xticklabels=[f"{n} $D_{{{d}}}$" for n, d in pairs],
           yticks=[0, 25, 50], ylim=(-12, top))
    ax.grid(axis="y", color=".88", lw=.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9, pad=1)
    fig.text(.015, .973, "Coverage change (pp)", va="top")
    fig.legend(handles=ax.get_legend_handles_labels()[0], loc="upper right",
               bbox_to_anchor=(.99, .995), ncol=2, frameon=False,
               columnspacing=.95, handlelength=1.3, handletextpad=.4, borderpad=.1)
    save(fig,"fig6_sim2real_decomp")


if __name__=="__main__":
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8.5,
        "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.65,"lines.linewidth":1.15,"lines.markersize":4,
        "ytick.major.size":2,"xtick.major.size":2,"xtick.major.pad":2,"ytick.major.pad":2})
    for path,d in read(BASE/"wp6_tables/protocol.json")["inputs"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==d,path
    for build in [overview,budgets,selectivity,priors,multisource,decomposition]:
        build()
