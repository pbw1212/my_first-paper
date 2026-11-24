"""Export the budget, contact and masking figures from the sealed paper-table sources."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from closure_main_tables import BASE, ROOT, read, digest

OUT = BASE / "wp6_tables/report/figures"
COLORS = ["#21669A","#BE641B","#218378","#895299","#645844","#424853"]
MARKERS = ["o","s","^","D","v","P"]
NAMES = {"company":"Company","python-dev":"Python-Dev","python-ideas":"Python-Ideas",
         "r-devel":"R-devel","r-package-devel":"R-package-devel","libc-alpha":"GNU libc 2021–22",
         "libc-alpha-2024":"GNU libc 2024","samba-technical-2020":"Samba 2020"}
DEPTHS = ["0","25","50"]
CAPTIONS = {
"T1_budgets":"Both-source coverage of the frozen additive model for three fixed list sizes (left) and two fractions of the legal candidate pool (right). Fractional budgets are rounded up per event. Points are means over five training seeds. Bars are 95% intervals from paired resampling of seeds and circular three-window blocks. Horizontal offsets separate networks within each budget.",
"T2_contact_auc":"Each point represents one simulated network or one real mailing-list period. The horizontal axis is the mean within-event AUC of the 28-day contact count. The vertical axis is the gain in Both-in-12 for the model with recent contacts, relative to personal history for simulations or post volume for real discussions. Lines are unweighted least-squares fits within each group. Real AUCs use scorable discussions with both targets on the roster; hit rates include all eligible discussions.",
"T6_mask_depth":"Both-in-12 across three masking depths. A uses all eligible discussions at each depth. B follows the same discussions eligible at all three depths. Panel titles give event counts in D0/D25/D50 order in A and one fixed count in B. The shared abstention rule applies at each depth. Lines connect observed point estimates; coincident markers denote equal rates. Each community has the same vertical scale in A and B. Python results use February–April 2019; the R lists use 2021–22."}


def export(fig,name):
    for ext in ("pdf","png"):
        kw = {"metadata":{"Title":name,"Subject":CAPTIONS[name],"CreationDate":None,"ModDate":None}} if ext=="pdf" else {"dpi":600}
        fig.savefig(OUT/f"{name}.{ext}",**kw)
    plt.close(fig)


def budgets(data):
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.35),sharey=True,gridspec_kw={"width_ratios":[3,2]})
    fig.subplots_adjust(left=.085,right=.98,top=.76,bottom=.18,wspace=.12)
    for ax,budgets,title in zip(axes,[["6","12","24"],["5%","10%"]],["Fixed list size","Fraction of candidate pool"]):
        for i,(name,s) in enumerate(data.items()):
            for x,b in enumerate(budgets):
                stat=s["methods"]["Both-in-"+b]
                mean=100*stat["mean"]
                lo,hi=np.array(stat["ci95"])*100
                xpos=x+(i-2.5)*.085
                ax.vlines(xpos,lo,hi,color=COLORS[i],lw=.95)
                ax.plot([xpos],[mean],marker=MARKERS[i],color=COLORS[i],ms=4,linestyle="none")
        ax.set(xticks=range(len(budgets)),xticklabels=budgets,xlim=(-.38,len(budgets)-.62),ylim=(-1.5,53),yticks=[0,10,20,30,40,50])
        ax.set_xlabel(title)
        ax.grid(axis="y")
    axes[0].set_ylabel("Both-in-budget (%)")
    handles=[Line2D([],[],marker=MARKERS[i],color=COLORS[i],ls="none",label=NAMES.get(n,n)) for i,n in enumerate(data)]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,.96),ncol=6,frameon=False,columnspacing=1,handletextpad=.3)
    export(fig,"T1_budgets")


def contacts(summary):
    fig,axes=plt.subplots(1,2,figsize=(7.55,3.5),sharex=True,sharey=True)
    fig.subplots_adjust(left=.09,right=.98,top=.88,bottom=.17,wspace=.14)
    offsets={"company":(-5,-14),"CollegeMsg":(1,14),"Dept1":(4,-13),"Dept4":(-7,6),"Dept2":(-6,6),
             "Enron":(-4,-8),"GNU libc":(4,-10),"libc-alpha":(-2,-15),"libc-alpha-2024":(-6,6),
             "r-devel":(4,9),"r-package-devel":(-2,-16),"python-ideas":(5,4),"samba-technical-2020":(2,-23)}
    for i,(kind,ax,title) in enumerate(zip(["sim","real"],axes,["Simulations · vs. personal history","Discussions · vs. post volume"])):
        records=[(key.split(" ",1)[1],s["auc"],100*s["gain"]) for key,s in summary["networks"].items() if key.startswith(kind+" ")]
        x,y=np.array([(a,b) for _,a,b in records]).T
        grid=np.linspace(x.min(),x.max())
        ax.axhline(0,lw=.75,color="#80878D")
        ax.plot(grid,np.polyval(np.polyfit(x,y,1),grid),color=COLORS[i],lw=1.15)
        ax.scatter(x,y,marker=MARKERS[i],s=20,color=COLORS[i],edgecolor="white",linewidth=.45,zorder=3)
        for n,a,b in records:
            dx,dy=offsets.get(n,(4,6))
            ax.annotate(NAMES.get(n,n),(a,b),xytext=(dx,dy),textcoords="offset points",fontsize=7.3,ha="right" if dx<0 else "left",
                        bbox=dict(facecolor="white",edgecolor="none",pad=.25))
        ax.set(xlim=(.482,.785),ylim=(-27,26),xticks=[.50,.55,.60,.65,.70,.75],yticks=[-20,-10,0,10,20],xlabel="Mean within-event contact AUC")
        ax.grid(axis="y")
        ax.set_title(title,fontsize=9.2,loc="left",fontweight="bold",pad=8)
    axes[0].set_ylabel("Both-in-12 gain (percentage points)")
    export(fig,"T2_contact_auc")


def bridge(data):
    fig=plt.figure(figsize=(7.55,8.05))
    grid=fig.add_gridspec(4,3,left=.092,right=.985,top=.87,bottom=.06,hspace=.87,wspace=.29)
    styles=[("volume","Post volume","#465663","-","o",6.8),("roles","Role history","#187864","--","s",5.4),
            ("reply","Roles + replies","#BD691C",":","^",4.2),("full","Full temporal model","#7C448E","-.","x",4)]
    handles=[Line2D([],[],label=label,color=c,ls=ls,marker=m,ms=size,markerfacecolor="none",lw=1.2) for _,label,c,ls,m,size in styles]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,1),ncol=4,frameon=False,columnspacing=1.4)
    communities=data["real"]
    for g,(cohort,title) in enumerate([("all","A   All eligible discussions"),("common","B   Discussions eligible at every depth")]):
        for i,(name,ds) in enumerate(communities.items()):
            ax=fig.add_subplot(grid[2*g+i//3,i%3])
            stats=[ds[d][cohort] for d in DEPTHS]
            ymax=max(100*ds[d][part]["methods"][m]["mean"] for d in DEPTHS for part in ("all","common") for m,*_ in styles)
            ceiling=next(u for u in (1,5,10,15,20,25,30,40,50,100) if u>=ymax)
            for m,label,c,ls,sym,size in styles:
                ax.plot(range(3),[100*s["methods"][m]["mean"] for s in stats],marker=sym,ms=size,markerfacecolor="none",markeredgewidth=.95,
                        ls=ls,color=c,lw=1.15)
            ax.set(xticks=range(3),xticklabels=["D"+d for d in DEPTHS],xlim=(-.13,2.13),ylim=(-.055*ceiling,ceiling*1.06))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3,steps=[1,2,2.5,5,10],prune="upper"))
            ax.grid(axis="y")
            counts="/".join(str(s["n"]) for s in stats) if g==0 else str(stats[0]["n"])
            ax.set_title(f"{NAMES[name]}   ·   n={counts}",loc="left",fontweight="bold",fontsize=8,pad=4)
            if i%3==0:
                ax.set_ylabel("Both-in-12 (%)")
        top=grid[2*g,0].get_position(fig).y1
        fig.text(.092,top+.041,title,fontsize=11,fontweight="bold")
    export(fig,"T6_mask_depth")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.65,"axes.edgecolor":"#62717A","axes.axisbelow":True,"grid.color":"#DFE4E6","grid.linewidth":.55,
        "xtick.major.size":2.5,"ytick.major.size":2.5,"savefig.facecolor":"white"})
    inputs=read(BASE/"wp6_evidence/protocol.json")["inputs"]
    for wp,fn in [("wp1_budgets",budgets),("wp2_predictability",contacts),("wp5_sim2real",bridge)]:
        path=BASE/wp/"summary.json"
        assert digest(path)==inputs["repo/"+path.relative_to(ROOT).as_posix()]
        fn(read(path))
    captions="\n\n".join(f"**{n.split('_')[0]}**  {caption}" for n,caption in CAPTIONS.items())
    (OUT/"captions.md").write_text(captions+"\n",encoding="utf8")


if __name__=="__main__":
    main()

