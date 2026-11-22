"""Plot individual seed differences under a common decoder."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report_factorial_history import NAMES, OUT, contrast, fg


def main():
    d=fg.load(OUT/"summary.json")
    order=["Dept2","Dept2_new_windows","company","company_new_windows","Enron","Dept1","CopenhagenSMS","CollegeMsg"]
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.7),layout="constrained",sharey=True)
    for ax,metric,title in zip(axes,("m12","top32"),("两位真源均进入 12 人名单","真实源头对进入前 32")):
        ax.axvline(0,color="#555e66",linewidth=.8)
        for i,k in enumerate(order):
            diff=contrast(d[k]["groups"]["all"],"D","C",metric)
            jitter=np.linspace(-.15,.15,len(diff)) if len(diff)>1 else [0]
            ax.scatter(diff,i+np.asarray(jitter),s=22,color="#88a4af",alpha=.85)
            ax.scatter([diff.mean()],[i],marker="D",s=35,color="#00526a",zorder=3)
        ax.set_title(title)
        ax.set_xlabel("加入双人交互的差值（百分点）")
        ax.grid(axis="y",alpha=.18)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
    axes[0].set_yticks(np.arange(len(order)),[NAMES[k] for k in order])
    axes[0].invert_yaxis()
    axes[1].scatter([],[],s=22,color="#88a4af",label="单次训练")
    axes[1].scatter([],[],s=35,marker="D",color="#00526a",label="种子均值")
    axes[1].legend(loc="upper center",bbox_to_anchor=(.5,-.15),frameon=False,ncol=2)
    fig.savefig(OUT/"相同历史下的交互增益.png",dpi=195)
    plt.close(fig)


if __name__=="__main__":
    main()
