"""Plot the frozen shortlist comparison and the distribution of its paired window effects."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from distilled_source_pairs import OUT,load


def main():
    names=["company","Dept2","Dept1","CollegeMsg","CopenhagenSMS","company_new_windows","Dept2_new_windows"]
    labels=["制造企业","Dept2","Dept1","CollegeMsg","哥本哈根短信","制造企业 · 补充窗口","Dept2 · 补充窗口"]
    data=[load(OUT/name/"summary.json") for name in names]
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
    fig,(left,right)=plt.subplots(1,2,figsize=(10.8,4.1),gridspec_kw={"width_ratios":[1.35,1]},layout="constrained")
    y=np.arange(len(names))
    for ax in (left,right):
        ax.set_ylim(len(names)-.55,-.55)
        ax.axhline(4.5,color="#9ea5a9",lw=.8,ls="--")
        ax.grid(axis="x",alpha=.18)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
    methods=[("pooled","逐人提升树","#797e83","s"),("old_joint","旧组合模型","#cc7941","^"),("pair_soft","后验监督组合模型","#126b81","o")]
    for offset,(key,label,color,marker) in zip([-.16,0,.16],methods):
        rate=[100*r["total"]["both"][key]/r["total"]["n"] for r in data]
        left.scatter(rate,y+offset,c=color,marker=marker,s=39,label=label,zorder=3)
    left.set_yticks(y,labels)
    left.set_xlabel("12 人名单同时覆盖双源（%）")
    left.legend(loc="upper center",bbox_to_anchor=(.50,-.17),ncol=3,frameon=False,fontsize=8)
    right.axvline(0,color="#525a60",lw=.8)
    for pos,r in enumerate(data):
        wins=[record for key,record in r.items() if "-" in key]
        differences=[100*(d["both"]["pair_soft"]-d["both"]["old_joint"])/d["n"] for d in wins]
        right.scatter(differences,pos+np.linspace(-.13,.13,len(wins)) if len(wins)>1 else [pos],
                      marker="o",s=29,color="#126b81",alpha=.82)
    right.set_yticks(y,[])
    right.set_xlabel("每个接触窗口相对旧组合模型的增益（百分点）")
    fig.savefig(OUT/"双源名单与逐窗增益.png",dpi=220,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
