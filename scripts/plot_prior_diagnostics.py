"""Paper figure for selection bias and the conditional coverage ceiling."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]/"runs/source_prior_shift"


def main():
    data = json.loads((BASE/"results.json").read_text(encoding="utf8"))
    nets = ("Dept2","Dept4","Enron","company")
    labels = ("Dept2","Dept4","Enron","制造企业")
    ceiling = [json.loads((BASE/n/"ceiling/summary.json").read_text(encoding="utf8"))["both"]["one"] for n in nets]
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
    fig,axs = plt.subplots(1,2,figsize=(11.5,4.9),layout="constrained")
    x = np.arange(len(nets))
    for j,(key,title,color) in enumerate((("volume","发送量","#aa8b55"),("raw","原模型","#758599"),("calibrated32","32 例校准","#066b75"))):
        value = [data[n]["both"]["all"]["methods"][key]["both"] for n in nets]
        axs[0].bar(x+(j-1)*.24,value,width=.23,color=color,label=title)
    bottom = np.zeros(len(nets))
    for key,title,color in (("conditional","模型条件覆盖","#066b75"),("missing_anchor_loss","遗漏传播者的损失","#d59a50"),("partner_ranking_loss","伙伴排序损失","#a1afb9")):
        value = np.array([g["methods"]["calibrated"][key] for g in ceiling])
        axs[1].bar(x,value,bottom=bottom,width=.58,color=color,label=title)
        bottom += value
    for i,(total,g) in enumerate(zip(bottom,ceiling,strict=True)):
        assert np.isclose(total,g["cap"])
        axs[1].text(i,total+1.4,f"{total:.1f}",ha="center",fontsize=9)
    for ax in axs:
        ax.set_ylim(0,100)
        ax.set_xticks(x,labels)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.1),ncol=3,frameon=False,fontsize=8)
    axs[0].set_title("两位源头均偏向活跃者")
    axs[0].set_ylabel("全部合格事件的双源名单命中率（%）")
    axs[1].set_title("同一抽取条件下，仅一位源头直接传播")
    axs[1].set_ylabel("条件期望覆盖及其与上限的差距（百分点）")
    fig.savefig(BASE/"先验校正与沉默伙伴上限.png",dpi=200)
    plt.close(fig)


if __name__=="__main__":
    main()
