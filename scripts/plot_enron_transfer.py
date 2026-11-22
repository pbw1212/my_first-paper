"""Plot paired Enron results using the audited event counts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder=Path(__file__).resolve().parents[1]/"runs/temporal_transfer"
    data=json.loads((folder/"summary.json").read_text(encoding="utf8"))["Enron"]
    keys=sorted(k for k in data if k.startswith("Enron-"))
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":10,"axes.unicode_minus":False})
    fig,(left,right)=plt.subplots(1,2,figsize=(10.8,3.9),gridspec_kw={"width_ratios":[1.75,1]},layout="constrained")
    windows=np.arange(1,len(keys)+1)
    deltas=[data[k]["methods"]["pair_soft"]["both"]-data[k]["methods"]["fgssi"]["both"] for k in keys]
    left.bar(windows,deltas,color=["#126b81" if d>=0 else "#bd683b" for d in deltas],width=.72)
    left.scatter(windows[np.equal(deltas,0)],np.zeros(sum(d==0 for d in deltas)),color="#627079",s=12,zorder=3)
    left.axhline(0,color="#556169",lw=.8)
    left.set_xticks([1,5,10,15,20,26])
    left.set_xlabel("2001 年连续的 14 天窗口（每窗 60 例）")
    left.set_ylabel("较 FGSSI 净增的双源名单命中数")
    left.set_title("完整组合模型的逐窗增益",loc="left")
    for i,(method,label,color) in enumerate([("fgssi","FGSSI","#808c92"),("pair_soft","完整组合","#126b81"),("five_hard","五帧硬标签","#a46530")]):
        vals=[100*data[k]["methods"][method]["both"]/data[k]["n"] for k in ("emitters_1","emitters_2")]
        right.bar(np.arange(2)+(i-1)*.23,vals,width=.21,color=color,label=label)
    right.set_xticks([0,1],["仅一位直接传播\n1,109 例","两位都直接传播\n451 例"])
    right.set_ylabel("12 人名单同时覆盖双源（%）")
    right.set_title("按源头是否直接传播分层",loc="left")
    right.legend(frameon=False,fontsize=8)
    for ax in (left,right):
        ax.spines[["top","right"]].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis="y",alpha=.18)
    fig.savefig(folder/"Enron逐窗增益与传播源头分层.png",dpi=200)
    plt.close(fig)


if __name__=="__main__":
    main()
