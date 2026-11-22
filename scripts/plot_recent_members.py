"""Plot event-weighted, seed-averaged effects for the manuscript."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]/"runs/recent_member_evidence"


def read(path):
    return json.loads((OUT/path).read_text(encoding="utf8"))


def main():
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
    fig,(left,right) = plt.subplots(1,2,figsize=(10.5,4.15),layout="constrained")
    nets = ["Dept2","制造企业","Enron","Dept4"]
    cohorts = [read("Dept2/summary.json")["all"],read("company/summary.json")["all"],
               read("posthoc/Enron/summary.json"),read("posthoc/Dept4/summary.json")]
    x = np.arange(4)
    for j,(m,label,color) in enumerate([("person","个人历史","#839197"),("links","加近期联系","#e2a556"),("full","完整历史","#11637a")]):
        left.bar(x+(j-1)*.24,[c["methods"][m]["both_pct"] for c in cohorts],.23,label=label,color=color)
    left.set_xticks(x,nets)
    left.set_title("近期联系与双源名单命中率")
    left.set_ylabel("双源同时入选（%）")
    left.legend(loc="upper left",frameon=False)
    for n,label,color in (("Dept2","Dept2","#11637a"),("company","制造企业","#a75e19")):
        d = read("stress/"+n+"/summary.json")
        y = [d[k]["eligible"]["differences"]["full-FGSSI"]["pp"] for k in ("low","original","high","once")]
        right.scatter(x,y,s=48,label=label,color=color)
    right.set_xticks(x,["重复\n0.15","重复\n0.30","重复\n0.50","仅首次\n0.30"])
    right.set_ylim(0,6)
    right.set_ylabel("完整模型相对 FGSSI 的差值（百分点）")
    right.set_title("改变传播规则后的名单收益")
    right.legend(frameon=False,loc="upper right")
    for ax in (left,right):
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y",alpha=.16)
        ax.set_axisbelow(True)
    fig.savefig(OUT/"近期联系与传播条件.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
