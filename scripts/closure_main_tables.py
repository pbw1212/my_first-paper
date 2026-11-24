"""Render the six locked paper tables from WP6's authoritative source map."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/paper_closure"
OUT = BASE / "wp6_tables"
REPORT = OUT / "report/main_tables.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(content)


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()


def interval(point, bounds, scale=100):
    return f"{point*scale:+.2f} [{bounds[0]*scale:+.2f}, {bounds[1]*scale:+.2f}]"


def rate(s, model):
    return f'{100*s["methods"][model]["mean"]:.2f}'


def diff(s, a, b):
    z = s["differences"][f"{a} minus {b}"]
    assert math.isclose(z["mean"], s["methods"][a]["mean"]-s["methods"][b]["mean"], abs_tol=1e-12)
    return interval(z["mean"], z["ci95"])


def grid(columns, rows):
    lines = [columns, ["---"]*len(columns)]
    for row in rows:
        assert len(row) == len(columns)
        lines.append(row)
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in lines)


def budgets(data):
    methods = ["Both-in-6","Both-in-12","Both-in-24","Both-in-5%","Both-in-10%","Any-in-12"]
    rows = []
    for n,s in data.items():
        rows.append([n,s["n"],f'{s["mean_candidates"]:.1f}',*[rate(s,m) for m in methods],
                     f'{s["methods"]["Pair-MRR"]["mean"]:.5f}',diff(s,"Both-in-5%","Both-in-12")])
    return [grid(["网络","事件数","平均候选人数","双源@6","双源@12","双源@24","双源@5%","双源@10%","至少一源@12","对 MRR","5% 减 12 人（百分点）"],rows),
            "比例预算逐事件向上取整。每个网络都使用冻结的完整逐人模型。五个训练种子共享同一批测试事件。"]


def contacts(data):
    rows = []
    for n,s in data["networks"].items():
        base = s["baseline"]
        rows.append([n,s["n"],s["auc_n"],f'{s["auc"]:.3f}',f'{100*s["true_contact"]:.1f}',f'{100*s["other_contact"]:.1f}',
                     "发帖量" if base=="volume" else "个人历史",rate(s,base),rate(s,"links"),diff(s,"links",base)])
    c = data["correlations"]
    return [grid(["网络 / 社区","合格","可算 AUC","AUC","真源有联系 %","非源有联系 %","对照","对照命中 %","加入联系后 %","配对差（百分点）"],rows),
            "AUC 在同一事件内比较两位名册内目标与其他候选人，平分计半。名单命中率保留所有合格事件，包括弃权。真实社区在这一表统一使用 D0，至少需一名名册内可见成员。",
            grid(["AUC 与增益的相关","网络或时期数","Pearson","Spearman"],
                 [[label,c[k]["auc"]["n"],f'{c[k]["auc"]["pearson"]:.3f}',f'{c[k]["auc"]["spearman"]:.3f}']
                  for k,label in [("sim","模拟，近期联系减个人历史"),("real","真实，近期联系减发帖量"),("pooled_common","全部，近期联系减个人历史")]]),
            "相关系数以网络或时期等权计算。GNU libc 的模拟组和真实组共用社区背景，两个真实时期也有关联。"]


def robustness(data):
    trees = []
    for n,by_budget in data["tree"].items():
        for b,s in by_budget.items():
            trees.append([n,b,s["n"],*[rate(s,m) for m in ("person","links","full","MLP","FGSSI")],
                          diff(s,"links","person"),diff(s,"full","MLP")])
    stress = []
    for n,rules in data["stress"].items():
        for r,groups in rules.items():
            s = groups["all"]
            stress.append([n,r,s["n"],rate(s,"person"),rate(s,"links"),diff(s,"links","person"),
                           groups["common"]["n"],diff(groups["common"],"links","person")])
    return ["**模型容量**　HistGradientBoostingClassifier 使用与 MLP 相同的输入列、数据切分和五个种子。",
            grid(["网络","预算","事件数","个人树","联系树","完整树","完整 MLP","FGSSI","联系树减个人树","完整树减 MLP"],trees),
            "**传播规则**　下表使用冻结 MLP 和 12 人名单。右两列将抽取固定为四种规则全部合格的交集。",
            grid(["网络","规则","本规则合格","个人历史","加入联系","配对差","共同合格","共同抽取配对差"],stress),
            "SIR48 和 SIR72 的活跃期分别为 48 和 72 小时。noise 允许每位非源成员在一窗内以 2% 概率自发加入。"]


def priors(data):
    rows = []
    for cond,label in [("flat","均匀"),("mixed","三分量等权"),("mostly_active","双人活跃偏向占 80%"),("linked","近期直连偏向")]:
        s = data["Enron"][cond]
        m = s["methods"]
        d = s["paired_to_mixed_plain"]["mixed_offset"]
        simple = s["mixed_offset_minus_uniform_offset"]
        assert sum(s["window_counts"])==s["n"]
        assert abs(m["mixed_offset"]["both_pct"]-m["mixed_plain"]["both_pct"]-d["pp"])<1e-10
        assert abs(m["mixed_offset"]["both_pct"]-m["uniform_offset"]["both_pct"]-simple["pp"])<1e-10
        rows.append([label,s["n"],*[f'{m[k]["both_pct"]:.2f}' for k in ("mixed_plain","mixed_offset","uniform_offset")],
                     interval(d["pp"],d["ci95_pp"],1),interval(simple["pp"],simple["ci95_pp"],1)])
    return ["Enron 的四个抽取条件均来自同一历史归档。前两种模型使用相同的混合训练事件，第三种在均匀源头上训练、部署时校正。",
            grid(["抽取条件","事件数","混合直接训练","混合显式先验","均匀训练加校正","显式减直接","显式减简单校正"],rows)]


def sources(data):
    labels = {"three_uniform/all":"三源均匀","three_active/all":"三源活跃偏向","hidden_uniform/all":"未知 k，全组",
              **{f"hidden_uniform/{i}":f"未知 k，实际 {i} 源" for i in (1,2,3)}}
    rows = []
    for n,conditions in data.items():
        for k,label in labels.items():
            z = conditions[k]
            s = z["All-in-12"]
            assert s["n"]==z["Recall@12"]["n"]
            rows.append([n,label,s["n"],f'{100*z["silent_event"]["mean"]:.2f}',*[rate(s,m) for m in ("volume","static","full")],
                         diff(s,"full","static"),rate(z["Recall@12"],"full")])
    return [grid(["网络","抽取条件","事件数","有沉默源 %","全入围·发信量","全入围·静态","全入围·完整","完整减静态","完整模型逐源召回 %"],rows),
            "未知 k 在每次尝试时从 1、2、3 中均匀抽取，合格事件的源数分布受遮挡门槛影响。分层行已经计入全组。"]


def bridge(data):
    rows, common, mechanisms = [],[],[]
    for n,depths in data["real"].items():
        for d in ("0","25","50"):
            s = depths[d]["all"]
            rows.append([n,"D"+d,s["n"],s["scored"],s["scored_reachable"],
                         *[rate(s,m) for m in ("volume","roles","reply","full")],diff(s,"full","volume")])
        fixed = [depths[d]["common"] for d in ("0","25","50")]
        assert all(s["n"] == fixed[0]["n"] for s in fixed)
        row = [n,fixed[0]["n"]]
        for s in fixed:
            row.extend([rate(s,"volume"),rate(s,"full"),diff(s,"full","volume")])
        common.append(row)
    for n,sim in data["sim"].items():
        for d in ("0","50"):
            s = sim["common_draw_decomposition"][d]["uniform_model"]
            assert abs(s["prior"]["mean"]+s["role"]["mean"]-s["total"]["mean"])<1e-12
            mechanisms.append([n,"D"+d,s["n"],*[interval(s[k]["mean"],s[k]["ci95"]) for k in ("prior","role","total")]])
    return ["**真实讨论**　每个深度共用 98 天名册、12 人预算和至少三名名册内可见成员的打分门槛。可达列同时要求可打分及两位目标在候选名单内。两个 Python 邮件列表使用 2019 年 2–4 月，其余社区使用归档测试期。",
            grid(["社区 / 时期","深度","合格","可打分","可达","发帖量","历史角色","角色加回复","模拟完整模型","完整减发帖量"],rows),
            "**固定讨论**　仅保留三档深度均合格的讨论。每档仍执行共同弃权规则。",
            grid(["社区 / 时期","共同合格","D0 发帖量","D0 完整","D0 配对差","D25 发帖量","D25 完整","D25 配对差","D50 发帖量","D50 完整","D50 配对差"],common),
            "**模拟机制**　同一冻结完整模型，四种生成条件均合格的抽取。将先验与角色按两种干预顺序取平均，可使两项差值相加等于总变化。",
            grid(["网络","深度","共同抽取","首发活动先验（百分点）","首位激活者角色（百分点）","总变化（百分点）"],mechanisms),
            "真实标签为根作者与首位不同回应者。模拟效应仅适用于表中共同合格的抽取。Python 测试期只有三个自然月，恰为一个循环块，区间无法刻画跨月份的不确定性。"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    args = parser.parse_args()
    master = BASE/"master_summary.json"
    wp6 = BASE/"wp6_evidence"
    assert read(wp6/"verification.json")["checks"]["master"]==digest(master)
    seals = read(wp6/"protocol.json")["inputs"]
    tables = read(master)["tables"]
    assert [t["id"] for t in tables]==[f"T{i}" for i in range(1,7)]
    inputs = {t["path"]:digest(ROOT/t["path"]) for t in tables}
    assert all(seals["repo/"+path]==value for path,value in inputs.items())
    protocol = {"scope":"Descriptive rendering of the six previously locked WP6 main tables",
                "master_sha256":digest(master),"inputs":inputs,"script_sha256":digest(Path(__file__)),
                "statistics":"Archived paired estimates and intervals, no fitting or reselection; T4 already in percent and pp"}
    def artifact(p,content):
        if args.verify:
            assert p.read_bytes()==content,p
        else:
            write_once(p,content)
    artifact(OUT/"protocol.json",json_bytes(protocol))
    sections = ["# 双源定位实验主表", "命中率和召回率以百分比表示。模型、预算或机制间的配对差以百分点表示，方括号内为 95% 区间。事件数按合格事件计，五次训练共用同一分母。"]
    for t,render in zip(tables,[budgets,contacts,robustness,priors,sources,bridge],strict=True):
        path = ROOT/t["path"]
        sections.append(f'## {t["id"]}　{t["claim"]}')
        sections.extend(render(read(path)))
        rel = Path(os.path.relpath(path,REPORT.parent)).as_posix()
        sections.append(f'[本表的权威统计文件]({rel})')
    sections.append("从仓库根目录运行 `python -B scripts/closure_main_tables.py --verify` 可逐字复核此表，并对照六份输入的协议指纹。")
    content = ("\n\n".join(sections)+"\n").encode()
    artifact(REPORT,content)
    lines = content.decode().splitlines()
    data_rows = sum(line.startswith("| ") and i >= 2 and lines[i-1].startswith("| ")
                    and not line.startswith("| ---") for i,line in enumerate(lines))
    result = {"passed":True,"tables":len(tables),"source_hashes_checked":len(inputs),
              "rendered_data_rows":data_rows,"report_sha256":digest(REPORT),
              "protocol_sha256":digest(OUT/"protocol.json")}
    artifact(OUT/"verification.json",json_bytes(result))
    print(f'{result["tables"]} tables, {result["rendered_data_rows"]} data rows, six archived source fingerprints verified.')


if __name__=="__main__":
    main()
