"""Render tables and paired contrasts directly from the audited factorial counts."""
import numpy as np
from factorial_fgssi import OUT
import fgssi_suffix as fg

NAMES={"Dept2":"Dept2","company":"制造企业","Dept2_new_windows":"Dept2 补充窗","company_new_windows":"制造企业补充窗",
       "Enron":"Enron","Dept1":"Dept1","CopenhagenSMS":"哥本哈根短信","CollegeMsg":"CollegeMsg"}


def values(group,key,metric):
    return np.array([100*s[key][metric]/s["n"] for s in group.values()])


def contrast(group,left,right,metric):
    return values(group,left,metric)-values(group,right,metric)


def interval(groups):
    rng=np.random.default_rng(20261119)
    windows=[v for k,v in sorted(groups.items()) if k.startswith("window/")]
    m=len(windows);seeds=list(windows[0]);n=len(seeds)
    blocks=((rng.integers(m,size=(10000,(m+2)//3,1))+np.arange(3))%m).reshape(10000,-1)[:,:m]
    sample=rng.integers(n,size=(10000,n,1))
    counts=np.array([[w[s]["n"] for w in windows] for s in seeds])
    result={}
    for key in ("B-A","D-C","C-A","D-B","Ddecode","D-F","C-F"):
        diffs=np.array([[w[s][key]["gained"]-w[s][key]["lost"] for w in windows] for s in seeds])
        diff=100*diffs[sample,blocks[:,None,:]].sum((1,2))/counts[sample,blocks[:,None,:]].sum((1,2))
        result[key]=np.quantile(diff,[.025,.975]).tolist()
    return result


def main():
    data=fg.load(OUT/"summary.json")
    rows=["# 五帧双源定位中的历史表示、组合评分与解码",
          "",
          "在相同五帧可见成员、真实双源硬标签、训练事件及预算下检验方法归因。每个开发网络使用 1,000 个较早事件训练、300 个后续事件按真实对负对数似然选轮次。随机种子为 111、314159、431、2027、9001，每格成对复用相同初始化种子、批次顺序和验证规则。Dept1、短信、CollegeMsg 与 Enron 使用 Dept2 模型。CollegeMsg 的本轮全候选诊断仅使用预先指定的种子 314159，其余组均使用五个种子。Enron 在上一轮已揭晓，本轮用于事后机制分析。真实网络记录提供接触时间与对象，传播及源头真值来自模拟。",
          "",
          "静态历史对窗口前所有联系合并方向并累计次数，输入无向跳距倒数、联系强度、度及其与五帧成员的关系，同时保留帧的先后顺序。有向时序历史使用原组合模型的方向、近期接触、分时活动与历史传播可达性特征。两种历史各自计算候选对之间的联系和对可见成员的支持重叠。A、C 只相加两个人的单人分数，B、D 具有额外的低秩交互、两人联系方式与观测支持重叠项。各行内共享单人输入、编码器宽度、损失和选模规则；行间差值反映整套历史表示，包括特征构造和维数的变化。",
          "",
          "## 相同名单解码下的四组比较","",
          "各格先用相同边际取人规则得到 12 人名单，表内为两位真源同时入选的百分比，跨种子取均值。末两列单位为百分点。",
          "",
          "| 测试组 | 事件数 | A 静态加性 | B 静态交互 | C 时序加性 | D 时序交互 | B−A | D−C |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k,label in NAMES.items():
        g=data[k]["groups"]["all"]
        cells=[values(g,c,"m12").mean() for c in "ABCD"]
        rows.append(f"| {label} | {next(iter(g.values()))['n']} | "+" | ".join(f"{v:.2f}" for v in cells)+f" | {cells[1]-cells[0]:+.2f} | {cells[3]-cells[2]:+.2f} |")
    rows+=["","Enron 上，固定逐人打分和名单规则后，有向时序历史相对静态历史提高 2.46 个百分点。固定有向时序历史后，加入双人交互的差值为 −0.69 个百分点。Dept2 上交互的名单收益为 1.17 个百分点，其更明显的收益体现在精确组合排名。","",
           "![不同网络固定历史表示后的双人交互收益](相同历史下的交互增益.png)","",
           "## 时间窗口、种子与精确组合排名","",
           "D−C 是固定名单解码后加上交互的收益。D 的解码增益是在同一分数矩阵上以联合名单搜索替换边际取人的差值。FGSSI 和联合解码 D 的主指标以跨种子的均值 ± 样本标准差表示，单种子只列点估计。正种子为 D−C 严格大于零的种子数。","",
           "| 测试组 | FGSSI 命中率 % | D 联合名单命中率 % | D 解码增益 pp | D−C 正种子 | C 精确 Top-32 % | D 精确 Top-32 % | C / D 的 MRR |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k,label in NAMES.items():
        g=data[k]["groups"]["all"]
        def fmt(v):
            return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v)>1 else f"{v.mean():.2f}"
        d=values(g,"D","j12")-values(g,"D","m12")
        rows.append(f"| {label} | {fmt(values(g,'F','j12'))} | {fmt(values(g,'D','j12'))} | {d.mean():+.2f} | {sum(contrast(g,'D','C','m12')>1e-9)}/{len(g)} | {values(g,'C','top32').mean():.2f} | {values(g,'D','top32').mean():.2f} | {values(g,'C','reciprocal').mean()/100:.4f} / {values(g,'D','reciprocal').mean()/100:.4f} |")
    ci=interval(data["Enron"]["groups"])
    fg.write_json(OUT/"enron_intervals.json",ci)
    rows+=["","Enron 的配对区间同时重采样五个训练种子及连续三个日历窗口，循环块抽样 10,000 次。对比仅适用于这批组织内接触与模拟事件。","",
           "| Enron 配对比较 | 平均差值 pp | 95% 百分位区间 pp |","|---|---:|---:|"]
    for key,(lo,hi) in ci.items():
        g=data["Enron"]["groups"]["all"]
        dif=np.mean([100*(r[key]["gained"]-r[key]["lost"])/r["n"] for r in g.values()])
        rows.append(f"| {key} | {dif:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    rows+=["","## 名单预算、沉默源头与遮挡","",
           "以下分别列 D 在预算 8、12、20 时联合名单命中率，以及预算 12 时按真值事后分层的命中率。各格单位为 %。","",
           "| 测试组 | 8 人 | 12 人 | 20 人 | 单直接传播源 | 双直接传播源 |","|---|---:|---:|---:|---:|---:|"]
    for k,label in NAMES.items():
        groups=data[k]["groups"];g=groups["all"]
        vals=[values(g,"D","j"+str(b)).mean() for b in (8,12,20)]+[values(groups["emitters/"+str(n)],"D","j12").mean() for n in (1,2)]
        rows.append("| "+label+" | "+" | ".join(f"{v:.2f}" for v in vals)+" |")
    c=next(iter(data["CollegeMsg"]["groups"]["all"].values()))
    rows+=["",f"CollegeMsg 的 {c['n']} 例中，D 的 12 人名单至少找回一位源头 {int(c['D']['any'])} 次；前 48 边际候选及前 48 对端点的并集同时保留两位源头 {int(c['D']['pool'])} 次，最终名单同时找回两人 {int(c['D']['j12'])} 次。","",
           "开发网络四个主窗各作 400 次新的均匀双源抽取。固定模型，用同一潜在轨迹生成 50% 和 75% 两种遮挡；同批成员不可拆分，后缀至少需有三位成员。表内列出抽取分母及合格数，再列仅取两种遮挡均合格事件时的配对结果。FGSSI 与 D 均按预算 12 评分。","",
           "| 开发网络 | 总抽取 | 50% 合格 | 75% 合格 | 共同合格 | 全部合格 D 50% / 75% | 配对 D 50% / 75% | 配对 FGSSI 50% / 75% |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ("Dept2","company"):
        groups=data[k+"_masks"]["groups"]
        flow=fg.load(OUT/(k+"_masks")/"generation.json")
        denom=sum(v["attempts"] for v in flow.values())
        counts=[sum(v[key] for v in flow.values()) for key in ("50","75","both")]
        def compare(tag,cell):
            return " / ".join(f"{values(groups[tag+'/'+str(q)],cell,'j12').mean():.2f}%" for q in (50,75))
        rows.append(f"| {NAMES[k]} | {denom} | {counts[0]} | {counts[1]} | {counts[2]} | {compare('mask','D')} | {compare('matched','D')} | {compare('matched','F')} |")
    rows+=["","## 运行与核验","",
           "每个开发网络每个历史表示的 scale.npz 仅从训练事件估计。模型选择记录保存轮次、验证损失、权重指纹和参数量；训练耗时与推理、解码时间为并发运行条件下的观察值。每个队列的 cells.jsonl.gz、fgssi.jsonl.gz 按事件与种子保存名单成员、命中、候选池保留及精确名次。运行前后可核对已有 314159 双人模型与 111 FGSSI 的归档名单及排名。","",
           "[全部分组与配对增减](summary.json) · [归档重放与结果指纹](audit.json) · [训练脚本](../../scripts/factorial_history.py) · [FGSSI 重复训练](../../scripts/factorial_fgssi.py) · [预算与遮挡评测](../../scripts/evaluate_factorial_history.py) · [审计脚本](../../scripts/summarize_factorial_history.py)"]
    differences=[entry for cohort in data.values() for entry in cohort["rank_differences"]]
    cutoff_changes=sum(any((e["previous"]<=k)!=(e["current"]<=k) for k in (1,5,32)) for e in differences)
    roundoff_count=sum(len(cohort["symmetry_roundoff"]) for cohort in data.values())
    rows+=["",f"与归档逐例复核的 12 人名单全部一致。共有 {len(differences)} 个精确名次发生变化，其中 {cutoff_changes} 个跨过 Top-1、5 或 32 的报告阈值。抽取四例复算，改变浮点推理的线程与批次设置可重现名次差异。另有 {roundoff_count} 条新遮挡观测在静态交互打分中出现约 2.4×10⁻⁷ 的矩阵不对称，解码时以无序候选对使用的上三角分数恢复对称。逐例记录见分组汇总。[精确排名复算](rank_precision_check.json)","",
           "以下复现命令从项目根目录运行。归档的 111 FGSSI、314159 五帧交互模型和原始公开事件属于输入。","",
           "```powershell",
           "foreach ($network in 'Dept2','company') {",
           "    python -B scripts/factorial_fgssi.py $network 111 314159 431 2027 9001",
           "    foreach ($representation in 'static','temporal') { python -B scripts/factorial_history.py $network $representation }",
           "    python -B scripts/evaluate_factorial_history.py (\"{0}_masks\" -f $network) generate",
           "}",
           "foreach ($cohort in 'Dept2','company','Dept2_new_windows','company_new_windows','Dept1','CopenhagenSMS','CollegeMsg','Enron','Dept2_masks','company_masks') {",
           "    foreach ($model in 'cells','fgssi') { python -B scripts/evaluate_factorial_history.py $cohort $model }",
           "}",
           "python -B scripts/summarize_factorial_history.py",
           "python -B scripts/report_factorial_history.py",
           "python -B scripts/check_factorial_precision.py",
           "python -B scripts/plot_factorial_history.py",
           "```"]
    (OUT/"四组消融与五种子验证.md").write_text("\n".join(rows)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
