"""WP6. Resolve paper evidence and independently audit event-weighted paired block intervals."""
from functools import lru_cache
from scipy.stats import rankdata
from closure_common import *

F=BASE/"wp6_evidence"
DOC=Path(os.environ.get("CLOSURE_DOC_ROOT",REPO/"docs"))/"closure_evidence_pack.md"
WPS=["wp1_budgets","wp2_predictability","wp3_robustness","wp4_ksources","wp5_sim2real"]
PREFIX="runs/paper_closure/"
TABLES=[
 ("T1","固定与比例预算、软召回和精确对排名",WPS[0]+"/summary.json","网络 → methods / differences"),
 ("T2","近期联系区分度、结构统计与名单收益",WPS[1]+"/summary.json","networks / correlations；模拟和真实基线分别列明"),
 ("T3","三档输入的树模型与非 SI 机制",WPS[2]+"/summary.json","tree / stress；不同规则分母和共同抽取分别展示"),
 ("T4","源头先验变化与同样本训练对照","runs/trained_source_mixtures/summary.json","Enron → mostly_active / linked；使用归档的 pp 单位"),
 ("T5","多源名单、逐源召回与沉默事件",WPS[3]+"/summary.json","网络 → 抽取条件/真实 k；对应 All-in-12、Recall@12、silent_event"),
 ("T6","模拟与真实角色的桥接",WPS[4]+"/summary.json","real 各深度、共同讨论与共同可打分；sim 各条件与共同抽取机制分解")]
FIGS=[
 ("F1","固定人数预算曲线",WPS[0]+"/budget_curve.png","对应 T1"),
 ("F2","AUC 与增益的跨网络散点",WPS[1]+"/auc_gain.png","对应 T2；按比较基线分两条回归线"),
 ("F3","源数与沉默事件、名单召回",F.name+"/ksource_boundary.png","对应 T5；未知 k 组按真实 k 分层"),
 ("F4","真实讨论遮挡深度扫描",WPS[4]+"/depth_scan.png","对应 T6；各深度以各自全部合格线程为分母")]
APP=[
 ("Dept3 早期可行性","runs/dept3_pilot/summary.json","用于说明基线与任务演变。"),
 ("128 路径补前缀","runs/dept3_forward/sampling_check.json","观察路径波动后的追加敏感性分析。"),
 ("盲估计 EM","runs/blind_source_prior/summary.json","未知先验估计的探索性诊断。"),
 ("稳健下界","runs/robust_source_list/summary.json","目标是模型内后验覆盖，实测覆盖可能下降。"),
 ("BOSouL 单快照","runs/recent_member_evidence/single_snapshot/Dept2/summary.json","Dept2 与 company 各 100 例；搜索预算、观测、候选均有适配。"),
 ("BOSouL 单快照制造企业","runs/recent_member_evidence/single_snapshot/company/summary.json","与 Dept2 分列；同为每例两人精确输出。"),
 ("完整父节点与精确激活时刻","runs/activation_time_ceiling/verification.json","特权信息上限与小图穷举。"),
 ("历史 Dept4 前瞻比较","runs/history_components/Dept4/protocol_lock.json","首次冻结比较的出处；后续简化模型与本轮复评均标事后。")]


def resolved(path):
    return REPO/path if path.startswith("runs/") else BASE/path


def canonical(path):
    return path if path.startswith("runs/") else PREFIX+path


@lru_cache(None)
def bootstrap_design(w,s):
    # Independently form multiplicities of sampled windows and seeds (no indexing the
    # event-score table with a 3D broadcast, as in the producer).
    rng=np.random.default_rng(20261120)
    origins=rng.integers(w,size=(10000,(w+2)//3))
    windows=np.column_stack([(origins[:,i//3]+i%3)%w for i in range(w)])
    seeds=rng.integers(s,size=(10000,s))
    return np.stack([(windows==i).sum(1) for i in range(w)],1),np.stack([(seeds==i).sum(1) for i in range(s)],1)/s


def check_ci(rows,keys,arr,stat):
    values=np.asarray(arr,float).reshape(len(rows),-1)
    nw,ns=len(keys),values.shape[1]
    wc,sc=bootstrap_design(nw,ns)
    at=[keys.index(r["window"]) for r in rows]
    totals=np.bincount(at,minlength=nw)
    val=np.zeros((nw,ns))
    for i,w in enumerate(at):
        val[w]+=values[i]
    den=wc@totals
    take=den>0
    sampled=((wc@val)*sc).sum(1)[take]/den[take]
    assert int(take.sum())==stat["valid"]
    assert abs(val.sum()/values.size-stat["mean"])<1e-12
    assert np.allclose(np.percentile(sampled,[2.5,97.5]),stat["ci95"],atol=1e-12,rtol=0),stat
    return 1


def check_group(rows,s):
    assert len({r["id"] for r in rows})==len(rows)==s["n"]
    if not rows:
        return 0
    assert dict(Counter(r["window"] for r in rows))==s["window_n"]
    n=0
    for m,stat in s["methods"].items():
        n+=check_ci(rows,s["keys"],[r["scores"][m] for r in rows],stat)
    for name,stat in s["differences"].items():
        a,b=name.split(" minus ")
        arr=[np.subtract(r["scores"][a],r["scores"][b]) for r in rows]
        assert np.allclose(arr,[r["delta"][name] for r in rows])
        n+=check_ci(rows,s["keys"],arr,stat)
    return n


def independent():
    d=[load(BASE/f/"summary.json") for f in WPS]
    counts=[0]*5
    for name,s in d[0].items():
        rows=read_rows(BASE/WPS[0]/(name+".jsonl.gz"))
        counts[0]+=check_group(rows,s)
        assert s["n"]==d[1]["networks"]["sim "+name]["n"]
    for name,s in d[1]["networks"].items():
        rows=read_rows(BASE/WPS[1]/(name+".jsonl.gz"))
        counts[1]+=check_group(rows,s)
        audit=[r["diagnostic"] for r in rows if "diagnostic" in r]
        assert len(audit)==s["auc_n"] and abs(np.mean([x["auc"] for x in audit])-s["auc"])<1e-12
    for group,vals in d[1]["correlations"].items():
        points=[z for n,z in d[1]["networks"].items() if group.startswith("pooled") or n.startswith(group)]
        for v,coeff in vals.items():
            x=[z[v] for z in points]
            y=[z["gain"] if group!="pooled_common" else z["differences"]["links minus person"]["mean"] for z in points]
            assert np.allclose([np.corrcoef(x,y)[0,1],np.corrcoef(rankdata(x),rankdata(y))[0,1]],[coeff["pearson"],coeff["spearman"]])
    for name,d0 in d[2]["tree"].items():
        rows=read_rows(BASE/WPS[2]/("tree-"+name+".jsonl.gz"))
        for b,stat in d0.items():
            counts[2]+=check_group([r for r in rows if r["budget"]==b],stat)
            assert stat["n"]==d[0][name]["n"]
    for name,dr in d[2]["stress"].items():
        rows=read_rows(BASE/WPS[2]/("stress-"+name+".jsonl.gz"))
        common={k for k,v in Counter(r["id"] for r in rows).items() if v==4}
        for rule,ss in dr.items():
            part=[r for r in rows if r["rule"]==rule]
            counts[2]+=check_group(part,ss["all"])+check_group([r for r in part if r["id"] in common],ss["common"])
    for name,stats in d[3].items():
        rows=read_rows(BASE/WPS[3]/(name+".jsonl.gz"))
        for k,s in stats.items():
            arm,num=k.split("/")
            rr=[r for r in rows if r["arm"]==arm and (num=="all" or r["k"]==int(num))]
            for metric in ("All-in-12","Recall@12"):
                counts[3]+=check_group([r[metric] for r in rr],s[metric])
            counts[3]+=check_ci(rr,s["All-in-12"]["keys"],[r["silent"]>0 for r in rr],s["silent_event"])
    for n,stats in d[4]["real"].items():
        rows=read_rows(BASE/WPS[4]/("real-"+n+".jsonl.gz"))
        common={k for k,v in Counter(r["id"] for r in rows).items() if v==3}
        ready={k for k,v in Counter(r["id"] for r in rows if r["ready"]).items() if v==3}
        for dep,ss in stats.items():
            rr=[r for r in rows if r["depth"]==int(dep)]
            for label,part in [("all",rr),("common",[r for r in rr if r["id"] in common]),("common_ready",[r for r in rr if r["id"] in ready])]:
                counts[4]+=check_group(part,ss[label])
        paired=read_rows(BASE/WPS[4]/("real-depth-pairs-"+n+".jsonl.gz"))
        comps=load(BASE/WPS[4]/("depth-comparisons-"+n+".json"))
        for cmp,s in comps["paired"].items():
            for method,stat in s["methods"].items():
                rr=[r for r in paired if r["method"]==method and f'D{r["deeper"]}-D{r["shallower"]}'==cmp]
                assert len(rr)==s["n"]
                counts[4]+=check_ci(rr,stats["0"]["all"]["keys"],[r["delta"] for r in rr],stat)
    for n,stats in d[4]["sim"].items():
        rows=read_rows(BASE/WPS[4]/("sim-"+n+".jsonl.gz"))
        for key,s in stats["cells"].items():
            cell,dep=key.split("/")
            counts[4]+=check_group([r for r in rows if r["depth"]==int(dep) and r["cell"]==cell],s)
        for dep,models in stats["common_draw_decomposition"].items():
            contrib=read_rows(BASE/WPS[4]/(n+"-contributions-D"+dep+".jsonl.gz"))
            grouped={}
            for r in rows:
                if r["depth"]==int(dep):
                    grouped.setdefault(r["id"],{})[r["cell"]]=r
            for m,z in models.items():
                rr=[r for r in contrib if r["model"]==m]
                assert len(rr)==z["n"]==sum(len(v)==4 for v in grouped.values())
                for r in rr:
                    v=grouped[r["id"]]
                    a,b,c,h=[np.array(v[cell]["scores"][m]) for cell in ("uniform_symmetric","active_symmetric","uniform_asymmetric","active_asymmetric")]
                    assert np.array_equal(r["prior"],(b-a+h-c)/2) and np.array_equal(r["role"],(c-a+h-b)/2)
                if rr:
                    for k in ("prior","role","total"):
                        counts[4]+=check_ci(rr,stats["cells"]["uniform_symmetric/"+dep]["keys"],[r[k] for r in rr],z[k])
    return dict(bootstrap_intervals_by_wp=counts,bootstrap_intervals=sum(counts),replicates_each=10000)


def sealed(include_self=True):
    for folder in [*(BASE/name for name in WPS),*([F] if include_self else [])]:
        check_protocol(folder)
        v=load(folder/"verification.json")
        assert v["passed"]
        for rel,h in v["sha256"].items():
            assert sha(folder/rel)==h,folder/rel
    if include_self:
        for path,h in [(DOC,"document"),(BASE/"master_summary.json","master")]:
            assert sha(path)==load(F/"verification.json")["checks"][h]


def run():
    sealed(False)
    lock(F,dict(tables=6,figures=4,independent="reconstruct event/seed differences and 10000 block-by-seed intervals via independent multiplicities",
        earlier_results="keep original data units, distinguish unchanged archive from post-hoc frozen inference or intervention"),
        [Path(__file__),Path(__file__).with_name("closure_all.py"),*(resolved(e[2]) for e in TABLES),*(resolved(e[1]) for e in APP)])
    stats=load(BASE/WPS[3]/"summary.json")
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.6),sharey=True)
    for ax,(name,groups) in zip(axes,stats.items()):
        for label in ("silent_event","All-in-12","Recall@12"):
            z=[groups["hidden_uniform/"+str(k)] for k in (1,2,3)]
            q=[r[label] if label=="silent_event" else r[label]["methods"]["full"] for r in z]
            y=np.array([v["mean"] for v in q])*100
            ax.plot([1,2,3],y,"o-",label=label.replace("_"," "))
            ax.fill_between([1,2,3],[100*v["ci95"][0] for v in q],[100*v["ci95"][1] for v in q],alpha=.1)
        ax.set(title=name,xlabel="True source count",xticks=[1,2,3],ylim=(-4,104))
    axes[0].set_ylabel("Eligible events / source recall (%)")
    axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(F/"ksource_boundary.png",dpi=160);plt.close(fig)
    mapping=lambda items:[dict(id=i,claim=c,path=canonical(p),scope=k) for i,c,p,k in items]
    audit=independent()
    master=dict(tables=mapping(TABLES),figures=mapping(FIGS),appendix=[dict(topic=c,path=p,note=k) for c,p,k in APP],
        packages=[dict(wp=i+1,path=PREFIX+w+"/summary.json",verification=PREFIX+w+"/verification.json") for i,w in enumerate([*WPS,F.name])],
        audit=audit,units="WP1–WP5 rates, differences and CIs are fractions; multiply by 100 for percent or pp. Pair-MRR and correlations unitless. T4 archive already percent / pp.",
        denominators="unique mask-eligible events within network × generator × depth. Replicated seeds, paired counterfactuals, alternative masks and overlapping archival events must not be pooled as new independent events.",
        reproduce="python -B scripts/closure_all.py reproduce",verify="python -B scripts/closure_all.py verify")
    dump(BASE/"master_summary.json",master)
    dump(F/"summary.json",dict(audit=audit,tables=6,figures=4,unique_authority_per_table=True))
    lines=["# 双源定位实验的主文证据包","","本轮为已有网络的协议锁定复评。六张主表各有一个数值来源。同一事件的五次训练预测只构成一个事件分母。","","## 主表与主图","","|编号|支持的判断或图形|唯一数据文件|取数位置与口径|","|---|---|---|---|"]
    for e in master["tables"]+master["figures"]:
        lines.append(f'|{e["id"]}|{e["claim"]}|[{Path(e["path"]).name}](../{e["path"]})|{e["scope"]}|')
    lines.extend(["","## 分母与可恢复性边界","",
        "- T1 的六网依次为 Dept2 1,200、company 1,200、Dept4 1,308、Enron 1,560、CollegeMsg 480、GNU libc 2,750。GNU libc 使用原均匀抽取组。候选池按事件计算，比例预算向上取整。",
        "- T2 的 AUC 只纳入可打分且两位目标均在名册内的事件，真实命中率保留弃权及名册外事件。真实部分统一 D0，至少一名可见名册成员；Python 为 2017–18，模型只在模拟事件上训练。真实发帖量对照与模拟个人历史对照不同；合并分析优先使用同一 links−person 差值。每个社区或时期只有一个相关分析点。",
        "- T3 的树与 MLP、FGSSI 在同一事件、预算、种子上配对。树采用 sklearn 直方图梯度提升。模拟规则按各自合格事件统计，另列四种规则共同合格的抽取。失活区间从激活时刻起算，外部加入为每名非源每窗 2%。",
        "- T4 的 Enron 80% 双人活跃偏向条件有 2,607 个合格事件；同样本显式先验相对直接训练提高 5.70 个百分点，95% 区间 [2.88, 9.39]。近期直连偏向条件回落 2.24 个百分点。其生成分布和分母独立于 T1 的 Enron 1,560 例。",
        "- T5 的未知源数对每次尝试均匀抽取 k；合格率随 k 改变。单源事件在该遮挡门槛下必有传播。Dept2 均匀三源的完整模型 All-in-12 为 1.30%，Enron 为 0.74%。一般 k 的完整父节点轨迹条件后验包含沉默集合内部接触的修正项，公式在 WP4 锁中；数值穷举覆盖 k=1、2、3。",
        "- T6 的 Python 角色系数来自 2017–18，测试仅用 2019 年 2–4 月。真实 D0/D25/D50 全部要求至少三名名册内可见成员才打分，D0 的弃权门槛严于历史仅遮两人的实验。深度改变合格集合，跨深度差值使用同一讨论 ID。D25/D50 对完整早期批次按人数向下截取。",
        "- T6 的模拟角色频数来自预测窗前已经结束的历史模拟，逐人排序只读取公开后缀。四种生成条件共同合格的 D50 抽取只有 Dept2 110 次、Enron 42 次。分解估计这批模拟抽取的先验与角色效应。真实共同发起身份未被观测，真实差距的因果占比无法从这些比较识别。",
        "- 配对区间按训练种子和长度 3 的循环相邻窗块共同重采样 10,000 次，空窗保留，仅剔除整次抽中零事件的重复。四窗比较的时间不确定性估计较弱。Python 三个月恰好构成完整循环块，其区间不能提供月份间不确定性；确定性方法间差值可退化为点。其余真实社区的单一时间段同样限制外推。",
        "- T1–T3 的 Dept4、Enron、CollegeMsg 和 GNU libc 复评属于已使用网络上的比较。早期冻结 Dept4 主比较单独保留。AUC、相关、Pair-MRR 与命中率的量纲各异。T1 的预算差只解释同一 Both 指标在不同预算间的变化。",
        "","## 附录与降级清单",""])
    lines.extend(f'- {c}。{k} [归档记录](../{p})' for c,p,k in APP)
    lines.extend(["","## 复现与独立复核","","从 Git 仓库根目录运行。完整原始数据、邮件头和冻结权重沿用原工作目录；默认自动识别父目录，也可设置环境变量 `CLOSURE_ARCHIVE` 指向带原始 scripts、data、runs 的完整工作目录。仅凭精简 Git 仓库无法重建大制品。","","```powershell","python -B scripts/closure_all.py reproduce","```","","该命令在临时目录完整重建六包，逐项对照科学产物后清理临时副本。只核验已有制品可用以下命令。","","```powershell","python -B scripts/closure_all.py verify","```","",f"独立核对 {audit['bootstrap_intervals']} 个点估计与配对或单方法区间，并检查工作包间的相同队列分母。五包机器可读结果以单位比例记录，乘 100 转为百分比或百分点。历史先验实验已经使用百分比及百分点。","","[总索引](../runs/paper_closure/master_summary.json) · [第六包复核](../runs/paper_closure/wp6_evidence/verification.json)"])
    DOC.parent.mkdir(parents=True,exist_ok=True)
    DOC.write_text("\n".join(lines)+"\n",encoding="utf8")


def verify():
    sealed(False)
    m=load(BASE/"master_summary.json")
    a=independent()
    assert a==m["audit"]==load(F/"summary.json")["audit"]
    for item in m["tables"]+m["figures"]+m["appendix"]+m["packages"]:
        p=item["path"]
        assert (BASE/p.removeprefix(PREFIX) if p.startswith(PREFIX) else REPO/p).is_file(),p
    old=load(REPO/"runs/trained_source_mixtures/summary.json")["Enron"]
    for k,n,delta in [("mostly_active",2607,5.70),("linked",1267,-2.24)]:
        z=old[k]
        assert sum(z["window_counts"])==z["n"]==n and round(z["paired_to_mixed_plain"]["mixed_offset"]["pp"],2)==delta
    evidence(F,dict(**a,document=sha(DOC),master=sha(BASE/"master_summary.json"),prior_archive=True))


if __name__=="__main__":
    if "--sealed" in sys.argv:
        sealed()
    else:
        if "--verify" not in sys.argv:
            run()
        verify()

