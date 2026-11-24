# 双源定位实验的主文证据包

本轮为已有网络的协议锁定复评。六张主表各有一个数值来源。同一事件的五次训练预测只构成一个事件分母。

## 主表与主图

|编号|支持的判断或图形|唯一数据文件|取数位置与口径|
|---|---|---|---|
|T1|固定与比例预算、软召回和精确对排名|[summary.json](../runs/paper_closure/wp1_budgets/summary.json)|网络 → methods / differences|
|T2|近期联系区分度、结构统计与名单收益|[summary.json](../runs/paper_closure/wp2_predictability/summary.json)|networks / correlations；模拟和真实基线分别列明|
|T3|三档输入的树模型与非 SI 机制|[summary.json](../runs/paper_closure/wp3_robustness/summary.json)|tree / stress；不同规则分母和共同抽取分别展示|
|T4|源头先验变化与同样本训练对照|[summary.json](../runs/trained_source_mixtures/summary.json)|Enron → mostly_active / linked；使用归档的 pp 单位|
|T5|多源名单、逐源召回与沉默事件|[summary.json](../runs/paper_closure/wp4_ksources/summary.json)|网络 → 抽取条件/真实 k；对应 All-in-12、Recall@12、silent_event|
|T6|模拟与真实角色的桥接|[summary.json](../runs/paper_closure/wp5_sim2real/summary.json)|real 各深度、共同讨论与共同可打分；sim 各条件与共同抽取机制分解|
|F1|固定人数预算曲线|[budget_curve.png](../runs/paper_closure/wp1_budgets/budget_curve.png)|对应 T1|
|F2|AUC 与增益的跨网络散点|[auc_gain.png](../runs/paper_closure/wp2_predictability/auc_gain.png)|对应 T2；按比较基线分两条回归线|
|F3|源数与沉默事件、名单召回|[ksource_boundary.png](../runs/paper_closure/wp6_evidence/ksource_boundary.png)|对应 T5；未知 k 组按真实 k 分层|
|F4|真实讨论遮挡深度扫描|[depth_scan.png](../runs/paper_closure/wp5_sim2real/depth_scan.png)|对应 T6；各深度以各自全部合格线程为分母|

## 分母与可恢复性边界

- T1 的六网依次为 Dept2 1,200、company 1,200、Dept4 1,308、Enron 1,560、CollegeMsg 480、GNU libc 2,750。GNU libc 使用原均匀抽取组。候选池按事件计算，比例预算向上取整。
- T2 的 AUC 只纳入可打分且两位目标均在名册内的事件，真实命中率保留弃权及名册外事件。真实部分统一 D0，至少一名可见名册成员；Python 为 2017–18，模型只在模拟事件上训练。真实发帖量对照与模拟个人历史对照不同；合并分析优先使用同一 links−person 差值。每个社区或时期只有一个相关分析点。
- T3 的树与 MLP、FGSSI 在同一事件、预算、种子上配对。树采用 sklearn 直方图梯度提升。模拟规则按各自合格事件统计，另列四种规则共同合格的抽取。失活区间从激活时刻起算，外部加入为每名非源每窗 2%。
- T4 的 Enron 80% 双人活跃偏向条件有 2,607 个合格事件；同样本显式先验相对直接训练提高 5.70 个百分点，95% 区间 [2.88, 9.39]。近期直连偏向条件回落 2.24 个百分点。其生成分布和分母独立于 T1 的 Enron 1,560 例。
- T5 的未知源数对每次尝试均匀抽取 k；合格率随 k 改变。单源事件在该遮挡门槛下必有传播。Dept2 均匀三源的完整模型 All-in-12 为 1.30%，Enron 为 0.74%。一般 k 的完整父节点轨迹条件后验包含沉默集合内部接触的修正项，公式在 WP4 锁中；数值穷举覆盖 k=1、2、3。
- T6 的 Python 角色系数来自 2017–18，测试仅用 2019 年 2–4 月。真实 D0/D25/D50 全部要求至少三名名册内可见成员才打分，D0 的弃权门槛严于历史仅遮两人的实验。深度改变合格集合，跨深度差值使用同一讨论 ID。D25/D50 对完整早期批次按人数向下截取。
- T6 的模拟角色频数来自预测窗前已经结束的历史模拟，逐人排序只读取公开后缀。四种生成条件共同合格的 D50 抽取只有 Dept2 110 次、Enron 42 次。分解估计这批模拟抽取的先验与角色效应。真实共同发起身份未被观测，真实差距的因果占比无法从这些比较识别。
- 配对区间按训练种子和长度 3 的循环相邻窗块共同重采样 10,000 次，空窗保留，仅剔除整次抽中零事件的重复。四窗比较的时间不确定性估计较弱。Python 三个月恰好构成完整循环块，其区间不能提供月份间不确定性；确定性方法间差值可退化为点。其余真实社区的单一时间段同样限制外推。
- T1–T3 的 Dept4、Enron、CollegeMsg 和 GNU libc 复评属于已使用网络上的比较。早期冻结 Dept4 主比较单独保留。AUC、相关、Pair-MRR 与命中率的量纲各异。T1 的预算差只解释同一 Both 指标在不同预算间的变化。

## 附录与降级清单

- Dept3 早期可行性。用于说明基线与任务演变。 [归档记录](../runs/dept3_pilot/summary.json)
- 128 路径补前缀。观察路径波动后的追加敏感性分析。 [归档记录](../runs/dept3_forward/sampling_check.json)
- 盲估计 EM。未知先验估计的探索性诊断。 [归档记录](../runs/blind_source_prior/summary.json)
- 稳健下界。目标是模型内后验覆盖，实测覆盖可能下降。 [归档记录](../runs/robust_source_list/summary.json)
- BOSouL 单快照。Dept2 与 company 各 100 例；搜索预算、观测、候选均有适配。 [归档记录](../runs/recent_member_evidence/single_snapshot/Dept2/summary.json)
- BOSouL 单快照制造企业。与 Dept2 分列；同为每例两人精确输出。 [归档记录](../runs/recent_member_evidence/single_snapshot/company/summary.json)
- 完整父节点与精确激活时刻。特权信息上限与小图穷举。 [归档记录](../runs/activation_time_ceiling/verification.json)
- 历史 Dept4 前瞻比较。首次冻结比较的出处；后续简化模型与本轮复评均标事后。 [归档记录](../runs/history_components/Dept4/protocol_lock.json)

## 复现与独立复核

从 Git 仓库根目录运行。完整原始数据、邮件头和冻结权重沿用原工作目录；默认自动识别父目录，也可设置环境变量 `CLOSURE_ARCHIVE` 指向带原始 scripts、data、runs 的完整工作目录。仅凭精简 Git 仓库无法重建大制品。

```powershell
python -B scripts/closure_all.py reproduce
```

该命令在临时目录完整重建六包，逐项对照科学产物后清理临时副本。只核验已有制品可用以下命令。

```powershell
python -B scripts/closure_all.py verify
```

独立核对 1119 个点估计与配对或单方法区间，并检查工作包间的相同队列分母。五包机器可读结果以单位比例记录，乘 100 转为百分比或百分点。历史先验实验已经使用百分比及百分点。

[总索引](../runs/paper_closure/master_summary.json) · [第六包复核](../runs/paper_closure/wp6_evidence/verification.json)
