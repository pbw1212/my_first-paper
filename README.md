# 早期参与记录缺失时的双源定位

[论文 PDF](paper_dasfaa/main.pdf) · [LaTeX 源码与构建](paper_dasfaa/README.md) · [数据与复现](docs/reproduction.md)

当一个群体只留下后期的参与记录，能否用此前的通信历史，把最早的两位参与者缩小到一份 12 人名单？一位源头可能没有直接带来任何新成员，因而多对不同的初始成员都能解释同一段后期记录。项目研究哪些历史联系有辨识度，源头选择方式改变后方法是否仍有效，以及增加观测能把候选范围缩小到什么程度。

## 研究对象

| 实验 | 模型可用信息 | 要恢复的身份 |
| --- | --- | --- |
| 时序网络上的双源模拟 | 窗口前的有向通信、历史名册、遮挡后的加入批次 | 模拟器抽取的两位初始成员 |
| 真实邮件讨论 | 截止时刻前的讨论历史、遮挡后仍可见的参与成员 | 根邮件作者和首位随后发言的不同作者 |

主指标 **Both@12** 衡量 12 人名单同时覆盖两位目标的比例。实验记录中使用 Both-in-12。精确二人组另用 Top-1、Hit@32 和 MRR 评价。真实邮件可核对先后发言角色，真实共同发起者身份仍需另外取得带标注的数据。

## 目前的结果

| 问题 | 已有证据 |
| --- | --- |
| 历史时序信息能否改善名单？ | 从 Dept2 冻结迁移至同机构的 Dept4，在 1,308 条模拟事件上，完整时序模型命中 23.49%，静态模型为 20.87%，五帧 FGSSI 适配为 17.87%。[同队列对照](paper_dasfaa/evidence/top12.json) |
| 近期联系在哪些网络有效？ | 加入候选人与可见成员的近期联系，Dept2 由 1.28% 升至 14.68%，制造企业由 2.15% 变为 2.17%。[Dept2](runs/recent_member_evidence/Dept2/summary.json) · [制造企业](runs/recent_member_evidence/company/summary.json) |
| 源头选择偏向活跃成员时如何变化？ | Enron 中，当 80% 的源头抽取使用双人活跃度权重，显式先验相对同样本直接训练由 34.05% 升至 39.75%。改为偏向近期有直接联系的二人组后，显式先验低 2.24 个百分点。[训练对照](runs/trained_source_mixtures/summary.json) |
| 完整记录能提供多少信息？ | 可计算沉默伙伴的条件后验。在拥有完整接触流及精确加入时刻的 12,280 条模拟轨迹中，额外提供成功父节点只改变了 3 条轨迹的最优名单条件覆盖值。[沉默伙伴](runs/source_prior_shift/Enron/ceiling/summary.json) · [加入时刻上限核验](runs/activation_time_ceiling/verification.json) |
| 方法能否迁移到真实讨论？ | 初稿的较深遮挡实验中，模拟训练模型未稳定超过近期发帖量。后续在 1,793 条真实讨论中只遮住最早两人，GNU libc 两个时期的历史角色模型均超过发帖量，新增关联特征在不同社区和时期的增益有正、有零、也有负。[初稿实验](runs/trained_source_mixtures/early_speakers/summary.json) · [后续留出](docs/experiments.md#真实讨论只遮住最早两人) |

## 仓库导航

| 目录 | 内容 |
| --- | --- |
| [paper_dasfaa/](paper_dasfaa/README.md) | 16 页 LNCS 投稿稿、LaTeX 源码、[六张投稿表](paper_dasfaa/tables/)与[六幅矢量图](paper_dasfaa/figures/) |
| [docs/](docs/) | [研究背景](docs/research.md)、[实验索引](docs/experiments.md)、[数据与复现](docs/reproduction.md) |
| [scripts/](scripts/) | 实验实现、图表生成和审计脚本 |
| [runs/](runs/) | 实验协议、冻结汇总与核验结果。参见[主要实验与统计口径](docs/closure_evidence_pack.md) |

## 历史记录

[实验底稿](docs/论文实验结果与写作底稿.md)记录各阶段实验与复现顺序。[WP6 归档表](runs/paper_closure/wp6_tables/report/main_tables.md)记录工作包收尾时的结果。[早期 Markdown 初稿](https://github.com/pbw1212/my_first-paper/blob/cd6b4e222703067b8d527a5cdc43c237cd96bfb4/%E8%AE%BA%E6%96%87%E5%88%9D%E7%A8%BF.md)保存在对应提交。
