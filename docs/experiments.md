# 实验进展与证据索引

更新于 2026-11-22。[早期 Markdown 初稿](https://github.com/pbw1212/my_first-paper/blob/cd6b4e222703067b8d527a5cdc43c237cd96bfb4/%E8%AE%BA%E6%96%87%E5%88%9D%E7%A8%BF.md)记录了模拟主实验和第一轮真实讨论迁移。[实验底稿](论文实验结果与写作底稿.md)保留详细分母、逐窗差值和复现顺序。较新的 GNU libc、R 与 Samba 试验见下文。

## 实验阶段

| 阶段 | 完成内容与结论 | 报告或统计 |
| --- | --- | --- |
| Dept3 可行性 | 生成遮挡场景，建立随机、活跃度、有向距离覆盖基线。依次测试节点候选检索、双源排序、隐藏前缀与正向评分，检查候选池遗漏和并列。 | [首轮基线](../runs/dept3_pilot/首轮基线报告.md) · [候选生成](../runs/dept3_candidates/候选生成实验报告.md) · [集合排序](../runs/dept3_pair_ranker/双源集合排序实验报告.md) · [正向评分](../runs/dept3_forward/隐藏前缀与正向评分实验报告.md) |
| 时间与跨部门留出 | 在 CollegeMsg 复核并拆分时间段，开展预算与双源可见证据分层。Dept3 原定末期完整窗无接触，记录空留出并启动事先固定的 Dept1 迁移。 | [CollegeMsg 复核](../runs/collegemsg_replication/CollegeMsg跨网络复核报告.md) · [多预算候选](../runs/conditional_candidates/候选筛选与预算实验.md) · [双源可见组](../runs/visible_dual_source/双源可见证据条件实验报告.md) · [冻结留出与复核](../runs/thesis_holdout/reproduction.json) |
| 跨机构、沉默源头与部分识别 | 在制造企业比较联合回放、BOSouL 适配与候选空间；评估第一位源头、沉默伙伴和同为 12 人的各种名单解码。 | [制造企业](../runs/masked_pair_benchmark/制造企业跨机构双源实验.md) · [部分识别](../runs/partial_source_identification/部分源头识别与候选名单实验.md) · [沉默源头](../runs/ordinary_source_roles/沉默源头识别实验.md) |
| 模型与公开基线 | 实现历史时序证据、无序组合打分与轨迹后验监督，分别比较 GCNSI、双中心、五帧 FGSSI、BOSouL 的后缀适配。冻结后首次评测 Enron 并保留后来使用已看网络的五种子消融。 | [时序证据](../runs/temporal_pair_evidence/方法与投稿判断.md) · [组合学习](../runs/distilled_source_pairs/双源组合学习与预算实验.md) · [同快照对照](../runs/matched_source_baselines/公开基线与后缀顺序消融.md) · [首次 Enron](../runs/temporal_transfer/时序公开方法与独立网络验证.md) · [因子消融](../runs/factorial_history/四组消融与五种子验证.md) |
| 历史特征与传播规则 | 拆分有向时序历史、发信量与候选人对后期可见成员的近期联系。Dept4 冻结比较验证完整模型，之后的简化特征迁移属于事后分析。压力测试覆盖两个开发网络各四个窗口和四种规则。 | [历史特征与冻结 Dept4](../runs/history_components/summary.json) · [活动量控制](../runs/activity_control/summary.json) · [联系区分度](../runs/recent_member_evidence/contact_diagnostic_and_audit.json) · [Dept2 传播规则](../runs/recent_member_evidence/stress/Dept2/summary.json) · [制造企业传播规则](../runs/recent_member_evidence/stress/company/summary.json) |
| 源头先验及可恢复性 | 完成已知先验校正、32 例有标签校准、盲估计、模型内稳健名单、同训练样本的显式先验比较，以及完整轨迹的精确似然与名单上限核验。 | [先验变化](../runs/source_prior_shift/results.json) · [盲估计](../runs/blind_source_prior/summary.json) · [名单后验下界](../runs/robust_source_list/summary.json) · [精确时刻与上限](../runs/activation_time_ceiling/verification.json) · [混合训练](../runs/trained_source_mixtures/summary.json) |
| 新社区的全套模拟 | 在 GNU libc 2021–22 的 24 个月，冻结 Dept2 权重，评测均匀、等权混合、活跃度偏向、近期直连偏向四类抽取。合格模拟事件共 26,245 条。双人活跃度偏向占 80% 时，混合直接训练为 41.06%，显式模型为 43.78%，简单校正为 43.67%，发帖量排序为 44.16%。 | [GNU libc 冻结协议](../runs/heldout_libc_alpha/protocol.json) · [结果](../runs/heldout_libc_alpha/summary.json) · [独立核验](../runs/heldout_libc_alpha/verification.json) |
| 真实讨论迁移及诊断 | 初稿覆盖 Python-Dev、Python-Ideas、GCC 和 FFmpeg。后续新增两个 R 列表及 GNU libc，完成回复方向、角色配对、成员选择消融、GNU libc 时间留出和 Samba 社区留出。 | [Python 首轮](../runs/public_thread_transfer/summary.json) · [Python 后续期](../runs/public_thread_transfer/calendar_followup/summary.json) · [GCC 与 FFmpeg](../runs/historical_thread_roles/summary.json) · [名册上限](../runs/real_roster_ceiling/summary.json) · [R 列表](../runs/heldout_r_lists/summary.json) · [回复方向](../runs/heldout_r_lists/direction/heldout.json) · [角色配对](../runs/heldout_r_lists/speaker_pairs/summary.json) · [最新收尾](../runs/heldout_r_lists/early_only/role_evidence/closure/实验收尾记录.md) |

## 模拟主结果与边界

- **冻结 Dept4。** 1,308 条合格事件中完整时序模型的 Both-in-12 为 23.49%，相对 FGSSI 五帧适配提高 5.61 个百分点，配对 95% 区间为 [4.44, 6.98] 个百分点。Dept4、Dept2 来自同一机构。首次 Enron 冻结测试用的是早期组合模型，后来用已看过的 Enron 比较 9.47%、7.01% 和 6.27% 属于新模型的事后分析。[Dept4 协议](../runs/history_components/Dept4/protocol_lock.json) · [首次 Enron 协议](../runs/temporal_transfer/Enron/protocol_lock.json)
- **近期联系。** 加入四项候选人与可见成员的联系特征，Dept2 从 1.28% 升至 14.68%，而制造企业从 2.15% 变为 2.17%。真源和其他候选人在过去 28 天至少发出一次此类联系的比例，在 Dept2 为 52.2% 对 8.6%，在制造企业为 67.7% 对 55.1%。[Dept2 消融](../runs/recent_member_evidence/Dept2/summary.json) · [制造企业消融](../runs/recent_member_evidence/company/summary.json)
- **先验变化。** Enron 在 80% 双人活跃度抽取条件下，显式模型在同样本直接训练的 34.05% 基础上达到 39.75%，相对简单校正的差值为 +0.83 个百分点，95% 区间跨零。偏向近期直连组合时，显式模型比直接训练低 2.24 个百分点。无标签先验估计和提高模型内最差后验覆盖下界都曾降低实测命中。[混合训练](../runs/trained_source_mixtures/summary.json) · [盲估计](../runs/blind_source_prior/summary.json) · [稳健名单](../runs/robust_source_list/summary.json)
- **沉默源头。** 在 Enron 仅一人直接传播的组中，完整轨迹条件下的平均最优名单覆盖，从均匀先验的 14.22% 变为双人活跃先验的 51.73%，对应各自生成的事件人群。完整时间戳基础上再增加父节点只对极少数轨迹改变条件覆盖。这些是模拟规则和观测权限给定时的上限分析。[伙伴上限](../runs/source_prior_shift/Enron/ceiling/summary.json) · [加入时刻分析](../runs/activation_time_ceiling/Enron/summary.json)
- **精确点名。** 较大的 CollegeMsg 名册中，单种子组合模型在 480 条事件内命中至少一位 52 次、双人 0 次。开发测试的 Dept2 与制造企业单快照比较各 100 例，双方精确 Top-1 都为 0。[五种子报告](../runs/factorial_history/四组消融与五种子验证.md) · [Dept2 单快照](../runs/recent_member_evidence/single_snapshot/Dept2/summary.json) · [制造企业单快照](../runs/recent_member_evidence/single_snapshot/company/summary.json)

## 真实讨论的较深遮挡

在这组实验中，隐藏最早的参与批次后，主要分母是符合后缀条件的全部线程。目标是根邮件作者和接下来 14 天最早的不同作者。2017–2018 年 Python-Dev 与 Python-Ideas 各有 174 和 243 个合格事件。近期联系模型的 Both-in-12 分别为 5.86% 和 3.46%，近期发帖量基线为 14.37% 和 7.00%。2019 年后续月份只有 51 条合格线程，真实校准后的联系模型仍未超过发帖量。四个社区的 524 条同任务合格线程上，更新的混合先验模拟训练也未稳定超过发帖量。GCC 和 FFmpeg 的历史讨论相似性检索没有新增双人命中。扩大名册提高了两位目标同时可入选的上限，实际命中增长很小。[Python 首轮](../runs/public_thread_transfer/summary.json) · [2019 配对审计](../runs/public_thread_transfer/calendar_followup/audit.json) · [真实迁移](../runs/trained_source_mixtures/early_speakers/summary.json) · [角色检索](../runs/historical_thread_roles/summary.json) · [名册诊断](../runs/real_roster_ceiling/summary.json)

## 真实讨论只遮住最早两人

后续实验只遮住首发者与首位不同回应者，保留其后的可见成员。各行使用同一 98 天历史名册、12 人名单和弃权规则，名册外与无法打分事件也进入分母。模型在较早的 Python 邮件讨论上拟合。GNU libc 2024 为时间留出，Samba 2020 为另外一个社区的冻结留出。下表报告两人同时进入名单的事件数。[冻结协议](../runs/heldout_r_lists/early_only/role_evidence/closure/protocol.json) · [核验和分层结果](../runs/heldout_r_lists/early_only/role_evidence/closure/summary.json)

| 社区与时期 | 合格讨论 | 28 天发帖量 | 历史角色 | 直接回复 | 完整模型系数＋新增特征置零 | 完整模型 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R-devel，2021–22 | 170 | 2 | 6 | 6 | 6 | 6 |
| R-package-devel，2021–22 | 227 | 9 | 12 | 12 | 12 | 15 |
| GNU libc，2021–22 | 880 | 354 | 375 | 376 | 376 | 383 |
| GNU libc，2024 | 437 | 122 | 136 | 132 | 134 | 134 |
| Samba，2020 | 79 | 14 | 21 | 23 | 23 | 22 |

1,793 条讨论来自四个社区，GNU libc 有两个时期。历史角色在 GNU libc 两时期都比发帖量多命中。固定完整模型系数后开启两项新增特征，GNU libc 2021–22 净增 7 例，2024 净变为 0，Samba 净减 1 例。Samba 仅有 41 条同时可打分且两位目标均在名册内，完整模型相对发帖量的差值为 +10.13 个百分点，95% 区间为 [−1.12, +27.42]。在固定候选范围的另一组消融中，按模型打分挑选额外可见成员未稳定超过均匀选择。该任务衡量真实讨论的早期角色恢复，两种深度的遮挡各有自己的分母和信息条件。[收尾记录与独立核验](../runs/heldout_r_lists/early_only/role_evidence/closure/实验收尾记录.md) · [成员选择](../runs/heldout_r_lists/early_only/member_ablation/summary.json)
