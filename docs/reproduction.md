# 数据与复现说明

## 最终论文和实验主表

[最终论文工程](../paper_dasfaa/README.md)可使用仓库内的图表直接编译，也可用冻结汇总重新生成图表。[主结果核验](closure_evidence_pack.md)说明如何用原始归档检查逐事件证据与完整复跑。

## 从首轮实验开始

从仓库根目录执行以下 PowerShell 命令。该流程只使用 Python 标准库和 SNAP 的 Dept3 数据，先生成公开观测与独立真值，再计算基线并逐例复核。归档的 [汇总](../runs/dept3_pilot/summary.json) 和 [审计](../runs/dept3_pilot/audit.json) 可先用于对照。若需保留归档文件原样，在仓库副本内运行。

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
Invoke-WebRequest -Uri 'https://snap.stanford.edu/data/email-Eu-core-temporal-Dept3.txt.gz' -OutFile 'data/raw/email-Eu-core-temporal-Dept3.txt.gz'
python -B scripts/build_dept3_pilot.py
python -B scripts/evaluate_dept3_pilot.py
python -B scripts/check_dept3_pilot.py
```

以上流程已在 Python 3.13.5 的独立目录用重新下载的数据执行。训练与验证各生成 100 个合格事件，`summary.json` 与 `audit.json` 均与归档逐字节一致。

## 数据来源和输入版本

模拟数据的原始文件放入 `data/raw/`，各实验报告给出样本纳入条件、窗口和时间切分。运行前可用 `Get-FileHash -Algorithm SHA256 data/raw/*` 对照指纹。

| 输入文件 | 来源 | SHA-256 |
| --- | --- | --- |
| `email-Eu-core-temporal-Dept1.txt.gz` | [SNAP 部门邮件](https://snap.stanford.edu/data/email-Eu-core-temporal.html) | `d914eb16d63cf48db86bb197fd939231092013937d04e80c8e0298bd20248fbf` |
| `email-Eu-core-temporal-Dept2.txt.gz` | [SNAP 部门邮件](https://snap.stanford.edu/data/email-Eu-core-temporal.html) | `8e8dcdb9049a671f2afcb4bc91ff8ad77423a40612ce8318ef0825aa8b44b53d` |
| `email-Eu-core-temporal-Dept3.txt.gz` | [SNAP 部门邮件](https://snap.stanford.edu/data/email-Eu-core-temporal.html) | `b1f92288e855044bbc74bacac39d42b12e59ec2a0130ad8df419786eb46a7bff` |
| `email-Eu-core-temporal-Dept4.txt.gz` | [SNAP 部门邮件](https://snap.stanford.edu/data/email-Eu-core-temporal.html) | `82756f2f2811cf94d4d58b89b95e2fbdb6ce9fa516a3701cec4001e0c6c64b6c` |
| `CollegeMsg.txt.gz` | [SNAP CollegeMsg](https://snap.stanford.edu/data/CollegeMsg.html) | `50ae2d98ed3bad9ddb18dbd495a89e5e10cfb8f7e86932827db29fc41b41f9fa` |
| `ia-radoslaw-email.zip` | [Network Repository 的时间戳版本](https://networkrepository.com/ia-radoslaw-email.php) | `8dbaf26082d8954e31a5dade6c39ac3d87b92143963388207b058df013606878` |
| `enron-igraph.rda` | [igraphdata 固定版本的 enron.rda](https://raw.githubusercontent.com/igraph/igraphdata/d61dabac88607cf726323dcc68109fb5753aebc7/data/enron.rda)，保存时采用本列文件名 | `68df616583a009c323131574c833f615443756087dda9a97e27a0ad8d658fd5a` |
| `copenhagen-sms.csv.zip` | [Copenhagen Networks Study 的 SMS 数据](https://www.nature.com/articles/s41597-019-0325-x) | `9d51b590273320c42fd4d5d106e74e28e9f8d1254ba67c71131688bbb6435aa7` |

真实讨论的原始输入从各项目官方归档按协议中的月份下载。解析脚本使用发件人、回复关系和时间重建线程，以散列标识处理邮箱及消息 ID。归档指纹和解析计数可在 `runs/` 中查看。GNU libc 邮件同时提供真实角色标签与独立的模拟接触流，两个实验使用各自的真值。[Python-Dev](https://mail.python.org/pipermail/python-dev/) · [Python-Ideas](https://mail.python.org/pipermail/python-ideas/) · [GCC](https://gcc.gnu.org/pipermail/gcc/) · [FFmpeg](https://ffmpeg.org/pipermail/ffmpeg-devel/) · [R-devel](https://stat.ethz.ch/pipermail/r-devel/) · [R-package-devel](https://stat.ethz.ch/pipermail/r-package-devel/) · [GNU libc](https://sourceware.org/pipermail/libc-alpha/) · [Samba](https://lists.samba.org/archive/samba-technical/)

## 主实验的依赖与执行顺序

主要数值依赖包括 NumPy、SciPy、scikit-learn、PyTorch 和 threadpoolctl。绘图使用 Matplotlib，个别诊断使用 Numba、NetworkX 和 joblib。首轮实验之外的代码使用过 CPU 版本，已检查的本地主要版本为 Python 3.13.5、NumPy 2.1.3、SciPy 1.15.3、scikit-learn 1.6.1、PyTorch 2.7.1。Enron 原始 R 数据解析使用 `rdata==1.1.0`。单快照 BOSouL 适配还需要 BoTorch 和 GPyTorch。

| 阶段 | 从哪里开始 |
| --- | --- |
| 数据生成、候选检索和正向回放 | 先跑 Dept3 首轮，再按[候选生成](../runs/dept3_candidates/候选生成实验报告.md)、[组合排序](../runs/dept3_pair_ranker/双源集合排序实验报告.md)、[正向评分](../runs/dept3_forward/隐藏前缀与正向评分实验报告.md)和 [CollegeMsg 复核](../runs/collegemsg_replication/CollegeMsg跨网络复核报告.md)的顺序生成前置制品。 |
| 模型、快照和跨网评测 | 依次查看[制造企业跨网评测](../runs/masked_pair_benchmark/制造企业跨机构双源实验.md)、[时序模型](../runs/temporal_pair_evidence/方法与投稿判断.md)、[双源组合训练](../runs/distilled_source_pairs/双源组合学习与预算实验.md)和 [FGSSI 与 Enron](../runs/temporal_transfer/时序公开方法与独立网络验证.md)的复现命令。 |
| 历史特征、先验与上限 | 用前一阶段模型和观测，按[实验底稿](../论文实验结果与写作底稿.md)第 12–18 节的顺序冻结协议、生成公开后缀、训练和评测。留出标签或完整轨迹由校准、评测或诊断入口按各自协议读取。 |
| 真实讨论的最近一轮 | 先依据[实验底稿](../论文实验结果与写作底稿.md)第 15–18 节及[实验索引](experiments.md)建立历史输入，再按[实验收尾记录](../runs/heldout_r_lists/early_only/role_evidence/closure/实验收尾记录.md)运行冻结、下载、预测和独立核验。 |

FGSSI 适配从 `runs/temporal_transfer/fgssi_upstream/` 载入固定提交中的 `model.py`、`args.py`，脚本检查文件 SHA-256。GCNSI 对照还需要 `torch-geometric`，按来源记录下载固定实现并保存为 `runs/temporal_pair_evidence/graphsl_gcnsi_model.py`。运行 `masked_bosoul.py` 时，第二个参数为 BOSouL 固定提交的本地检出目录。获取上游文件时保留原始行尾以通过校验。[FGSSI 固定提交](https://github.com/cgao-comp/FGSSI/tree/cb3cbaf3a14fe5fa9cd488ffce203c674eb42edc) · [GCNSI 来源记录](../runs/temporal_pair_evidence/gcnsi_source.json) · [BOSouL 固定提交](https://github.com/XGraph-Team/BOSouL/tree/e891f84f8d5e1c829b511672da7d946aeba06def)

仓库归档了自编脚本、实验报告、汇总、协议和输入指纹。原始通信数据、逐邮件或逐事件记录、二进制模型权重及第三方代码从数据源取得或按各阶段生成。报告标为「本地制品」的路径对应原工作目录或重跑输出。后期的冻结脚本有时要求历史制品的字节级指纹一致，可使用原版本环境或从已有前序制品启动。这次仓库整理重跑了 Dept3 首轮流程，其余实验引用原有统计和复核记录。
