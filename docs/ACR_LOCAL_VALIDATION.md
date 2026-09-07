# ACR-SA 本地验证报告

2026-09-08，Windows、Python 3.9.25、PyTorch 2.7.1+cu118、RTX2060 6GiB、仓库 Ultralytics 8.4.21。**没有启动正式训练；没有在 AutoDL 执行命令。**

完整结构化结果见 [acr_local_validation.json](acr_local_validation.json)，全部逐键/数值证据压缩在 [acr_evidence](acr_evidence/inventory.json)。两个最终审计的源码规范化 SHA256 清单与交付代码逐文件相等。报告生成于提交前，其中 git parent 是 C17；服务器必须在交付提交上重跑，不能拿本机报告启动。

## 最终通过结果

| 项目 | C22 C2＋ACR | C23 C17＋ACR |
|---|---|---|
| nc=1 整网参数 | 20,084,620 | 20,111,532 |
| 相对各自原模型增量 | 1,848 | 1,848 |
| 公共 C2 源状态逐键映射 | 533/533 精确 | 533/533 精确；CSCEF 原状态保留 |
| 新 ACR 状态 | 12，三层参数独立 | 12，三层参数独立 |
| 实际 self-attention | 三层 CoverageRelationSelfAttention | 三层 CoverageRelationSelfAttention |
| 原 cross-attention | 三层 MSDeformAttn | 三层 MSDeformAttn |
| Decoder 输入 | [19,22,25] | [20,23,26] |
| 原生 AMP helper | passed（非 skipped） | passed（非 skipped） |
| CPU/CUDA FP32、CUDA AMP、实际 half | 640 和 320×640 初始整网输出通过 | 同左 |
| 受控训练前向 | batch2、640、含 DN 通过 | 同左 |
| nc=1/80 实际 train API 重建 | 公共状态、分类状态、RNG 一致 | 同左 |
| 11 项回归测试 | 6 项机制＋5 项工具通过 | 同左 |
| 真实训练 smoke | 3 个真实增强 batch2 更新 | 同左，超出组合最少1批要求 |
| 初始真实 loss/DN/匹配 | 与 C2 对齐，4次 matcher 调用 | 与受控 C17 对齐，4次 matcher 调用 |
| checkpoint | 模型、optimizer 重载精确；half 有限 | 同左 |
| 真实末批 | 单图前反向通过，无额外 optimizer 更新 | 同左 |
| 正式初始化/EMA 隔离 | 初始化 hash 未变，初始 optimizer 空、EMA updates=0 | 同左 |

整网、loss、DN/匹配、checkpoint/optimizer 比较的全部记录分别为 1,116 和 1,131 个张量，**全部逐位相等，最大绝对误差 0**。FP32 预设 `atol=1e-6, rtol=1e-5` 未放宽。

专项测试覆盖：相同/包含/横向与纵向分离/近零宽高框、独立标量公式核对十维顺序、方向性、对角零偏置、平移/尺度适用条件、bool/float 及 B/H 展开 mask、padding、非零偏置下 regular→DN 始终禁止、dropout 前归一化、CPU/CUDA RNG、dropout RNG 对齐、几何 detach、两步学习、原参考框/特征梯度保持。

第一步全部 ACR 输出头 weight 梯度有限非零，上游几何投影梯度为零；三次更新后所有新增参数张量均有有限非零梯度。新增 weight/bias 各6个张量，weight decay 分别为 .0001/0，公共 optimizer 组不变，无缺失或重复。C23 的 CSCEF 五个权重仍在原组且可训练；未冻结。

真实数据是从原本地裂缝数据集复制的32张训练图和4张 val/4张 test，逐个校验 SHA256，所有缓存写入新 worktree 的 `outputs/local_data`。来源、文件清单和 hash 在证据中，图片/标签本身未提交。smoke 用原生在线增强和损失，只将 batch/workers/device/output/val/plots/save 等改为独立诊断设置，完整差异表保存在 smoke 报告。正式109字段配方仅改 `model/name/save_dir`。

| 真实 smoke | 三批 loss | DN splits |
|---|---|---|
| C22 | 63.05761337, 36.92552567, 35.99574661 | [200,300], [192,300], [198,300] |
| C23 | 63.05761337, 36.93144608, 36.00249863 | [200,300], [192,300], [198,300] |

这些是短 smoke 的数值，不是收敛或收益证据。正式 scaler 沿用 C2；smoke-only 初始 scale128，审计确认每个有状态参数均实际进行了3次更新，没有把溢出跳步当成训练成功。

## 资源与工具

未融合、FP32、batch1、640随机输入，预热3次、10次计时的一次探测：

| 比较 | 原模型 ms/image | ACR ms/image | 原模型峰值 bytes | ACR 峰值 bytes |
|---|---:|---:|---:|---:|
| C2→C22 | 42.38 | 59.04 | 187,776,512 | 187,787,264 |
| C17→C23 | 38.89 | 36.04 | 191,655,936 | 193,685,504 |

短测受本机负载、时钟和顺序影响，两组时间波动明显，**不据此宣称速度提升或确定开销比例**。峰值是整网最高分配，不代表 ACR 单独的工作区占用；原特征路径可能主导峰值。真实 smoke 各步耗时和峰值另见 JSON；首步额外执行 FP32 backward，不能作为常规训练速度。服务器 prepare 必须记录其自己的 batch16 smoke 和640资源；当前没有 AutoDL 显存/速度证据，也不报告测得 GFLOPs。

两组 smoke 权重均通过各4图的真实 val/test 流程，输出路径、权重 hash、指标和数值预测/GT 均校验。此4图工具 smoke 不是正式实验 val/test，不参与结构、上限或模型选择；相应 mAP 无研究意义。排序 mask 问题在这几张低步数 smoke 预测上未触发，专门构造的排序测试确实触发并验证修正。历史 C2/C17 完整 val 是否受影响尚待统一重评。

结果包测试使用明确的单元测试 fixture，验证文件清单/hash、≤20MiB、排除权重和原图、只允许训练曲线 PNG、防覆盖和未运行 test 标记；**尚无正式训练结果包**。Bash语法、Python编译、配方锁、重复启动锁、子进程退出码7保留测试通过。AutoDL 的 conda/tmux/完整 prepare/start 尚未在服务器运行。

## 开发中已解决的问题与剩余边界

- 首次 AMP helper 不在新 worktree；从可信本地副本补齐 bus.jpg/helper 后，两份最终报告均为原生 passed，未关闭 AMP。
- 隔离的基线 loss fixture 缺少真实 trainer 设置的 `nc` 属性，已按 head.nc 补齐该 fixture；不改损失。
- optimizer step 标量重载时可在 CPU，诊断改为双方都复制到 CPU 后做精确比较，未改变 optimizer 计算。
- C23 的启动锁单元 fixture 需要隔离组合评审依赖；生产组合启动仍检查完整 val 和用户明确排期。
- 原 CUDA grid_sample backward 有 `deterministic warn_only` 提示，沿用 C2；初始前向逐位一致不意味着未来正式训练轨迹完全可重复。
- AutoDL PyTorch 2.1.2、原数据划分、batch16、正式200轮、完整同口径 C2/C17/C22/C23 val、最终锁定后 test 仍待服务器运行。没有任何精度收益结论。

复现入口和输出位置见 [ACR_AUTODL.md](ACR_AUTODL.md)。
