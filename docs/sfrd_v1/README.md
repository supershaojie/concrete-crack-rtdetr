# SFR-D v1

本候选只替换 `model.6.blocks.1.branch2b` 和 `model.7.blocks.1.branch2b`。S4 的 C/r 为 256/256，S5 为 512/384；第一层完整 C→C 3×3、两个原 BN 的状态、identity shortcut、各阶段首块和所有 Neck 引用保持原义。主候选为 `cbr_lif_sfrd_v1`；`cbr_lif_sfr_control_v1` 是无 gate 参数的纯 SFR 控制；`sfrd_v1` 是不含 CBR/LIF 的独立消融。没有占用新的 C 编号，没有叠加 NBR-G。

## 唯一前向

```text
H1 = original_branch2a(X)
Z = Conv1x3[C→r](H1)
G = 1 + 0.5*tanh(DW1x5(Z) + DW5x1(Z))
Y = ReLU(X + original_BN2(Conv3x1[r→C](Z*G)))
```

A/B 是无 bias 的普通跨通道卷积，只有 gate 是逐通道卷积。DW1x5 无 bias，DW5x1 有 bias；其 3 个参数张量仅在构造时清零，两处共 7040 参数。没有额外 BN/激活、可学习 alpha、通道选择、池化、detach 或额外损失。纯 SFR 直接把 Z 送入 B，模块和 state_dict 中均不存在 gate。

SVD 和 gate 零初始化仅用于从原始 ImageNet 初值首次构建候选。load、Trainer 重建、EMA、推理和 fuse 不重算 SVD、不重置门控。新增模块在原阶段完成构造之后，以局部 RNG scope 创建，因此后续公共层及原生 80→1 分类头的随机数消费不变。

## 来源与隔离

独立工作树为 `D:/MyProjects/Crack_RTDETR/outputs/worktrees/Crack_RTDETR-sfrd-v1`，分支为 `exp-rtdetr-r18-lite-sfrd-v1`。基点 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 经真实历史核对，包含 `da600f3` 原 CBR＋LIF 实现、`216bba0` 融合诊断修复、`8bdc14f` 截止位 tie 证据修复和跨平台审计哈希修复。该 SHA 只作为源码基点，不宣称是最终训练/test版本。

父工具按此基点最小复用：受控公共参考、数据指纹、原生优化器、CBR/LIF probe 和候选感知 fuse capture/replay。`lif_down.py`、`cbr.py`、criterion、Decoder/topk/gather、原有实验脚本与 YAML 均不修改。NBR-G 工作树不作为本实现来源。

指定模块包实际位于 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip`，SHA256 与指定 `1c120da7-...zip` 一致。参考审计见 [reference_audit.json](reference_audit.json)。尤其 `Strip_Attention.forward` 中 `spatial_gating_unit` 的调用确实被注释，不能把类名当执行证据。没有整包导入或覆盖 nn 目录。

空间低秩卷积已有研究：[Jaderberg et al.](https://arxiv.org/abs/1405.3866)。PConv/FasterNet 仅处理部分通道，与此处保留完整第一层、分解第二层不同：[FasterNet](https://arxiv.org/abs/2303.03667)。用户给定的 [EMAFG-RTDETR](https://doi.org/10.3390/drones10010006) 作为参考出处保留，本轮 DOI 页面抓取失败，未声称复核全文。SVD、残差和条形卷积均非新概念；SFR-D 是待实验验证的具体组合，尚无学术原创性结论。

## 受控初始化

唯一原始权重 SHA256 为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`，epoch=-1、nc80、无训练状态。原工程把 torchvision ResNet18 ImageNet V1 的 layer1–4 迁移到骨干，未直接复制 7×7 stem 到三层 3×3 stem。

CPU FP64 展开固定为 `M[(o,h),(i,w)] = W[o,i,h,w]`，符号取 U 每列绝对值最大元素为正。固定 r256/384；完整秩为 3C。实际 FP32 因子缓存供三配置复用，文件与逐张量哈希均校验，源 SHA/r/算法不符或已有输出时拒绝覆盖。因子不入 git，谱、阈值、重建和 BN2 加权误差见 [svd_audit.json](svd_audit.json)。

| 位置 | 谱能量保留率 | FP64 核相对误差 | BN2 加权相对误差 |
|---|---:|---:|---:|
| S4 | 0.9192342617 | 0.2841931355 | 0.2646402007 |
| S5 | 0.9703677628 | 0.1721401673 | 0.1721640276 |

这是核矩阵 Frobenius 误差，不是裂缝特征、检测损失或语义重要性最优；不据此改变 r。初始化逐键和 BN 属性审计见 [initialization_audit.json](initialization_audit.json)。正式 nc1 由原生 Trainer 构造器产生，再严格加载完整映射，仅 9 个原类别适配键允许改变。

## 预检与融合

可执行入口：`tools/init_sfrd_v1.py`、`tools/preflight_sfrd_v1.py`、`tools/train_sfrd_v1.py`。后者区分 `plan` 和显式 `train`；初始化/预检不会派发正式训练。每个预检使用新目录和独立模型，所有预检 optimizer_steps=0，不回写正式初值。

覆盖：FP64 完整秩/截断核映射、非方形与边界、多通道；零 gate 与纯 SFR 一致；非零 gate 的显式公式、空间变化、真实路径梯度；真实源迁移、三变体公共初值；新 Python 进程重载和非零 gate 重载/恢复；原生 Trainer setup、优化器与 EMA、含有效目标的真实 batch/非空匹配；真实整网 fuse。

B＋BN2 由 PyTorch eval fusion 折叠，A/gate 保留；LIF 原共同 BN 保护不变。基点的整网 fuse 不处理原 `ConvNormLayer`，这里也不全局改变旧实验行为。新 B＋BN2 单独融合比提示词的融合参数估计再少 C4+C5=768 个参数；按实际实现记录，不为对齐表格隐藏差异。

大通道真实块的 FP32 BN 融合存在累加/抵消误差，初始 `rtol=1e-5, atol=1e-6` 检查并非全部通过。额外 FP64 折叠恒等证明及基于 `gamma_(3r)` 的逐点浮点舍入界定位原因，保留原始失败记录。整网继续使用父项目 FP32 `atol=2e-5, rtol=2e-4` 和原候选选择/重放判定，不改变 topk、不跳过诊断、不扩大整网容差。原 fusion_fix 中尚未证明的历史根因仍保持 NOT_PROVEN。

真实结果与缺项以 [preflight.json](preflight.json)、[complexity.json](complexity.json) 和 [SERVER_HANDOFF.md](SERVER_HANDOFF.md) 为准。THOP 采用实际 640 输入、2 FLOPs/MAC，不包含 tanh、逐元素 gate、部分 functional attention/grid sampling 等未注册算子。延迟/FPS未测。

本机最终预检为 PASSED：CPU 与 CUDA FP32 整网初始/非零 gate 融合、CUDA FP32/AMP 检测反向、640/batch16/AMP 单批均通过，optimizer_steps=0。B16 的实际 CUDA peak allocation 为 6,532,433,408 bytes；Windows WDDM 分配行为不能证明 Linux 服务器或两个同时训练任务的容量。没有宣称真 FP16 或 AMP 模式融合前后完整等价，只核验了融合后 AMP 推理有限。

| nc1 配置 | 未融合参数 | 融合参数 | 未融合 GFLOPs | 融合 GFLOPs |
|---|---:|---:|---:|---:|
| 原 CBR＋LIF | 20,149,765 | 19,944,965 | 58.672512 | 57.564954 |
| CBR＋LIF＋纯 SFR | 18,773,509 | 18,567,941 | 57.099648 | 55.987174 |
| CBR＋LIF＋SFR-D | 18,780,549 | 18,574,981 | 57.110912 | 55.998438 |
| 基线＋SFR-D | 18,713,556 | 18,507,732 | 56.715008 | 55.599258 |

同口径未融合 GFLOPs 净减 1.5616，与卷积解析估计一致。融合后还消除了两个 BN2 的计算，不能与未融合58.7G混为同一口径。

## 训练与评估边界

固定 200 epochs、patience50、batch16、640、AdamW lr0=0.0005、AMP、seed42、原在线增强。完整109字段对比见 [recipe_diff.json](recipe_diff.json)。预检的小 batch/CPU fixture 仅用于工程检查，不能代替正式 CUDA B16；训练入口要求该容量证据、代码/初值/数据指纹均匹配，否则拒绝启动。pretrained=true 保留，实际模型源是受控本地 pt，不再下载替代检测权重。

另外读取了用户最新 CBR＋LIF 完整结果包中的实际训练 args、authoritative C2 args 和 launch runtime，与成员清单哈希核对；109字段与本基点配方一致，原组合差异仅 model/name/save_dir。证据见 [parent_recipe_evidence.json](parent_recipe_evidence.json)。

首轮槽位 A 由另一任务负责 CBR＋LIF＋NBR-G；槽位 B 为本主候选。纯 SFR 控制和单模块消融后续补跑。无正式训练、完整 val/test、性能提升或 FPS 结论。历史 test 仅在最终协议下比较，不能选择 r、gate 或 checkpoint。如果四项 P/R/mAP50/mAP50:95 全降，不能仅凭减参认定成功。
