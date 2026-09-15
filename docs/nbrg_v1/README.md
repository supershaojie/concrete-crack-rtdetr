# NBR-G v1 实现与验证

本次交付实现、受控初始化、有限预检和独立实验分支；**没有启动正式训练，没有执行最终 test，也没有精度提升结论**。

## 代码基点与边界

- 分支：`exp-rtdetr-r18-lite-nbrg-v1`。
- 基点：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。经本地历史及成功组合的 `metadata/source` 核验，包含 original CBR、LIF-Down 及后续 fixture/replay、cutoff tie 判定修复。选择依据是源码和修复继承关系，不是把 launch initialization 的 commit 当作最终训练 commit。
- 隔离工作树：`D:/MyProjects/Crack_RTDETR/outputs/worktrees/nbrg-v1`。原主工作目录仍在 CSCEF-v4，用户的未提交文件未移动或删除。
- `BasicBlock`、`ConvNormLayer`、CBR、LIF、head、transformer、criterion 和原预检数值判定源码保持不变。实际源码差异和继承记录见 `base_audit.json`。
- 未分配新的 C 编号；没有加入其他注意力、卷积变体、训练阶段或完整八组消融。

## 三个配置

YAML 均位于 `ultralytics-main/ultralytics/cfg/models/rt-detr/`。

| variant | YAML | 用途 |
|---|---|---|
| `cbr_lif_nbr_control_v1` | `rtdetr-resnet18-lite-cbr-lif-nbr-control-v1.yaml` | 首轮纯 NBR 压缩对照，无 gate 参数 |
| `cbr_lif_nbrg_v1` | `rtdetr-resnet18-lite-cbr-lif-nbrg-v1.yaml` | 首轮 NBR-G 候选 |
| `nbrg_v1` | `rtdetr-resnet18-lite-nbrg-v1.yaml` | 第三模块独立消融入口 |

只替换 `model.6.blocks.1` 和 `model.7.blocks.1`：S4 为 256→192→256，S5 为 512→256→512。两处 `blocks.0` 的 stride=2 主分支、AvgPool+1×1 shortcut 原样保留。27 个顶层节点、S3/S4/S5 接口以及 Decoder 的 `[19,22,25]` 输入不变。

窄块仍是两个 dense 3×3、BN、原位置 ReLU 和恒等 shortcut。新模块在 `ultralytics/nn/modules/nbr_g.py`，注册仅改 `modules/__init__.py` 和 parser 对 `NBRStage` 的识别。

## 门控定义与参考来源

X、F 按原连续通道划分为 8 组。每组沿通道计算 `mean(abs(X))`、`amax(abs(X))`、`mean(abs(F))`、`amax(abs(F))`；通过 `stack(dim=2).reshape(B,32,H,W)` 按组交错排列。`Conv2d(32,8,3,padding=1,groups=8,bias=True)` 后使用 `1+0.5*tanh(logits)`，只乘相应组的残差，再与 X 相加并 ReLU。

每处 gate 296 参数，合计 592。卷积和偏置仅在构造时清零；加载已学习 gate、原生模型重建和模型 EMA 均保留其值。没有 detach、输入原地修改、可学习 alpha 或精度补丁。

实际阅读的模块包为 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip`，虽已改名，其 SHA256 为文档指定的 `b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc`。已核对包内源码与解压源码相同，阅读 `Partial_conv3`、`Faster_Block_Rep_EMA`、其父类和 `EMA`，并对照 `rtdetr-r18.yaml`。逐文件哈希见 `reference_audit.json`。

参考模块使用部分通道 RepConv、pointwise MLP、DropPath 和 EMA。参考 EMA 将组折入 batch，组间共享卷积，并含 H/W 池化、全局池化、GroupNorm、softmax 和 sigmoid。NBR-G 保留两层普通 3×3，使用独立分组空间卷积和双输入描述，不导入模块包的 `extra_modules`。

这是输入引导的分组残差空间校准。描述图不是裂缝语义标签，门控不保证恢复被删除的信息；零 gate 等价于同初值 pure NBR，不等价于原宽块。该实现不确认学术新颖性。

## 受控初始化

源权重实际路径：`D:/rtdetr跑结果/c2 200e在线/c2_rtdetr_r18_lite_e200_b16_onlineaug_20260830_215053/weights/rtdetr_r18_lite_imagenet_backbone_init.pt`。

SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。来源为项目 torchvision ResNet18 IMAGENET1K_V1 layer1–4 迁移；三层 3×3 stem 未移植原 7×7 stem。未使用任何已训练 best/last 或替代 COCO 权重。

1. 复用 `init_c19_lif_v1.controlled_models` 构建共同 nc=80 C2/CBR+LIF 参考状态，校验 original CBR/LIF 公共构造值。
2. 仅从固定源 checkpoint 参数、BN running statistics 和真实 eps，在 CPU float64 按文档公式评分。并列按原索引升序；top-m 后再按索引升序保存。正 beta 的常量通道不会被错误排除。
3. 同步切片 W1 输出、BN1 向量、W2 输入；BN1 计数器、完整 BN2 和其余状态逐值复制，保留 BN 属性。完整字典 `strict=True` 加载，未知缺失/多余/形状变化报错。
4. `NBRStage` 先消耗原完整宽 stage 的构造 RNG，再在局部 RNG 作用域内替换第二块。后续 Decoder、CBR、LIF 和类别头保留原构造随机数顺序。
5. 保存 nc=80 初值，继续使用原 Trainer 的 nc=1 适配时机。并行参考构造在隔离 RNG 中核验；仅放行 9 个分类张量，逐值比较原生 C2/CBR+LIF nc=1 初值。两组组合的 **552 个公共参数及 BN buffers** 在最终 nc=1 模型中相同。
6. 实际 `RTDETR.train` API 调用受控 Trainer 的 `get_model`，在优化器前校验，再检查真实 AdamW 参数覆盖。模型均在加载后创建优化器。正式启动重新读取未更新的 clean checkpoint 并按原 seed=42 初始化流程运行。

`channel_selection.json` 保存两组完整 score、真实索引及算法版本；`initialization_audit.json` 保存切片/公共状态哈希摘要和初值 checkpoint 路径，逐张量明细在其引用的无损 `.full.json.gz`。CPU/CUDA 报告同样保留可读摘要与完整压缩 JSON，避免把候选 ID 数组重复展开到代码 diff。权重位于忽略目录 `outputs/nbrg_v1_initial/`，不进入 Git。JSON 中初始产物的 raw SHA256 指创建时的真实文件；跨平台源码/交付文件另以 LF 规范化哈希审计。

## 验证与数值边界

详见 `preflight.json` 和 `evidence/`。本机是 Python 3.9.25、PyTorch 2.7.1+cu118、RTX 2060 6 GiB；不是历史服务器的 Python 3.10.13 / PyTorch 2.1.2+cu121。

- CPU：三种结构、640 输出接口、组交错与广播、零 gate 等价、输入不变、梯度、评分边界、严格初始化、真实 Trainer API/优化器、零及非零 gate 新进程重载和模型 EMA。
- 两组组合：原 criterion 前后向、原生融合和显式顺序 Conv+BN 融合。融合副本激活了 CBR/LIF、非平凡 BN 和非零 gate。
- CUDA：真实训练图片的小 batch AMP 前后向；组合的 FP32、AMP、true-half 原生融合。沿用原 fixture、候选 ID 对齐和固定查询重放；没有改变 topk 或放宽容差。
- B16/640/AMP：在独立对象上按真实检测 batch 运行。具体每组结果、显存、图像/标签哈希见容量报告。这是有限工程检查，未执行正式训练 epoch，也未证明长期训练收敛或服务器等价。
- 本地 train/val/test 划分路径与标签指纹和原成功组合一致。未用 val/test 图片评分或选通道，未执行全量 val/test。

真实预训练权重的 masked-wide FP32 对照出现少量超过建议 `rtol=1e-5, atol=1e-6` 的元素；S5 包含明显抵消，dense reduction 的宽度改变带来不同累加顺序。保留原 FP32 错差记录，用同一实际源权重/输入的 FP64 对照确认至 `rtol=1e-10, atol=1e-11`，另以受控正值权重验证两种实际通道尺寸的 FP32 masked-wide 关系。全宽回归与 NBR/零 gate FP32 等价仍使用原建议容差。没有改变训练模型精度或全局容差。历史 fusion_fix 中未证明的根因仍不作已证明表述。

## 实测复杂度

全部为 nc=1、640、batch1；THOP 2.0.18 实际整网前向，乘加各计一次。动态归约、abs、stack、tanh、广播以及部分 functional attention/grid_sample 未被完整计入，不能据此推导 FPS；本次未测速。

| 结构 | 未融合参数 | 原生融合参数 | 所有顺序 ConvNormLayer 另行融合 | THOP GFLOPs |
|---|---:|---:|---:|---:|
| 原 CBR+LIF | 20,149,765 | 19,944,965 | 19,940,037 | 58.6725120 |
| CBR+LIF+pure NBR | 17,494,917 | 17,290,117 | 17,285,509 | 55.8397184 |
| CBR+LIF+NBR-G | 17,495,509 | 17,290,709 | 17,286,101 | 55.8408704 |
| 原基线+NBR-G | 17,428,516 | 17,223,460 | 17,218,852 | 55.4449664 |

提示词的三个窄模型融合值各比当前原生计数高 320：原生 `BaseModel.fuse()` 保留全部 `ConvNormLayer.norm`，因此窄化参数差仍为 2,654,848。若在两边一致地额外融合这些顺序 BN，pure NBR 减少 2,654,528，与理论差值一致；此时原组合基准也应减少 4,928，而窄模型减少 4,608。不能将原生基准与额外融合的窄化差值混用。此任务未修改原生融合行为，LIF 共同 BN 始终保留，动态 gate 始终保留。

两处普通卷积少 1.4155776 GMAC；gate 卷积加 0.000576 GMAC。THOP 差值还包含其支持的 BN 运算；详见 `complexity.json`。

## 配方、交付与后续

权威 C2 全字段快照在 `authoritative_c2_args.yaml`；两组 `plan` 命令已执行，差异仅为模型/数据环境路径和输出身份。正式配方仍为 200 epochs、patience=50、batch16、640、AdamW、lr0=0.0005、AMP=True、seed42，以及原在线增强。实际正式 Trainer args 将在 `on_train_start` 再次逐字段核验并落盘，当前状态为 `PENDING_FORMAL_START`。

服务器操作见 [SERVER_HANDOFF.md](SERVER_HANDOFF.md)。`start` 必须同时收到 CPU、CUDA、B16 通过报告；报告绑定源权重、初值文件、源码、数据指纹和 CUDA 环境。服务器应生成本机自己的初值和预检报告。所有预检更新对象均与正式初值隔离。

最终完整 HEAD、remote、push 结果见交付目录的 `delivery.json` 和 `DELIVERY_SHA.txt`，避免在某个 commit 内自引用该 commit 的 SHA。正式训练、最终 test、metadata 分目录；后续正常 patience=50 早停可以完成，必须记录实际 epoch 和原因；崩溃/中断不会标为完成。最终 test 和 PR/混淆矩阵等留待后续指令。

最终目标仍是相对原 CBR+LIF 的精度与复杂度权衡：如果 P、R、mAP50、mAP50:95 全面下降，该候选未达目标。当前只有实现证据，不使用历史 test 指标作为训练期门槛。
