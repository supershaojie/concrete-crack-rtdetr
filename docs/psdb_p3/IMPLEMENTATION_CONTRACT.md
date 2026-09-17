# PSDB-P3 v1：Codex 完整实施提示词

版本：2026-09-17。本文是完整实施合同，可独立交给 Codex 执行。PSDB 是 SDB 路线的第二个实现；为保持实验身份清晰，代码命名使用 psdb_p3_v1。

## 0. 任务与执行范围

请在用户的混凝土裂缝 RT-DETR 项目中，完成 PSDB-P3 实现、受控初始化、本机可完成的检查、独立分支提交和推送核验，交付实际可执行的服务器命令与轻量交接材料。不要停留在建议、示例或待办清单。

主实验是“原 R18-Lite＋原 CBR＋原 LIF-Down＋PSDB-P3”。另外准备 C2＋PSDB 的单模块消融配置。首轮服务器命令默认仅运行主组合；单模块供后续消融，不同时派发两项正式训练。

本次实施允许创建独立 worktree、修改本实验代码、必要修复、提交及普通 push；不强推、不合并成功母版、不删除旧实验、不改用户未提交工作。不要登录服务器、停止已有训练、启动正式 200 轮训练或执行最终 test。服务器预检和正式启动由用户按交付命令执行。本机缺 CUDA、真实数据或统一源权重时，完成其他工作并明确标记对应 PENDING，不伪造 PASSED。

已有 SDB-P3 正在运行，必须保留其代码、环境、输出和进程。PSDB 是第三模块的替代候选，不是“CBR＋LIF＋SDB＋PSDB”四模块。不要载入任何已训练 best.pt/last.pt 来初始化本实验。原骨干、CBR、LIF、AIFI、Decoder、损失和训练配方保持。

本次目标是验证收益，不承诺涨点，不把新名字或工程检查通过当作论文新颖性与精度证据。

## 1. 只读核对项目与来源

### 1.1 项目和固定母版

先读取适用的 AGENTS.md，检查 git status、remote、HEAD、worktree 和现有初始化/训练工具。目标是 concrete-crack-rtdetr；已知远端为 https://github.com/supershaojie/concrete-crack-rtdetr，使用前与实际 remote 核对，不能进入 YOLO26 项目。

- 固定成功母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
- 主组合母版配置：`rtdetr-resnet18-lite-cbr-lif-down.yaml`。
- 单模块消融母版配置：`rtdetr-resnet18-lite.yaml`，使用原 C2 下采样和 Decoder，不含 CBR/LIF。
- 新分支：`exp-rtdetr-r18-lite-psdb-p3-v1`，直接从上述固定母版派生。
- 建议服务器 worktree：`/root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1`。

同名分支/目录存在时核对身份并继续未完成工作，不 reset 或覆盖。原模块 LF 归一化 SHA256（CRLF 转 LF 后计算）：

| 文件 | SHA256 |
|---|---|
| lif_down.py | 26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7 |
| cbr.py | d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787 |

不要修改这两个模块来适配 PSDB。基点/源码不符时先查明，不换成其他同名版本。

### 1.2 SDB 参考实现与模块包

已交付的 SDB 参考分支为 `exp-rtdetr-r18-lite-sdb-p3-v1`，提交为 `f976779bcdb114178f83a3d2fa959baf6d9655ec`。可用 git show 或独立只读路径检查其 `sdb_p3.py`、解析器和生命周期工具，不切换/修改正在使用的 SDB worktree，也不把整个实验分支合入新分支。复用经过审查的代码时记录来源与本次修改。

模块包：
- 文件名：`dc2eda42-6ea1-467d-ab90-02d26882d4d0.zip`。
- SHA256：`b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc`。
- ChatGPT 侧历史路径：`/workspace/scratch/00484fe92169/upload/dc2eda42-6ea1-467d-ab90-02d26882d4d0.zip`，不保证在用户 Windows 主机存在。
- 包内参考：`RTDETR-main/ultralytics/nn/extra_modules/block.py` 的 `SPDConv`；按定义位置查阅包内 `CARAFE`，仅参考内容相关权重思想，不搬入整个算子。

有界查找附件/项目已有模块包，记录实际读取路径和哈希。参考包或 SDB 提交不可见时如实记录；本文数学合同足以继续实施，不因缺参考材料伪称已读取或改用其他结构。不要执行参考包安装/训练脚本，不整体替换 ultralytics，不将大包提交 Git。

理论参考：[SPD-Conv](https://arxiv.org/abs/2208.03641)、[CARAFE++](https://arxiv.org/abs/2012.04733)。它们分别提供空间重排、内容相关重组的先例，不是本模型涨点证明。

### 1.3 研究定位

原 SDB 已经有压缩后的 P3 语义门控。PSDB 唯一新增机制是：压缩前，由未修改的 P3 引导四相位有界重加权，然后保留各相位身份继续拼接。

待验证假设是：先静态压缩再门控，可能削弱细裂缝的局部位置证据；在压缩前增加温和选择，可能改善细节使用。不能宣称当前指标已经证明该因果关系。P3 对弱裂缝的语义遗漏也可能误导选择，权重下限只限制衰减，不保证收益。

## 2. 固定实验身份

| 项目 | 主组合 | 单模块消融 |
|---|---|---|
| variant | cbr_lif_psdb_p3_v1 | psdb_p3_v1 |
| YAML | rtdetr-resnet18-lite-cbr-lif-psdb-p3-v1.yaml | rtdetr-resnet18-lite-psdb-p3-v1.yaml |
| run name | cbr_lif_psdb_p3_v1_rtdetr_r18_lite_e200_b16_onlineaug | psdb_p3_v1_rtdetr_r18_lite_e200_b16_onlineaug |

- 新模块：`ultralytics-main/ultralytics/nn/modules/psdb_p3.py`。
- 核心类：`PSDBP3`；接线包装类：`PSDBRepC3`。
- 建议工具前缀：`init_psdb_p3.py`、`check_psdb_p3.py`、`train_psdb_p3.py`、`eval_psdb_p3.py`、`pack_psdb_p3_light.py`，具体 CLI 以真实实现为准。
- 元数据与预检目录：`outputs/psdb_p3/<variant>/`；文档目录：`docs/psdb_p3/`。
- 正式训练 project：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series`。

## 3. 精确数学与数值合同

### 3.1 输入和四相位

核心接口为 `PSDBP3.forward(p2, original_p3)`，返回修正后的完整 P3。

- `X2 = p2`：B×64×160×160（输入图640时）。
- `T3 = original_p3`：B×256×80×80，必须是原 RepC3 的完整输出。
- detail_channels=32；phase_channels=8。
- 两个 GroupNorm 固定 groups=4、channels=32、eps=1e-5、affine=True。

固定四相位顺序为 **00、10、01、11**，每相位64通道：

```python
S0 = X2[..., 0::2, 0::2]
S1 = X2[..., 1::2, 0::2]
S2 = X2[..., 0::2, 1::2]
S3 = X2[..., 1::2, 1::2]
```

必须按 phase-major 拼接。`pixel_unshuffle` 的默认通道次序未经重排不能直接替代。P2 奇数高/宽时，只在右/下各补最多1像素，replicate padding，然后分相位；T3高宽必须等于ceil(H2/2)、ceil(W2/2)。不匹配时明确报错，禁止用插值掩盖错误接线。

### 3.2 主公式

以下是数学定义用伪代码；实现要正确处理 dtype、autocast、half 参数和广播：

```python
Q = GN_Q(W_q(T3))                         # B,32,H3,W3
q = W_phi_q(Q)                           # B,8,H3,W3
keys = [W_phi_k(Si) for Si in phases]     # SAME W_phi_k for all four

# normalize over embedding channels, NOT over phases or spatial dimensions
q_hat = normalize_fp32(q, dim=1, eps=1e-6)
k_hat = [normalize_fp32(ki, dim=1, eps=1e-6) for ki in keys]
scores = stack([(q_hat * ki).sum(dim=1) for ki in k_hat], dim=1)
a = softmax_fp32(2.0 * scores, dim=1)     # B,4,H3,W3; four-phase softmax
w = 0.75 + a

S = cat([w[:, i:i+1] * phases[i] for i in range(4)], dim=1)
D = SiLU(GN_D(DW3(W_d(S))))               # B,32,H3,W3
g = sigmoid(W_g(cat([D, Q, D * Q], dim=1)))
delta = W_o(g * D)                        # B,256,H3,W3
T3_new = T3 + delta
```

固定层配置：

| 名称 | 配置 | 参数量 |
|---|---|---:|
| W_d | Conv2d(256,32,1,bias=False) | 8,192 |
| DW3 | Conv2d(32,32,3,padding=1,groups=32,bias=False) | 288 |
| GN_D、GN_Q | 两个 GroupNorm(4,32,eps=1e-5,affine=True) | 128 |
| W_q | Conv2d(256,32,1,bias=False) | 8,192 |
| W_g | Conv2d(96,32,1,bias=True) | 3,104 |
| W_o | Conv2d(32,256,1,bias=False) | 8,192 |
| W_phi_q | Conv2d(32,8,1,bias=False) | 256 |
| W_phi_k | 共享 Conv2d(64,8,1,bias=False) | 512 |
| 合计 | 相对各自父模型新增 | **28,864** |

所有卷积 stride=1；未注明 padding 的1×1卷积为0；没有额外相位 bias、可学习温度、零标量、位置编码、dropout、BN、辅助损失或新 attention 层。

必须满足：

- `W_phi_k` 是一份共享参数，不是4个独立卷积；q只计算一次。
- 每个位置、每个相位只有一个标量权重，广播到该相位全部64通道。
- softmax仅沿四相位维度。四组加权后仍拼接成256通道，禁止先求和/平均变成64通道。
- `0.75 <= w_i <= 1.75`，各位置四权重之和为4。均匀a=0.25时w=1，恢复原 SDB 拼接输入。权重和固定不等于特征能量不变。
- 不添加 grid_sample、偏移预测、曲线采样、Haar、边缘算子；不把本次设计替换为包中完整 CARAFE。
- Q来自原T3，不能来自已被PSDB修正的T3，避免循环依赖。

### 3.3 初始化与精度

仅 `W_o.weight` 全零；其余卷积使用固定的普通非零初始化（Xavier uniform），`W_g.bias=0`，两个GN的weight=1/bias=0。不将q/k零初始化，不叠加新的零门控系数。

在新增模块构造时隔离CPU RNG，保证后续原网络初始化不受影响；主组合与单模块的新支路初值逐项一致。若可读取固定SDB参考，实现中共用的W_d/DW3/GN/W_q/W_g/W_o尽量沿用其初始化顺序，并将新增q/k放入独立随机流，核对共有新层初值；无法核对则如实说明，不声称和历史SDB内部完全相同。

初始W_o=0时，有限输入下完整PSDB输出等于T3。q/k非零随机初始化通常不会产生均匀attention，因此不能宣称内部D初值等于SDB。只有在诊断中固定均匀权重时才可验证退化为原SDB。

q/k的相位投影、L2归一化、点积、softmax及权重形成使用局部FP32数值路径，显式禁用该区域autocast。相位加权结果回到主细节分支合适的dtype，最终delta与T3 dtype一致。能读取原SDB实现时保留其D/Q/g/W_o的既有精度边界，仅增加上述选择计算的FP32路径；不能为实现便利顺手改变原父网络精度。主网络继续原生CUDA AMP。

必须兼容显式 `model.half()`：不能直接用half参数的Conv接收FP32输入。必要时使用可微的 `F.conv2d(..., weight.float(), bias.float())` 或等价实现；若GN也进入FP32区域，其affine参数同样处理。不得在forward中调用module.float()、替换Parameter、detach投影或缓存与输入不匹配的keys。保持优化器、EMA、保存重载和梯度路径正常。

### 3.4 可核对的计算量

640输入时，可学习卷积分析值：
- 原SDB部分：178,790,400 MAC。
- 新q/k部分：(32×8 + 4×64×8)×80×80 = 14,745,600 MAC。
- 合计：193,536,000 MAC，即按2 FLOPs/MAC为 **0.387072 GFLOPs**。

这不是实测整网GFLOPs；L2归一化、GN、点积、softmax、激活及逐元素运算另计。不得把函数式FP32卷积漏计，也不得与模块hook重复计数。只统计已覆盖算子时明确写PARTIAL和遗漏范围。

## 4. 接回母版：只改变model.19的包装与输入来源

先以真实YAML核对以下图：model.4为P2(64通道)，model.18为融合Concat(512通道)，model.19为原RepC3(256输出、内部3次RepConv、e=0.5)，model.20为LIFDown，model.22/25为最终P4/P5，model.26为RTDETRDecoderCBR，from=[19,22,25]。

`PSDBRepC3`继承原RepC3，原`cv1/cv2/m/cv3`路径与计算不变：

```python
def forward(self, inputs):
    fusion_input, p2 = inputs
    original_p3 = super().forward(fusion_input)
    return self.psdb(p2, original_p3)
```

推荐YAML中的model.19为：

```yaml
- [[18, 4], 3, PSDBRepC3, [256, 0.5, 32, 8]]
```

建议构造接口：`PSDBRepC3(c1,c2,n=3,e=0.5,p2_channels=64,detail_channels=32,phase_channels=8)`。解析器专门处理双输入，按`ch[18]`和`ch[4]`传入对应通道；内部repeat只传一次，外部repeat设1。不能将输入列表直接用于ch[f]或重复串联双输入包装器。确保model.4加入save列表。

新增状态仅在`model.19.psdb.*`，不把原RepC3包进parent子模块导致共有key迁移。整网仍27个节点；model.20/22/25/26层号与from不变，修正P3同时供原LIF和原Decoder消费。单模块版同理，但保留C2原下采样与原Decoder。

不要修改model.17侧向卷积、AIFI、原ResNet层、head查询初始化、查询数300、hidden_dim256、3层decoder、eval_idx2、CBR rho/normal_fraction=0.10。没有额外P2检测头。

真实nc=1参数预期（需实测核对，不同融合口径不得混合）：

| 模型 | 未融合 | 融合后 |
|---|---:|---:|
| 原CBR＋LIF | 20,149,765 | 19,944,965 |
| CBR＋LIF＋PSDB | **20,178,629** | **19,973,829** |
| 原C2 | 20,082,772 | 19,877,716 |
| C2＋PSDB | **20,111,636** | **19,906,580** |

## 5. 受控初始化与真实Trainer加载

优先审查并复用基点`tools/init_c19_lif_v1.py`、`tools/init_lif_down.py`的受控初始化流程。

统一未训练源：
- `/root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt`。
- SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
- nc=80、epoch=-1，无已训练optimizer/EMA/scaler状态。
- 历史C2提交：`67c3078e54a657fd96d65fee657a75fbb1dae0d6`。

构建受控C2、CBR＋LIF父模型，再严格复制共有状态至对应PSDB模型。所有公共参数及buffers必须逐key、形状、数值核对，不仅看Transferred数量或seed相同。输出COMMON、NEW_TRAINABLE、NEW_BUFFER、MISSING、UNEXPECTED、SHAPE_MISMATCH和分类适配清单。

真实调用RTDETRTrainer.get_model/训练模型重建路径核验nc=1。nc=80→1时允许适配的9项是：denoising_class_embed.weight，enc_score_head.weight/bias，以及3层dec_score_head的weight/bias。它们仅是相对nc=80源的形状适配例外；匹配RNG后，父/目标的9项nc=1张量仍须逐值相等。父参考构造不能消耗目标分类适配的随机状态。

原CBR/LIF初值和新增PSDB初值都不得被Trainer重建覆盖。保存初值后立即重载审计；start只使用该初值；resume只从真实未完成训练checkpoint恢复已学状态。forward/load/EMA/fuse/resume不得重置PSDB、CBR或LIF。

## 6. 正式配方与数据

权威配方是本轮成功CBR＋LIF的 **200轮在线增强** 配方。旧150轮、lr0=0.01、关闭在线增强的记录不适用于本实验。完整快照见附录A；与项目成功母版实际args核对，存档并生成逐字段差异表。

关键固定值：epochs=200，patience=50，imgsz=640，batch=16，nbs=64，seed=42，workers=8，device=0；AdamW，lr0=0.0005，lrf=0.01，momentum=0.937，weight_decay=0.0001；warmup_epochs=5，warmup_momentum=0.8，warmup_bias_lr=0.1；cos_lr=True，deterministic=True，amp=True，cache=False。

在线增强：hsv_h=0.015，hsv_s=0.5，hsv_v=0.35，degrees=5，translate=0.1，scale=0.4，shear=1.5，perspective=0.0002，flipud=0.2，fliplr=0.5，mosaic=0.8，mixup=0.05，close_mosaic=10；其余字段继承附录/成功父记录。

仅模型标识、输出目录/name、经核实等价的数据绝对路径可变。损失、DN、matcher、学习率计划、查询配置、增强策略全部继承；不要依赖升级后的库默认值。保持exist_ok=False，已有run明确区分start/resume，不生成name2冒充同一实验、不删除旧目录重试。

数据配置：`/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml`；nc=1，当前train=6048、val=1728、test=864。核对实际划分与标签身份，数量只作辅助，不重划分或重新增强数据。读取test清单不等于执行test推理；训练和候选筛选不运行test。

## 7. 有限且有效的验证

所有工程预检使用一次性模型/优化器副本和独立临时目录，不覆盖受控初值、正式训练输出或既有实验。按能力区分本地/服务器，缺失环境标记PENDING。修复后只重跑受影响检查及依赖项，不反复扩展无关测试。

### 7.1 核心机制验收

1. 手工编号张量逐元素核查四相位顺序、矩形与奇数右/下padding；真实hook确认model.19确实接到model.4的P2，而非伪造张量。
2. 核对q/k通道、共享k、softmax维度、权重广播、相位拼接，w有限且在边界内，各位置sum(w)=4。
3. 均匀相位的诊断退化：在**W_o已非零**的一次性副本中固定a=0.25，以逐值相同的共有参数核对PSDB与本文原SDB公式的delta及最终输出一致，必要时逐层比较S/D/g定位差异。禁止用零W_o状态的输出等价替代退化验证。此诊断不能变成正式forward中的强制均匀、缓存权重或测试专用默认分支。
4. 在受控非退化输入上改变Q或某相位Si，确认a能够随输入改变；不要求随机网络已表现为真实前景选择。W_o非零时，对比正常相位选择与均匀相位表达式应能产生差异，避免新增q/k只被注册却没参与输出。
5. W_o=0的初始核心输出等于T3；共有状态一致时，整网相对各自父模型的初始输出/loss在相同精度与匹配RNG下核对，报告容差和实际误差。
6. 首次有效更新先检W_o梯度；上游新层首步零梯度是预期。W_o更新后，在随后有效反向中检查W_phi_q、共享W_phi_k、W_d、DW3、W_q、W_g等参与训练，梯度有限且在非退化样本上有非零范数。不能把GradScaler跳步算成有效更新，也不要求每个参数元素在每个batch均非零。
7. 用W_o非零的核心副本、固定T3和独立可求导P2检查delta对P2的梯度。P2在原骨干本就有梯度，单看整网P2.grad非零不足以证明旁路连通。

### 7.2 模型、训练与恢复

- 实际导入的ultralytics和新模块必须来自当前worktree。核对层索引、原模块源码、公共初值、参数量及两个变体新状态一致性。
- 核心奇数尺寸与整网stride对齐矩形分开检查；不强求父模型本不支持的奇数整图。
- CPU FP32、可用时CUDA FP32/native AMP使用真实检测loss和有效GT，覆盖DN路径。比较train/DN时同步CPU/CUDA RNG与BN状态，不把独立DN随机噪声误判为模型差异。
- 服务器使用真实训练数据、原在线增强、B16/640/native AMP做前后向和至少2次有效optimizer更新。记录GT/DN、loss、scale、跳步/有效更新、梯度和峰值显存。优化器须恰好覆盖每个可训练参数一次，包含q/k与GN参数。
- 非零分支状态下验证save/reload、EMA、原生resume（optimizer/scaler/epoch和已学PSDB均恢复）、整网fusion及显式CUDA half有限推理。GN不能当BN折叠，PSDB不能被fusion丢掉。
- 不要求CPU half整网作为CUDA训练的通过条件；若做了该诊断，如实标注能力限制，不能伪称所有精度均通过。
- 服务器正式start必须匹配当前执行提交、关键源码/配置哈希、初值/源权重哈希、数据、variant及PASSED服务器预检。不得手改旧报告status或commit来通过。

### 7.3 必须防住已出现的工程故障

**warmup有限输入：** 检查AutoBackend.warmup是否使用torch.empty；若会在本环境产生非有限输入，最小修复为torch.zeros并记录。保留数值异常检测，不用nan_to_num掩盖问题。

**AMP与half：** 保留原生GradScaler流程，缩放梯度Inf可能触发正常降scale/跳步，需结合后续有效更新判断。禁止用全局amp=False、固定scale=1或降低batch作为通过方案；局部FP32只能处理指定数值路径。复用本地AMP检查资源，资源缺失明确列出，不自动升级ultralytics或跳过AMP检查。

**FP32融合与TF32：** CSR已有证据显示TF32可让严格融合比较超容差。严格FP32融合诊断可在局部context关闭matmul/cuDNN TF32，记录原flags并用try/finally恢复；记录默认精度表现，不能据此改正式训练全局策略或单纯放大容差。融合比较用非零支路。

**RT-DETR top-k离散差异：** 低精度融合可能改变候选top-k，最终解码差异不能一概判为连续特征错误。记录父/目标默认top-k及连续特征，必要时仅诊断使用固定候选重放。不得把固定候选通过冒充原生输出等价，也不得修改正式候选选择绕过问题。NaN/Inf、明确的支路丢失和真实恢复错误必须失败。

**THOP计数：** total_ops兼容int/Tensor，不无条件访问.device/.new_tensor；共享k的四次调用应计4次计算但只计一份参数，函数式卷积不能漏计或重复计数。对父/目标使用同一nc=1/640与同一融合口径，解析估算与实测/部分计数分开。

**训练回调：** on_train_start不能假定trainer.epoch存在；按start_epoch/真实生命周期记录。训练完成轮数与final_eval状态分别保存，200轮后验证失败不应触发自动重训。tmux退出不等于成功；保留pipefail与真实退出码，明确RUNNING、COMPLETED_200、EARLY_STOPPED、INTERRUPTED、FAILED。

CUDA非确定性warning如实记录，沿用父实验deterministic策略。无需为消除warning换算子或改训练参数。

## 8. 生命周期、评估和轻量包

复用经过审查的生命周期工具，只扩展PSDB需要的接口。初始化/预检、plan、start、resume、独立val/test、pack应明确分离。每条交付CLI都要用实际--help或对应一次性检查核实，不能交付猜测参数。

主run默认200轮、原patience=50。中途同轮观察不自动改变学习率、总轮数或停止规则；用户决定是否止损。不要用中期截图与母版最终指标混比，也不要把提前停止作为完整200轮消融。

准备独立val/test工具，但本次不执行最终test。固定训练验证选出的同一个best.pt，记录其SHA256；先独立val，再对同一权重test，使用既有`corrected_sorted_conf_mask_v1`协议：imgsz=640、batch=16、workers=0、half=False、conf=0.001、iou=0.7、max_det=300、augment=False、rect=False、seed=42。审查原RTDETRValidator/修正排序后的置信度掩码实现，不加额外NMS或test调阈值。

输出P/R/AP50/AP75/mAP50–95、逐IoU AP、速度和完整精度JSON，保存PR/混淆矩阵及训练曲线。轻量结果包包含args/results、评估与初始化/预检/训练状态、关键日志尾、必要源码配置、提交和SHA256清单；排除数据、权重、大型预测与参考包，默认目标<20MiB。打包不触发训练/test，不自动下载失败实验。日志被allowlist排除时将有限尾部存为.txt纳入证据。

## 9. Git与服务器交付

1. 在独立worktree完成实现，检查diff/status后提交本实验源码、配置、必要检查与文档。不要提交数据、权重、运行结果、大模块包或用户无关改动。
2. 普通push到本实验分支，独立核对远端SHA与本地HEAD；不强推、不合并母版。网络失败如实写PUSH_PENDING，提供相对固定母版的小型增量bundle，并在一次性仓库实际验证可导入，不能声称推送成功。
3. 最终执行代码提交确定后，使验证报告绑定该代码身份；生成填入完整40位HEAD的交接命令。填SHA的交付副本可作为不再反向改变代码HEAD的产物，避免提交文档又使固定SHA过期的循环。
4. 服务器主目录：`/root/autodl-tmp/projects/Crack_RTDETR`。激活`/root/miniconda3/etc/profile.d/conda.sh`中的现有rtdetr环境；PYTHONPATH指向PSDB worktree的ultralytics-main，不改其他环境/正在训练的worktree，不擅自pip升级。
5. 同步先git cat-file检查目标对象；缺失才fetch指定分支。保持HTTP/1.1、--progress、--no-tags、GIT_TERMINAL_PROMPT=0、http.lowSpeedLimit=1、http.lowSpeedTime=60，单次timeout120秒、最多5次。失败后给经过验证的bundle导入方案；不clone覆盖主目录。
6. 创建独立worktree并核对完整HEAD。已有目录先核验，禁止rm -rf清理旧实验。每段命令自行source本实验环境文件，不依赖其他终端变量；代码版本改变后重新生成/核对环境文件，不能继续引用断言旧SHA的环境脚本。
7. 交付命令分成“同步与环境”“初始化与预检”“显式正式start”“resume”“val/test”“轻量打包”短段。每段在子bash中set -Eeuo pipefail，heredoc结束符完整，避免长代码截断导致终端一直等待。预检不自动启动训练。
8. 先查看GPU占用；SDB仍运行时不能杀进程腾显存。不要默认同时启动多个高显存预检。容量不足记录并安排资源，不降低正式B16/640/AMP配置。
9. 交付至少包括：实施说明/结构合同、两个YAML、源码和可用工具、公共状态审计、实际检查报告、父配方及差异表、固定SHA的SERVER_COMMANDS.md、必要时已验证bundle。缺失项列清具体原因与下一步，不空泛写“检查通过”。

最终回复简明列出：分支、固定母版、本地/远端SHA与推送状态；新增参数28,864；真实通过项与PENDING项；交付文件及下一条用户命令。明确“正式训练NOT_STARTED，最终test NOT_RUN”。不要声称访问过未登录的服务器或已经获得精度提升。

## 附录A：成功CBR＋LIF完整训练args快照

以下是已核对的成功父实验记录，仅作为配方来源；模型/输出身份应替换为PSDB的对应值。不要把父模型权重路径直接作为新模型配置，亦不要把旧实验目录作为输出。与项目实际成功记录有实质冲突时查明并报告，不能无声改配方。

```yaml
task: detect
mode: train
model: /root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix/weights/c19_lif_v1_controlled_init.pt
data: /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml
epochs: 200
time: null
patience: 50
batch: 16
imgsz: 640
save: true
save_period: -1
cache: false
device: '0'
workers: 8
project: /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series
name: c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug
exist_ok: false
pretrained: true
optimizer: AdamW
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: true
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
half: false
dnn: false
plots: true
end2end: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
int8: false
dynamic: false
simplify: true
opset: null
workspace: null
nms: false
lr0: 0.0005
lrf: 0.01
momentum: 0.937
weight_decay: 0.0001
warmup_epochs: 5
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 7.5
cls: 0.5
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.35
degrees: 5
translate: 0.1
scale: 0.4
shear: 1.5
perspective: 0.0002
flipud: 0.2
fliplr: 0.5
bgr: 0.0
mosaic: 0.8
mixup: 0.05
cutmix: 0
copy_paste: 0
copy_paste_mode: flip
auto_augment: null
erasing: 0
cfg: null
tracker: botsort.yaml
save_dir: /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug
```
