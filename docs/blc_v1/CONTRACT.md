# BLC：Codex 完整实施提示词

版本：2026-09-19。用户已确定名称为 **BLC（Bilateral Line Contrast，双侧线对比增强）**。P3 只表示插入位置，不作为模块名称后缀。本文件独立说明结构、配方、验证与交付要求。

## 1. 任务与范围

请在用户的混凝土裂缝检测项目实现 **原 RT-DETR-R18-Lite＋原 CBR＋原 LIF-Down＋BLC**，同时准备 **原 C2＋BLC** 单模块消融配置。持续完成项目核对、独立 worktree、源码与配置、受控初始化、实际 Trainer 重建核验、必要的有限验证、服务器入口、Git 提交、普通 push、独立远端 SHA 核实及固定 SHA 交接，不要停在计划或伪代码。

本次交付实现与可执行入口。本次不登录服务器，不启动正式 200 轮训练，不执行最终 test，不自动训练单模块消融。缺少本机 CUDA、真实数据或指定源权重时，完成其余工作，将依赖项准确标为 PENDING，并提供服务器补检命令。不要把缺环境、只跑合成输入或旧实验通过记录写成本次通过。

仅增加本文 BLC。原 CBR、原 LIF-Down、AIFI、decoder、查询选择、loss、matcher、DN、数据划分和训练配方保持。所有公共可训练参数继续正常训练，不冻结骨干。不得从母版或 PBI 等实验已训练的 best.pt/last.pt 微调。PBI 已被用户中止，不恢复其训练，也不修改或删除其结果。

BLC 属于最近 DPR/PBI 阶段的小幅增参、精度优先实验，不宣称满足早期骨干替换“减参至少 10%”的目标。零初始化只保护初始行为，工程验证不等于涨点。

先读取适用 AGENTS.md。保留未提交修改、已有输出和其他实验；只在本实验 worktree 改动。不得强推、合并母版、reset/clean、停止其他训练或无声升级依赖。若存在实际冲突，说明具体位置并完成不受影响部分，不用删除文件解决冲突。

## 2. 项目、母版和实验身份

目标仓库身份：`supershaojie/concrete-crack-rtdetr`。核对当前目录、Git HEAD、status、origin、worktree；识别 HTTPS/SSH/可选 `.git` 等等价形式，不无声修改 origin 或泄露凭据。不要操作 YOLO26 项目。Windows 本地路径由当前工作区核实，下面服务器路径不能直接套用到本机。

| 项目 | 固定内容 |
|---|---|
| 成功母版提交 | `a0459d6a652cb702699087c88fa39a3e4c4087ec` |
| 主组合父配置 | `rtdetr-resnet18-lite-cbr-lif-down.yaml` |
| 单模块父配置 | `rtdetr-resnet18-lite.yaml` |
| 新分支 | `exp-rtdetr-r18-lite-blc-v1` |
| 服务器主目录 | `/root/autodl-tmp/projects/Crack_RTDETR` |
| 服务器 worktree | `/root/autodl-tmp/projects/Crack_RTDETR-blc-v1` |
| 主变体 | `cbr_lif_blc_v1` |
| 单模块变体 | `blc_v1` |
| 主组合新配置 | `rtdetr-resnet18-lite-cbr-lif-blc-v1.yaml` |
| 单模块新配置 | `rtdetr-resnet18-lite-blc-v1.yaml` |
| 主 run name | `cbr_lif_blc_v1_rtdetr_r18_lite_e200_b16_onlineaug` |
| 单模块 run name | `blc_v1_rtdetr_r18_lite_e200_b16_onlineaug` |
| 正式 project | `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series` |
| 新模块建议位置 | `ultralytics-main/ultralytics/nn/modules/blc.py` |

从固定母版建立实验，不从最近的 PBI/DPR/其他失败实验 HEAD 派生。同名分支或目录存在时先核实身份，再决定安全复用，不覆盖其他工作。

原模块按 CRLF→LF 归一后应保持下列 SHA256；其他公共源码的改动只能是必要的注册、解析或有证据的兼容修复，逐项说明：

| 文件 | SHA256 |
|---|---|
| `lif_down.py` | `26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7` |
| `cbr.py` | `d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787` |

主组合保持原 CBR `rho/normal_fraction=0.10`、hidden_dim=256、queries=300、3 层 decoder、eval_idx=2；以源码真实字段核对。单模块版保留 C2 原 decoder 和下采样，不包含 CBR/LIF。

## 3. 模块包与来源说明

用户最新补发 ZIP：`c0a84cd5-02a6-4827-be29-317c8cc5990b.zip`。

- 已核对大小：21,038,947 bytes。
- SHA256：`b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc`。
- ChatGPT 本次附件路径：`/workspace/scratch/00484fe92169/upload/c0a84cd5-02a6-4827-be29-317c8cc5990b.zip`，仅作来源线索，不保证 Codex 主机存在。

在项目、当前附件和用户已有下载位置有界查找同名或同哈希文件，记录实际路径。只读相关源码：

- `RTDETR-main/ultralytics/nn/extra_modules/attention.py`：`LSKBlock`、`CAA`；
- `RTDETR-main/ultralytics/nn/extra_modules/dynamic_snake_conv.py`：`DySnakeConv`、`DSConv`；
- 当前母版的 `BasicBlock`、`Blocks`、`parse_model` 和融合/Trainer 加载流程。

参考用于核对已有方向建模、上下文选择与本设计的区别，不把上述完整模块直接拷入 BLC。不整体 import extra_modules，不执行包内脚本，不替换项目 ultralytics，不安装模块包 requirements。找不到参考包就如实记为“本机参考包未核对”；数学合同完整，不因此停止其余实现。

来源边界：线/脊检测、中心与周围区域对比、方向算子和残差适配均已有研究。BLC 是本项目待验证的具体组合设计，不宣称发明这些基础思想、不承诺精度提升、不将工作名等同于已证明的论文新颖性。

- Steger：<https://mv.in.tum.de/_media/members/steger/publications/1996/fgbv-96-03-steger.pdf>
- Dynamic Snake Convolution：<https://arxiv.org/abs/2307.08388>
- LSKNet：<https://arxiv.org/abs/2303.09030>

## 4. 唯一插入位置与连接合同

只在原骨干 `model.5` 的 **第二个 BasicBlock 完整输出之后** 增加一次 BLC。这里是原 `Blocks` 阶段的末尾，已经执行原残差相加和原末端 ReLU。禁止把 BLC 插入两个 BasicBlock 各一次、换成 neck 的 model.19、加到 decoder，或替换原 BasicBlock 的卷积。

640 输入时：model.4 为 64×160×160；原 model.5 输出 X 为 128×80×80；BLC 输出 Y 仍为 128×80×80。Y 同时进入原 model.6（骨干 P4）以及原 model.17（neck 的 P3 侧向投影），两条路径都必须接到增强后的结果。

建议用 `BLCBlocks` 继承原 `Blocks`，保留 `.blocks` 的原层次和参数键，新增成员仅为 `.blc`：

```python
class BLCBlocks(Blocks):
    # 接口在核实母版后实现：
    # (ch_in, ch_out, block, count, stage_num, act='relu', r=32, variant='d')
    # 先原样 super().__init__(..., variant=variant)，
    # 再仅在隔离 CPU RNG 的作用域构造和初始化 self.blc。
    def forward(self, x):
        original_p3 = super().forward(x)
        return self.blc(original_p3)
```

两份 YAML 都只将原 model.5 的这一行替换为：

```yaml
- [-1, 1, BLCBlocks, [128, BasicBlock, 2, 3, "relu", 32]]
```

其余层号、from、原 Blocks 个数、内部两块 BasicBlock、stage_num=3、variant=d、通道和重复次数保持。`parse_model` 对 BLCBlocks 参照原 Blocks 分支解析 block_type、注入 ch_in、计算 ch_out×expansion，并传入 r=32；禁止按普通 Conv 的参数规则解释本行或错误重复整个包装器。

原公共状态仍在 `model.5.blocks.0.*` 和 `model.5.blocks.1.*`，不能改成 `model.5.parent.blocks.*`。新增可训练状态只有 `model.5.blc.Wd.weight`、`Wg.weight`、`Wg.bias`、`Wo.weight` 四项；上层 checkpoint 可能另有统一前缀。所有其他公共键保持。需要固定方向信息时用不可变 Python 元组即可，不增加可学习方向或采样偏移。

保持 model.17/19/20/22/25/26 原连接。主组合 model.20 仍是原 LIFDown，model.26 仍是原 CBR decoder；单模块版对应位置保持 C2 原结构。

## 5. BLC 唯一数学定义

### 5.1 维度与可训练层

输入 `X: B×128×H×W`，中间通道固定 `r=32`。仅有三个 1×1 卷积，全部 stride=1、padding=0、groups=1：

```text
Wd = Conv2d(128, 32, kernel_size=1, bias=False)
Wg = Conv2d(40,   9, kernel_size=1, bias=True)
Wo = Conv2d(32, 128, kernel_size=1, bias=False)
Z  = Wd(X)
```

不添加 BN/GN/LN、投影后激活、额外 DWConv、Haar/频域分支、DropPath、可学习 alpha、额外监督或归一化。下文指定的 ReLU、绝对值、minimum 和 softmax 是公式组成部分。Z 保留正负信息。

### 5.2 方向、间距和边界

坐标用 `(行, 列)`，即 `(dy, dx)`。方向与法向的顺序固定：

| 索引 | 方向 | 沿线 t | 横向 n |
|---|---|---|---|
| 0 | 水平 | (0, 1) | (1, 0) |
| 1 | 垂直 | (1, 0) | (0, 1) |
| 2 | 主对角 | (1, 1) | (1, -1) |
| 3 | 副对角 | (1, -1) | (1, 1) |

两种间距 `s∈{1,2}`，候选顺序为 `(方向0,s1),(方向0,s2),(方向1,s1),…,(方向3,s2)`。这里 s 是整数网格步长，斜向距离包含两个坐标分量，不把它宣传为与轴向相同的欧氏距离，也不代表原图真实裂缝宽度。

定义 `T(Z,dy,dx)[...,y,x] = Z[...,clamp(y+dy,0,H-1),clamp(x+dx,0,W-1)]`。边界固定为 replicate/clamp。严禁用 torch.roll 的周期绕回，或插值/grid_sample/可变形采样。

建议对 Z 一次 replicate pad=3 后按合成偏移切片，最大坐标偏移为3；所有切片保持 H×W。每一个沿线点都按其**最终合成坐标**对原 Z 做 clamp。不要先对中心短线平均图做边界 clamp 再移位，这会改变边界定义。

### 5.3 三点中心短线与两侧短线

对每个方向 θ 和间距 s：

```text
Cθ   = [ T(Z, -tθ)       + Z             + T(Z, +tθ)       ] / 3
Bθs+ = [ T(Z, -tθ+s*nθ)  + T(Z, s*nθ)    + T(Z, +tθ+s*nθ)  ] / 3
Bθs- = [ T(Z, -tθ-s*nθ)  + T(Z, -s*nθ)   + T(Z, +tθ-s*nθ)  ] / 3
```

以上加减是二维整数向量运算；Cθ 可在两个间距间复用。每张图均为 `B×32×H×W`。这是固定的三点汇集，没有可学习空间核。

### 5.4 保留正负号的双侧线对比

```text
d+ = Cθ - Bθs+
d- = Cθ - Bθs-

Rθs = minimum(relu(d+),  relu(d-))
    - minimum(relu(-d+), relu(-d-))
```

逐元素计算：

- 中心同时高于两侧时为正，幅度由较弱一侧限制；
- 中心同时低于两侧时为负，保留低响应线索；
- 两侧差异异号或一侧为零时为零。

禁止只保留正响应、改成差分相乘、中心减双侧均值、绝对值直接求和或另设手工阈值。minimum/ReLU 在折点使用框架原生次梯度，不引入 detach、STE 或自定义 backward。真实裂缝可能不符合这一局部先验，原残差路径保留全部原特征。

### 5.5 八种响应与一个零响应选项

按固定顺序记八张带符号响应为 R0…R7。对每张响应沿通道求绝对值均值，得到八张单通道描述：

```text
ei = mean(abs(Ri), dim=channel, keepdim=True)    # B×1×H×W
G  = concat([Z, e0, e1, ..., e7], dim=channel)  # B×40×H×W
L  = Wg(G)                                     # B×9×H×W
a  = softmax(L, dim=channel)
R  = sum(a[:, i:i+1] * Ri for i in range(8))    # B×32×H×W
Δ  = Wo(R)
Y  = X + Δ
```

第9个权重 `a[:,8:9]` 对应严格为零的新增响应，不再另建可训练分支。它通过 softmax 分母减少其他八组权重；不能将前八组重新归一化到和为1，否则会抹掉这一选择。权重在每个空间位置独立产生，八组方向权重对32通道共享；softmax 只沿9个候选维度，不能沿通道32、空间HW或batch归一化。

以上八个响应不能先取 abs 再融合，abs 只用于门控描述 ei。BLC 输出不额外加 ReLU，保持完整的 `X+Δ`。

### 5.6 初始化与精度

- Wd.weight：Xavier uniform，gain=1。
- Wg.weight、Wg.bias：全零，初始九个 softmax 权重均为1/9；零 logits 不等于零门控。
- Wo.weight：全零，使有限输入初始 `Y=X`。
- 不在外面再乘一个零初始化 alpha，不把 Wd 也置零。
- 仅在新增层构造/初始化范围隔离 CPU RNG，不能消耗后续公共层随机数；两个变体使用同一套 BLC 新增初值，但不共享可变 Parameter/storage。

BLC 内部的投影、差分、minimum、softmax、融合和残差求和统一在局部工作精度计算：输入 FP16/BF16 时使用 FP32；输入 FP32/FP64 时分别保持 FP32/FP64。只在 BLC 内禁用 autocast，使用可微的参数 dtype 转换和 functional conv2d，最后将 Y 转回输入 X.dtype。例如 Wd 的计算可用 `F.conv2d(X.to(work_dtype), Wd.weight.to(work_dtype))`，Wg/Wo 同理，Wg.bias 也转换。

不得在 forward 中修改 Parameter 的 `.data`、新建脱离优化器的 Parameter、detach、缓存过期的转换后权重，或执行重新初始化。模型本身的 AMP、GradScaler、TF32策略、全网 dtype 和训练配方不变。局部 FP32 是 BLC 固定设计的一部分，必须在训练、eval、EMA、显式 half、融合和恢复后共用同一前向实现。

这样设置不保证输入或输出一定有限。真实 NaN/Inf 应报告原因，不能用 nan_to_num、输出裁剪或静默跳过分支掩盖。

## 6. 参数与计算量口径

新增参数：Wd=4096，Wg=40×9+9=369，Wo=4096，合计 **8,561**。不存在其他可学习权重。基于当前母版已确认计数，预期为：

| 配置 | 未融合参数 | 原生融合后参数 |
|---|---:|---:|
| C2＋BLC | 20,091,333 | 19,886,277 |
| CBR＋LIF＋BLC | 20,158,326 | 19,953,526 |

实际构造后分别计数并核对差异来源。不要改变结构凑数；不混用未融合 Params 与融合 GFLOPs。

在 B=1、P3=128×80×80 时，三个可学习1×1投影合计54,732,800 MACs，即按2 FLOPs/MAC约0.1094656 GFLOPs，**不含**固定三点汇集、差分、abs/mean/minimum/ReLU/softmax、门控融合、残差加法和类型转换。它不是新增模块完整 FLOPs，也不是实测整网增量或速度。

functional conv2d 可能绕过子 Conv2d 的 THOP hook。若报告 GFLOPs，应显式核算三个投影，避免漏算和重复计数，说明固定算子覆盖范围。只检查当前安装版本的计数器，不为本次实验安装多个 THOP 版本或搭建历史 DPR 的大回归矩阵。

BLC 有输入相关门控和非线性，不能无损折叠成原卷积。原网络 Conv-BN/RepConv 融合照常进行，融合后保留一次完整 BLC；不宣称零推理开销。

## 7. 公共初始化与真实 Trainer 重建

审查并复用母版已有初始化逻辑，优先核对 `tools/init_c19_lif_v1.py`、`tools/init_lif_down.py`。

统一公共未训练源：

- 服务器路径：`/root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt`。
- SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
- 源历史 nc=80、epoch=-1，不带已训练 optimizer/EMA/scaler；对应历史 C2 提交 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。

分别构造 C2 父模型和原 CBR＋LIF 父模型，再向对应 BLC 候选逐 key/shape/value 严格迁移所有公共参数与 buffers。不能只加载 backbone、只看 transferred 数量，或凭相同 seed 推断初值一致。

输出 COMMON、NEW_TRAINABLE、NEW_BUFFER、MISSING、UNEXPECTED、SHAPE_MISMATCH 清单。新增可训练键恰为上述四项；所有公共键保持。BLC 无需新增 tensor buffer；若实现确有固定 buffer，明确列出用途和生命周期，不作为可学习参数。

必须通过实际 RTDETRTrainer.get_model 及正式 train 的重建/加载路径验证 nc=1，而非只验证 RTDETR(checkpoint) 推理构造。相对于 nc80 源，允许原生适配9个分类状态项：denoising_class_embed.weight、enc_score_head weight/bias、3层 dec_score_head weight/bias。适配后的父/候选 nc1 张量仍应逐项一致，不能把这9项作为随意重初始化例外。

保存受控初值后重载核查。start 使用未训练受控初值；resume 使用本实验真实 checkpoint，不重复执行零初始化。原公共分类层、CBR/LIF 和 BLC 初值均不得被 Trainer 重建覆盖。隔离参考模型构造的 RNG，保护目标实际初始化流程。

本机缺指定源时标 PENDING，不能拿随机权重、其他同名模型或已训练 best 替代后宣称完成。

## 8. 正式训练协议

本次沿用最近成功母版与 PBI/DPR 的 **200e／在线增强** 协议，不能回退到早期150e、lr0=.01、关闭在线增强的旧记录。文末附完整109字段历史 args 快照；结合当前母版实际 args 核对并归档。只允许模型/输出身份和经核实等价的数据路径不同；resume 字段仅在显式续训入口改变。

核心参数：epochs=200，patience=50，imgsz=640，batch=16，nbs=64，seed=42，workers=8，device=0，AdamW，lr0=.0005，lrf=.01，momentum=.937，weight_decay=.0001，warmup_epochs=5，warmup_momentum=.8，warmup_bias_lr=.1，cos_lr=True，deterministic=True，amp=True，cache=False，freeze=None，rect=False。

保持 hsv_h=.015、hsv_s=.5、hsv_v=.35、degrees=5、translate=.1、scale=.4、shear=1.5、perspective=.0002、flipud=.2、fliplr=.5、mosaic=.8、mixup=.05、close_mosaic=10。保持原 loss、matcher、DN、queries、梯度累计、优化器分组规则、学习率、EMA、epoch验证和 best 选择规则。所有新增可训练参数在优化器中恰好出现一次，不另设学习率/weight decay组。

数据配置：`/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml`；根目录：`/root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det`；现有 train/val/test=6048/1728/864，nc=1。核对现有清单与标签身份，不重新划分、增强覆盖或交换 val/test，不只凭数量判断一致。不增加种子搜索、交叉验证或 holdout。

## 9. 必要验证：检查真实风险，避免无关扩张

本实验建立 BLC 的验证记录，不修改或重写 DPR/PBI 的旧准入规则、历史失败或通过报告。优先复用已审查的项目工具；不要整套搬入旧实验的模型或未解释的验收断言。检查范围在本节预先确定，不能看到结果后改阈值、改数据或反复重采样寻找通过。

### 9.1 数学、连接与梯度

1. 用独立的直接索引参考公式核查四方向、两间距、最终坐标clamp、候选顺序、带符号响应、9路softmax与零响应选项。覆盖平坦场、线性坡度、单侧阶跃、正/负细线和1/2/3网格宽度；分开报告内部区域与边界。阶跃某一方向为零不等于所有真实图像背景都能被抑制，不夸大理想图验证。
2. 核心覆盖小矩形、奇数H/W及边界。整网只要求母版已支持的stride对齐形状，不新增母版本来不支持的任意奇数整图功能。
3. 使用非零Wo副本比较实际前向与独立公式，并检查Wd/Wg/Wo及输入梯度。FP64导数检查避开minimum相等、abs=0、ReLU=0等不可微折点；不能因折点的次梯度选择不同误报数学错误。严格参考推荐FP64 atol=1e-8/rtol=1e-6，FP32 atol=2e-5/rtol=2e-4；记录实际误差。若已有项目更严格的适用规范则说明并遵循。
4. 原公共状态相同时，Wo=0的BLC核心输出应逐元素等于输入；父/候选对应P3/P4/P5连续特征与原生输出做初始化核验。公共层参数路径、两条P3消费路径和唯一BLC调用必须正确。
5. 使用原生RT-DETR真实检测loss验证梯度启动：首个有效反向Wo应有有限非零梯度；Wd/Wg首步为零是Wo=0的预期结果。Wo有效更新后，再检查Wd/Wg在非退化样本上的梯度与参数更新。不能要求每个标量、每个位置每一步都非零；也不能用weight decay造成的参数变化代替非零梯度证据。

### 9.2 有界真实训练与容量

服务器使用现有 rtdetr Conda、Python3.10、PyTorch2.1.2+cu121、RTX4090。先核对 ultralytics 导入来自本worktree；不升级环境来通过检查。CPU检查与本机可用CUDA检查分开记录；服务器默认只给主组合做一次实际容量验收，两份配置都需通过结构/初始化核验。

容量使用真实train数据、有效GT、原DN、原在线增强、B16/640/native AMP和原生优化器/累计，最多16个batch，目标至少2次有效optimizer更新；记录batch数、loss、GradScaler跳步/回退、有效更新、梯度有限性、显存和耗时。到预算停止，不自动继续200轮。单次scaler溢出回退不直接等于模型失败；持续非有限或预算内无足够有效更新应如实失败，不把step函数调用次数冒充更新。

预检用临时模型和独立输出，不能把预检更新后的模型覆盖正式controlled_init。不得降低batch、关闭全网AMP、固定scale=1、缩小图像或无限重试来制造通过。用户允许并发：可显示GPU占用，不设置“有GPU进程就退出”的空闲门槛，不kill其他任务。真实OOM如实报告。

### 9.3 非零分支的保存、恢复、EMA与融合

必须在真实有效更新后或明确的非零测试副本上检查，只有零初始化状态通过没有意义：

- state_dict和完整模型保存重载；实际Trainer重建后四个BLC状态项均保留，输出存在可验证的非零增量。
- 原生optimizer参数名/分组绑定、moments/step、scaler、epoch、EMA及updates按文件实际保存内容正确恢复。沿用母版保存策略，记录其FP16/FP32量化行为；不要把保存前live FP32与原生half文件的量化差异当成丢状态，也不通过偷偷改保存精度修改本实验行为。
- 用两个从同一checkpoint恢复的副本核对文件状态和可变张量无共享storage；在相同外部梯度下执行原生更新，检验恢复和优化器绑定。它是保存恢复诊断，不能替代真实独立反向训练验证。
- 原生resume入口能加载真实未完成checkpoint，恢复epoch/optimizer/scaler/EMA，并继续进行有效更新；不得把受控初值或deploy-only文件当续训文件。
- 整网原生fuse后BLC仍被调用一次，非零响应和新增参数保留；不能由forward_fuse绕过。BLC本身不做卷积重参数化折叠。
- 验证CUDA FP32、native AMP，以及实际原生epoch验证所用的half EMA路径和AutoBackend保存/加载路径。half EMA会影响best选择，不能删除其检查。CPU不支持half时记录环境边界，不替代CUDA实测。

比较恢复前后输出必须使用相同输入、eval状态、dtype和相同文件量化后的参考状态。half与FP32、不同融合顺序是不同的数值路径，分别记录原始max_abs、relative_L2、allclose及有限性，不把“整网逐元素跨精度一致”作为文件是否正确恢复的定义。

若融合误差存在，用非零BLC、原P3/P4/P5连续特征、encoder分数和候选索引定位。严格FP32诊断可临时关闭TF32并在finally恢复；不能永久改正式训练精度。RT-DETR近同分top-k可能变序，固定候选重放仅用于定位，不能改正式查询选择或用重放输出替代自然验证。

CUDA含原网络及本模块padding等算子的独立反向可能有非确定性；保留父/候选原始梯度差异，不要求两个独立CUDA反向结果全张量allclose才算文件恢复正确。只有同文件状态、同精度前向、相同梯度更新和实际训练有限性均有证据，且额外数值差异有定位依据时，才可单列PRECISION_NOTE。丢参数、丢分支、错误绑定、真实非有限、同文件同路径无法解释的差异和缺证据均不能放行。不要仅以“父模型也不一致”解释候选的所有异常。

### 9.4 验证组织与状态

保留PASSED/FAILED/PENDING/PRECISION_NOTE原始记录，失败后不覆盖旧报告。证据按当前代码/配置内容哈希、源初值/数据身份、variant、环境关联；start检查相关证据，缺必需服务器证据时明确指出，不把存在某个许可文件当通过。检查通过只表示可开始实验，不自动运行训练。

必要的forward/gradient hooks在deepcopy/torch.save前移除，finally清理；不要把局部闭包hook序列化。一次性pt放工具独占临时目录，仅清理本次明确拥有的临时产物，保留必要报告和正式初值/best/last。成功证据在相关实现与输入未变时复用；修复后只重跑受影响项及其依赖，不机械重复整个矩阵。

如父AutoBackend确实因未初始化torch.empty热身输入产生异常，允许最小改为有限zeros并记录理由；不能用这一修复掩盖真实输入非有限。使用Torch2.1可用的GradScaler接口，不假定存在新版本torch.amp.GradScaler(device)签名。

## 10. 训练、评估与轻量打包入口

建议提供 `tools/blc_server.sh`，通过 `BLC_VARIANT=cbr_lif_blc_v1|blc_v1` 显式选变体，默认主组合。可复用现有工具组织，不要求为每个动作另建一个大脚本。应提供真实可用的 environment、init、init-preflight、plan、start、resume、val、test、pack 或等价入口，并核对实际help。运行时不得依赖PBI/DPR等其他worktree。

每段服务器命令独立加载环境：现有 `/root/miniconda3/etc/profile.d/conda.sh`、`conda activate rtdetr`、当前worktree的PYTHONPATH，断言ultralytics实际导入路径。处理Conda与set-u兼容，保留真实退出码和实时日志。

start与resume分离；resume恢复真实本实验未完成checkpoint，不能重复零初始化。正式run名已存在时不覆盖、不静默追加成新实验名；报告当前状态及明确resume方式。保留原epoch验证和best选择，200轮结束但final_eval失败时不能自动从头再训。

训练完成后按原规则固定best.pt及SHA，再独立val、独立test；本次仅实现入口，不执行最终test。评估继续使用项目已核验的 `corrected_sorted_conf_mask_v1`：imgsz640、batch16、workers0、half=False、conf=.001、iou=.7、max_det300、augment=False、rect=False、seed42。保留原一对一匹配和排序后置信度掩码，不增加NMS、不调test阈值。若固定母版未含该修正，定位已有审查实现，移植最小必要修正并记录来源，不能把指标口径悄悄换回旧版本。

保存P/R（各模型最大F1工作点）、AP50、AP75、mAP50–95、十个IoU AP、图数/GT、完整精度JSON、PR/混淆矩阵和训练曲线。中期val只与同协议val比较，不能与母版最终test混比。历史母版数字从原始评估JSON及十个AP核对后引用，不直接采用旧聊天中冲突的52.20/56.51记忆值。

有效性比较以原CBR＋原LIF为对照，完整报告P/R/AP50/AP75/mAP50–95和开销；不能凭单项上升称全面超过。结构和checkpoint选择依据val，最终test只做冻结后的确认。主组合有收益后再安排C2＋BLC单模块消融，本次不自动训练。临时关闭分支的输出差异只能作诊断，不能替代重新训练的消融。

轻量包目标<20MiB：必要源码/YAML、args.yaml/results.csv、初始化/重建审计、预检与训练状态、已实际产生的val/test指标和曲线、必要日志尾、Git身份与SHA256清单。排除权重、数据、参考ZIP和大型中间张量。pack不触发train/test、不自动下载、不删旧输出；打印archive绝对路径、大小、哈希及missing/omitted。未运行的评估诚实标NOT_RUN，不伪造文件。

## 11. Git与最终交付

完成必要实现和本机可做验证后，选择性git add、commit并普通push本实验分支；不混入用户改动、大权重、数据或其他实验。独立 `git ls-remote` 核实本地与远端完整40位SHA，网络失败保留提交并写PUSH_PENDING，不声称远端已存在。

服务器同步沿用用户固定方法：先 `git cat-file` 查目标SHA，缺失才fetch指定分支；HTTP/1.1、`--progress --no-tags`、`GIT_TERMINAL_PROMPT=0`、`http.lowSpeedLimit=1`、`http.lowSpeedTime=60`；单次timeout120秒、最多5次，显示进度/退出码，有限重试后回终端。保持已核实origin，不改全局Git配置，不默认要求离线包。

创建/复用独立worktree时核对仓库公共目录、完整HEAD、母版祖先关系和已有修改。保留untracked/ignored结果，不reset/clean或覆盖冲突文件。使用固定目标提交的同步脚本，保证主目录和其他实验工作树不被切换。

交付简洁实现说明、验证报告、父配方差异和完整固定SHA交接文档。提交确定后再写交接SHA，避免仅修改文档HEAD而无休止反复提交。

最终回复明确列出：

1. BLC的实际位置、两份配置、8,561新增参数设计值与实测值、融合统计口径；
2. 母版/分支/本地和远端完整SHA及核验证据；
3. 已通过、FAILED、PENDING及有依据的PRECISION_NOTE，不隐瞒失败；
4. 文件入口；直接在聊天给“同步与环境”“初始化与有限预检”两段可复制服务器指令；
5. 完整start/resume/val/test/pack命令写入交接文档，供用户随后显式调用。start给出tmux方式：检测已在会话内则直接运行；否则创建或进入明确的BLC会话，避免重复启动训练；
6. **正式训练NOT_STARTED；最终test NOT_RUN；本次未登录服务器。**

交付前核对所有命令都指向BLC真实已实现入口，不能只把PBI/DPR名字替换后留下无效参数、旧SHA或旧run路径。

## 附录：成功母版109字段args快照

下面来自已读取的成功组合原始 `training/args.yaml`，保留完整字段供比较。父model路径和run身份是历史来源，不能直接作为BLC训练输入；新实验按上述受控初始化和独立run替换。实际原始文件与快照冲突时，先查明来源，不静默混用。

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
