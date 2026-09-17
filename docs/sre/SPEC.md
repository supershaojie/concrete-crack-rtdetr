# SRE：Codex 完整实施提示词

版本：2026-09-17。本文可独立交给 Codex 执行。模块正式名称只有 **SRE（Support-based Response Enrichment，支持驱动响应补偿）**，不要在模块名后添加 P3、P4 或其他尺度后缀。v1 仅作代码和实验版本标记；特征尺度只用于说明接线。

## 0. 目标与执行范围

请在用户的混凝土裂缝 RT-DETR 项目中，完整实现 SRE、受控初始化、本机可完成的预检、独立分支提交与普通推送核验，交付固定 SHA、可复制执行的服务器命令。不要停在计划或示例代码。

主实验：**原 R18-Lite＋原 CBR＋原 LIF-Down＋SRE**。同时准备 **C2＋SRE** 单模块消融配置。首轮服务器命令默认只预检、训练主组合；不同时派发两个正式训练。

允许创建独立 worktree、修改本实验代码、必要修复、提交及普通 push；不强推、不合并成功母版、不覆盖用户未提交改动、不删除旧实验。本次不登录服务器、不停止已有进程、不启动正式训练、不运行最终 test。缺 CUDA、真实数据或统一源权重时，完成其他工作并将依赖检查列为 PENDING，提供补检命令，不伪造 PASSED。

SRE 是独立的第三模块候选，不叠加 SDB、PSDB、CSR、BFR、CPI、BSC、TRC、LSRT 或 RCS-Q。保持原骨干、CBR、LIF、AIFI、Decoder、损失和原200轮配方。不加载成功母版已训练 best.pt/last.pt 作为新实验初值；从统一未训练源公平构建。

本实验针对召回：假设一部分漏检来自已有弱响应未获得邻域支持。目标是提高相同 Precision 水平下的 Recall，同时观察 AP75、mAP50–95 与新增误检。不能把潜在响应称为裂缝概率，不能承诺涨点，也不能把工程检查通过当成精度证据。

## 1. 核对项目、母版与参考材料

先读适用 AGENTS.md，检查 git status、remote、HEAD、worktree 及初始化/训练工具。目标是 `concrete-crack-rtdetr`，已知远端 https://github.com/supershaojie/concrete-crack-rtdetr；与真实 remote 核对，不进入其他项目。

- 固定成功母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
- 主组合父配置：`rtdetr-resnet18-lite-cbr-lif-down.yaml`。
- 单模块父配置：`rtdetr-resnet18-lite.yaml`，保留原 C2 下采样及 Decoder。
- 独立新分支：`exp-rtdetr-r18-lite-sre-v1`。
- 服务器建议 worktree：`/root/autodl-tmp/projects/Crack_RTDETR-sre-v1`。

同名目录或分支已存在时核对身份、继续未完成工作，不 reset 或覆盖。原模块 LF 归一化 SHA256（CRLF 转 LF 后计算）：

| 文件 | SHA256 |
|---|---|
| lif_down.py | 26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7 |
| cbr.py | d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787 |

不要修改这两个模块来适配 SRE。若基点或源码不符，查明而非换成其他同名版本。

参考模块包：`dc2eda42-6ea1-467d-ab90-02d26882d4d0.zip`，SHA256 为 `b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc`。历史 ChatGPT 路径 `/workspace/scratch/00484fe92169/upload/dc2eda42-6ea1-467d-ab90-02d26882d4d0.zip` 不保证在当前主机存在。

有界查找实际附件/项目包，记录读取路径与哈希。包内参考：

- `RTDETR-main/ultralytics/nn/extra_modules/RFAConv.py` 的 RFAConv：参考感受野内的内容相关加权；原包使用 softmax 与生成特征，本方案不直接复制其完整结构。
- `RTDETR-main/ultralytics/nn/extra_modules/block.py` 的 ContextGuidedBlock：参考局部与周围上下文的区分；不整体搬入该模块。

参考包不可见时如实记录，继续按本文数学合同实现；不伪称读过、不执行包内安装/训练脚本、不整体替换 ultralytics、不提交大包。背景来源：[Neighborhood Attention](https://arxiv.org/abs/2204.07143)、[RFAConv](https://arxiv.org/abs/2304.03198)。这些只提供相关技术背景，不证明本设计首创或有效。

与旧 BSC 的区别必须准确：BSC 使用固定方向的成对邻居、双侧均值/差异；SRE 使用完整局部邻域的内容相似关系，逐通道只接收更强响应，可接受单侧支持。二者都属于局部残差增强，不能声称完全没有关联；不能把 SRE 实现回双侧平均或方向卷积。

## 2. 统一命名

| 项目 | 主组合 | 单模块消融 |
|---|---|---|
| variant | cbr_lif_sre_v1 | sre_v1 |
| YAML | rtdetr-resnet18-lite-cbr-lif-sre-v1.yaml | rtdetr-resnet18-lite-sre-v1.yaml |
| run name | cbr_lif_sre_v1_rtdetr_r18_lite_e200_b16_onlineaug | sre_v1_rtdetr_r18_lite_e200_b16_onlineaug |

- 源码：`ultralytics-main/ultralytics/nn/modules/sre.py`。
- 核心类：`SRE`；接线类：`SRERepC3`。
- 工具建议：`init_sre.py`、`check_sre.py`、`preflight_sre.py`、`train_sre.py`、`eval_sre.py`、`pack_sre_light.py`；具体 CLI 必须与实现一致。
- 文档：`docs/sre/`；元数据：`outputs/sre/<variant>/`。
- 正式训练 project：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series`。

不要生成带尺度后缀的 SRE 类名、文件名、分支、YAML 或实验名。

## 3. 接线：原 model.19 完整输出后串联 SRE

先核对真实 YAML。640输入时 model.19 是最终 P3，256×80×80；model.20 是原 LIFDown；model.22、25 是最终 P4/P5；model.26 是原 RTDETRDecoderCBR，from=[19,22,25]。

SRE 只接收原 model.19 RepC3 的完整输出，不读取 P2、其他尺度、预测框或 Decoder 分数。

```python
class SRERepC3(RepC3):
    # 原 RepC3 构造后，隔离 RNG 创建 self.sre
    def forward(self, x):
        original_p3 = super().forward(x)
        return self.sre(original_p3)   # SRE 自身已含原输入残差，不再加一次
```

建议构造器：`SRERepC3(c1,c2,n=3,e=0.5,latent_channels=32,descriptor_channels=8,window_size=5)`。

建议 model.19 YAML 行：

```yaml
- [-1, 3, SRERepC3, [256, 0.5, 32, 8, 5]]
```

解析器正确注入 c1/c2，将内部 repeat=3 传入构造器，外部 repeat=1。使用 base_modules/repeat_modules 的解析器要注册到两者或实现等价逻辑。只保留一个 SRE 包装，内部仍是父模型原3个 RepConv；不能外部串联3个包装器。

保留原 RepC3 的 cv1/cv2/m/cv3 状态键，新可训练状态只在 `model.19.sre.*`。不得包成 parent 子模块导致旧 key 迁移。原 from=-1、节点总数27和其余层号保持；修正输出沿原路供 model.20 与 Decoder 使用，无额外检测支路。

不修改原 Conv/RepC3 的全局行为。SRE 核心继承 nn.Module，投影用原生 nn.Conv2d，避免原 Ultralytics Conv 融合逻辑替换其 forward。单模块版同位置接入，但保留 C2 原下采样和原 Decoder。

保持原骨干、model.17、AIFI、CBR rho/normal_fraction=0.10、queries=300、hidden_dim=256、3层 Decoder、eval_idx=2。原有权重从公共初值开始，允许像父实验一样正常训练，并非冻结母版权重。

## 4. 精确数学合同

### 4.1 输入、投影与描述子

核心接口 `SRE.forward(X)`，X 为 B×256×H×W，当前 H=W=80。输出同形状、同 dtype/device。

```text
Z = GN(W_d(X))
Q = normalize(W_q(Z), p=2, dim=channel, eps=1e-6)
V = softplus(Z, beta=1, threshold=20)
```

| 名称 | 固定配置 |
|---|---|
| W_d | Conv2d(256,32,1,stride=1,padding=0,bias=False) |
| GN | GroupNorm(4,32,eps=1e-5,affine=True) |
| W_q | Conv2d(32,8,1,stride=1,padding=0,bias=False) |
| W_o | Conv2d(32,256,1,stride=1,padding=0,bias=False) |

Q 为 B×8×H×W，V 为 B×32×H×W。Q 沿8维描述子归一，不沿空间、batch 或邻居维归一。V 是非负潜在特征，不是概率或已验证的前景置信度。

### 4.2 24个有效邻居及相似权重

固定邻域：`(dy,dx)∈{-2,-1,0,1,2}²`，排除 `(0,0)`。每个位置最多24邻居，只有一次聚合，不迭代传播，不额外加膨胀尺度、方向对、学习偏移或位置偏置。

对有效邻居 j：

```text
cos_pj = sum_k Q[b,k,p] * Q[b,k,j]
w_pj   = max(0, cos_pj)^2
```

仅为浮点误差可将 cos 夹紧至[-1,1]；不引入新阈值、可学习温度或 softmax。w 每位置每邻居一个标量，对32个潜在通道共享。

边界使用有效邻居掩码：越界邻居的 w 必须为0，不能通过 replicate/reflect padding 让同一个边缘像素重复投票，不能用循环 roll 跨图像边缘通信。若切片/移位实现，要核对偏移方向与重叠区域。H或W为1、矩形和奇数尺寸都应支持；1×1输入没有邻居。

### 4.3 只接收更强的相似响应

```text
s_p  = sum_j w_pj
D_pc = sum_j [w_pj * relu(V_jc - V_pc)] / (1 + s_p)
Y    = X + W_o(D).to(dtype=X.dtype)
```

逐通道正差分在加权求和前计算，不能换成 `relu(weighted_mean(V)-V)`，两者一般不等价。分母严格为1+s，不是s+1e-6，也不按有效邻居个数再平均。这个“1”是固定的零消息权重，不是可学习参数。

可审查的伪代码如下；实际实现需处理参数 dtype、有效切片和设备：

```python
Z = group_norm(conv1x1(X.float(), Wd))
Q = normalize_fp32(conv1x1(Z, Wq), dim=1, eps=1e-6)
V = softplus_fp32(Z, beta=1, threshold=20)
numerator = zeros_like(V)
support = zeros_like(V[:, :1])
for dy, dx in fixed_24_offsets:
    Qj, Vj, valid = shifted_values_with_valid_mask(Q, V, dy, dx)
    cosine = (Q * Qj).sum(dim=1, keepdim=True).clamp(-1, 1)
    w = relu(cosine).square() * valid
    numerator = numerator + w * relu(Vj - V)
    support = support + w
D = numerator / (1.0 + support)
delta = conv1x1(D, Wo)
return X + delta.to(X.dtype)
```

所有邻域值来自同一份原始 Q/V，不将已经补偿的位置用于更新下一个位置；计算结果不应依赖扫描顺序。不要 detach Q、V、w 或 D，不缓存输入相关张量，不使用破坏 autograd 的原地更新。

### 4.4 应满足的性质及限制

记 `m_pc=max_j relu(V_jc-V_pc)`，无邻居时m=0。精确算术下：

```text
0 <= w_pj <= 1
0 <= D_pc <= (s_p/(1+s_p)) * m_pc <= m_pc
```

- 无正相似支持时 D=0。
- 邻域 V 同值时 D=0。
- 中心某通道不弱于所有邻居时，该通道 D=0。
- 单侧邻居也可以提供支持，不要求另一侧存在。
- w 的成对相似性是对称的，但正差分让每通道强→弱的潜在补偿具有方向性；同一对位置可能在不同通道互相提供支持，不能把它解释成按一个前景分数排列所有位置。
- 总支持很弱时，固定零消息权重降低补偿；不会强制把邻域权重归一成总和1。

这些性质仅约束潜在 D。W_o 有符号，最终特征、分类分数和 Recall 都不保证单调增加。相似背景纹理也可能产生补偿；目标在原特征中完全消失时，不能声称能恢复真实细节。

## 5. 初始化、精度和代价

### 5.1 初始化及梯度启动

仅 W_o.weight 全零；W_d、W_q 使用非零 Xavier uniform；GN weight=1、bias=0。不再叠加零标量或零门控，不将 W_q 置零。

新增模块构造隔离 CPU RNG，不消耗父模型后续层的初始化随机序列。若用 fork_rng(devices=[])，只设置 CPU generator，不能无保护地同时改 CUDA RNG。两个变体的新 SRE 初值逐项一致。

有限输入、W_o=0 时 Y=X。首个有效 backward 检查 W_o 有限非零梯度；首步上游梯度为0是预期。一次真正 optimizer step 使 W_o 非零后，再检查 W_d、W_q、GN 的有效梯度。GradScaler跳步不算更新，不要求每个元素在每个 batch 非零。不能把 BFR 的 DC/GN.bias 梯度例外套用到 SRE，本模块没有 DC 排除。

### 5.2 数值和内存

原网络保持 native CUDA AMP。新增 SRE 的投影、GN、L2归一、Softplus、点积、权重、正差分及累计采用局部 FP32，显式禁用该区域 autocast；出口将 delta 转回 X.dtype。

兼容显式 model.half()：使用可微 F.conv2d/F.group_norm 与参数.float()，或验证等价实现；不能让 half 参数直接接收 FP32 输入，不在 forward 中 module.float()、替换 Parameter 或修改优化器对象。half checkpoint 已量化，.float()不恢复量化前精度；重载比较须同量化口径。

禁止自动安装 NATTEN、自定义 CUDA 或额外大依赖。标准 PyTorch 即可实现。先实现数学正确且可审计的版本，再处理明确的性能瓶颈；不为性能改公式。

完整 B16×32×24×80×80 FP32 邻域张量约300MiB，尚不含反传和其他张量。可采用分块或逐偏移累计，避免无必要的重复展开；不能声称循环就消除了 autograd 保存量。必须实测 B16/640/native AMP 的峰值显存与耗时，不以参数少代替容量检查。

### 5.3 可核对的参数与计算量

| 新层 | 参数量 |
|---|---:|
| W_d：256→32 | 8,192 |
| GN affine | 64 |
| W_q：32→8 | 256 |
| W_o：32→256 | 8,192 |
| 合计 | **16,704** |

640输入、B=1时，三个投影的主要计算：`(256*32 + 32*8 + 32*256)*80*80 = 106,496,000 MAC`，按2 FLOPs/MAC为 **0.212992 GFLOPs**。邻域点积、权重/正差分、累计、GN、归一化与激活另计；这不是完整整网GFLOPs。THOP的函数式算子不能漏计或双算。

nc=1 参数预期（同融合口径实测核对）：

| 模型 | 未融合 | 融合后 |
|---|---:|---:|
| 原 CBR＋LIF | 20,149,765 | 19,944,965 |
| CBR＋LIF＋SRE | **20,166,469** | **19,961,669** |
| 原 C2 | 20,082,772 | 19,877,716 |
| C2＋SRE | **20,099,476** | **19,894,420** |

## 6. 受控初始化与实际 Trainer 加载

审查并优先复用固定母版 `tools/init_c19_lif_v1.py`、`tools/init_lif_down.py`。

统一未训练源：

- `/root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt`。
- SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
- nc=80、epoch=-1，无已训练 optimizer/EMA/scaler 状态。
- 历史 C2 提交：`67c3078e54a657fd96d65fee657a75fbb1dae0d6`。

构建受控 C2、CBR＋LIF 父模型，严格复制公共状态到对应 SRE 目标。共有参数与 buffers 逐 key、形状、数值核验，输出 COMMON、NEW_TRAINABLE、NEW_BUFFER、MISSING、UNEXPECTED、SHAPE_MISMATCH 与分类适配清单，不只检查 seed 或 Transferred 数量。

实际调用 RTDETRTrainer.get_model/训练模型重建，验证 nc=1。nc80→1 相对源允许适配9项：denoising_class_embed.weight、enc_score_head.weight/bias、3层 dec_score_head 的 weight/bias。父/目标这9项 nc1 张量仍须在匹配 RNG 后逐值一致；参考父构建不能消耗目标的分类适配随机状态。

Trainer 不能覆盖原 CBR/LIF 或新增 SRE 的受控状态。初值保存后立即重载审计；start 只使用该初值；resume 只恢复真实未完成训练的 checkpoint。forward/load/EMA/fuse/resume 不得重置 SRE、CBR 或 LIF。

## 7. 保持原200轮配方与数据

权威配方是成功 CBR＋LIF 的 **200轮在线增强**，完整快照见附录A。旧150轮、lr0=0.01或关闭在线增强的记录不适用。与实际成功父记录核对、存档，生成逐字段差异表。

- epochs=200、patience=50、imgsz=640、batch=16、nbs=64、seed=42、workers=8、device=0。
- AdamW；lr0=0.0005、lrf=0.01、momentum=0.937、weight_decay=0.0001；warmup_epochs=5、warmup_momentum=0.8、warmup_bias_lr=0.1。
- cos_lr=True、deterministic=True、amp=True、cache=False。
- hsv_h=0.015、hsv_s=0.5、hsv_v=0.35、degrees=5、translate=0.1、scale=0.4、shear=1.5、perspective=0.0002、flipud=0.2、fliplr=0.5、mosaic=0.8、mixup=0.05、close_mosaic=10。

只允许模型标识、输出/name、经核实等价的数据绝对路径改变。损失、DN、matcher、学习率计划、查询配置、增强继承；不为 Recall 调整类别权重、正负样本分配、检测头或训练阈值。不依赖新版本默认值，不新增辅助监督。不擅自多 seed、交叉验证、holdout、改轮数或额外提前停止策略。

保持 exist_ok=False，已有 run 区分 start/resume，不自动生成 name2，不删除旧输出重试。数据 `/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml`，nc=1，train=6048、val=1728、test=864；核对清单与标签身份，数量只作辅助，不重划分。

## 8. 有限且有意义的预检

使用一次性模型/优化器副本和独立临时目录，不覆盖受控初值、正式输出或其他实验。两个变体做本地可完成检查；服务器默认检查主组合。缺环境列 PENDING，修复后重跑受影响项及依赖，避免无关重复检查。

### 8.1 数学与结构

1. 用很小的手工 Q/V 和显式邻居循环参考验证数值；不要只拿同一聚合函数比自己。覆盖中心/边/角、矩形/奇数、1×1和单轴长度1，检查24偏移、排中心、有效邻居数与广播。
2. 手算零/负相似、正相似、零描述子、单侧更强邻居、邻域同值、中心局部最大。检查无支持D=0，正差分在求和前，分母严格1+s，支持强弱行为和D上界正确。
3. 固定V只改变Q，验证相似关系会改变D；固定Q改变邻居V，验证强→弱传递及不会从弱邻居向强中心增加D。人工构造两侧差分抵消的例子，确认没有误实现为先平均再ReLU。
4. 在非零W_o副本上核对参考聚合和正式实现的输出、输入/参数梯度；边缘重复padding和循环移位不得悄悄通过。
5. W_o=0 初始输出与父模型对齐；整网训练态比较同步CPU/CUDA RNG、BN和GT/DN。真实hook确认SRE接的是model.19原RepC3完整输出，并沿原连接供LIF和Decoder消费。
6. 至少两次有效更新验证梯度启动，检查W_o先变化、随后W_d/W_q/GN参与训练。梯度验证使用非退化样本；不把无支持样本的自然零梯度误判为错误。

### 8.2 实际训练与生命周期

- 记录Python/PyTorch/CUDA/ultralytics实际导入路径，确保来自当前worktree。核对层索引、原模块哈希、公共状态、参数量与两个变体新初值一致。
- CPU FP32、可用时CUDA FP32/native AMP使用真实检测loss、有效GT并覆盖DN。核心任意尺寸与整网stride对齐输入分开，不要求父模型不支持的整图尺寸。
- 服务器真实train数据、原在线增强、B16/640/native AMP，有限batch内至少2次有效optimizer更新；预先记录预算，报告batch数、GT/DN、loss、scale/backoff、跳步/有效更新、梯度、峰值显存和耗时。预算耗尽仍不稳定则明确失败，不无限重试。
- 原生GradScaler遇缩放梯度Inf可能正常回退，要结合后续有效更新判断。不用amp=False、固定scale=1或降低batch过检。优化器须恰好覆盖每个可训练参数一次。
- 用已学非零SRE状态验证save/reload、EMA、原生resume（optimizer/scaler/epoch/已学参数）、整网fusion与显式CUDA half有限推理。GN不折叠为BN，原RepC3融合不得丢失SRE。仅零W_o等价不算完整生命周期验证。
- half checkpoint比较使用相同量化口径；不把CPU half整网能力作为CUDA训练必过条件。
- 正式start匹配当前执行提交、代码/配置哈希、源/初值哈希、数据身份、variant和PASSED服务器预检。禁止改旧报告status/SHA放行。

### 8.3 防止重现已有故障

**预热：** 检查AutoBackend.warmup的torch.empty，必要时最小修复为torch.zeros并记录。保留真实数值异常检测，不nan_to_num掩盖错误。

**TF32/融合：** 严格FP32诊断可局部关闭matmul/cuDNN TF32，保存并try/finally恢复flags；同时记录默认精度表现。不靠单纯放宽容差或永久改训练全局策略通过。

**离散top-k：** 低精度融合可能改变候选选择，分别检查连续特征、候选与最终输出。固定候选重放只作诊断，不冒充原生输出等价，不改变正式查询选择；NaN/Inf、丢支路、恢复错误必须失败。

**计数：** THOP total_ops兼容int/Tensor，不无条件访问.device/.new_tensor。函数式投影和邻域运算不能漏计或双算；缺项明确PARTIAL。父/目标同nc1、同640、同融合口径。

**回调和退出：** on_train_start不假设trainer.epoch已存在；按真实生命周期记录start_epoch等。训练完成轮数与final_eval状态分开，200轮后验证故障不自动重训。保留pipefail和真实退出码，区分RUNNING、COMPLETED_200、EARLY_STOPPED、INTERRUPTED、FAILED；tmux退出不等于成功。

复用现有AMP检查资源，不擅自升级环境。CUDA非确定性warning如实记录，保留父策略，不为清除warning换算子或配方。

## 9. 召回评估：保留原协议，增加明确的验证集诊断

### 9.1 原正式指标

初始化/预检、plan、start、resume、独立val/test、pack分开。每条CLI用实际--help或一次性检查核对，不能交付猜测参数。

保留原训练验证和best.pt选择规则，不因本模块目标是Recall就改成挑R最高的一轮。中期比较必须同轮、同口径，不混比母版最终test；不自动调参/停止。

此次仅实现评估工具，不执行最终test。训练后固定原规则选出的best.pt、记录SHA256，先独立val，再对同一权重test。沿用 `corrected_sorted_conf_mask_v1`：imgsz=640、batch=16、workers=0、half=False、conf=0.001、iou=0.7、max_det=300、augment=False、rect=False、seed=42。审查原RTDETRValidator的排序/置信度掩码，不添加额外NMS。

保存P/R/AP50/AP75/mAP50–95、逐IoU AP、速度、完整精度JSON、PR/混淆矩阵和曲线。保留原P/R的“各模型最大F1工作点”口径说明，不用新诊断覆盖原指标。

### 9.2 固定Precision下Recall，仅用val做诊断

已核对成功父验证集记录：P=0.8655727028797452、R=0.835202492211838、mAP50=0.8956724997170779、AP75=0.5453045665315482、mAP50–95=0.5245427221919051；P/R来自父模型自身最大F1工作点。父best.pt SHA256为 `24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`。这些是历史参照，不是本次重新评估结果。

为检验“同样精确率下找回更多裂缝”，预先固定：

```text
split = val
matching_IoU = 0.50
precision_target = 0.8656
score_floor = 0.001
R_at_P = max R(t) over realizable thresholds t with P(t) >= precision_target
```

上述matching_IoU是TP/FP判定阈值，不是把正式CLI的iou=0.7改成0.5。父模型与候选必须按同一规则重新计算该诊断；不能直接将历史R=0.8352当作父R_at_P。

实现要求：

- 复用经审查的原验证器一对一匹配口径与有效预测集合，保存协议版本；每个GT至多匹配一次、每个预测至多匹配一次，不能逐GT独立取max IoU而重复计TP。本诊断明确复用原验证器在固定候选池上已经产生的逐预测TP@0.50标记，筛选时不逐阈值重新匹配，不混用两套规则；结果注明为原验证器固定匹配标记的PR诊断。
- 采用相同score_floor/max_det及类别口径，披露候选池已受conf=0.001预截断。扫描原始唯一分数阈值，使用score>=threshold，对整份val预测累计TP/FP；同分预测必须整体纳入/排除，不能先四舍五入或利用同分内部排序取得不存在的阈值工作点。Recall分母是同一整份val的GT数。
- 输出P、R、阈值、TP/FP/FN、GT数、预测数、目标Precision、父/候选checkpoint SHA、val数据身份及精确匹配规则。对达到最大R的并列阈值，固定选择最高Precision，再并列选最高阈值。
- 不用平滑或插值曲线伪造达到目标的真实阈值；无非空预测工作点满足目标时记录NOT_ACHIEVED，阈值和R_at_P均为null，不能用空预测的未定义Precision冒充成功，也不能伪报为有效的零召回工作点。
- 同时记录父/候选阈值分别是多少；这是匹配Precision的PR诊断，不是宣称使用了同一confidence阈值。
- 父预测证据不可用时列PENDING，提供对指定父权重的独立val命令；不能凭汇总JSON编造R_at_P，也不能用训练中的新结构加载父权重后误当父模型。
- 用少量人工匹配/分数案例验证重复匹配、同分和无达标点，不必另写一套与原检测器不同的完整匹配算法。必要的并列反例：GT=2，scores=[0.9,0.8,0.8]、TP=[1,1,0]、目标P=0.8656，正确R_at_P=0.5、阈值0.9；若先计入0.8组中的TP而报告R=1就是错误。

test不用于选择阈值、模块超参数或训练轮数。若后来用户要求部署阈值的test结果，必须使用事先在val确定并冻结的阈值，报告test实际Precision/Recall，不能再在test找满足目标Precision的最优点。原正式AP评估继续原协议。

只有R_at_P变好仍不足以宣布全面成功，还需看AP75、mAP50–95、原指标和新增误检；仅默认R提高、同时P明显下降，不能归因为漏检能力改善。不要写死保证涨点或为了通过验收调指标口径。

## 10. Git、服务器命令与轻量交付

1. 独立worktree完成实现，检查git diff/status，提交本实验代码、两个YAML、工具和必要文档；不混入用户改动、数据、权重或大参考包。
2. 普通push，独立验证远端完整SHA与本地HEAD一致；不强推、不合母版。网络失败如实标PUSH_PENDING，提供相对固定母版的增量bundle，并在一次性仓库实际验证可导入。
3. 最终代码提交确定后绑定报告、生成填入完整40位HEAD的SERVER_COMMANDS.md。固定SHA交接副本可作为独立产物，避免为更新文档反复提交导致SHA失效。
4. 服务器主目录 `/root/autodl-tmp/projects/Crack_RTDETR`；source `/root/miniconda3/etc/profile.d/conda.sh`，使用现有rtdetr环境；PYTHONPATH指向SRE worktree的ultralytics-main。不修改其他环境或正在训练的worktree，不擅自pip升级。
5. 先git cat-file检查目标对象；缺失才fetch指定分支。保持HTTP/1.1、--progress、--no-tags、GIT_TERMINAL_PROMPT=0、http.lowSpeedLimit=1、http.lowSpeedTime=60；单次timeout120秒，最多5次。失败后给已验证bundle导入，不clone覆盖主目录。
6. 创建独立worktree后核验完整HEAD。已有目录检查身份，不rm -rf。每段自行source本实验环境文件，不依赖其他终端变量；代码改变后同步更新环境文件与预检身份。
7. 交付“同步与环境”“初始化与预检”“显式正式start”“resume”“val/test与召回诊断”“轻量打包”短段。每段子bash使用set -Eeuo pipefail，完整闭合heredoc；预检不自动启动训练。
8. 先检查GPU占用，不停止其他实验，不默认并发高显存预检。容量不足记录资源问题，不改B16/640/AMP正式条件。
9. 轻量包包含args/results、初始化/预检/训练状态、原评估及召回诊断JSON、关键日志尾、必要源码/配置、提交和SHA256清单。默认目标<20MiB，排除权重、数据、大型预测和模块包；原始大预测可留服务器，用清单记录来源。pack不触发训练/test、不自动下载失败实验。
10. 交付结构合同、两个YAML、可用工具、公共状态审计、数学/运行报告、父配方与差异表、固定SHA服务器命令、必要的bundle。各CLI实际核验，缺项明确原因和下一步。

最终回复简明列出：分支、母版、本地/远端SHA与推送状态；新增参数 **16,704**；真实通过项与PENDING；交付文件及下一条用户命令。明确 **正式训练 NOT_STARTED，最终test NOT_RUN**，不声称已登录服务器或获得精度提升。

## 附录A：成功 CBR＋LIF 完整训练 args 快照

以下仅作为配方来源，模型与输出身份替换为SRE对应值；不能把父checkpoint路径直接当成新模型结构，不能写入父输出目录。有实质冲突时查明并报告，不无声改配方。

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
