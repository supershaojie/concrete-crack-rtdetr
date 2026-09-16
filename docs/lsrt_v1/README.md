# LSRT v1：局部语义残差输运

首轮主实验为 **原 RT-DETR-R18-Lite + original CBR + 原 LIF-Down + LSRT**。另准备原 R18-Lite + LSRT 单模块消融；本轮不启动正式训练或最终 test，不含 RCS-Q、NBR-G、SFR-D，不冻结或压缩 R18 骨干。

完整且可恢复的用户设计规格保存在 [DESIGN_SPEC.md](DESIGN_SPEC.md)，该文件逐字复制原始 `Codex_LSRT_v1_Full_Prompt.md`。本页解释实现契约和入口；历史证据与实际核验哈希见 [reference_audit.json](reference_audit.json)。当前 LSRT 检查结论应以新实验生成的报告为准，不能使用历史组合 PASS 替代。

## 机制与精确接入

LSRT 只修正 CCFM 自顶向下 P4→P3 的高层注入。低层侧向特征 L 为细位置提供内容依据，上游 P4 中间特征 H=Y4 提供语义。它保持低层坐标以及原最近邻上采样、拼接顺序和 RepC3。

| 层 | 语义 | 处理 |
|---|---|---|
| 15 | 上游 P4 中间特征 Y4，即 H | 保留；不能改用最终 PAN P4 的22层或 backbone S4 |
| 16 | 原 nearest×2 上采样 U | 保留 |
| 17 | backbone S3 的原1×1投影 L | 保留 |
| 18 | 原无参数 Concat | 同索引替换为 `LSRTConcat([U,L,H])`，输入索引 `[16,17,15]` |
| 19 | 原 RepC3 输出 P3 | 保留 |
| 20 | 原组合 LIFDown；单模块为原 Conv | 保留各自原模型语义 |
| 21–25 | 原 bottom-up PAN | 保留 |
| 26 | 原组合 RTDETRDecoderCBR；单模块为 RTDETRDecoder | `from=[19,22,25]`、300 normal queries、3层 Decoder 及原 DN/loss/matching 均保留 |

YAML 仅替换18层融合节点：

```yaml
- [[16, 17, 15], 1, LSRTConcat, [32, 8.0]]
```

parser 传入真实三路通道 `[256,256,256]`；输出通道是 `256+256=512`，不是三路求和768。save列表必须保留15/16/17。原 Concat 的运算语义仍存在，但 YAML 的层类名确实改变。新增参数路径仅 `model.18.lsrt.*`，后续层号及公共状态键不变。

## 唯一数学定义

输入 `L,U ∈ R[B,256,2h,2w]`、`H ∈ R[B,256,h,w]`。严格2倍关系，允许矩形和 `h=1` 或 `w=1`；非法形状明确报错，不隐式 resize。

```text
d = 32；单头；固定逆温度 tau = 8.0
q_proj:   Conv1x1(256,32,bias=False)
k_proj:   Conv1x1(256,32,bias=False)
v_proj:   Conv1x1(256,32,bias=False)
out_proj: Conv1x1(32,256,bias=False)
rel_bias: 9个可学习参数，初值全0

p = (y,x)
c(p) = (floor(y/2), floor(x/2))
候选顺序 = (-1,-1),(-1,0),..., (1,1)，中心index=4
Q_p = q_proj(L)_p
K_j = k_proj(H)_j
V_j = v_proj(H)_j
qhat = Q / max(||Q||_2, 1e-6)
khat = K / max(||K||_2, 1e-6)
score_pj = 8 * dot(qhat_p,khat_j) + rel_bias[j]
a_pj = softmax_j(masked_score_pj)
DeltaV_p = sum_j a_pj * (V_j - V_c(p))
delta_p = out_proj(DeltaV)_p
U'_p = U_p + delta_p.to(U.dtype)
P3 = 原RepC3(原Concat(U',L))
```

tau 是余弦logit乘数，不再除 `sqrt(32)`，也不除8。只归一化Q/K，不归一化V。差分在加权前显式计算，中心取自同一投影V，中心差分精确为0。同粗位置对应的2×2细位置共享候选集合，但各自具有独立Q，不能将L下采样后共享一组权重。

只在粗尺度的32通道K/V上 unfold 3×3候选，细尺度Q用2×2相位重排匹配；不展开256通道高分辨率特征，不用逐像素Python循环。图外候选在softmax前设为 `-inf`，无效value差分用 `where` 置零；不把padding当候选，不clamp重复边界。中心始终有效，包括1×1粗特征。

整个新分支在局部 `autocast(enabled=False)` 下用FP32计算，函数式 `conv2d` 与可微 `.float()` 转换保留梯度和参数对象，兼容native AMP和half推理。最后delta才转换为U.dtype。禁止在forward里 `.float()`/`.half()` 修改模块，禁止detach、原地修改共享输入或 `nan_to_num` 隐藏真实NaN/Inf。整网其余部分沿用原AMP。

不增加BN/LN、激活、门控、Dropout/DropPath、null槽、多头、连续offset、额外位置MLP、新loss或外部扩展。

## 性质、初始化与公共状态

- 高层H在邻域内相同时，全部有效值差分为0，非零Wo也得到全图零修正，包括边角。中心one-hot聚合也为0；非中心one-hot应返回对应中心差分。
- L恒定不要求修正为0。注意力归一化不意味着经过Wo和残差后的U'是H凸组合，也不意味着最优传输或严格质量守恒。
- 仅新建实验时q/k/v使用Xavier uniform，rel_bias为0，out_proj为0，没有第二个零总门控。新节点构造隔离RNG，防止后续RepC3、CBR和分类头初值漂移；还要逐值核验公共参数及BN buffers。
- 起点delta=0，与对应原模型初始前向等价。首步Wo应有有限非零梯度；此时q/k/v/rel_bias梯度为0是预期。用隔离副本受控激活Wo或诊断更新后验证上游可学习。诊断副本不能成为正式初值，正式optimizer更新次数必须为0。
- load、EMA、fuse、resume、eval不重复清零Wo。零及非零Wo均需重载和融合验证。原LIF两路相加后的共同BN保护保持不变。
- 源nc80到任务nc1沿原Trainer构造/加载时序适配，白名单恰为 `model.26.denoising_class_embed.weight`、`enc_score_head.{weight,bias}` 和三层 `dec_score_head.i.{weight,bias}` 共9键；其余公共状态必须逐值一致，新增参数全部恰好进入AdamW一次。

## 基点、参考和限制

基点为 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。这是正式original CBR+LIF源码，保留既有融合、受控初始化、FP16候选临界值诊断和训练入口修复。组合归档的启动/初始化/预检/打包元数据均指向该SHA；十个关键源码与配置的Git blob逐项一致，结果CSV到200轮。证据范围与陈旧运行状态的限制记录在reference audit中，不将单独初始化记录等同完整训练版本证明。

固定源 `rtdetr_r18_lite_imagenet_backbone_init.pt`：

```text
SHA256 fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e
```

只使用此源，不使用训练后的best/last或其他实验权重，不需要factors、通道索引或额外预训练。保留原ImageNet初始化；原三层3×3 stem不能描述为直接照搬torchvision的7×7 stem。

参考包实读路径为 `D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip`，SHA256 `b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc`，与用户指定包相同。DySample预测连续offset；SFC_G2重采样两路；CARAFE重组内容。LSRT分别以固定粗邻域的低层Q/高层K匹配、固定L坐标和中心相对值差分与它们区分。不导入整个extra_modules，不引入MMCV/DCN/Mamba/CUDA扩展，不复制另一套Ultralytics。不引入历史CSCEF的Scharr/一致性门控，也不重复LIF的Haar/Predict/Update。

[FaPN](https://arxiv.org/abs/2108.07058)支持高低层特征对齐动机；[FreqFusion](https://arxiv.org/abs/2408.12879)支持融合中类内一致性和边界问题的动机。它们不能证明本项目存在同一瓶颈、LSRT必然涨点或首次对齐/attention/残差创新。LSRT改变融合后P3，LIF和CBR也依赖P3，因此三模块组合改善需要实验验证。

## 配方和数据

完整109字段配方直接继承 `docs/c19_lif_v1/c2_args.yaml`，不是只复制核心子集。已核验归档args SHA256为 `ab0594ac3758b53421dc0a5adbea693505fc9e2591d8e668a25ba950d70234fd`。核心为200轮、patience50、640、B16、seed42、workers8、device0、AdamW、lr0=.0005、lrf=.01、momentum=.937、weight_decay=.0001、5轮warmup、cos_lr、AMP和deterministic=True。在线增强保持原值。允许变化仅模型身份、run名、输出目录和核验后的环境路径。

原数据train/val/test为6048/1728/864图，45573/12840/6663框，单类crack。原清单只有路径和标签内容指纹，不足以证明历史全部图像字节相同；新plan、preflight和start需核验真实图像及标签指纹一致，不重新划分数据。测试图像不得用于挑选容量批次。

## 入口及验收

以下命令在已同步工作树根目录、现有rtdetr环境中执行。新终端的完整环境设置、交付SHA与服务器顺序命令以 `SERVER_HANDOFF.md` 及提交外生成的交付文件为准。

```bash
python tools/init_lsrt_v1.py --help
python tools/preflight_lsrt_v1.py --help
python tools/train_lsrt_v1.py --help
python tools/check_lsrt_module.py --help
```

初始化接受 `--variant {cbr_lif_lsrt_v1,lsrt_v1} --source PATH --output PATH --report PATH`，分别生成两个受控nc80初值并检查真实Trainer nc1适配。已有初值不得覆盖。

预检接受 `--variant ... --source PATH --initialized PATH --output NEW_DIRECTORY --device cpu|cuda [--data DATA_YAML] [--capacity]`。本地CPUFP32可完成的检查与服务器PENDING分开报告。只有CUDA、真实训练数据、完整相应模型、B16/640/native AMP/原DN完整前后向才可确认容量，记录GT/DN和峰值显存，不执行正式optimizer.step；OOM必须如实报告，不减batch/imgsz/query或关闭AMP。

训练工具提供 `plan/start/resume/resources/inventory/status`。`plan`接受 `--variant --source --initialized --data --output`，可指定 `--project`、`--c2-args`、`--baseline-inventory`；输出完整逐字段diff，不创建正式run。`resources --main MAIN_REPOSITORY --output NEW_JSON` 使用原生AMP参考权重和bus.jpg准备流程，不绕过check_amp。`start --plan PLAN_JSON --preflight REPORT_JSON --output NEW_LAUNCH_DIRECTORY` 是后续获授权才运行的正式入口，校验variant、源码/配置/源/初值/真实数据指纹和已通过服务器预检。`resume`另需 `--checkpoint --previous-launch`，保留已学习状态。

零/非零Wo的checkpoint/state_dict/EMA/eval/fuse、公共状态、形状/边界/数学/梯度、nc1真实Trainer、AdamW覆盖、CPUFP32、CUDAFP32/AMP/half均纳入必要验收。严格FP32融合诊断关闭并恢复TF32，沿原容差，不因失败放宽。deterministic warn_only的实际警告应记录；不宣称采样算子CUDA反向逐bit确定。正式训练默认不保存大批特征图，仅可选汇总注意力/熵/delta与U范数比/Wo范数/关键梯度。

新报告置于被忽略的 `outputs/lsrt_v1`。训练状态区分200轮完成、patience早停、异常、OOM和手工中断，不能仅由进程退出判定完成。本次交付正式训练为 `NOT_STARTED`，最终test为 `NOT_RUN`。

## 参数和算量口径

新增参数精确 `4×256×32+9=32,777`。640输入时，Q和Wo各52,428,800 MAC，K和V各13,107,200 MAC，9邻居点积与聚合各1,843,200 MAC，主要计算合计134,758,400 MAC，即按2×MAC为0.2695168 GFLOPs。

上述数值不含全部归一化、softmax、mask、差分和内存搬运，不是整网实测算量或速度。整网对照须同nc1、640、同工具/模式，并区分融合前后。函数式投影使用显式THOP统计规则，避免漏计或重复计数；局部FP32与内存访问的延迟影响需实测。
