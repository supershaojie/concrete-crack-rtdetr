# LSRT v1：Codex 完整执行提示词

请在用户的混凝土裂缝检测 RT-DETR-R18-Lite 项目中直接完成本任务。先完整阅读本文和真实项目指令，再检查仓库、实现、验证并交付。本文完整保存已确定的 LSRT 方案及本轮实现规格，不依赖聊天上下文，也不依赖另一份 RCS-Q 提示词。

本次完成源码、组合与单模块配置、受控初始化、必要预检、训练/评估/轻量打包入口、独立分支提交推送与服务器交接。暂不执行正式训练、最终 test、远程服务器同步。不只返回建议，不同时实现或合入 RCS-Q。

## 1. 目标、已定方案与实验身份

LSRT = Local Semantic Residual Transport，中文“局部语义残差输运”。它针对的问题是：细位置已有低层细节，但直接上采样的高层语义可能没有对应到合适的细位置。

用户已确定的结构核心必须保留：

1. 仅放在 CCFM 自顶向下的 P4→P3 融合连接。
2. 低层侧向特征 L 提供细位置的内容依据；上游 P4 尺度中间特征 H=Y4 提供语义。不是最终 PAN P4，也不是 Decoder 输出。
3. 每个 P3 细位置匹配其原对应粗位置附近 3×3、共9个高层候选位置；Q/K 压到32维。
4. 只输运相对原中心的高层差异：Δv_p = Σ_j a_pj Wv(H_j−H_c)。低层特征坐标不动。
5. U′ = 原 Upsample(H) + Wo Δv，再原样 Concat(U′,L) 和 RepC3。
6. 仅最后 Wo 零初始化，使新模型从原融合行为开始学习。

本提示词将原方案尚未写死的工程细节固定为：单头32维、余弦相似度、固定逆温度 τ=8、9个可学习相对位置偏置初值为0、所有1×1投影无bias、严格2倍最近邻中心对应、图外邻居显式屏蔽。它们是本轮实现取值，不是已验证的最优超参数，不自动搜索多组取值。

首轮主实验必须是：

    原 RT-DETR-R18-Lite + original CBR + 原 LIF-Down + LSRT

另准备原 R18＋LSRT 单模块配置，但本次不训练。RCS-Q 是另一个独立候选，不能放进这个模型；NBR-G/SFR-D 的骨干压缩亦不采用。本次保留 R18 结构和原 ImageNet 初始化，不承担减参10%的任务，不冻结骨干。

互补目标：LSRT改善向细尺度传递语义时的局部对应；LIF处理后续下采样的细节；CBR细化最终框。必须同时说明：LSRT改变融合后的P3，LIF和CBR都依赖它，组合是否改善需要实验验证。不能保证单模块或三模块涨点，不能把“有注意力/残差”直接写成论文创新证据。

## 2. 独立工作树与代码基点

先读取实际 AGENTS.md，记录 git status --short、分支、完整HEAD、remote和git worktree list。保护用户未提交内容及其他实验；不reset/clean/stash、不force push、不改写历史、不删除已有工作树。

建议独立分支 exp-rtdetr-r18-lite-lsrt-v1，工作树 Crack_RTDETR-lsrt-v1。同名存在时先核对用途和状态，不覆盖。

基点采用已经验证的 original CBR＋LIF-Down 正式源码，保留适用的既有融合、初始化和训练入口修复。不要从含RCS-Q/NBR-G/SFR-D的模型继承，亦不要直接合并整个其他实验分支。

历史定位线索如下，均需实际核对，不能当作当前HEAD直接套用：

- C2：67c3078e54a657fd96d65fee657a75fbb1dae0d6。
- original C19/CBR：025997e3c51eaf6933534308a95da6ebf97bff53。
- LIF：0e95bbade3558b0d2b77c5531483c60810391d88。
- 成功组合启动初始化记录：a0459d6a652cb702699087c88fa39a3e4c4087ec；不单独证明最终训练/test代码版本。
- Windows项目历史路径：D:/MyProjects/Crack_RTDETR。
- 服务器主仓库：/root/autodl-tmp/projects/Crack_RTDETR。
- 历史组合工作树：/root/autodl-tmp/projects/Crack_RTDETR-c19-lif-v1-gatefix。

从真实组合YAML、CBR/LIF源码、Git历史和运行记录确认基点，记录base SHA及选择理由。可以最小复用其他分支已经核验的通用工程修复，但不能带入它们的模型、初始化变换、专属限制或训练配方。

## 3. 实际阅读项目与模块包

正式项目至少阅读：

- 原R18-Lite、LIF单模块、original CBR及正式CBR＋LIF YAML。
- ultralytics-main/ultralytics/nn/tasks.py：parse_model、save列表、前向路由及模型融合。
- ultralytics-main/ultralytics/nn/modules/conv.py、block.py：Concat、Conv、RepC3。
- ultralytics-main/ultralytics/nn/modules/cbr.py、lif_down.py、head.py、transformer.py。
- tools/init_c19_lif_v1.py、train_c19_lif_v1.py、preflight_c19_lif_v1.py、check_c19_lif_v1.py、sync_c19_lif_v1.sh，及相关已验证修复。按实际仓库定位，不假设旧文件一定存在。
- 原成功组合的完整C2 args、data、初始化和预检记录。

用户本次参考模块包：

- 名称：d562592a-72d7-4703-8ea6-1e958de67292.zip。
- SHA256：b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc。
- 对话工作区路径：/workspace/scratch/00484fe92169/upload/d562592a-72d7-4703-8ea6-1e958de67292.zip。
- 先前审计定位的用户电脑路径：D:/7.1 rtdetr改/魔鬼面具_RTDETR/RTDETR-20260623.zip；哈希与本次包相同，是定位线索，不保证当前仍在。

桌面Codex通常不能访问对话路径。先在当前项目、上述已知位置及合理相邻目录用rg定位，不扫描整盘。相同SHA的旧文件名可以作为同一包。记录真实路径、SHA和实际阅读的类；不可见时完成可依据项目源码做的工作，将参考核对标为待完成，不虚构已阅读。

包内重点：RTDETR-main/ultralytics/nn/extra_modules/block.py 中 DySample、SFC_G2、CARAFE。

| 参考对象 | 必须保持的区别 |
| --- | --- |
| DySample | LSRT用低层Q与高层K匹配固定粗邻域，不由高层单独预测连续offset |
| SFC_G2 | 不同时重采样两路特征，不移动L，只修正高层注入支路 |
| CARAFE | 不直接复制内容重组模块；采用低层引导匹配和相对原中心的值差分 |
| 历史CSCEF | 不加Scharr、跨尺度一致性门控或整套历史融合模块 |
| LIF-Down | 不重复Haar、Predict/Update或下采样模块 |

背景研究：[FaPN](https://arxiv.org/abs/2108.07058)讨论高低层特征对齐；[FreqFusion](https://arxiv.org/abs/2408.12879)研究融合中的特征一致性和边界问题。这些支持问题动机，不证明本项目的瓶颈或本模块必然有效。准确归属，不声称首次特征对齐、首次局部attention或首次残差。

不直接import整个extra_modules，不新增MMCV/DCN/Mamba/CUDA扩展，不复制另一套Ultralytics。

## 4. 精确接入：保留索引、原Concat操作和RepC3

实际正式组合若与下列拓扑一致，优先采用这一最小接法；先核真实YAML，不把LIF-only YAML误当成组合YAML。

| 层 | 原含义 | 本次处理 |
| --- | --- | --- |
| 15 | 上游P4中间特征Y4，256×40×40 | 保留，作为H |
| 16 | Y4的原2倍nearest上采样，256×80×80 | 保留，作为原U |
| 17 | backbone S3的原1×1投影，256×80×80 | 保留，作为L |
| 18 | Concat([16,17])，512×80×80 | 同索引换为薄适配器LSRTConcat([16,17,15]) |
| 19 | 原RepC3输出P3，256×80×80 | 原封不动保留 |
| 20 | 组合模型中的原LIFDown | 保留，输入仍来自19 |
| 21–25 | 原bottom-up PAN路径 | 保留 |
| 26 | 组合中的original RTDETRDecoderCBR，from=[19,22,25] | 保留 |

建议新增 ultralytics-main/ultralytics/nn/modules/lsrt.py：

- LocalSemanticResidualTransport：forward(L,H)返回高分辨率256通道修正量delta。
- LSRTConcat：forward([U,L,H])执行 torch.cat((U + delta.to(U.dtype), L), dim=1)。它内部可使用原Concat(1)完成拼接；不能换顺序、改成相加，或删掉原融合。
- 新参数路径建议 model.18.lsrt.*，内部投影命名q_proj/k_proj/v_proj/out_proj和rel_bias。

保留layer16原Upsample，薄适配器只在拼接前插入高层修正。因此原Concat的运算语义保留，但YAML中第18层类名改变。不得在文档中误写成“YAML完全不动”。

建议YAML仅在18层改为：

```yaml
- [[16, 17, 15], 1, LSRTConcat, [32, 8.0]]  # d=32, fixed tau=8
```

为LSRTConcat显式注册导入与parser分支。构造参数由parser提供真实三路通道，可采用LSRTConcat([c_u,c_l,c_h], d=32, tau=8.0)。本轮断言三者均256，输出通道c2=c_u+c_l=512，不能因from有三项而错误sum成768。不要把该类注册成通用卷积模块导致额外width/depth缩放或重复实例。核对save列表确实保存15、16、17。

这一接法保持后续层号及所有原可训练层的state_dict路径。若实际项目层号不同，按上述真实拓扑映射并记录，不强行采用历史数字；优先同索引替换参数为空的融合节点。

不能把H接到最终PAN P4（通常22），这会把后续特征接回上游并形成错误依赖；不能取backbone原S4代替Y4。不得detach L/H或原地修改U/L/H。低层坐标和L的前向张量保持原样，训练梯度可因新支路而变化。

## 5. 唯一数学定义与张量结构

本轮只做单头局部匹配：C=256、d=32、3×3邻域、stride关系严格2。输入L和U为[B,256,2h,2w]，H为[B,256,h,w]。

```python
q_proj   = nn.Conv2d(256, 32, 1, bias=False)
k_proj   = nn.Conv2d(256, 32, 1, bias=False)
v_proj   = nn.Conv2d(256, 32, 1, bias=False)
out_proj = nn.Conv2d(32, 256, 1, bias=False)
rel_bias = nn.Parameter(torch.zeros(9))
tau = 8.0  # 固定常量，不是可训练参数
```

不加BN、LN、激活函数、额外门控、Dropout、DropPath、null槽、多头、额外位置MLP、连续offset或新loss。RCS-Q的空槽/区域均值设计不能挪进本模块。

对细位置p=(y,x)，与原nearest×2一致的粗中心是：

    c(p) = (floor(y/2), floor(x/2))

邻域按row-major顺序取(-1,-1),(-1,0),…,(1,1)，中心index=4。同一个粗中心对应的2×2细位置共享候选集合，但各自的q来自不同L_p，所以权重可以不同。不得先把L平均下采样后令这四个细位置共用一组attention。

投影及余弦匹配：

    Q_p = q_proj(L)_p
    K_j = k_proj(H)_j
    V_j = v_proj(H)_j
    qhat_p = Q_p / max(||Q_p||_2, 1e-6)
    khat_j = K_j / max(||K_j||_2, 1e-6)
    score_pj = 8 * dot(qhat_p, khat_j) + rel_bias[j]
    a_pj = softmax_j(masked_score_pj)

这里τ是余弦logit的乘数（逆温度），不要又除sqrt(32)，也不要写成softmax(score/8)。Q/K做通道L2归一化，V保持原投影值，不对V归一化。

值差分与输出：

    Δv_p = Σ_j a_pj * (V_j - V_c(p))
    delta_p = out_proj(Δv)_p
    U′_p = U_p + delta_p
    P3 = 原RepC3(原Concat(U′, L))

因为Wv线性且无bias，V_j−V_c与Wv(H_j−H_c)一致。必须以原nearest中心为参照，不换成区域均值，也不改成只输出加权高层特征。

尽量显式先计算V_j−V_c再加权，保持中心差分精确为0；避免先加权后相减引入不必要的抵消误差。中心投影值必须从相同V张量取得，不另跑随机变换。

## 6. 邻域边界、实现效率与性质

在粗尺度先投影H得到32通道K/V，再用unfold/gather提取3×3邻域。无效的图外邻居score设为-inf，在softmax前屏蔽。中心始终有效；即使H只有1×1，仍至少有中心。可以unfold全1掩码获得有效性；固定几何索引/buffer应正确随设备移动，不缓存上个batch的特征。

不能将零padding当成真实候选参与softmax，不能clamp图外邻居索引让边界样本重复参与竞争。无效value差分用where显式置零后聚合。输入真实NaN/Inf应诊断报错，不用nan_to_num隐藏。有限输入下softmax和梯度必须有限。

保持原nearest×2，不改成bilinear或nearest-exact。若输入U/L与H不满足2倍空间关系，报出明确形状错误；不偷偷resize修补原网络的错误。矩形的合法2倍尺寸需要支持。推理640对应h=w=40，不硬编码80×80。

只展开32通道粗K/V；禁止在256通道高分辨率特征上展开9份邻域。不要用Python循环逐个处理6400个像素。可用向量化unfold/gather，或等价的2×2相位重排避免重复展开K/V；优化必须与朴素参考公式核对，不能改变四个细位置独立匹配的含义。

必须准确解释并检查：

- 邻域高层特征相同时，所有值差分为0，修正为0，包括图像边缘。
- 选择中心的one-hot权重时，修正为0。这个性质可直接测试聚合函数，不要求有限softmax恰好产生one-hot。
- 低层L为常量不必导致修正为0，不要写这个错误验收条件。
- 注意力权重之和为1，但U′因有Wo变换和残差，不保证是H的凸组合；“输运”不等于最优传输或严格质量守恒。
- 不强制复制原中心，也不强制去掉中心；中心是可选的“保持原对应”候选。

## 7. 初始化、梯度与公共权重保护

仅首次建立新实验初值时：q_proj/k_proj/v_proj使用Xavier uniform，rel_bias全0，out_proj全0。没有第二个零总门控，不能同时清零内部特征来关闭支路。

因此起点delta=0、U′=U；在公共状态相同条件下，新单模块对应原基线，新组合对应原CBR＋LIF，应保持初始前向等价。

首个反传要求out_proj有有限非零梯度（用非均匀输入与非退化测试损失）。out_proj为0时，q/k/v/rel_bias首步梯度为0是预期。用独立测试副本做少量诊断更新，或设置受控非零out_proj，再核对上游参数梯度有效。

诊断副本不能保存成正式初值，正式训练优化器更新次数必须为0。不要在forward、load、EMA、fuse、resume或eval中重复清零out_proj；非零状态保存重载后必须保留。注意项目全局初始化可能覆盖新模块初值，检查最终真实模型而非只检查构造函数。

新增层构造必须隔离RNG或用共同参考状态迁移，避免多抽随机数改变后续RepC3、CBR、类别头等公共初值。仅写seed=42不是公共权重一致证据。

## 8. 配置、源权重与真实Trainer

| variant | 结构 | 本次用途 |
| --- | --- | --- |
| cbr_lif_lsrt_v1 | 原R18＋original CBR＋原LIF＋LSRT | 首轮主候选，准备完整服务器训练入口 |
| lsrt_v1 | 原R18基线＋LSRT | 单模块消融，准备配置、初始化与预检 |

对照使用原组合和原基线，不做带冻结/乘零LSRT参数的假对照。两两组合可后续补，本次不自动开展八组训练。

lsrt_v1必须从真正的原基线生成：其layer20仍是原Conv，Decoder仍是原RTDETRDecoder，不能保留LIF或CBR再称为单模块。cbr_lif_lsrt_v1则从真实组合生成，保留LIFDown和RTDETRDecoderCBR。

固定源：rtdetr_r18_lite_imagenet_backbone_init.pt；SHA256：

    fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e

历史路径：D:/MyProjects/Crack_RTDETR/weights/ 或 /root/autodl-tmp/projects/Crack_RTDETR/weights/ 下的该文件。

不从已训练best/last、RCS-Q或压缩骨干权重初始化。原三层3×3 stem不是直接复制torchvision的7×7 stem，不虚构“骨干每层完全照搬”。

沿用原源nc=80→实际任务nc=1的类别适配时间与规则，列出精确shape变化白名单，其他missing/unexpected逐项处理，不用宽泛strict=False掩盖漏载。

必须核对：

- 两个新模型相对各自原模型的所有公共参数和BN buffers逐值一致，层号和state_dict键保留。
- 两个新配置的LSRT初值逐值一致，新增键仅在计划的LSRT节点。
- original CBR和LIF结构、零初始化与融合保护不变；组合的Decoder确实是CBR版本。
- 真正进入Trainer的nc=1模型再次核对，不能只比较事先生成的nc=80文件。
- 新参数在构建AdamW之前注册，全部requires_grad参数恰好进入优化器一次。
- Decoder查询选择、300 normal queries、3层Decoder、DN数量/噪声/掩码、loss/matching和返回结构全部继承原模型。LSRT不修改DN生成器。

本方案不需要factors.pt/factors.json、通道索引或新的外部预训练。在服务器用固定源权重和Git源码生成初值即可。缺源时完成可完成的源码和合成检查，将受控初始化核验准确标为待源权重。

## 9. 原训练配方保持一致

以成功CBR＋LIF所用完整C2 args为事实来源，读取并输出逐字段差异。当前已核验核心值如下；不要用早期150e/lr0=0.01/关闭在线增强的旧记忆，也不要混入YOLO26的batch32/MuSGD。

```text
epochs=200, patience=50, imgsz=640, batch=16
seed=42, workers=8, device=0, deterministic=True
optimizer=AdamW, lr0=0.0005, lrf=0.01
momentum=0.937, weight_decay=0.0001
warmup_epochs=5, warmup_momentum=0.8, warmup_bias_lr=0.1
cos_lr=True, amp=True, cache=False, close_mosaic=10
hsv_h=0.015, hsv_s=0.5, hsv_v=0.35
degrees=5, translate=0.1, scale=0.4, shear=1.5
perspective=0.0002, flipud=0.2, fliplr=0.5
mosaic=0.8, mixup=0.05, copy_paste=0, erasing=0
```

其余参数从完整权威args继承，不只复制这个子集。允许变化限于模型身份、实验名、输出目录及核验后的环境路径。数据split、类名/nc、图像/标签指纹沿用正式对照，不重划分、不多种子/CV/holdout、不改增强。

建议主run名：cbr_lif_lsrt_v1_rtdetr_r18_lite_e200_b16_onlineaug。训练、预检、test、审计分目录，exist_ok=False，不覆盖已有结果或自动name2/name3。

## 10. 精度策略和既有故障防护

沿用现有环境，已知服务器Python3.10、PyTorch2.1.2+cu121、4090，最终记录实际版本。不升级torch/ultralytics，不关正式AMP掩盖错误。

为明确首版行为，新增LSRT小分支整体使用局部FP32：四个投影、L2归一化、dot/logits、softmax、值差分和聚合在autocast(enabled=False)中计算，最后delta转换成U.dtype相加。采用函数式conv2d及输入/参数的可微float转换，兼容常规模型、native AMP及half推理模型；不能detach参数，也不能在forward里module.float()/half()修改参数对象。模型其余部分保持原native AMP策略。

CPU预检用FP32，不在旧torch启用CPU float16 autocast。CUDA严格FP32融合诊断关闭matmul/cudnn TF32并记录/恢复状态；保留项目既定绝对/相对容差，不能仅因失败就放宽或跳过。

LIF两支相加后共同BN的融合保护必须继承已验证修复，不能只折叠base分支。新LSRT没有BN，不需要发明新的重参数化；原生fuse不能删除LSRT或重置其out_proj。

复用现有ensure_amp_resources或同等已验证资源准备，核验实际入口使用的参考权重及ultralytics/assets/bus.jpg，避免启动卡下载/缺图。不得monkeypatch check_amp=lambda:True。

沿用原deterministic=True和既有warn_only行为，记录实际警告；不声称含原CBR/LIF采样算子的整网CUDA逐bit确定。

## 11. 必要预检与验收

预检报告记录variant、实际源码指纹、配置/源权重/初始化哈希、设备、环境和每项状态。缺CUDA/真实数据时标PENDING，完成可完成内容并提供服务器补检命令，不把skip当pass。

### A. 图与初始化

- 两配置正确构建；主配置CBR/LIF/LSRT都存在，R18未改，无RCS/NBR/SFR遗留。
- 18层输入顺序为[U,L,H]，输出512通道，拼接前256通道为U′、后256通道为原L；19/20/26及Decoder from保持原语义。
- 新增参数精确32,777，公共state_dict逐值一致；真实nc=1 Trainer和optimizer覆盖正确。
- out_proj=0时与对应原模型初始输出等价；训练模式如有DN随机性须独立副本和同一RNG状态，防止BN更新/随机噪声伪差异。

### B. 数学与梯度

- 用小型可辨认坐标特征核对nearest中心floor(y/2),floor(x/2)、3×3顺序、中心index4，确认同一粗中心的2×2细位置仍有独立Q。
- 非方形合法尺寸、h或w=1、边角邻域都正常；shape关系错误应明确失败。
- 常量非零H在非零out_proj下全图零修正，包括边角；不要只测Wo=0的无效条件。
- 聚合函数one-hot中心为零，选非中心时得到对应V差分；无效邻居权重为零，有效权重和为1。
- 非均匀L/H和受控参数下输出随局部匹配改变；核对实现与朴素参考公式及梯度。
- 首步Wo梯度非零；隔离副本少量诊断更新后q/k/v/rel_bias可学习。全局共同平移bias不改变softmax，不能要求9个bias方向都可辨识。
- forward不原地修改共享输入、不新增detach；对H/L的梯度能沿原路和新路传播。

### C. 重载、融合与真实容量

- 零与受控非零out_proj都覆盖checkpoint/state_dict重载、EMA、eval和原生融合，确认已学习参数不被清零。
- CPU FP32 smoke；CUDA FP32、native AMP的整网前向、原loss与反向；项目使用half推理时补模块与整网对应检查。
- 正式容量检查必须真实训练数据、完整主组合、imgsz640、batch16、AMP、原DN，至少一次完整前向/反向，不做正式optimizer.step。记录GT和DN数量、峰值显存。
- 可用GT较多的真实train批次补检动态DN容量；不能从test挑样本。CPU/小图/batch2不能替代B16/640容量通过。
- OOM真实报告；不自动减batch/imgsz/queries、关AMP或改配方。不要通过更换运行数据掩盖容量问题。

训练入口正式更新次数保持0；将测试副本诊断更新单独标注。达到这些必要验收后停止无关扩测。

## 12. 复杂度和轻量诊断

本规格新增参数：

    4 × (256 × 32) + 9 = 32,777 ≈ 0.032777M

640输入、P3=80×80、H=40×40时，主要计算核对：

| 算子 | MAC |
| --- | ---: |
| Q投影：256→32，80×80 | 52,428,800 |
| K投影：256→32，40×40 | 13,107,200 |
| V投影：256→32，40×40 | 13,107,200 |
| 输出投影：32→256，80×80 | 52,428,800 |
| 9邻居×32维匹配点积 | 1,843,200 |
| 9邻居×32维值聚合 | 1,843,200 |
| 合计 | 134,758,400 |

按2×MAC约0.2695168 GFLOPs；不包含全部L2归一化、softmax、mask、差分、gather及内存搬运，不能直接充当整网实测算量或速度。FP32局部分支和内存访问会影响实际延迟。

在同nc=1、640、同工具/模式下报告原组合与新组合、原基线与单模块；未融合/融合参数分别列清，不混用口径。函数式投影若不被THOP自动hook识别，应实现明确的统计规则并避免重复计数，不可把漏计结果当真实额外开销。

诊断入口只保存汇总：中心attention权重、邻域权重/熵、delta与U范数比、Wo范数及关键梯度。默认正式训练关闭大量诊断，不保存逐批完整特征或巨量attention图。

## 13. 训练、评估与轻量打包工具

复用原c19_lif成熟入口，任务专用脚本作为薄包装，建议：

- tools/init_lsrt_v1.py：固定源→两配置初值、公共权重审计。
- tools/preflight_lsrt_v1.py：明确variant、CPU/CUDA、真实容量与报告路径。
- tools/train_lsrt_v1.py：plan/start分离，plan不占用正式训练目录，主variant默认cbr_lif_lsrt_v1。
- tools/sync_lsrt_v1.sh：保护已有worktree，指定分支及完整交付SHA。
- test入口：以后从真实best.pt/原EMA协议执行split=test，沿用原conf/iou/max_det及FP32/AMP等评估设置；保存P/R/AP50/AP75/AP50:95、原协议速度、PR/混淆矩阵及配置。本次只准备，不执行。
- 打包入口：train/test分目录收集CSV、args、metrics、曲线、必要源码和Git/初始化/预检/参数审计，生成manifest及SHA256；默认不打包数据集、参考模块包、best/last或巨量逐图预测。

入口名称可按实际项目习惯，但最终文档里的命令必须对应真实文件且--help可用，不能虚构命令。

start检查预检对应的variant、真实源码/配置/源/初始化指纹一致，拒绝缺失或失败预检；不能手工改报告状态。resume单独处理，保留非零已训练LSRT，不重新生成零初值或要求resume模型仍与baseline初始等价。

训练状态区分200轮完成、patience早停、异常、OOM、手工中断，不因为进程退出就记为完成。未经后续用户指令，本次不运行正式start/test。

## 14. 服务器交接和Git交付

不自动登录服务器，不更改任何正在运行的RCS-Q或其他实验。复用既有Conda rtdetr、tmux和同步方式，交接命令从新终端可顺序执行：

1. 固定完整40位交付SHA；先查本地Git对象，缺失才fetch指定分支。
2. fetch沿用HTTP/1.1、--progress、--no-tags、GIT_TERMINAL_PROMPT=0、lowSpeedLimit=1、lowSpeedTime=60，单次timeout120秒，最多5次，不抓全仓、不无限重试。
3. 网络失败可用精简Git bundle导入同一SHA，核对base；不覆盖已有分支/工作树。
4. 先创建新worktree并确认HEAD等于交付SHA，再创建其weights/outputs目录，不能让用户先找一个还不存在的工作树路径。
5. 激活现有rtdetr环境，设置新worktree的PYTHONPATH，检查真实import路径、torch和GPU。
6. 核对固定源权重与数据，用源码在服务器生成初值，不要求上传新的factors文件。
7. 处理AMP资源，执行服务器补检并生成plan。
8. 最后单独提供tmux正式start命令，主variant为cbr_lif_lsrt_v1。单模块命令另列，不默认同时启动所有配置。

每段显式设置所需任务变量或source同一env文件，不依赖另一个终端残留变量。正确引用路径，set -o pipefail避免tee吞失败。双任务并行需按实际空闲显存核对，不自动杀其他进程，不自行改batch。

建议docs/lsrt_v1/README.md、SERVER_HANDOFF.md、reference_audit.json；运行报告放被忽略的outputs/lsrt_v1。README应包含本文原始机制与固定规格，后续可从文件恢复完整方案。

完成必要验证后依次git status、检查diff、只add本任务文件、commit、push独立分支并核对远端SHA。沿用既有授权流程；遇到权限/网络限制，完成其余工作并真实报告，不绕过限制。

Git只提交源码和必要小文本，不上传权重/数据/参考包。最终交付目录可含DELIVERY_SHA.txt、SERVER_HANDOFF_DELIVERED.md、同步脚本和小bundle；实际最终HEAD写入提交外生成的交付文件，避免“提交包含自己最终SHA”的循环。

最终简洁汇报：

- 分支、base SHA、最终完整HEAD、push/远端核验状态。
- 首轮为CBR＋LIF-Down＋LSRT，单模块仅准备，RCS-Q不包含在内。
- 实际改动层、原R18/CBR/LIF保留情况、新增参数及算量口径。
- 已通过/失败/PENDING检查及原因；正式训练NOT_STARTED，最终test NOT_RUN。
- 服务器交接文件与完整可复制命令位置。

执行至可复核、可同步、可在服务器按顺序补检和启动的状态。
