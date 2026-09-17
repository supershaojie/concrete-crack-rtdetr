# SDB-P3 v1：Codex 完整实施提示词

版本：2026-09-17。本文可独立交给 Codex 执行，不依赖另一份方案文档。

## 任务边界与执行方式

你正在用户的混凝土裂缝 RT-DETR 项目中实施一项结构实验。请直接完成代码实现、受控初始化工具、可用环境中的预检、独立分支提交与推送核验，并交付实际可执行的服务器命令。不要只给建议、示例代码或待办清单。

本次不登录服务器，不启动正式 200 轮训练，不运行最终 test。真实服务器预检由用户执行；本机缺 GPU、数据或统一源权重时，完成可完成的工作并将对应项目标记 PENDING，不能伪造 PASSED。普通实现、修复、创建独立工作树和本实验分支提交推送属于本次执行范围；不得强推、删除其他实验或改动用户未提交工作。

这是针对已成功的“原 CBR＋原 LIF-Down”增加第三项的独立实验。同时准备单模块消融配置，但首轮服务器命令默认只预检和训练三模块主组合。不要混入 SFR-D、NBRG、RCS-Q、LSRT、BSC-Rep、TRC 或本轮另一个新模块。不要重新设计 CBR/LIF、替换 R18 骨干、修改 Decoder、检测损失或训练增强。

本文件定义实现合同；名字不构成新颖性证明，预检通过不代表精度提升，不承诺一定涨点。

## 1. 先核查项目、母版与模块包

### 1.1 项目与母版

1. 读取当前项目适用的 AGENTS.md；只读检查 git status、remote、当前 HEAD、已有 worktree，以及相关初始化/训练/验证工具。
2. 目标项目为 concrete-crack-rtdetr，已知远端为 https://github.com/supershaojie/concrete-crack-rtdetr。核对实际 remote 后使用，不要进入另一个 YOLO26 项目。
3. 固定成功组合的代码基点：
   - a0459d6a652cb702699087c88fa39a3e4c4087ec
   - 主组合模型配置：rtdetr-resnet18-lite-cbr-lif-down.yaml
   - 原 R18-Lite 单模块消融母版：rtdetr-resnet18-lite.yaml
4. 用 git 对象和真实 YAML/源码确认母版，不从最近失败的分支 HEAD 直接堆叠。在指定基点派生独立分支和独立 worktree；同名分支/目录已存在时先核对，不能 reset 覆盖。
5. 已知基点原模块 LF 归一化 SHA256（读取源码核验，CRLF 转 LF 后计算）：
   - lif_down.py：26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7
   - cbr.py：d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787
   不修改这两个文件来“兼容”新模块。遇到基点/哈希不符，查明版本关系并报告，不擅自采用同名的其他版本。

### 1.2 模块包来源

用户本次附件：
- 文件名：dc2eda42-6ea1-467d-ab90-02d26882d4d0.zip
- SHA256：b91dbaf5f74e35ff23079fe0854140e7fd82207c21455b564abc594636aba2cc
- ChatGPT 本轮可见路径：/workspace/scratch/00484fe92169/upload/dc2eda42-6ea1-467d-ab90-02d26882d4d0.zip
- 包内根目录：RTDETR-main

上面的 /workspace 路径不保证在用户的 Windows/Codex 主机存在。优先检查用户附件和当前项目已保存的模块包，按文件名或包内路径有界搜索；记录实际读取的绝对路径和 SHA256。同版本旧附件可作为参考，但必须核对内容。确实找不到时明确记录“模块包本机不可见”，按本文已给出的数学合同继续可完成的实现，不能声称读过文件。

模块包只供机制、接口和坐标实现参考；不要执行包里的训练/安装脚本，不要整体替换项目 ultralytics，也不要把第三方包全部提交到仓库。本次模块对应的必读文件在后续专属章节列出。

### 1.3 已核对的原组合图

640 输入时，下面是原语义节点，写代码前以真实 YAML 再核对一次：

| 原节点 | 内容 | 输出 |
|---|---|---|
| model.4 | R18 P2 | 64×160×160 |
| model.5 | R18 P3 | 128×80×80 |
| model.6 | R18 P4 | 256×40×40 |
| model.7 | R18 P5 | 512×20×20 |
| model.9 | 原 AIFI | 256×20×20 |
| model.15 | 自顶向下中间特征 Y4 | 256×40×40 |
| model.17 | P3 侧向 1×1 投影，原 act=False | 256×80×80 |
| model.18 | 上采样特征与侧向 P3 的 Concat | 512×80×80 |
| model.19 | 原 RepC3，内部 n=3、e=0.5 | 256×80×80 |
| model.20 | 原 LIFDown | 256×40×40 |
| model.22 | 最终颈部 P4 | 256×40×40 |
| model.25 | 最终颈部 P5 | 256×20×20 |
| model.26 | 原 RTDETRDecoderCBR，from=[19,22,25] | 检测输出 |

注意 Y4 是中间特征，不是最终 P4。保留 decoder hidden_dim=256、num_queries=300、3 层、eval_idx=2，保留 CBR rho=normal_fraction=0.10。单模块消融从 C2 母版派生，其下采样和 decoder 使用原基线版本，不带 LIF/CBR。

## 2. SDB-P3 专属身份与结构

- 全称：Semantic-guided Detail Bypass，语义引导细节旁路。
- 分支：exp-rtdetr-r18-lite-sdb-p3-v1。
- 建议服务器 worktree：/root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1。
- 主变体：cbr_lif_sdb_p3_v1。
- 单模块变体：sdb_p3_v1。
- 配置：rtdetr-resnet18-lite-cbr-lif-sdb-p3-v1.yaml，以及 rtdetr-resnet18-lite-sdb-p3-v1.yaml。
- 主 run name：cbr_lif_sdb_p3_v1_rtdetr_r18_lite_e200_b16_onlineaug。
- 输出 project：/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series。
- 新模块建议文件：ultralytics-main/ultralytics/nn/modules/sdb_p3.py。

必读参考：模块包 RTDETR-main/ultralytics/nn/extra_modules/block.py 中 SPDConv；当前项目原 RepC3、parse_model、LIFDown、RTDETRDecoderCBR。原论文 https://arxiv.org/abs/2208.03641 仅作空间重排依据，不把它当作本设计涨点证明。

设计问题：给现有 P3 增加一个此前没有直接进入检测颈部的 P2 信息来源，再用已有 P3 语义决定补充强度。没有额外 P2 检测头，不改变 decoder 的三尺度输入。

### 2.1 核心模块的精确定义

输入顺序固定为核心 SDBP3.forward(p2, original_p3)：
- X2=p2，B×64×160×160；
- T3=original_p3，B×256×80×80；
- detail_channels=32，GN groups=4，eps=1e-5，affine=True。

以下为定义用伪代码，不是要求复制某个未经验证的实现：

    S = cat([
        X2[..., 0::2, 0::2],
        X2[..., 1::2, 0::2],
        X2[..., 0::2, 1::2],
        X2[..., 1::2, 1::2],
    ], dim=1)                         # phase-major 00,10,01,11

    D = SiLU(GN_D(DW3(W_d(S))))        # B,32,H3,W3
    Q = GN_Q(W_q(T3))                 # B,32,H3,W3
    g = sigmoid(W_g(cat(D,Q,D*Q)))    # B,32,H3,W3
    delta = W_o(g*D)                  # B,256,H3,W3
    T3_new = T3 + delta

固定层配置：
- W_d：Conv2d(256,32,1,bias=False)。
- DW3：Conv2d(32,32,3,stride=1,padding=1,groups=32,bias=False)。
- GN_D、GN_Q：GroupNorm(4,32,eps=1e-5,affine=True)。
- W_q：Conv2d(256,32,1,bias=False)。
- W_g：Conv2d(96,32,1,bias=True)。
- W_o：Conv2d(32,256,1,bias=False)。
- 不增加 BN、dropout、额外 learnable scale、位置编码、边缘/Haar 支路、辅助监督或 detach。

初始化：
- W_o.weight 全零。
- 其他卷积权重用固定的普通非零初始化，建议显式 xavier_uniform_；W_g.bias=0。
- GN weight=1、bias=0。
- 隔离新增层的随机数消耗；两个变体新模块状态相同。
- 不能再叠加一个零初始化标量乘在 W_o 外面，否则可能锁死支路。

准确新增参数：
W_d：8192；DW3：288；两个GN：128；W_q：8192；W_g：3104；W_o：8192。合计28,096。
640 输入主要卷积 MACs=178,790,400；按2 FLOPs/MAC为0.3575808 GFLOPs，不含GN/激活/逐元素等开销。以上不是实测整网 FLOPs。

### 2.2 与原图连接：保持层号和公共参数路径

为减少层号迁移，推荐在 model.19 用 SDBRepC3 继承原 RepC3；这只是封装接线，不修改原 RepC3 计算：

    def forward(self, inputs):
        fusion_input, p2 = inputs
        original_p3 = super().forward(fusion_input)
        return self.sdb(p2, original_p3)

- model.19 的 from 从 -1 改成 [18,4]，其原输入通道仍为 model.18 的512。
- model.19 原 cv1/cv2/m/cv3 参数路径保留，不包进新的 parent 子模块。
- 新增状态仅放在 model.19.sdb.*。
- model.19 内部仍是原3次 RepConv、e=0.5；返回256通道。
- model.20、model.22、model.25、model.26 层号及 from 保持，head26仍从[19,22,25]取输入。因此修改后的 P3 同时进入原 LIF 和原 decoder。
- core SDB 在语义上确实位于原 RepC3 输出之后，不是先修改 concat 输入再做原 RepC3。
- 不另加一个只能被 Decoder 消费而绕开 LIF 的 P3 支路。

建议将model.19这一行写为（其他行保留）：

    [[18, 4], 3, SDBRepC3, [256, 0.5, 32]]

可采用构造接口SDBRepC3(c1,c2,n=3,e=0.5,p2_channels=64,detail_channels=32)。解析器将c1=ch[18]、c2=256、内部n=3、e=0.5、p2_channels=ch[4]、detail_channels=32传入；实际通道以母版核对为准。它内部先构造原RepC3，再在隔离RNG作用域构造新支路。

parse_model 专门处理该双输入模块：
- 分别读 ch[f[0]] 和 ch[f[1]]；不能将列表直接 ch[f]。
- 原 depth-scaled n 作为内部 repeat 只传一次，外部 repeat 置1；否则会错误地串联多个双输入包装器。
- P2 的 model.4 必须进入保存列表以便此处访问，禁止 detach 或偷偷换来源。
- 推断输出通道为256，验证下游所有尺度及通道一致。
- 两个新 YAML 只作本模块规定的更改；单模块版的父结构是原 C2，不包含 CBR/LIF。

### 2.3 空间尺寸合同

- 四相位顺序明确为00、10、01、11；torch.pixel_unshuffle 默认排列不能未经重排就替代。
- 支持640方图和父模型支持的stride对齐矩形。
- 核心模块接受奇数P2尺寸时，只在右/下各补最多1像素，固定replicate padding，随后重排；要求T3尺寸恰好是ceil(H2/2)、ceil(W2/2)。
- 不满足上述关系时给出清晰尺寸错误，不允许插值偷偷补救接线。
- 进行随机父/目标比较时对齐CPU/CUDA RNG与BN状态；新模块不引入随机forward操作。
- 父模型本身不支持的任意奇数整图不必扩展；分别报告核心模块奇数单测和整网对齐矩形测试。

## 3. SDB 专属验收

1. 使用手工编号的小张量逐元素核查四相位次序、矩形及右/下padding，不能只看shape。
2. 原路径状态全部相同且W_o=0时，core输出等于T3，整网输出与各自父模型一致；报告CPU/CUDA/AMP对应误差与容差。
3. 首次检测loss反向，W_o应在非退化输入上获得有效梯度；D/Q/gate等上游新层零梯度是此时的预期行为。
4. 在一次性副本更新W_o后，检查W_d、DW3、W_q、W_g逐步获得有限梯度，不要求每个标量每个batch非零。P2在原骨干路径本就有梯度，不能仅凭P2.grad非零证明新旁路连通：在W_o非零的核心副本中，以独立可求导P2和固定T3，针对新增delta单独核查P2梯度；此隔离诊断不改变正式forward。
5. 构造W_o已非零的副本，比较禁用补充分支与正常分支确有输出差异；该禁用仅用于诊断，不进入正式结构或额外搜索变体。
6. 保存重载、EMA及整网fusion使用非零分支复检。GN不能被当作BN融合；保留原Conv/RepConv融合及LIF原有保护，不丢.sdb。
7. 未融合nc=1主组合参数应为20,177,861，单模块应为20,110,868；若原父计数确认一致，融合后预期主组合19,973,061、单模块19,905,812。逐项计算验证，不能把混合口径写成结果。
8. 解释性输出最多保留少量门控分布和补充残差相对T3幅度统计，必要时在同一验证checkpoint临时关闭支路观察影响。它们不能替代完整训练消融，也不能把gate称为前景真值。

## 4. 受控初始化：比较的是结构，不是追加训练

优先复用并审查基点 tools/init_c19_lif_v1.py、tools/init_lif_down.py 及既有 nc=1 Trainer 加载流程。不要因为名字相似就复用失败实验的模型配置或已训练 checkpoint。

### 4.1 统一源

服务器统一未训练源：
- /root/autodl-tmp/projects/Crack_RTDETR/weights/rtdetr_r18_lite_imagenet_backbone_init.pt
- SHA256：fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e
- 既有源模型 nc=80、epoch=-1；不带已训练 optimizer/EMA/scaler 状态。
- 源对应的历史 C2 提交：67c3078e54a657fd96d65fee657a75fbb1dae0d6。

不能使用 CBR＋LIF 的 best.pt/last.pt 初始化新实验，也不要求用户上传额外预训练权重或 factors 文件。源文件在本机缺失时报告该项 PENDING，服务器命令引用现有文件。

### 4.2 完整公共状态与新支路

- 构建受控 C2 父模型及原 CBR＋LIF 父模型，再向两个目标变体逐键严格加载共有状态。
- 不能只迁移 backbone、只看 transferred 数量，或依赖“都 seed=42”推断共有初始化一致。
- 所有原有参数和 buffers 必须逐键、逐形状、逐值相等；本实验公共路径不应发生编号迁移。
- 新模块构造使用隔离的 CPU RNG 状态；不要消耗后续原 head 的构造随机数。主组合和单模块的新支路初值逐项一致。
- 输出 COMMON、NEW_TRAINABLE、NEW_BUFFER、MISSING、UNEXPECTED、SHAPE_MISMATCH 和允许的分类适配清单。除规定的 nc 分类适配外，缺失/多余/形状不符应为零。
- 初值只在新模型构造时设置。forward、load_state_dict、resume、验证和融合均不得重置新模块、CBR 或 LIF。
- 基点父模型 verify_model 可能要求精确旧拓扑，不要把目标强行当成未改动父模型验证；新增目标拓扑验证器，并复用原模块源码合同。

### 4.3 真实 nc=1 Trainer 重建

必须通过实际 RTDETRTrainer.get_model 及 model.train 的模型加载路径核验，而非只测试 RTDETR(checkpoint)：

- 固定数据 nc=1。
- 对 nc=80→1 的原生分类层适配明确列出允许的 9 个状态项：denoising_class_embed、enc_score_head 的 weight/bias、3 层 dec_score_head 的 weight/bias。
- 上述9项只是相对于nc=80源的形状适配例外，不是允许两个nc=1模型初始化不一致。目标与对应父模型经匹配RNG的实际Trainer重建后，这9项nc=1张量也必须逐值相等。
- 其余所有共有张量与对应受控父模型完全一致；原 CBR/LIF 和新模块初值不能被随机重建覆盖。
- 父模型参考构造需放在隔离 RNG 作用域，不能改变实际目标的原生分类适配随机状态。
- 保存受控初值后立即重载核对。正式 start 使用该初值；resume 只加载实际续训 checkpoint，并保留已学习的新支路状态。

## 5. 正式训练配方：继承这轮成功组合的 200 轮设置

本轮权威对照是成功 CBR＋LIF 的 200e/onlineaug 配方。更早“150e、lr0=0.01、关闭在线增强”的旧记录不适用于本轮。

读取成功组合的完整 args.yaml，随实验存档并生成逐字段差异表。已核对关键值如下：

| 参数 | 固定值 |
|---|---|
| epochs / patience | 200 / 50 |
| imgsz / batch / nbs | 640 / 16 / 64 |
| seed / workers / device | 42 / 8 / 0 |
| optimizer | AdamW |
| lr0 / lrf | 0.0005 / 0.01 |
| momentum / weight_decay | 0.937 / 0.0001 |
| warmup_epochs / warmup_momentum / warmup_bias_lr | 5 / 0.8 / 0.1 |
| cos_lr / deterministic / amp | True / True / True |
| cache / rect / multi_scale | False / False / 0.0 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.5 / 0.35 |
| degrees / translate / scale / shear | 5 / 0.1 / 0.4 / 1.5 |
| perspective / flipud / fliplr | 0.0002 / 0.2 / 0.5 |
| mosaic / mixup / close_mosaic | 0.8 / 0.05 / 10 |
| cutmix / copy_paste / erasing / bgr | 0 / 0 / 0 / 0 |
| auto_augment | null |
| fraction / freeze / compile | 1.0 / null / False |

损失、matcher、DN 及其余全部字段继承父实验。仅允许模型标识、输出路径/name、已核实等价的数据绝对路径发生变化。不要依赖升级后的库默认值，不要顺手换优化器、增强、查询数或训练策略。

服务器数据：
/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml

检查与父实验相同的数据配置和划分/标签清单。当前记录 train=6048、val=1728、test=864，类别数=1；数量只是交叉核对，不替代内容身份核验。不得重新划分或生成新数据。

正式 run 使用指定唯一目录，exist_ok=False；若已有结果，保留并明确要求 start/resume 语义，不能产生 name2 后还当原实验，也不能删除失败目录重试。

## 6. 有实际意义的预检，重点堵住过去的故障

预检在独立临时输出和一次性模型副本上执行。临时优化器更新不算正式训练；受控初值文件和正式输出目录不得被改变。缺失硬件/数据分别标记 PENDING，而不是把小输入的通过等同服务器 B16/640 通过。

### 6.1 必须完成的检查

1. 核对实际导入的 ultralytics 和新模块绝对路径，必须来自本 worktree。
2. 核对拓扑、层索引、原模块源码及共有初始化；两个变体的新增状态一致。
3. 在方形与 stride 对齐矩形上检查模块和整网，覆盖 train/eval 两条路径；形状检查之外完成专属章节的数学/梯度检查。
4. 初始整网等价优先在eval固定输入比较；若比较train/DN输出、loss或公共输入梯度，使用同一有效batch、相同CPU/CUDA RNG及一致BN状态，不能把两次独立DN噪声误判为结构差异。使用真实检测 loss 和有效 GT，包含原 DN 路径，做 CPU FP32、可用时 CUDA FP32/native AMP 检查。不要用 output.sum() 代替所有真实 loss 验证。
5. 使用真实数据和原在线增强做服务器 B16/640/native AMP 前后向与少量真实 optimizer steps；保留父配方，并确认至少 2 次有效更新、所有新参数被优化器恰好覆盖一次。
6. 正确理解各模块分阶段梯度：初始零梯度不必然是断路。更新一次性副本后检查新分支启动；禁止为“通过梯度检查”改变规定初始化。
7. 保存重载、EMA、恢复训练和整网融合均验证。尤其在新分支已经非零时比较融合前后；初始恒等状态通过不足以证明融合正确。
8. 覆盖显式 model.half() 推理；局部必要的 FP32 运算要兼容已半精度化参数，不得 forward 中替换 Parameter 或改变 dtype 身份。
9. 报告实际 torch/CUDA/cuDNN/ultralytics、参数量、复杂度口径、峰值显存和耗时。分析估算与实测分开。
10. 预检失败输出错误、traceback、关键张量/梯度与阶段；不能吞异常后写 PASSED。

### 6.2 已出现过的问题必须提前处理

- warmup：检查当前 AutoBackend.warmup 是否用 torch.empty 作为输入。在 deterministic 环境下可能包含非有限值；若存在，做最小修复为确定的有限 torch.zeros，单独记录并提交。不要删除模型有限值检查或用 nan_to_num 掩盖问题。
- CPU AMP：CPU FP32 检查不要调用 dtype=float16 的 CPU autocast；按实际设备选择路径，不能全局关闭正式 CUDA AMP。
- GradScaler：缩放后的梯度出现 Inf 不应立即等同模型失败；按原生动态 loss scaling、unscale、step/update 路径诊断，记录 scale 与跳过/有效更新。禁止把 AMP=False 或固定 scale=1 偷换成正式配方来通过。
- AMP 资源：复用服务器已有兼容的本地检查权重和 assets/bus.jpg；缺失时给出具体资源准备命令。不要因网络失败自动 pip upgrade ultralytics 或跳过原生 AMP 检查。
- THOP：custom hook 的 total_ops 可能是 int 或 Tensor。显式兼容实际类型，不无条件调用 .device/.new_tensor；小型计数单测覆盖该问题，避免重复/漏计底层卷积。
- 生命周期：on_train_start 不能假定 trainer.epoch 已存在。按 start_epoch 和实际生命周期记数；覆盖全新启动和续训。修复日志代码时不能改变训练行为。
- tmux：worker 保存真实退出码，set -o pipefail 保留训练错误；结束状态区分 COMPLETED_200、EARLY_STOPPED、FAILED、INTERRUPTED。不能把 [exited] 当成功。
- 环境：使用已有 rtdetr Conda 环境和 worktree PYTHONPATH，不修改其他 worktree，不强制卸载/升级依赖。
- CUDA 非确定性 warning 单独记录；不为了消除 warning 改变父实验 deterministic 策略。
- 若正式训练最后 final_eval 出错，状态应同时保留“已完成训练轮数”和“最终验证失败”，不要把已有 200 轮权重当不存在或自动重训。

预检报告绑定当前代码提交及必要源码/配置哈希、源权重/初值哈希、数据身份、variant 和精度设置。修复提交后重新运行受影响的检查；不得修改旧报告的 status/commit 字段伪装匹配。所有影响执行的修复应在最终服务器预检前完成。

### 6.3 复杂度的统一口径

同一 nc=1、640 输入、同一融合状态、同一统计工具下，报告父模型与新模型及增量。融合前和融合后分别列出，不能混用。

既有参照（用于交叉核对，不作为本次新测量）：
- 原 CBR＋LIF：未融合 20,149,765 参数；融合后 19,944,965 参数，历史日志约 58.7 GFLOPs。
- 原 C2：未融合 20,082,772 参数；融合后 19,877,716 参数，历史约 58.3099248 GFLOPs。

新模块不因 BN 融合而丢失；若数量差异与公式不符，应解释和检查，不得改公式只为对齐数字。没有现成算子计数时给出明确的分析补项及计数范围，禁止漏掉 grid_sample 等开销后宣称完整 GFLOPs。

## 7. 训练、验证和打包工具

优先复用已核查的生命周期工具；只增加本实验必要的入口和校验。CLI 名称和参数由实际实现决定，文档中的每条命令必须用 --help 或一次性预检实证，不能交付想象的参数。

需要覆盖：
- 统一环境与指定提交同步；
- 受控初始化；
- 本地和服务器预检；
- plan/start/resume；
- 独立 val/test；
- 轻量结果包。

正式训练期间在 40/80 轮输出与父实验同期 val 记录的观察信息即可，不额外修改学习率计划，也不新增自动淘汰规则；用户决定是否止损，原 patience=50 保持。不得拿新模型早期结果直接对比父模型第 200 轮后宣称成功/失败。

独立评估固定训练验证选出的同一 best.pt，遵守已有修正排序后置信度掩码协议 corrected_sorted_conf_mask_v1：
- imgsz=640，batch=16，workers=0，half=False；
- conf=0.001，iou=0.7，max_det=300；
- augment=False，rect=False，seed=42；
- 原 RTDETRValidator/既有修正 postprocess，无额外 NMS、阈值优化或 test 选模型。
- 先独立 val，再对同一 checkpoint/hash 做 test；训练中禁止 test。
- 输出 P、R、AP50、AP75、mAP50–95、逐 IoU AP 和速度；保存完整精度 JSON，不只保留终端三位小数。

轻量包应包含训练 args/results/曲线、val/test 指标及 PR/混淆矩阵、初始化/预检/训练状态、关键日志尾、实际源码/配置、SHA256 清单和提交信息。明确排除数据集、权重、巨大预测和参考压缩包；默认目标小于 20 MiB，超过时列明大文件供处理，不静默丢关键证据。如果打包 allowlist 排除 .log，应将有限日志尾另存 .txt 纳入证据。不自动打包下载失败实验。

## 8. Git 与服务器交接

1. 保留用户当前工作区和其他分支。只提交本实验实现、必要测试、配置与文档；权重、数据、实际结果和大模块包不入 Git。
2. 提交前检查 diff/status；必要修复完成后形成清晰提交。push 到本实验独立分支，不合并到成功母版、不 force push。
3. push 后独立核对远端分支 SHA 与本地 HEAD 相同；网络失败如实记录，提供基于成功母版的小型增量 git bundle 并实际验证可导入，不能宣称已推送。
4. 服务器工作根：/root/autodl-tmp/projects/Crack_RTDETR；Conda：/root/miniconda3/etc/profile.d/conda.sh，环境 rtdetr。
5. 提供固定完整 HEAD SHA 的服务器同步指令：先 git cat-file 检查对象；缺失才 fetch 指定分支，HTTP/1.1、--progress、--no-tags、GIT_TERMINAL_PROMPT=0、http.lowSpeedLimit=1、http.lowSpeedTime=60，单次 timeout 120 秒、最多 5 次。仍失败提供验证 bundle 后导入的替代命令，不 clone 覆盖主项目。
6. 创建独立 worktree，核验 HEAD；已有目录先核验，不能 rm -rf。每段命令都自行 source 本实验环境文件，不能依赖另一个终端的变量。
7. 服务器命令分成“同步与环境”“初始化与预检”“显式正式 start”“resume”“val/test”“轻量打包”几个短段。每段用子 bash 包裹 set -Eeuo pipefail，保证错误不会关闭交互终端。
8. 每段 heredoc 必须完整包含结尾标记，不能出现上一轮 python tools/train 这种截断命令；不把会大量占显存的两个预检默认同时启动。
9. start 必须核对匹配的 PASSED 服务器预检、初值、数据和当前代码身份；不能自动降低 batch/imgsz/AMP 逃过容量检查。
10. 最终答复列出：分支、母版 SHA、最终本地/远端 SHA、新增参数、真实通过项、PENDING 项、交付文件路径和下一条用户命令；明确“正式训练 NOT_STARTED，test NOT_RUN”。

请持续完成上述已授权工作。不要把待实现项包装成完成，不要声称在未访问的服务器上执行过命令。

## 附录 A：已核对的成功组合完整 args 快照

以下为原成功 CBR＋LIF 训练参数记录，供本机缺少原结果目录时核查，不是新实验直接照抄的模型/输出路径。生成新实验参数时仅替换允许的身份字段；同时检查本地项目中成功父实验的记录，若有实质冲突应指出具体字段，不能无声改配方。保存完整快照和差异表。

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
