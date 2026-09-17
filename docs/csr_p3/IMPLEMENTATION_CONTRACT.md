# CSR-P3 v1：Codex 完整实施提示词

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

## 2. CSR-P3 专属身份与结构

- 全称：Curved Sampling Residual，曲线采样残差。
- 分支：exp-rtdetr-r18-lite-csr-p3-v1。
- 建议服务器 worktree：/root/autodl-tmp/projects/Crack_RTDETR-csr-p3-v1。
- 主变体：cbr_lif_csr_p3_v1。
- 单模块变体：csr_p3_v1。
- 配置：rtdetr-resnet18-lite-cbr-lif-csr-p3-v1.yaml，以及 rtdetr-resnet18-lite-csr-p3-v1.yaml。
- 主 run name：cbr_lif_csr_p3_v1_rtdetr_r18_lite_e200_b16_onlineaug。
- 输出 project：/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series。
- 新模块建议文件：ultralytics-main/ultralytics/nn/modules/csr_p3.py。

必读参考：模块包 RTDETR-main/ultralytics/nn/extra_modules/dynamic_snake_conv.py 中 DSConv/DSC，以及 block.py 的 DySample 坐标约定；当前项目原 Conv、parse_model 和 fuse 路径。原论文 https://arxiv.org/abs/2307.08388 仅提供细长曲折结构自适应采样的动机；其分割损失和完整效果不属于本实验。

不得直接复制包内DSConv：已发现 offset.detach().clone() 与 range(1,center) 组合，在默认k=3时循环为空，偏移采样的梯度连接被切断；较大核也有端点遗漏问题。使用可微分cumsum和统一grid_sample重新实现。不要引入第三方CUDA扩展。

### 2.1 固定位置与核心层

输入L为原model.17的1×1侧向投影输出，B×256×H×W，640时H=W=80。输出L'=L+delta在同一位置交给原Concat18；不改model.19的RepC3，不改高层上采样支路。

    Z = W_i(L)                         # B,32,H,W
    o = offset_pw(SiLU(offset_dw(Z))) # B,12,H,W
    increments = tanh(o.float())
    curved_grid = four_independent_cumsums(increments)
    D = weighted(curved_samples - straight_samples)
    L_new = L + W_o(D)

固定层：
- W_i：Conv2d(256,32,1,bias=False)。
- offset_dw：Conv2d(32,32,3,padding=1,groups=32,bias=True)。
- 激活：SiLU，非inplace。
- offset_pw：Conv2d(32,12,1,bias=True)。
- theta：可学习Tensor，shape[2,32,7]；沿最后的7点维softmax。
- W_o：Conv2d(32,256,1,bias=False)。
- 没有额外BN/GN、可学习残差缩放或gate，也没有额外loss。

初始化：
- 只有offset_pw的weight和bias全零。
- theta全零，因此实际聚合权重是非零均匀1/7。
- W_i、offset_dw、W_o使用显式普通非零初始化，建议xavier_uniform_；offset_dw.bias=0。
- 特别注意W_o不能置零，聚合权重不能全部为零。
- 新层RNG隔离，两个变体的.csr初值一致。

准确新增参数：
8192 + (288+32) + (384+12) + 448 + 8192 = 17,548。
注意参数量不代表采样速度；不得按普通卷积计数直接宣称整网开销。

### 2.2 曲线与直线采样的精确定义

所有坐标先用P3特征图像素坐标，p=(x,y)，整数位置代表像素中心。K=7，r∈[-3,-2,-1,0,1,2,3]。

将o重排为[B,2,2,3,H,W]，顺序固定：
- axis=0：水平链；axis=1：垂直链；
- side=0：负侧；side=1：正侧；
- step=0,1,2：距中心1、2、3步。

各轴各侧独立从中心向外累加：
d[a,s,m] = sum(i=1..m, tanh(o[a,s,i]))。

- 水平链：q_x,r=(x+r, y+d[x,side(r),abs(r)])。
- 垂直链：q_y,r=(x+d[y,side(r),abs(r)], y+r)。
- r=0时两个链都等于p，不使用偏移。
- 负侧的增量自身有正负值，不再额外乘一个负号。
- 每一步正交方向偏移绝对值小于1，最外端累计绝对值小于3。
- 用可微分cumsum；从自然顺序[距离1,2,3]生成负侧后，在最终r顺序中排列成[距离3,2,1]。
- 直线参考q0使用同一坐标定义，仅令全部偏移为0；不能直接拿另一个池化/卷积分支冒充直线参考。

令S(Z,q)为双线性采样。每通道聚合：
D_c(p) = 0.5 * sum_a sum_r softmax_r(theta[a,c,r]) *
         [S(Z,q_a,r)_c - S(Z,q0_a,r)_c]。

两个采样使用同一个Z、同一组聚合权重和相同采样实现。中心差分恒为0，但其softmax权重通过分母调节其他点的总贡献，因此保留7点theta；不要换成7个互不归一化的自由中心权重。

### 2.3 坐标、精度和内存

- 归一化x：u=2*(x+0.5)/W-1；归一化y：v=2*(y+0.5)/H-1。
- grid最后维顺序为(x,y)，width用于x、height用于y，不能正方形测试通过后就忽略矩形。
- 两路都用mode=bilinear、padding_mode=border、align_corners=False。
- 偏移预测的最后投影、tanh/cumsum、网格、softmax、采样及差分累加在局部FP32下计算。必要时对半精度参数做可微分float视图，不可替换Parameter。
- 差分进入W_o及最终相加时明确dtype转换，返回dtype与原L一致；不全局关闭AMP。
- 曲线偏移和采样特征不能detach。固定参考网格可以不求梯度，但直线采样的Z不能detach，否则初始公共输入梯度的两路抵消被破坏。
- 优先按轴、采样点或小块累加，避免把完整输入重复成[B,C,2,7,H,W]甚至更大张量。必须测真实B16/640训练峰值显存；不要用较小batch替代。
- 固定坐标网格不注册成随H/W变化而进入state_dict的buffer；可按实际device/size创建，缓存如使用需明确dtype/device/shape失效规则且不改变结果。

### 2.4 集成：保留原Conv的状态与融合行为

推荐CSRConv继承原Conv，原构造参数保持，包括model.17原本act=False：
- 原conv/bn/act的状态键完整保留；不包成新的parent子模块。
- forward先执行原Conv，再执行self.csr(L)。
- 新增状态只出现在model.17.csr.*。
- model.17仍from=5，输出256；Concat18、RepC3 19、LIF20、decoder26及其from都不变。
- parse_model以原Conv方式推断通道，并加上明确的新模块参数，不能顺手改激活/卷积stride。
- 两个配置都在相同侧向位置插入CSR；单模块版是原C2＋CSR，没有CBR/LIF。

建议model.17配置为（其余原行保持）：

    [5, 1, CSRConv, [256, 1, 1, None, 1, 1, False, 32, 7]]

解析器先补入c1=ch[5]，其后保留原Conv的c2/k/s/p/g/d/act参数，最后两个参数分别是detail_channels=32、kernel_points=7。确认不把最后的7错误传成原Conv的膨胀率。

必须显式覆盖forward_fuse，并在其中先算原融合Conv再执行CSR。Ultralytics可能在fuse时把forward重绑到forward_fuse；如果继承了原Conv.forward_fuse而没覆盖，推理时可能静默丢掉新支路。需要非零偏移验证此项，不能只看零初始化输出。

## 3. CSR 专属验收

1. 用手工坐标/水平斜坡/垂直斜坡/单点特征核对采样轴、正负侧点序、中心固定、累计幅度、矩形归一化和边界。不是只检查shape。
2. offset_pw=0时，曲线和直线网格相同，D=0，模块输出及整网输出与父模型一致；同时比较公共输入梯度。给出具体精度容差。
3. 首次真实loss反向在非退化输入上检查offset_pw获得有限有效梯度。W_i、offset_dw、theta、W_o首步可能没有任务梯度，这是预期分阶段启动，不是失败。
4. 一次性副本执行多个有效更新，检查偏移开始变化、D变为非零，其余新层逐步获得梯度。不能为了让所有层首步非零而改掉初始化，也不能把scale=1诊断当正式AMP通过。
5. 不要求常量/平坦输入也有非零偏移梯度；应对多张有效真实输入做适当检查，定位到具体层。
6. 给偏移设已知小的非零值核对曲线采样差分，再用学习后非零状态完成checkpoint/EMA/half/整网fusion及forward_fuse验证，防止融合时丢支路。
7. 未融合nc=1主组合预期20,167,313参数，单模块20,100,320；在父计数确认一致时，融合后主组合19,962,513、单模块19,895,264。不同口径分别报告。
8. 额外记录有限的偏移绝对值/累计端点分位数、接近tanh饱和比例和残差相对L的幅度。不要把可视化采样曲线自动解释为已识别真实裂缝。
9. 不宣称具有拓扑连续性保证，不增加分割标注/拓扑loss，不把模型退化简单归因于训练轮数不足。

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
