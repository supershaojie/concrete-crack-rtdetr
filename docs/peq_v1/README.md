# PEQ v1：精修证据质量校准

本实验在指定母版 RT-DETR-R18-Lite＋原 LIF-Down＋原 CBR 上增加 PEQ。新增分支读取同一 query 的 CBR 精修前后局部视觉证据，只校准最终评分。当前交付是本地实现和验证；服务器 B16/640 预检、200 轮正式训练、新实验独立 val/test 均未执行，不能把工程 PASS 当作涨点证据。

## 实验身份

- 母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
- 分支：`exp-rtdetr-r18-lite-peq-v1`，独立 worktree：`Crack_RTDETR-peq_v1`。
- 仓库：[concrete-crack-rtdetr](https://github.com/supershaojie/concrete-crack-rtdetr)。
- 模型：`rtdetr-resnet18-lite-cbr-lif-down-peq-v1.yaml`。
- 正式 run：`peq_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`；tmux：`peq-v1-training`。
- 真实交付 SHA 在普通提交、推送并核对远端后写入 `outputs/peq_v1/DELIVERY.md` 和 `server_commands.md`；避免把文件自身的提交 SHA 写进同一个提交而形成循环。

原文件 LF 规范化 SHA256：

```text
lif_down.py  26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7
cbr.py       d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787
```

上述两个原文件及其公共参数路径均保留。只修改三个公共注册点：`nn/modules/__init__.py`、`nn/tasks.py`、`models/rtdetr/model.py`；其他实现集中在新文件和专用子类。

## 信息流与接口

母版 YAML 节点 17 仍从骨干节点 5 经 Conv 投影为 256 通道，act=False；节点 18 拼接上采样分支，节点 19 为 Neck P3，640 输入时为 80×80。节点 20 保留原 LIFDown；节点 22/25 为 P4/P5，节点 26 输入仍为 [19,22,25]。三层 decoder、300 普通 query、encoder 选择和 DN 调度保留。

新 `RTDETRDecoderCBRPEQ` 在同次原 decoder forward 取得 final_query、b0，调用原 `self.cbr` 一次得到 b1。原 CBR 的 36 点、rho=.10、普通 query 和 DN 精修规则不变。PEQ 按真实 `dn_num_split` 分离普通 query，只处理这 300 个 query；没有再次 forward、筛选高分项或缓存带图状态。

`PrecisionEvidenceQuality` 的实际路径：

1. F.detach() 经无 bias 1×1 Conv(256,16)，q.detach() 经无 bias Linear(256,32)＋SiLU。
2. b0/b1 共享投影特征、3×3 点和聚合器。相对点为 {-1/3,0,1/3}²；grid=2×(center+wh×relative)−1，FP32 bilinear/border/align_corners=False。非方形宽高坐标分别处理，原框不裁剪。
3. 每点 feature16＋position2 经 Linear(18,16)＋SiLU＋无 bias Linear(16,1)，9 点 softmax 和原采样特征加权和使用 FP32，得到 E0/E1。
4. 拼接 E1、E1−E0、Q、8 维 detached 几何、裁剪至 ±10 的原 logit 上下文，共 73 维；Linear(73,64)＋SiLU＋Linear(64,10)。
5. 仅最后一层 weight/bias 全零；delta=tanh(u)，quality_logits=z.detach().float()+delta.float()，评分为十项 sigmoid 均值。基准 z 不裁剪，十个输出不加单调约束。

新增参数逐组为 4,096＋8,192＋320＋4,736＋650＝**17,994**。nc=1、未融合总参数 **20,167,759**（原模型 20,149,765）。原 552 个状态项全部保留，仅增加 9 个 PEQ 状态项。构造允许 nc=80 用于公共初始化，实际 PEQ forward 必须 nc=1。

训练 raw 为母版五项＋本次调用 payload；原 dec_scores 未替换。推理输出仍 B×300×5，为原 b1＋PEQ 评分。无标签 predict 复用原 RTDETRPredictor 正确的过滤和排序路径。`enabled=false` 直接走原 CBR 和原 loss；零输出只表示初始评分相等，仍会学习质量损失。

## 目标、原损失与梯度

原 RT-DETR VFL 已包含 detached IoU 质量监督，PEQ 并非给一个“没有质量监督”的母版首次添加质量。

`PEQDetectionLoss` 显式调用原 matcher 取得真正用于 final 主项的匹配，以原 z 和最终 b1 匹配；主项复用该索引，encoder/decoder aux 继续各自匹配，DN 保留原匹配。逐项验证原 12 项 native loss、插入顺序和求和。原 loss_gain 为 class=1、bbox=5、giou=2；matcher cost 为 class=2、bbox=5、giou=2，未套用通用 args 的 box/cls/dfl 重新配权。

新目标为实际 assigned GT 的普通 IoU 在 .50:.05:.95 上的 >= 指示值；未匹配 query 全零。IoU、框、GT 和原输入全部 detached。匹配组和未匹配组分别对 query×10 求 BCE 均值，两组存在时各 0.5；只有一组时权重为 1。只新增一个 `loss_peq`，诊断分项不重复相加；空图保留有效负例，非有限值明确失败。

使用原单个 AdamW 和 GradScaler。unscale 一次后，原参数集合和 PEQ 集合分别 max_norm=10，再按原顺序 step/update/zero_grad/EMA。原模型实际返回 sum(losses.values())，Trainer 再 .sum()，固定母版没有额外 B 倍率。新增损失不会直接反传到原参数；共享 scaler 溢出跳步、浮点原子累加、验证变化仍可能间接改变训练轨迹，不能承诺全过程逐位一致。

## 必须披露的工程差异

- 新 head/model/criterion/Trainer 扩展 payload 和 loss 路由；公共注册点按新头选择专用实现。
- 原全局梯度裁剪改为两个互斥且完整的集合各自裁剪，保持一个优化器。
- 本实验 validator 实际实现母版独立评估工具的 `corrected_sorted_conf_mask_v1`，修复原每轮 validator 中“排序后框配未排序 mask”的错误。每轮 fitness 仍是原 metrics fitness，但每轮过滤索引修正属于真实工程差异，须纳入历史曲线比较的解释。
- 原每轮验证精度策略保留；独立 val/test 为 FP32。AutoBackend/fuse 已以非零 PEQ/CBR 验证。融合可能改变几乎并列的 encoder top-k query 顺序，测试按相同 anchor 候选身份比较框/分数，并记录重排。
- 为避免原 Trainer 在 OOM 后静默减 batch，专用 Trainer 拒绝此降级。AMP 仍真正执行母版 check_amp；额外检查日志必须出现 checks passed，捕获异常后跳过不能算 PASS。
- 仅支持 eager PyTorch .pt/AutoBackend；enabled PEQ 的未验证导出后端明确抛 NotImplementedError。未声称 ONNX/TensorRT/TorchScript 导出可用。

## 初始化、配方与数据身份

`init_peq_v1.py` 复用母版 `controlled_models`，源必须匹配公共未训练源 SHA：
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。母版 best 仅用于只读 probe，未用于新实验训练初值。

新增层构造隔离 CPU RNG，诊断保存并恢复 Python/NumPy/CPU/CUDA RNG。初始化逐 key 核对公共值，再经实际 Trainer get_model/setup_model 路径复核。nc80→nc1 仅允许 9 个类别相关状态尺寸适配；未知缺失、额外状态或 shape 差异立即失败，未用 strict=False 吞差异。审计见 [initialization.json](initialization.json)。

[mother_args.yaml](mother_args.yaml) 来自实际成功母版 training/args.yaml，109 项全部保留。正式配置仅 model/name/save_dir 改为本实验身份；[args_diff_109.json](args_diff_109.json) 列出每一字段。研究配置独立为 [research.yaml](research.yaml)。关键值：200 epoch、patience50、B16、640、nbs64、seed42、workers8、device0、AdamW、lr0=.0005、lrf=.01、weight_decay=.0001、warmup5、cos_lr/AMP/deterministic=true；没有额外 warmup。

[dataset_identity.json](dataset_identity.json) 记录本地同源数据的配置、逐图清单、图像内容与标签聚合哈希：train6048/45573 GT，val1728/12840，test864/6663。此步骤只核验文件身份，未做 test 推理。服务器 prepare 在实际路径重新计算全部内容身份；路径迁移与字段差异分别记录。已知同源增强图跨 split，结论仅适用于同协议比较，不作为独立原图泛化或多种子显著性证据。


### prepare 配置比较修复

首版 prepare 将 YAML 的整数类别键 `0` 与审计 JSON 的字符串键 `"0"` 直接比较，导致相同类别配置误报。已用实际 `dataset_identity.json` 与母版 `c2_data.yaml` 复现：排除允许迁移的顶层 `path` 后，仅 `names` 存在表示差异，train/val/test 值完全一致。现在双方先转换为一致 JSON 表示，再逐字段严格比较；真实类别、划分、新增/缺失或其他字段变化会显示字段及 server/audited 两侧值，规范化后重名键会明确拒绝。

全部数据数量、路径清单、图像及标签哈希断言保持原样，数据集和审计基准未修改。定向检查包含 4 个通过场景和 13 个拒绝场景，并接入既有 preflight 工作流检查；详见 [prepare_fix_validation.json](prepare_fix_validation.json)。这是本地配置比较回归验证，服务器完整 prepare/preflight 仍需同步修复后重跑，未启动正式训练。

## 已执行验证与机会诊断

[validation.json](validation.json) 汇总真实执行范围；[checks_cpu.json](checks_cpu.json)、[checks_cuda.json](checks_cuda.json) 保存逐项证据，包含：

- 非方形、多图、边界、极窄框的手算采样，E0=E1 与局部视觉变化。
- 实际 assigned GT、阈值等号、全局 GT 偏移、空 GT、G>Q、动态 DN。
- 非零分支评分路由、原框/raw 不变，新增输入 detach，第一步输出层及更新后上游梯度。
- 原 native loss 与 matcher 调用、原梯度及分集合裁剪对照；CUDA 另记录原损失重复反传的累加噪声作对照，不冒充位级相同。
- 实际 Trainer 构造、AMP check、optimizer_step、EMA、原生保存/加载、恢复 optimizer/scaler/epoch、val、predict、AutoBackend/fuse。
- 状态未启动/运行/完成/失败、旧 dispatch exit 隔离、tee 失败与非有限 JSON。
- 补充工作流 fixture 验证两套分数各自排序后的原生指标、已完成 test 强制复用、归档 manifest、独立命令块 Bash 语法。这些合成指标不是实验成绩。

本地环境：Python3.9.25、torch2.7.1+cu118、CUDA11.8、RTX2060 6GB、工作树 Ultralytics8.4.21。服务器历史环境为 Python3.10.13/torch2.1.2+cu121/RTX4090，但本次未登录验证；使用 torch2.1 已有接口，没有安装或升级依赖。小 B/小尺寸检查不能作为服务器 B16/640 容量 PASS。

完整母版 probe 使用历史 best SHA
`24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`，FP32/640、全部300 query、无增强、只读 val。1,728 图、12,840 GT、518,400 query，IoU≥.5/.75/.9 候选覆盖分别 **98.3567% / 71.9782% / 19.3380%**。3,390,944 对高质量(≥.75)/低质量(<.5)候选中有 807,866 对低质量得分更高（**23.8242%**），1 对并列。原分数与 max-IoU proxy Pearson=0.55184。BN 和 best 文件未改，详见 [probe_summary.json](probe_summary.json)。

这些数值表明存在排序空间，不能保证 PEQ 改善；高 IoU=.9 的候选覆盖仍有限。候选覆盖不是正式 Recall/AP，max-IoU 不是一对一 TP，也不是 AP 上界。没有阈值/网格/通道扫描，没有依据 test 调参，没有自动启动训练。

## 服务器操作与证据

完整、真实 SHA 的独立命令块由 `tools/peq_v1_delivery.py` 生成到 `outputs/peq_v1/server_commands.md`。同步脚本必须从交付 SHA 下载后真正执行 sync 建立身份，不能拿漂移 branch head 代替。所有计算动作指定绝对 Python/worktree 和 PYTHONPATH、PYTHONUNBUFFERED=1、YOLO_AUTOINSTALL=false，并验证 ultralytics.__file__。

推荐顺序为 sync→prepare→preflight→probe；用户查看 probe 后显式 start。preflight 创建独立子进程，最多16 micro-batch/900秒，B16/640/真实 AMP/AdamW/原累积，要求至少两次实际有效更新且输出层更新后新增投影有梯度。只有诊断 scaler init_scale=128；其状态全部丢弃，正式训练沿用母版 scaler。源/配置/数据/环境实质不变时复用 PASS，不因报告时间戳失效。

start 使用独立 tmux，实时显示并 tee 到唯一 dispatch 日志，记录 Python 与 tee 各自退出码。status 只读真实进程、子进程、epoch、阶段和日志尾部。已有 run 不覆盖，同实验活跃拒绝重复启动，不操作其他实验。resume 只接受有效 last 中 optimizer/scaler/EMA/epoch 和 PEQ 配置，继续同一 run。仅启动失败 args/空 CSV、无权重且无活跃进程时，archive-failed 通过重命名保留现场。

每 epoch 机制汇总和事件分别在 mechanism_epochs.jsonl / mechanism_events.jsonl；记录组别计数、十阈值正例率、L_Q 分项、delta、分数变化和单调违例率。delta 分位数由400 bin直方图近似（宽 .005）；首步和每100 micro-batch 抽样证据差/越界比例，并在下一次 optimizer_step 记录梯度/参数范数，带分母/次数，不保存整批 P3。

独立 val 必须读取完成正式 run 的 best；同次 forward 得到相同最终框，分别按 PEQ 与 raw 分数计算原生指标，前者为正式分数，后者只作诊断。固定 FP32、640、B16、workers0、conf=.001、iou=.7、max_det300、seed42、无增强。记录十阈值 AP、AP50/AP75/mAP、P/R 和各自最大平滑 F1 工作点。val 锁定 checkpoint/source/config/data/模式，test 仅用户显式执行该身份；相同已完成 test 即使 --retry 也复用，不能切模式或换 best 选 test。

pack 只打包已有证据，LIGHT 保留所有机制 JSONL、源码快照/母版 diff、配置、完整 args/109 diff、初始化/预检/状态/指标/CSV、文件身份与日志摘要。大 .pt、数据集、完整预测流、原日志不进入 LIGHT；列出排除文件大小和 SHA。--include-predictions 单独打包已有 train/val 错误分析流，既不重跑，也不加入 test。逐图流含真实尺寸、坐标约定、query ID、框、raw/PEQ 分数及 GT。原文件保留，归档写 manifest 并复核每项大小/哈希。

FileZilla 目录：`/root/autodl-tmp/projects/Crack_RTDETR-peq_v1/outputs/peq_v1/packages/`；命令输出精确完整包路径并写 last_package.json。训练 tmux 脱离为 Ctrl+B 后按 D。未执行项明确 NOT_RUN/PENDING。

## 文献对应与限制

[Rank-DETR，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html) 讨论定位质量与分类排序不一致，并包含 rank-adaptive 分类、query 排名与损失/匹配设计。PEQ 只借鉴研究动机，不复现其整套设计，不改变母版 matcher。

[Cascade-DETR，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Ye_Cascade-DETR_Delving_into_High-Quality_Universal_Object_Detection_ICCV_2023_paper.pdf) 的期望 IoU 分支以匹配 query 的 L2 目标预测条件定位质量，并结合分类分数。PEQ 使用精修前后共享局部证据及十阈值残差 logit、匹配/未匹配组 BCE；不只是把 IoU head 改名，也不据此宣称新颖性或性能优势。上述对应来自实际论文内容，不将论文结论移植为本实验成绩。

提供的 PEQ 目录未发现 RFAConv.py 模块包；按合同固定公式独立实现9点共享加权聚合，没有搬入完整 RFAConv 特征展开。

TQC/RSC 的框演化信息、ROR 的训练期排序约束、CQS 的 decoder 候选作用与本实验信息流不同；这些区别仍需实验支持。可能失败原因包括准确候选不足、query 已含全部有效信息、border 重复采样、新评分变差。PEQ 不能生成缺失的准确框。历史母版 val52.4543%、test52.2009%仅作历史参照，本次新实验指标尚不存在。
