# DTR v1 实施交付

唯一实验为 **RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR + DTR v1**。
本地实现和有限验证不等于服务器 B16/640 AMP 通过，也不代表涨点。本次没有启动正式 200 轮训练，没有运行 test。

## 代码和隔离

- 实际母版/分支基座：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
- 独立分支：`exp-rtdetr-r18-lite-dtr-v1`。
- 核实 remote：`https://github.com/supershaojie/concrete-crack-rtdetr.git`。
- 本地工作区：`C:/Users/o'v'o/.codex/worktrees/dtr-v1/Crack_RTDETR`。原 CSCEF 工作区、未提交文件及其他实验工作区保持。
- 未发现非空适用 AGENTS.md。没有合并主分支、reset/clean、强推或改写历史。
- 功能提交：`bb5cc3987c6b0df3c65ee9253e2bc63ba725a461`（包含初始实现 `82ae3098bea0931efcbcd6cc9132b9000e6c7453` 和 Windows 初始化路径修复）。后续纯文档提交不改功能代码；服务器命令固定此功能提交，避免自引用提交 hash。见 `server_commands.md`。
- 基座直接取母版，没有引入 ARG/ARG v2/QCC/RMD/RDL/ROR/PEQ/TCR/GEO 算法或训练状态。

## 实际接入

| 文件 | 职责 |
|---|---|
| ultralytics-main/ultralytics/models/rtdetr/dtr_loss.py | 单类 DTR criterion、显式图像/GT 映射、FP32 逐边中位数和超额误差 |
| ultralytics-main/ultralytics/models/rtdetr/dtr_model.py | native loss 包装；从实际 batch img 传 H/W，finally 清理 |
| ultralytics-main/ultralytics/models/rtdetr/dtr_trainer.py | native get_model 后的 Python 类型专门化、epoch 同步、前四批机制日志、真实 optimizer step 观察 |
| ultralytics-main/ultralytics/models/rtdetr/dtr_val.py | 独立 FP32 corrected_sorted_conf_mask_v1 和同次全 query/GT 导出 |
| ultralytics-main/ultralytics/nn/autobackend.py | 唯一母版文件变更：warmup 的 torch.empty 改成有限 torch.zeros |
| tools/dtr_v1.py | prepare/preflight/status/start/resume/val/test/finish/pack 及内部 worker |
| tools/dtr_v1_common.py | 固定身份、完整配方、快照复用、原公共初始化工具调用 |
| tools/experiment_runtime.py | 小型共享 JSON/hash/进程边界/互斥工具，无实验算法 |
| tools/dtr_v1_preflight.py | 原生 GradScaler 的最多 16 micro-batch / 900 秒服务器短检 |
| tools/dtr_v1_eval.py | 评估锁、离线同阈值统计、完整包与故障包 |
| tools/sync_dtr_v1.sh | 限时 Git 同步和独立 worktree，不使用 raw 下载入口 |
| tools/check_dtr_v1.py、tools/check_dtr_v1_ops.py | 数学/路由/模型集成和 CLI/离线流程验证 |

真实路径：Trainer.get_model 原生构建/加载 → DTRDetectionModel.loss → 原 RTDETRDetectionModel.loss 拆普通/DN、拼 encoder → DTRDetectionLoss → 原全部 L0 → 只增加一次 loss_dtr → 原模型 sum(loss.values()) → 原 Trainer backward/optimizer。

原 CBR 在拆 DN 前对最后 decoder 层全部 queries 修框，因此普通端和正 DN 端均为同次前向、同一原 CBR 后的最终框。没有第二次前向或新 Hungarian。criterion 仅在原最终普通层 _get_loss 捕获原匹配；encoder/auxiliary/DN 仍按原分配。捕获在 finally 清除。

## 母版保留证据

- 原 YAML 不变：3 层 decoder、300 普通 queries、CBR 36 点、rho=0.10、原 detach 和推理路径。
- LF 归一化 SHA256：
  - lif_down.py：`26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7`
  - cbr.py：`d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787`
- 同 nc=1 未融合口径：**20,149,765 参数、552 state_dict 张量键**，参数名称/形状和初始化值与母版一致；损失不增加可训练参数/部署张量。AutoBackend 融合后的参数数不同属于原融合口径，不拿它冒充未融合计数。
- 原 L0 保留 class/VFL、5×L1、2×GIoU、encoder/decoder auxiliary 和 DN。实对象 VFL alpha/gamma=0.25/1.5；matcher alpha/gamma=0.25/2.0，class/bbox/giou 代价=2/5/2。
- 公共 nc=80 源权重 SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。直接复用母版 init_c19_lif_v1.initialize 和原 Trainer nc=80→1 映射，552 键中 9 个类别相关张量按原生规则适配，其余 543 键精确加载。没有加载实验 best/last/smoke 作为正式初始化。
- 权威配方来自 docs/c19_lif_v1/resolved_formal_config.yaml，完整 **109 字段**。正式服务器仅 model/project/name/save_dir 路径身份变化；data 仍是母版 YAML。全部优化器、epoch/early stopping、B16/640、AMP、seed、累积和增强参数不变。见 formal_args.yaml 和 recipe_diff.json。
- 训练 run 位于 DTR worktree 自己的 runs/c_series 下；数据和公共权重从核验的主项目只读引用。

## 固定公式和边界

`DTR-v1-dnmedian-excess`：lambda=0.20，delta=0.05，一像素分母下限，ramp_start=5，ramp_full=20。e 是零基 trainer.epoch，ramp=clip((e-5)/15,0,1)，e≤5 为零，e=6 为 1/15，e≥20 为一；显示轮次=e+1。

对每个最终普通匹配，按**图像和 batch 展平 GT ID**，从原 get_dn_match_indices 取全部正 DN；负查询和 padding 不进入参照。用显式索引构造临时分组，不猜 reshape 顺序。半径是逐边 abs(DN.detach−GT.detach) 的组中位数；偶数组取两个中间次序统计量的平均。K=1 合法。

`u = relu(abs(x-g)-r) / S`；S 的左右分母 max(w,1/W)，上下 max(h,1/H)，H/W 来自增强及预处理后的 batch["img"]。计算使用 FP32 和局部禁用 autocast；phi 使用严格等价的 `u²/(sqrt(u²+0.05²)+0.05)`。

`raw=sum(phi)/(4*max(M,1))`，M 包含无正 DN/未激活匹配。无对应正 DN 直接给零，不把半径置零变成额外回归。只加入一份 `0.20*ramp*raw`，没有 aux/encoder/DN 新损失。原无 DN 补零完成后才追加新项，避免 loss_dtr_dn。

普通最终匹配框保留梯度；DN 参照、GT、尺度和索引 detach。对独立边，激活时方向为 `u/sqrt(u²+delta²)*sign(x-g)/S`，再乘平均、lambda 和 ramp；误差≤半径时梯度零，原 L0 仍工作。原 DN 主损失继续训练。DN 误差全零时退化为定义中的附加 GT 边回归，如实接受；不增加隐藏门控。

disabled/ramp=0 直接走原损失，不新增分组/中位数或 RNG；val/inference 关闭。非 nc=1 明确报错，损坏元数据/越界/非有限坐标报错。没有多卡补偿，DDP 未验证。

每轮最多前四个训练 micro-batch 记录 M、缺参照数量、K 分布、r/S、普通误差/S、u、激活比例、原损失、raw/weighted/ramp。按计数/求和聚合，边顺序 L/T/R/B；无样本的均值为 null、求和为零。ramp=0 的记录明确 skipped，不偷偷算统计。不据此自动改公式或停止实验。

## 可靠性复用来源与行为调整

1. 母版 a0459d6：公共初始化、完整 args、原 LIF/CBR/criterion/Trainer；corrected_sorted_conf_mask_v1 参考 tools/c19_lif_v1_results.py。
2. ARG 参考 `ef9cb7e05e5557f7dd06c95cf2361998a284adc9`：只抽取有限零值 warmup、JSON 类别键规范化、原生模型重建后专门化、严格 AMP 结果检查、真实 step/scale 观察、限时子进程与退出证据、原始 query 导出模式。未复制 ARG 损失/模型算法、整套训练工具或低 scale fallback。
3. 本版将快照复用、首次默认完整导出、缺锁补锁、status/pack 只读和一次完整包做成显式流程。已确认快照可复用时不再次全量 hash 图片/标签；后续校验 YAML/清单 hash、固定配置和六个 split 目录时间戳。**这不是连续文件监控；原地改写文件必须显式 prepare --refresh-data，不能声称每次重验所有图片内容。**
4. 原训练期 val、损失日志、fitness=mAP50–95 和 best 保存逻辑不变（原生相等 fitness 的保存规则也保留）。原 final_eval 在训练自然完成后原本还会 strip 并再次 val；DTR 仅记录完成，把这次正式 FP32 评估交给训练后显式 finish。保留 best/last 原生完整恢复状态，避免提前多跑一次不带导出的 val。没有增加训练预算或改变 best 选择。
5. 本地 prepare 实测发现原权重加载器会删除绝对 Windows 路径中的撇号；修复仅为在工作区内通过相对路径调用原初始化器，不改初始化值或母版框架。失败 checkpoint/错误记录保存在本实验 initialization_failures 下，修复后 prepare 已 PASS。

## 实际检查与 PENDING

详见 local_validation.json、local_initialization.json、data_identity.json。本地解释器 D:/miniconda3/envs/rtdetr/python.exe，Python 3.9.25，torch 2.7.1+cu118，CUDA 11.8，RTX 2060 6GB，导入来自本 DTR worktree。未升级依赖。

已完成：

- 小张量数值/autograd、奇偶中位数、反向偏差不抵消、K=1、零参照、容差边界、空图/无 DN、全 M 分母、窄框/矩形输入、稳定 phi、固定 teacher 的有限差分。
- 原 get_cdn_group 多图 [2,0,1] GT、3 组正负/padding 的可辨识映射，跨图同局部编号隔离；显式查询重排仍相同；损坏元数据报错。
- 固定 preds 下全部 L0/DN 数值及输出梯度完全一致；实际 matcher 调用 4 次，不增加调用；只有一个 loss_dtr；DTR-only 不直达 DN/logits/早层/未匹配框，原 DN 梯度仍存在。
- 原公共初始化到真实 Trainer.get_model 重建；本地 B2/128 的 CPU/CUDA FP32 前向及独立优化更新；恢复 RNG 后真实 DN 输出及 L0 完全一致。CPU 全部 345 组可训练参数梯度在 rtol=1e-5、atol=1e-6 内一致；仅 DN embedding 存在 5.59e-9 级差异，未声称全部 bitwise 一致。
- CUDA 全参数的相同严格逐元素梯度比较为 **PENDING**，没有放宽容差将其改为 PASS：DTR 对母版最大绝对差 2.53e-4；母版同一计算图重复反传对照自身也有差异，最大 3.59e-4，均涉及 226 组参数。该现象与原 CUDA 反传的非确定性一致，但不将这一推断当作精确一致性证明；CPU 和固定 preds 的路由证明、CUDA 有限梯度/更新/同权重推理仍独立记录。
- 同权重参数/张量键/推理输出/后处理一致；独立进程加载自定义 model/criterion，输出精确一致。
- 独立新进程本地 B2/160 FP32 warmup 使用有限全零输入，一批真实 val=2 图，导出 600 普通 queries 和全部 GT。诊断权重和结果不能当作正式评估/涨点证据。
- 每个 Python 主入口和所有子命令 --help 在新进程、非仓库 cwd 执行；同步脚本 bash -n 和 --help。
- 已提交功能版本实际 prepare=PASS；本地有界 preflight 的 CPU 项 PASS，其余服务器项 PENDING、start_eligible=false，正确拒绝将本机小批检查冒充服务器资格。
- 离线阈值统计（含空 GT/空预测）、排序/conf mask、缺锁补锁且不再推理、篡改拒绝、互斥锁、status/pack 不推理不扫描、默认含预测的 INCOMPLETE 包与 manifest 非自包含 hash。

待服务器实际运行：

- Linux 原解释器/数据/公共权重路径及 GPU 空闲检查。
- CUDA 严格逐元素全模型梯度重现性；原母版重复反传对照也未达到该容差。不得将本机该项记为 PASS。
- **B16/640、原 AMP/AdamW/累积及 native GradScaler 的真实有效更新**；最多 16 micro-batch / 900 秒。未取得有效更新为 PENDING，拒绝正式 start；不调小 batch、不替换 scaler、不以低 scale 诊断冒充原 scale PASS。
- 原 Trainer 完整 setup/save/resume，epoch/optimizer/scaler/EMA/scheduler 恢复，以及新进程 B16 FP32 一批真实 val。已提供可执行短检，当前机器的局部检查不替代它。
- 正式 200 轮/合法 patience=50 停止、完整 FP32 val/test、机制的长期行为、性能开销和精度收益；DDP。

## 完成后的评估与一次打包

finish 先核对原生训练完成证据及同一 best 的 run identity，再依次只补缺失成功结果的 val/test。协议：FP32、640、B16、workers=0、conf=0.001、iou=0.7、max_det=300、augment/rect=false、seed=42。保留原 RT-DETR 后处理，不添加 NMS/阈值调参。

首次每个正式 split 同次推理默认导出全部 300 普通 queries/图和 GT：唯一相对路径 ID、split、原尺寸、原 query index、框/分数/类别、xyxy/原图像像素坐标。原指标仅用 score>0.001，导出保留低于该阈值的 query。DN 不作为检测输出。

锁绑定 split、checkpoint SHA256、数据快照、评估代码内容/配置和协议；完整结果直接复用，完整报告缺锁只补锁，失败/部分评估不写成功锁。评估代码修复可独立恢复评估，并保存修复后的源码快照，无需重新训练。每次 finish 只生成一个完整包；身份与文件 hash 一致的再次 finish 复用上一个完整包。

指标包含 P/R/F1、AP50/AP75/mAP50–95、十个 IoU AP、PR 和 P/R/F1-confidence 曲线及原始统计数组。保留各 split 原最佳 F1 点的 P/R；另外仅从 val 确定共享阈值，离线重算 val/test 同阈值 TP/FP/FN/P/R/F1（IoU=.5，单类）。P 是 Precision，绝非含 TN 的 Accuracy。test 不参与阈值/模型选择。

包包含源码/Git 身份、公式、args/recipe_diff、环境/导入、初始化/数据身份、预检、机制日志、results.csv/训练日志/曲线、dispatch/PID/开始结束/真实 Python 与 tee 退出码、失败与恢复历史、完整评估曲线/预测/GT、best/last 路径大小 SHA256/best epoch/规则、评估锁和 manifest。manifest 不包含自身 hash。默认不带原始图片全集和大权重本体；明确记录权重未包含。

status/pack 仅读现有证据；不推理、不挑 best、不重扫数据。缺项列入 INCOMPLETE。训练已经完成但评估失败，用 val/finish 恢复；原框架/外部若 strip 掉完整训练状态，resume 明确拒绝。

## 研究边界

DN-DETR 提供去噪查询基础；teacher-bounded regression 已有学生比教师差时增强 GT 回归的思想；D-FINE 属于不同分布回归设计的定位自蒸馏。本候选增量仅为“同次正 DN + 逐边绝对误差中位数 + GT 中心容差 + 超额惩罚”。DN 不是无偏教师/真实不确定性，KD 原理不是本方案首创；不保证共享参数梯度与 L0 一致，不保证有限学习率无越界，不承诺涨点。不自动增加消融或参数变体。
