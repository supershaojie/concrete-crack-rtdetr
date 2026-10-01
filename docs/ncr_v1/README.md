# NCR v1 实施与实验边界

本分支从 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 派生。仅增加 Nested Center Rescue；原 RT-DETR-R18-Lite、CBR、LIF-Down v1、模型 YAML、HungarianMatcher、预测返回值保持母版源码。单类未融合模型 20,149,765 参数，300 普通 queries、3 层 decoder。CBR 仍只修正最后返回层（含原 DN 处理），原框梯度路径未改。

## 损失与接线

令 `b,g` 是本次最后普通分支的原 Hungarian 匹配框，归一化 `cx,cy,w,h`。每轴：

```text
d = abs(c - c_gt)
m = abs(length - length_gt) / 2
h = detach(relu(m - d) / (m + 1e-7))
s = max(length_gt, 4 / network_input_axis_pixels)
u = (c - c_gt) / s
rho = 0.5*u² if |u| <= 1 else |u|-0.5
L_ncr_raw = sum(h*rho) / (2*max(M,1))
lambda_eff = 0.25 * clip((epoch - 4)/15, 0, 1)
L_total = L0 + lambda_eff*L_ncr_raw
```

M 包括门控为零的全部普通匹配正样本。两轴独立。GT、完整 h 停止梯度，中心残差保留梯度；内部禁用 autocast、转 FP32。独立最终框的新增直接梯度只有中心分量；共享网络/CBR 的参数梯度仍可能耦合。

`DETRLoss._get_loss` 新增默认关闭的 `final_regular_hook`，只从 RTDETR 普通分支的最后层明确传入。该 hook 获取已经匹配的框，不重新匹配、不缓存匹配或训练图。aux 和 DN 调用不传 hook。`NCRDetectionLoss` 在原普通、aux、DN 所有项装配完后，加入一次 `loss_ncr`，从而不会生成 `loss_ncr_dn`。λ=0 或 ramp 关闭时不创建 hook/门控图，原项计算和 RNG 保持原路径；M=0 返回同设备 FP32 零诊断。

`RTDETRDetectionModel.loss` 从实际 `batch['img'].shape[-2:]` 传 H/W，真实总损失仍为 `sum(loss.values())`；显示项增加独立第四项 `ncr_loss`，前三项未改。配置只挂到模型的 `ncr_config` 并由 criterion 明确读取，没有向 `train()` 注入未知参数。

`NCRTrainer` 的 epoch 回调在原生 resume 恢复之后读取 `trainer.epoch`，同步普通模型和 EMA；不按 batch/验证次数/环境变量推算。e=0..4 关闭，e=5 为 0.25/15，e≥19 为 0.25。Huber 转折 1、4px 下限和 0.25 均为本方案初值，未调优。

原 L0 是 VFL（源码实查 α=0.25、γ=1.5）、5×L1、2×GIoU，加原 encoder、decoder auxiliary 和 DN；匹配成本 class=2/bbox=5/giou=2，原匹配策略不变。L1 原本已有中心梯度。NCR 不保证当前 IoU 立即提高，非零损失/中心更准/训练 loss 下降均不能证明检测收益。

## 完整配方与初始化

`parent_args.yaml` 是母版正式训练归档的完整 **109 个字段**，来源为用户本地 `C19＋LIF：原 CBR＋LIF-Down v1/training/args.yaml`。与母版已提交 `docs/c19_lif_v1/resolved_formal_config.yaml` 比较，仅历史 worktree 的模型路径不同。`resolved_formal_config.yaml` 给出 NCR 的服务器绝对路径版本；运行时再次从完整母版归档解析并保存完整 args 与差异。

公共初始化 SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。直接复用母版 `init_c19_lif_v1.initialize/build_training_model`：固定种子构建公共 nc80 与两模块模型，逐键严格合并，再沿母版原生 trainer 顺序转 nc1。仅原有 9 个分类形状适配项允许差异，其他 543/552 个状态精确加载；CBR/LIF 构造初始化及零初始化保留。不使用母版 best 开始训练。

固定训练：200 epochs、patience=50、640/B16、nbs=64、seed=42、workers=8、GPU0，AdamW（lr0=0.0005、lrf=0.01、weight_decay=0.0001、momentum=0.937），warmup_epochs=5、cos_lr=True、AMP=True、deterministic=True。所有增强及其余字段继承 109 字段归档；没有从当前默认值补齐配方。数据 train 6048/45573、val 1728/12840、test 864/6663（图/框），单类 crack。

## 服务器生命周期

入口：`tools/ncr_v1.sh preflight|start|status|attach|finish|pack`。详见仓库根目录 `server_commands_ncr_v1.md`。

- preflight：最多 900 秒、必要单测、严格初始化、同一 B16/640 真实批次的 FP32/AMP 各一步；两个独立临时模型，最多 2 个 optimizer steps。检查的 AMP GradScaler 初值 128 复用母版 `tools/check_c19_lif_v1.py` 的损失冒烟规则；正式 trainer 的原生 GradScaler 配置不改。这个检查不等于正式原生 scaler 首批没有 overflow。
- start：核对代码/配方/数据/初始化/环境完全相同的成功服务器预检；独立新进程从公共初始化重新构建。训练固定输出到主仓库的绝对 runs 路径。`start --resume` 仅安全续训本实验已有 last，原生恢复 optimizer/scaler/epoch；不会从已完成/已 strip 的模型继续训练。
- 锁仅锁 NCR 的运行身份，不锁 GPU。允许其他实验同时使用 GPU0。不检查“必须空闲”，不改卡、batch、分辨率或 AMP。原生 OOM 降 batch 重试已关闭。
- 训练 tmux `ncr-v1-training`、finish tmux `ncr-v1-finish`，仅各自窗口开启 remain-on-exit。worker 分别检查 Python 和 tee 退出码，保留死 pane、日志、最终退出码。status 根据退出证据和 pane 状态区分未启动/运行/成功/失败。
- 训练期 val、fitness、best 和 early stopping 使用原实现。训练结束仅原生 strip 权重；额外独立验证明确延后至 finish，让首次正式 FP32 val/test 同时保存全 query 证据。
- finish 仅在成功训练且 best SHA 固定后运行；先 val 再 test，同一个 best。成功且身份相同则复用；失败步骤可在下次显式 finish 时重做一次并保留旧失败目录。成功评估缺导出时明确拒绝悄悄重推理。
- pack/status 都不启动推理。完整包检查必需文件、初始化、配方、源代码、预检、best、评估及导出哈希；缺项命名 PARTIAL。已有相同内容包复用，tar 每个成员回读验证并输出 SHA256、manifest、清单。不会包括整个数据集/环境/其他实验，不自动下载或删除低分结果。

环境实查写入 manifest；服务器要求现有 Python3.10 / torch2.1.2+cu121 / NumPy1.26.4，不自动安装升级。母版 AMP 检查资源仅从现有主仓库复制，不自动下载。

## 评估与诊断

独立评估使用母版 `tools/c19_lif_v1_results.py` 的 `postprocess` 和 `image_record`（母版 a0459d6，原注明来自 C24 `beedcfa`），协议 `corrected_sorted_conf_mask_v1`。未从其他损失实验移植机制或整支提交。

FP32、640/B16、workers=0、conf=0.001、iou=0.7、max_det=300、augment=False、rect=False、seed=42。每次正式前向同步导出普通 300 queries：原 query index、完整 class logits/score、normalized cxcywh、精确网络 input xyxy、原图 xyxy、GT、图像 ID 和原图/网络尺寸。RT-DETR 使用 stretch，明确记录 x/y gain 与零 padding。导出不改变预测。

保存原评估器同一次 TP 标记；离线以原生 IoU 匹配重放校验，检测 TP/AP 不使用 Hungarian。另行 Hungarian 仅用于 NCR 机制诊断。按 GT 网络输入短边 `<4 / 4–16 / 16–32 / ≥32 px` 报告原 PR 工作点的 IoU50/75 recall、中心/尺寸误差与 FP/FN；这些是分组召回而非分组 AP。P/R/F1、AP50、AP75、全部 10 个 AP、mAP 保存原始小数与显示百分数。Precision 不标 Accuracy；F1=2PR/(P+R)。母版独立 FP32 val/test mAP50–95 为 52.454272% / 52.200902%，差异以百分点报告，AMP 训练验证峰值不混作独立分数。

每个训练 epoch 首个 batch 记录 detached 匹配数、h 激活率/分布、短边分组相对中心误差、raw/λ/加权项，以及有效轴额外中心梯度相对于原 5L1+2GIoU 梯度的统计。分母原梯度为零的轴单独计数，不用 epsilon 伪造比例；无激活样本如实为零。

可选 `ncr_v1_parent_diagnostic.py` 只读母版已有 val 导出的前 64 张图，无新增推理/test。老导出缺原 logits/query index，故其 IoU50 几何调查不声称代表训练 Hungarian 正样本比例。

## 验证与文件入口

核心：`ultralytics/models/utils/ncr.py`；显式作用域：`models/utils/loss.py`；真实装配：`nn/tasks.py`；importable trainer：`tools/ncr_v1_training.py`。

运行 `python tools/check_ncr_v1.py` 与 `python tools/check_ncr_v1_ops.py`。涵盖几何/解析与冻结门控有限差分、零系数母版逐项/梯度/RNG、aux/encoder/DN 隔离、真实 sum 装配、CBR 梯度、推理/参数一致、新进程加载、原生 resume epoch、完整配方、导出、退出码和 PARTIAL 复用。具体本地结果与未执行项见 `VALIDATION.md`，原始小型证据存于 `evidence/`。

交付文件清单（下列路径相对仓库根目录）：

| 文件 | 用途 |
|---|---|
| ultralytics-main/ultralytics/models/utils/ncr.py | NCR 数学、诊断、可导入 criterion/config |
| ultralytics-main/ultralytics/models/utils/loss.py | 最终普通分支的显式可选 hook |
| ultralytics-main/ultralytics/nn/tasks.py | criterion 配置、实际输入尺寸、一次总损失装配和四项日志 |
| tools/ncr_v1_training.py | 原生 trainer、初始化、真实 epoch/resume、低频记录 |
| tools/ncr_v1_common.py | 完整配方、源码/数据/环境身份 |
| tools/ncr_v1.py、tools/ncr_v1.sh | 预检、tmux、启动/恢复、状态、finish、离线完整性打包 |
| tools/ncr_v1_results.py | 原协议评估、全 query/GT/TP 导出、离线分析和指标页 |
| tools/check_ncr_v1.py、tools/check_ncr_v1_ops.py | 11 项核心测试、10 项操作/导出测试 |
| tools/ncr_v1_parent_diagnostic.py | 可选母版已有 val 导出的小样本离线调查 |
| tools/diagnose_ncr_v1_amp.py | 单批零 optimizer-step 的缩放梯度故障定位 |
| server_commands_ncr_v1.md | 可复制的服务器同步/预检/运行/收尾命令 |
| docs/ncr_v1/parent_args.yaml、resolved_formal_config.yaml、recipe_diff.json | 完整固定配方与允许的路径差异 |
| docs/ncr_v1/parent_dataset_inventory.json、PROVENANCE.json | 母版数据与设施复用来源 |
| docs/ncr_v1/VALIDATION.md、evidence/*.json | 通过/失败定位/NOT_RUN 与可复查本地证据 |
