# ROR v1：精修次序反转损失

这是研究候选。工程正确性、固定样本现象、正式训练和独立 AP 收益是四种不同证据。本交付不运行 200 epoch、不运行正式 val/test，也不保证优于原成功组合。

## 身份与保留项

- 唯一母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`；分支：`exp-rtdetr-r18-lite-ror-v1`。
- 正式 run：`ror_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`；工作树：`Crack_RTDETR-ror_v1`；独立 tmux：`ror-v1-training`。
- 原 YAML 实测为 LIF 20、head 26、head 输入 `[19,22,25]`；3 层 decoder、300 常规 query；CBR rho=0.10、36 点采样、原条件梯度、LIF/CBR 零初始化均保留。
- 模型仍为原 `RTDETRDetectionModel`，未用子类冒充母版审计、未改 `__class__`、未改 CBR/LIF 数学计算。nc=1 未融合参数 20,149,765，融合 19,944,965；ROR 新增参数和 buffer 均为 0，state_dict 仍为 552 项。
- 未更改优化器、增强、数据划分、原学习率 warmup、DN、推理后处理或 checkpoint 的 val fitness 选择方式。原参数全部联合训练。

## 精确定义与梯度

在每张图最终 decoder 常规匹配正样本中，用最终 b1 的一次原生 Hungarian 匹配，固定每个 query 的唯一 GT。b0 来自同次 forward 的 `details['before']`，不是上一 decoder 层，也不重新匹配 b0。

```text
q0_i = sg(aligned_IoU(b0_i,g_i)); q1_i = sg(aligned_IoU(b1_i,g_i))
R_n = {(i,j): 同图、同类、i≠j，q0_j-q0_i≥0.02，q1_i-q1_j≥0.02}
ell_i = log(clip(q1_i,1e-4,1-1e-4)) - log1p(-clip(q1_i,1e-4,1-1e-4))
m_ij = min(2,ell_i-ell_j); w_ij = q1_i-q1_j
d_ij = relu(m_ij-(z_i-z_j))       # z 是 GT 类别的 raw logit
phi(d) = 0.5*d² (d≤1)，否则 d-0.5
L_n = sum_R(w_ij*phi(d_ij)) / max(|R_n|,1)
L_ROR = sum_n(L_n)/实际图像数N
L = L0 + 0.10*clip((e-5)/15,0,1)*L_ROR
```

只对选中的反转对求和，不对称重复、不除 2、不按权重和或违约对数归约。没有 pair 的图像仍计入 N；已经满足 margin 的 eligible pair 仍计入图内分母。mask 和 w 使用未裁剪 q，裁剪只供 logit 计算。IoU 是 cxcywh 转 xyxy 后的 aligned 普通 IoU，使用一致的角点面积，避免同一框在 FP32 下得到略大于 1 的值。

q0/q1、mask、m、w 全部 detached。新项仅直接对 z 求导，违反 margin 时推动 i logit 增大、j 减小；不直接给 boxes 梯度，原 L0 到 CBR/框的梯度保留。分类梯度仍会经过共享特征，不能称模块解耦。局部关闭 autocast，IoU/log/log1p/margin/Huber/归约均 FP32。空集合使用当前 raw logits 的空切片求和连接计算图，不用可能溢出的 `sum(z)*0`。无 nan_to_num；无效框和非有限值明确报错。

规定例子的未加权损失约 0.06433845，z 梯度约 -0.09/+0.09。`torch.logit(q)` 与 `log(q)-log1p(-q)` 的 GPU FP32 末位舍入可留下约 1e-15 的损失；数学检查使用 1e-12 绝对容差，不改变实现中的 margin 或筛选规则。

e 是当前 epoch 开始前完成的 epoch 数：e=0..5 为 0，e=6 为 1/15，e=20 为 1。实际 Trainer 每轮开始显式设置 criterion，并同步 EMA 的标量元数据；不是按 batch/step 增长。resume 由 checkpoint.epoch+1 恢复，不重新 warmup。

## 原 L0 / matcher 核对与接线

核对的是母版实际 `ultralytics/models/utils/loss.py` 和 `ops.py`。VFL 已用最终框 detached IoU 监督质量，主项系数为 class/bbox/GIoU=1/5/2。`no_object=0.1` 虽在配置字典内，当前 VFL 分支不读取它，未额外乘入。无 GT 时仍按原 FL 路径。matcher 成本 class/bbox/GIoU=2/5/2；返回的 query 索引为图内索引，GT 索引包含 batch 展平偏移。

`tasks.py` 的默认预测返回值保持原样，仅增加显式 `return_cbr_details`。训练正权重路径同次 forward 携带结构化 details；DN 预测在 dim=2 拆分，details 在 dim=1 拆分。encoder 预测仍拼在 decoder 前面，未把它当作第四层 decoder。没有 hooks、全局 tensor 状态或本步 tensor 属性缓存。

母版 criterion 新增默认关闭的 `final_match_indices` 参数，只作用最终主损失；aux 保持逐层匹配，DN 保持原分配。ROR criterion 先计算一次最终匹配，再交回母版完整 L0，最后只追加一次 `loss_ror`。没有 `loss_ror_dn` 或 aux 变体。原主三项显示名不变，统计数值进入独立 `training_ror.jsonl`，不进入损失求和字典。

`tools/ror_v1_training.py` 的 RORTrainer 在真正 `get_model` 重建边界先调用原 `build_training_model` 严格审计，再安装稳定导入路径的无参数 criterion。安装前后核对全部 state keys 和参数名。checkpoint 保存标准模型及可导入 criterion，额外身份为普通 Python 元数据，不是 buffer。原框架训练结束会 strip optimizer/criterion，因此完成后的 best/last 可独立评估，但不可伪称还能 resume；中断时未剥离的 last 支持 resume。

验证器带 preds 调用 eval 模型 loss 时只计算 L0，不二次 forward。普通无标签推理、融合和导出不需要 GT 或 ROR details。推理模型输出仍 `[B,300,4+nc]` 及原 raw tuple。

## 初始化、配方、数据

公共初值 SHA256：`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。复用母版 epoch=-1/nc80 公共源到 pair 的受控初始化，再按原 Trainer 适配 nc1；仅 9 个分类相关 keys 改变形状，逐项报告。正式实验不从训练完成权重起步。

`prepare` 读取母版完整 109 字段 C2 args，并与服务器主项目实际 C2 args 逐字段核对，生成 `recipe_diff.json`；只允许模型/输出身份和同一数据的等价绝对路径变化。ROR 固定参数单独保存在 `ror_config.json`。batch16、nbs64、epoch200、patience50、seed42、native AMP、AdamW、全部原增强均锁定。

`parent_dataset_inventory.json` 来源于已核验成功组合归档的 `metadata/launch/dataset_inventory.json`。实测 train/val/test 图数为 6048/1728/864，GT 为 45573/12840/6663，路径和标签指纹均核对。既有同源增强图跨划分问题保留并披露；未重分数据或新增 holdout。数量一致不单独作为身份判据。

## 已训练母版诊断

使用 SHA256 为 `24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa` 的成功组合 best.pt，仅 eval、FP32、原 RT-DETR stretch640 无随机增强；每个 train/val split 使用 seed42 最多128张。`diagnosis.json` 保存统计，`train_samples.json` / `val_samples.json` 保存实际样本及图像/标签哈希、q0/q1、质量变化、反转 pair、margin、当前 raw logit 差、Huber 和加权量级。诊断 batch 可以适配显存，报告保留实际值；正式 batch16 不变。

缺失权重记为 PENDING_ASSET。全部 batch 为空或已满足 margin 记为 NO_OBSERVED_SUPPORT；该样本证据不足以直接投入长训，但不证明训练全过程没有反转。观察到少量 pair 时不设任意数量门槛，不放宽 0.02 或改成全部正样本排序。

VFL 对固定质量标签的独立理想输出是 p=q；q 未受端点裁剪影响时，其 logit 差满足本 margin。这只是输出变量相容性，不证明参数梯度永不冲突。Huber、排序损失、质量感知分类并非新发明；候选点是固定匹配身份下的精修前后反转选择与有限间隔。与旧 RSC-Head 的质量/排序动机相关，未额外修改分类网络。SDB 的 AP75 增加未保证整体 mAP 改善；不叠加 SRE/BFR/RCS-Q/NBR，也不把未确认的 GRA/DCC 结果写成失败。

## 分阶段服务器操作

服务器主根 `/root/autodl-tmp/projects/Crack_RTDETR`；历史成功归档已确认解释器 `/root/miniconda3/envs/rtdetr/bin/python`、Python3.10.13、torch2.1.2+cu121。入口会重新核验，版本不一致明确停止，不升级环境。这里没有远程服务器连接，B16/640 原生 AMP 容量必须由服务器 `preflight` 实测。

1. `sync_ror_v1.sh 完整交付SHA`：从已验证 origin 获取交付 commit，新建独立 sibling worktree。只允许 fast-forward，主/其他工作树不切换，不 force/reset/clean。
2. `bash /root/autodl-tmp/projects/Crack_RTDETR-ror_v1/tools/autodl_ror_v1.sh prepare`：环境/公共源/数据/配方检查并写受控初值，不训练。
3. 同入口 `preflight`：有限数学、CPU/CUDA 接线/生命周期、原 AMP 检查及隔离 B16/640 在线增强 batch 的前后向；容量检查不做 optimizer update，不污染正式初值。不因其他 GPU 进程存在而拒绝，只按实测容量判定。
4. 同入口 `diagnose`：默认从主项目 C19+LIF 正式 run 的 best.pt 取已训练母版，核对 SHA 后固定抽样。若资产移动，可传 `--weights '实际路径'`；不下载替代权重、不补训。
5. 同入口 `start`：读取当前身份一致的 prepare/preflight/diagnose 记录，仅此操作显式派发200轮训练。NO_OBSERVED_SUPPORT 不通过长训启动检查。tmux、run、日志、init、manifest 均独立；仅检测相同 ROR run 的并发冲突。
6. 同入口 `resume`：仅续本实验 last.pt；核对版本/配方/epoch/optimizer/scaler并恢复真实调度。已结束且 stripped 的 checkpoint 明确拒绝。
7. 同入口 `val`：默认本实验 best.pt，按 `corrected_sorted_conf_mask_v1` 完整独立 val，冻结路径/SHA/代码/指标/理由至 val_selection.json。训练 CSV 峰值不算独立 val。
8. 同入口 `test`：只读取固定 val_selection.json，不扫描或选择其他 checkpoint。协议为640、batch16、workers0、device0、halfFalse、conf0.001、iou0.7、max_det300、augmentFalse、rectFalse、seed42，以及母版其余设置。
9. 同入口 `pack`：生成 LIGHT 包并校验成员 SHA。源文件、身份/配置、完整原始数值、小报告和日志尾部（每份至多32KiB，明确字节范围）进入包；正式 best/last、数据和大型逐预测输出不进入 Git/LIGHT。

本地补充 CUDA 直接 optimizer 更新一致性为 REVIEW_REQUIRED（CPU 对照和原生 AMP 分别已通过）；详见 VALIDATION.md。服务器 preflight 若重现该未通过状态，也会写 REVIEW_REQUIRED，start 拒绝启动。GradScaler 的初期缩放溢出按原生 skip/backoff 保留并记录，不把回退后的 scale 固定为新训练参数。

每次调用 shell wrapper 均独立定位工作树、解释器与 PATH，不依赖前一段变量。最终回复给出填入真实完整 SHA 的 sync 和其余逐段可复制命令。交付 SHA 在提交后写入外部记录，避免文件内提交 SHA 的自引用。

## 结果解释

主比较为同协议独立 mAP50–95；辅以 AP50/AP75、十阈值 AP、P/R 及各模型最大F1工作点。历史成功组合独立 val 约52.4543%、test约52.2009%，仅为协议参照。不能拿部分 epoch 的 val 对比母版最终 test，不能用并发训练耗时当公平推理延迟，也不能把单 seed 小增益写成确定结论。

实际检查命令和 PASS/FAIL/SKIPPED 状态见 `VALIDATION.md`、对应 JSON。正式训练和正式 val/test 当前均 NOT_RUN。
