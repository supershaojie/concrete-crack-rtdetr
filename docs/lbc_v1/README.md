# LBC v1 独立实验

源码母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。分支：`exp-rtdetr-r18-lite-lbc-v1`。
组合固定为 RT-DETR-R18-Lite＋原 LIF-Down＋原 CBR＋LBC。
本交付是实现和有限工程验证，没有执行正式 200 epoch 训练，没有本实验的 val/test 性能结论。

## 定义与接线

`ultralytics-main/ultralytics/models/rtdetr/lbc.py` 定义稳定、可导入的训练模型类和辅助头。
先用母版工厂构建并执行原 `verify_model` 严格类型审计，再显式实例化训练子类，复制原 Module 状态。
没有运行时替换 `__class__`、全局 monkey patch、永久 hook、第二次模型前向或 live 特征缓存。
真实 `LBCTrainer.get_model` 重建后也执行此流程，nc=80→1 的允许适配仍是原来的 9 个具体类别 keys。

第 5 层是 backbone S3（640 时 B×128×80×80），不是第 19 层 Neck P3。
激活时 `predict_with_s3` 在一次原 traversal 中局部返回 `(原输出, F3)`；第 6 层和第 17 层仍读取原 S3。
公开 `predict` 直接继承母版。第 20 层 LIF、第 26 层 CBR（读取 19/22/25）和全部原损失源码均保持不变。

辅助头为无 bias 的 128→32 1×1 卷积与 32 维可训练 prototype，只有两个新 state keys：
`lbc_head.proj.weight`、`lbc_head.prototype`，共 4,128 参数，无新 buffer。
CPU 构造使用隔离 `fork_rng`，seed=424001；卷积使用 PyTorch 默认 Kaiming-uniform，prototype 标准正态。
在关闭 autocast 的 FP32 域中，投影及 prototype 按通道 L2 归一化（eps=1e-6）后点乘得到余弦分数 Z。
F3.float() 保留梯度，非有限特征/参数/得分会报错。辅助分数不乘回正式特征。

区域完全来自当前增强后的 `batch['bboxes']`（normalized cxcywh）、`batch_idx`、`cls` 和当前画布 H/W。
GT 不可导；转为像素 xyxy 后裁边；非法/退化区域跳过计数。非法 batch 格式明确报错。
网格点为 ((col+.5)W/Wf, (row+.5)H/Hf)，所有区间左上含、右下不含。
P 是框内正 bag；每个框 E 各边扩 max(一个网格单元,0.1×框边长)；目标外框 O 各边扩 max(四个单元,0.5×边长)。
N 为 O 内、同图所有 E 之外的点。重叠正 bag 不删除；无正点或 N<16 时跳过，不扩大极细正框。

P/N 各自按升序 row-major 取至多 128 个点；n>128 时采样序号为 floor(t(n−1)/127)，t=0..127。
不消耗 RNG，不按得分预筛。共享 k=min(4,|P_sampled|,|N_sampled|)，分别取 top-k 分数平均 s_pos/s_neg。

```text
ell_pair = softplus((0.20 + s_neg - s_pos) / 0.20)
ell_bg   = softplus(s_neg / 0.20)
L_LBC    = mean_over_valid_GT(ell_pair + 0.25 * ell_bg)
r(e)     = clip((e - 5) / 15, 0, 1)
L_total  = L0 + 0.05 * r(e) * L_LBC
```

mean 分母是当前 micro-batch 的有效目标数，按 GT 等权，不按图先平均。e 是 epoch 开始前已完成轮数；e=5/6/20 对应 0、1/15、1。
r=0 完全走母版 loss，不执行辅助头/区域/top-k。激活但 M=0 返回 L0，不产生 head 梯度，也不清除之前 micro-batch 的贡献。
诊断保存在独立普通数值字典，不进入 criterion 会自动 sum 的损失字典；原三项日志不增加维度。

调用链是 `BaseModel.forward(dict)` → `LBCDetectionModel.loss` → 原 `RTDETRDetectionModel.loss(batch,preds)` →
原 `RTDETRDetectionLoss` 完整 encoder/aux/DN/VFL/L1/GIoU 求和 → 加一次标量 LBC →
`BaseTrainer._do_train` 原 `loss.sum()` 和 DDP world_size 缩放 → native GradScaler backward。
这个母版外壳没有额外 batch 乘数；本实现没有加入 batch 乘除。
母版主项 1×VFL+5×L1+2×GIoU，matcher cost 2/5/2，未用通用 args 的 box/cls 改写它们。

LBC 单项直接梯度到 head 和 S3 及之前的骨干；第 6 层及以后、Neck、LIF、decoder、CBR 无直接附加梯度。
总损失仍通过 L0 训练全部原模块。联合训练会产生间接影响，不宣称后续层完全不受影响。

## 优化器、生命周期与部署

仍用同一个 native AdamW 和原参数组规则。两个新增参数均进入普通 `weight` decay 组：lr0=0.0005、decay=0.0001，
不使用特殊倍率或额外训练优化器。原 nbs=64、accumulation、warmup 和 cosine scheduler 保留。
真实母版 clip 阈值是 **10.0**：一次原 `unscale_` 后，原网络与 head 分别 clip_grad_norm_(10.0)，不再全体重复裁剪。
随后保留 scaler.step/update、zero_grad(set_to_none=True)、EMA 顺序；记录真实 overflow 跳步。
连续 16 个 optimizer 边界均 overflow 时明确失败，不改 batch/nbs/AMP。

母版已保存 scaler，但未保存完整 scheduler/累积窗口，且结束时会 strip best/last。
本实验保存 raw FP32 模型、native FP16 EMA、FP32 optimizer、scaler、scheduler、early stopper、epoch、有效更新数、
累积边界和未完成窗口的 scaled gradients、主进程 RNG 及 LBC 固定版本。
`trainer.py` 只增加两个默认行为不变的扩展点，允许实验恢复 last_opt_step 和训练开始时的 pending gradients。
正式结束仍保留 best/last 和 optimizer；不调用原地 strip。损失、训练循环步进/缩放和原验证选模规则未重写。
恢复只支持同配置、同 loader 长度的 epoch 边界；恢复 e=20 时 ramp=1。
不保证任意 batch 中断逐位复现，且多 worker 的预取/增强 RNG 不能仅凭主进程 RNG 完全恢复。

部署构建标准 `RTDETRDetectionModel`，严格加载全部 552 个原公共 state entries，仅去掉两个 head keys。
另外完整复制原 decoder 的 `shapes/anchors/valid_mask` 普通缓存属性，它们未注册在 state_dict 中；
遗漏缓存会使 FP16 EMA 保存后的 rounded anchors 与重新生成的 FP32 anchors 不同。
部署前后均在 eval/FP32/no-grad 路径比较，输出逐位一致后写独立 `_deploy.pt`，不覆盖 best/last。
参数数：原模型 20,149,765；训练模型 20,153,893；部署未融合 20,149,765；本次融合实测 19,944,965。
普通 predict/val 不调用 head；验证器传 preds 计算 loss 时只报告 L0，绝不补做前向。

严格融合检查局部禁用 autocast、cudnn TF32、matmul TF32，融合后再次 eval，finally 恢复所有精度设置。
仅复用 RDL `91abf119f92a0b930fee933f30a2e6950143f023` 的独立精度作用域修正思路，未合并或依赖其损失/训练文件。
使用母版工具原容差 atol=2e-5、rtol=2e-4。先保存原始逐行结果；仅在需要时调用母版候选 ID 验收。
集合漂移/不完整一一映射/真实超差均拒绝，不调整生产输出顺序。本地此次为普通 PASS_ORDERED。

## 配方、数据与结果边界

完整 109 字段配方来自 `docs/c19_lif_v1/c2_args.yaml`；prepare 写 `train_args.yaml`、`resolved_args.yaml`、逐字段 `recipe_diff.json`。
仅 model/name/project/save_dir 和等价数据路径作为身份字段变化。LBC_CONFIG 单独保存，不塞入通用 Ultralytics 参数。
本地核验实际解释器 Python3.9.25、torch2.7.1+cu118、CUDA11.8、RTX2060，加载的 Ultralytics 来自本 worktree。
服务器入口要求指定 Python3.10.13/torch2.1.2+cu121/RTX4090；不会自动升级依赖。
统一源 SHA256 为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`，epoch=-1、nc=80，
经原 pair 初始化和 Trainer nc=1 适配；不使用训练后 best/last 作为初值。

物理 train/val/test 数量 6048/1728/864，框数 45573/12840/6663；路径与标签指纹逐项匹配母版现有核验结果。
身份检查会读取三个 split 的路径和标签，不进行 test 推理；机制检查仅使用两个 train mini-batch。
原有增强同源图跨 split 的限制保留，不重新划分数据或改变增强。
画布几何背景可能包含 padding/漏标，本次图示确实看到部分选中负点位于灰色 padding 区，不能称为确认真实背景。
局部困难背景是候选假设，当前工程验证不能证明提高检测性能。

框监督 MIL 是已有思想，例如 [Hsu 等，NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/e6e713296627dff6475085cc6a224464-Abstract.html)。
本候选研究同图局部背景参照、稀疏高分聚合与检测骨干辅助监督的组合；没有像素 GT、逐行约束、SAM/Scharr 伪标签、预测框或 teacher。
任务书短检段出现的“LCD 零输出分支”属于另一实验，本 LBC 非零初始化定义优先，未引入 LCD。

## 操作入口

固定 worktree `/root/autodl-tmp/projects/Crack_RTDETR-lbc_v1`；解释器 `/root/miniconda3/envs/rtdetr/bin/python`；
固定 tmux `lbc-v1-training`；正式 run 父目录 `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series`，
run 名 `lbc_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`；工程输出本 worktree 的 `outputs/lbc_v1`。

`tools/sync_lbc_v1.sh FULL_SHA` 固定 origin/分支并要求实时远端 SHA 相同；支持首次创建、同 SHA 重入、干净前进更新。
先更新新代码再写交付记录；冲突的 tracked/untracked/ignored 文件均拒绝覆盖；备份旧交付记录，保留所有其他 worktree HEAD。
prepare 只核验并初始化；preflight 独立运行 CPU 检查与服务器有界容量检查，不派发正式训练。
容量检查为 B16/640/原 AMP/nbs64，e=20，最多 16 micro-batch、900 秒、2 次真实更新；至少 1 次有效更新才通过。
只取两个真实增强 batch；必要的后续累积复用它们，记录这一事实，最多 16 张示意图。父进程限制 900 秒。
短检模型/optimizer/scaler/目录均隔离，不会作为正式初值续训，不自动降低 batch/nbs/关闭 AMP。

start 只核对与源码/配置/初值对应的有效 preflight，不自动重跑；新建独立 tmux 并先报告 DISPATCHED。
实际 Python 进程在 session 内经 tee 写 `console_<UTC>.log`，保存 Python 退出码、时间及进程 token；首个真实训练 batch 后才记 RUNNING。
status 检查 PID/token、子进程、session、日志和结果 CSV；已有本实验进程/session 拒绝重复启动，不干预其他实验。
resume 校验版本、完整 last.pt、配方和身份后使用同一个独立 tmux。

val/test 使用母版 `corrected_sorted_conf_mask_v1` 独立评估标准部署模型；保持 640/B16/workers0/FP32/conf.001/iou.7/max_det300。
仍按母版 native val fitness 选 best。val 锁定 best 与 deploy SHA，test 对锁定权重只允许一次尝试，不扫描其他 checkpoint。
完整 AP50/AP75/十阈值 AP/P/R 和预测 GT 流由原评估工具输出。历史独立 val/test mAP50–95=52.454272%/52.200902%，本轮未复测。
母版评估函数仅增加默认关闭的结果回调；LBC 用它从同次 confidence curves 输出原 max-F1 工作点及 P≥0.80/0.90/0.95 时的最大召回，
这些是 IoU0.5 辅助指标，不改变 AP、选模、生产阈值或推理次数。
pack 只收集源码差异、配置、报告、CSV 和日志尾部；不带权重/数据，LIGHT 校验小于 8MB。

服务器各动作：`tools/lbc_v1.sh prepare|preflight|start|status|resume|val|test|pack`。
可直接使用固定解释器运行绝对路径的 `tools/lbc_v1.py`。交付回复提供含实际完整 SHA 的独立短命令。
查看空 pane 时应使用 `status`、`tail` 实际 console 文件，或 `tmux list-panes -t lbc-v1-training -F '#{pane_pid} #{pane_current_command} #{pane_dead}'`。

本地已有验证与服务器待办见 `VALIDATION.md`；源代码/报告的可追溯指纹见 `local_evidence.json`。
