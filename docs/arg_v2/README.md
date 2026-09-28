# ARG v2：ARG-v2-q2

本实验为 RT-DETR-R18-Lite + 原 LIF-Down + 原 CBR + ARG v2。
从数据身份修复后的 v1 代码 ef9cb7e05e5557f7dd06c95cf2361998a284adc9 分出
exp-rtdetr-r18-lite-arg-v2，母版基座 a0459d6a652cb702699087c88fa39a3e4c4087ec。
v1 源码、worktree、权重及 run 不被改写。v2 不继承 v1 的训练状态。
质量门控是本轮固定假设，尚未正式训练，不能宣称提升精度或首创 IoU 幂次加权。

## 唯一损失与接线

实现位于 ultralytics/models/rtdetr/arg_v2_loss.py。
参数固定 lambda_arg=0.20、epsilon_w=0.05、gamma_quality=2.0、eps_num=1e-7。
e 为已完成的 epoch 数，r(e)=clip((e-5)/15,0,1)；resume 延续 e。
只用最后一层普通 query 的最终 CBR 后框及其原 Hungarian 匹配。
最终层原匹配复用一次，encoder/早期 decoder 的原匹配与 DN 原路径保留。
L0 保留 VFL、5×L1、2×GIoU 及全部辅助/DN 项；matcher cost 保持 class/bbox/giou=2/5/2。

原归一化 cxcywh 在局部关闭 autocast 后转 FP32，再可导地转换 xyxy。
GT 停止梯度。对 b=(l,t,r,d)、g=(gl,gt,gr,gd)：

    q  = IoU(stopgrad(b), g)
    qx = IoU((gl,t,gr,d), g)
    qy = IoU((l,gt,r,gd), g)
    dx = max(qx-q,0); dy = max(qy-q,0)
    wx = 2*(0.05+dx)/(0.10+dx+dy)
    wy = 2*(0.05+dy)/(0.10+dx+dy)
    quality_gate = q*q

以上几何权重均在 no_grad 内计算，二维 union 分母保持 v1 的 clamp_min(1e-7)。
不裁剪、排序或扩大异常预测框。合法性检查保留匹配行、图像和 query 定位。
每轴区间 [u,v] 与 [gu,gv] 的可导项：

    I = max(min(v,gv)-max(u,gu),0)
    U = (v-u)+(gv-gu)-I
    C = max(v,gv)-min(u,gu)
    ell = 1-I/max(U,1e-7)+(C-U)/max(C,1e-7)
    per_match = 0.5*(wx*ell_x+wy*ell_y)
    raw_v2 = sum(quality_gate*per_match)/max(M,1)
    loss_arg = 0.20*r(e)*raw_v2
    L_total = L0+loss_arg

门控发生于逐匹配归约之前；分母为 M，不是门控和，不补偿 lambda。
q=0 时该框 ARG v2 值及框梯度为零，原 L1/GIoU 等仍监督它。
M=0 时新增项有限为零；r=0 时直接使用原 criterion 路径。
只把 loss_arg 一次加入优化字典，诊断与 v1 参考值不参与优化。
新增项位于原 batch/DDP 缩放之前。aux/encoder/DN/未匹配 query 无新项。
eval/predict 保持原输出、后处理、L0 和 fitness。

arg_v2_model.py 中的 ARGv2Trainer 在原生 get_model 重建及加载后改用可导入的
ARGv2DetectionModel/ARGv2DetectionLoss。没有新增参数、buffer 或另一个优化器。
3 decoder 层、300 queries、CBR 36 点/rho=0.10、Neck [19,22,25] 保持。
未融合 nc=1 参数量 20,149,765，state keys 552。
LIF BN 融合排除和已有 AutoBackend 零 warmup 输入修复保留。

原文件 LF SHA256：

- lif_down.py：26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7
- cbr.py：d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787

## 完整配方与初始化

[formal_args.yaml](formal_args.yaml) 与 [recipe_diff.json](recipe_diff.json) 来自母版完整
docs/c19_lif_v1/c2_args.yaml 的 109 字段。保持 B16/640、200 epoch 上限、patience50、
nbs64、AdamW、原学习率/warmup/增强、AMP、workers8、seed42。
源代码还逐值逐类型核对服务器原 C2 args（存在时）。
固定源权重 SHA256 为 fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e；
调用原 init_c19_lif_v1.initialize，沿用 nc80→nc1 原生构建/映射。
v2 初始化是独立文件，不能使用 v1 best/last 或短检权重。

## 采样诊断

仅每 epoch 前四个训练 batch 采样，保留 epoch/ramp/M、q/gate、wx/wy、
mean(abs(wx-1))、两轴 ell、门控前后 raw、weighted、原定位损失、保护计数。
保存 count/sum/sumsq，跨 batch 按匹配数计算均值/标准差。
四个 IoU 日志区间为 [0,.5)、[.5,.75)、[.75,.9)、[.9,1]；
各自保存计数、门控前后损失和，空区间均值为 null，不参与训练。
原生 optimizer post hook 记录真实更新数；区分当前 epoch 与当前进程计数。
r=0 不额外计算几何，因此 warmup 的采样为空，不伪称零匹配。
只保留 detach 后标量，不重复前向/反传；采样损失标量比不代表梯度比例。

## 固定数据快照与显式重验

保持 train6048/45573GT、val1728/12840GT、test864/6663GT。
prepare 优先复用已有 v2 快照，否则尝试主仓库同级 v1 的 outputs/arg_v1/prepare.json。
复用会校验保存清单的整体/分 split 哈希、计数、图像→标签路径映射、当前 YAML 的
完整配置及字节哈希，并验证目录元数据；不重新读取图像/标签内容。
旧 v1 清单没有目录监视记录时，要求相关目录 mtime 不晚于原 prepare 时间。
没有可用自动候选才执行一次完整 inventory；显式 --reuse-snapshot 失败会停止。
data_snapshot.json 记录 snapshot_id、manifest hash、reused_from、未复用原因与核验范围。
names 类别编号统一字符串键，键冲突报错。原 YAML 不改写。

start、resume、worker、preflight 和 evaluate 使用同一缓存身份与轻量配置/目录检查。
status/pack 不读取原始数据，也不调用 inventory。
本机制要求原始数据保持冻结：轻量检查能检测 YAML、目录替换/mtime 改动，
不能把未重读的同路径文件内容声称为刚刚全量验证。若原地编辑图像/标签或需重新确认，
显式运行 recheck-data。它执行完整校验、保存重验清单；真实差异使快照 INVALID，
保留原身份并阻止继续。数据恢复后再次显式重验，完全一致才恢复 PASS。

## 首次导出、正式评估和 finish

独立评估使用 corrected_sorted_conf_mask_v1，FP32、640/B16/workers0、
conf=.001、iou=.7、max_det300、augment=false、rect=false、seed42。
训练期 AMP val 仍按原策略运行，不用作正式 FP32 val 锁。
best 仍由原 val mAP50–95 fitness 选择，不使用 test 调参。

训练 final_eval 保留原生 last/best strip_optimizer 次序，在 best 最终字节固定后，
一次正式 FP32 val 同时产生指标、曲线、全部300 queries+GT 和 val_lock。
逐图导出包含 query 索引、图ID、原图尺寸、原图像素 xyxy、scores/classes、GT、
split、checkpoint/data/config/training/eval SHA 身份；不对导出内容做 conf 筛选。
指标保存 Precision、Recall、由两者计算的 F1、AP50、AP75、mAP50–95 和十阈值 AP。
P/R 工作点是原评估器在 IoU=.50 下平滑平均 F1-置信度曲线最大值；
记录可取得的阈值，并保存 PR/P/R/F1 数组及原生 plots。Precision 不是 Accuracy。
指标数值单位为 fraction，报告百分数时乘100。

finish：核对训练结束及 best → 复用完整正式 val 或执行缺少的首次 val →
对同一个锁定 best 执行尚未完成的 test → 一次完整打包。
成功重入不重复模型推理或选择 best；已成功的指标文件若只缺锁，会直接恢复锁。
失败的尝试与原 worker exit1/final_eval 错误保留，仅恢复缺失阶段。
已有成功结果却缺导出/曲线/身份时停止并要求显式 val/test --recover-export；
status/pack 不偷偷补推理。恢复仍锁定相同 checkpoint/data，并保存 previous_lock。
finish 会执行最终 test，应只在训练完成后由用户显式运行。

pack 始终只读取现有源码/记录/run/权重哈希，默认包含已有全部预测与 GT。
分析包包含源码及原训练快照、公式、实际 args/diff、数据快照/初始化身份、
预检、机制和训练日志/曲线、dispatch/PID/退出/恢复证据、指标/曲线、val/test 锁、
best/last 大小和 SHA256、逐成员 manifest。没有原始数据集或权重本体。
缺项时也能生成故障包，analysis_complete=false 并列出缺项。
包按输入指纹复用；完整包写完但缺 receipt 时可恢复 receipt；
中断的半包保留，后续显式调用只恢复打包阶段。

## 入口、资源与验证边界

入口 tools/arg_v2.py 的 prepare/preflight/status/start/resume/val/test/finish/pack/
recheck-data 均支持 --help。Linux 使用 tools/arg_v2.sh，
固定 /root/miniconda3/envs/rtdetr/bin/python 和本 worktree PYTHONPATH。
正式 tmux 为 arg-v2-training，pipefail 记录真实 Python/tee 退出码。
run 为 /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/arg_v2_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug。

服务器 preflight 总边界900秒/16训练micro-batch，B16/640、原 AMP/优化器/累积。
必须有真实 optimizer 更新和有限 ARG-only 梯度、实际新进程 FP32 warmup/val 与 resume。
沿用有依据的门禁：cpu/cuda_b16_amp/mechanism/new_process_val/resume 必须 PASS。
原生初始 scale 适应若 PENDING，须有单列的隔离 scale128 有效更新证据才能 start_eligible；
正式训练不改 scaler、从 e=0 开始，不继承诊断状态。
缺离线 yolo26n.pt/bus.jpg 资源即停止，不联网绕过 AMP 检查。

本地检查入口：

- tools/check_arg_v2.py：数学、自动微分、匹配路由、母版逐位模型/更新、新进程 CPU resume。
- tools/check_arg_v2_identity.py：JSON、碰撞、缓存复用、变更拒绝、显式重验。
- tools/check_arg_v2_ops.py：启动资格、锁、超时、评估身份、故障打包。
- tools/check_arg_v2_delivery.py：合成 evaluator 的一次导出/finish/恢复/打包，以及真实 exporter 的低分 query/GT。
- tools/check_arg_v2_lifecycle.py：单图真实 FP32 val/warmup，不能代替服务器 B16 预检。

本机 Windows/RTX2060 6GiB 与服务器环境不同。完整 val/test、200轮/早停训练、
GPU B16容量、CUDA scaler 恢复、训练开销与精度收益均不能用本机小测替代。
最终提交、验证证据与服务器分段命令见 DELIVERY.md 和 server_commands.md。
