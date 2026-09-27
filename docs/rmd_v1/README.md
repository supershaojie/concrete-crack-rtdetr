# RMD v1：原 LIF + 原 CBR + Residual-Matched Denoising Loss

本实验从 `a0459d6a652cb702699087c88fa39a3e4c4087ec` 派生，分支为 `exp-rtdetr-r18-lite-rmd-v1`。只替换原正 DN 框损失的回归系数；没有正式长训结果、完整 val/test 结果或涨点结论。完整实施规格保存在 [SPECIFICATION.md](SPECIFICATION.md)。发布时的固定 SHA 和分段命令见 [server_commands.md](server_commands.md)，验证范围见 [DELIVERY.md](DELIVERY.md)。

## 模型与公式

原 YAML `rtdetr-resnet18-lite-cbr-lif-down.yaml` 不变。LIF 仍为第 20 层；第 26 层 CBR 读取 Neck `[19,22,25]`，保留 36 点采样、rho=0.10、原修框和梯度规则。decoder 为 3 层、300 常规 query，num_denoising=100，DN 噪声与注意力掩码不变。LIF 仍排除在普通 Conv/BN 融合之外。

令 e 为当前 epoch 开始前已完成的 epoch 数：

```text
r(e) = clip((e - 5) / 15, 0, 1)
rho(b,g) = [(b.cx-g.cx)/(g.w+1e-7), (b.cy-g.cy)/(g.h+1e-7),
            log((b.w+1e-7)/(g.w+1e-7)), log((b.h+1e-7)/(g.h+1e-7))]
s_jk = exp(-||rho(d_jk,g_j)-rho(a_j,g_j)||² / (2 * 0.50²))
w_jk = 1 + 0.50 * r(e) * (s_jk - mean_k(s_jk))

L1_DN[ell]   = 5 * sum_jk(w_jk * sum_coord(abs(b_DN[ell,j,k]-g_j))) / max(Ndn,1)
GIoU_DN[ell] = 2 * sum_jk(w_jk * (1-GIoU(b_DN[ell,j,k],g_j))) / max(Ndn,1)
```

`a_j` 是最终 **CBR 后普通层** Hungarian 匹配选中的 query 在 decoder **第一次进入时**的参考框。`d_jk` 是同一次前向的正 DN 初始框。输入 logits 的 sigmoid 与母版 decoder 使用相同 dtype，只做一次；然后用 detached FP32 副本计算残差和权重。GT 来自当前增强后的 batch，不读取原始标签替代增强后坐标。

最终普通匹配只做一次；encoder 和普通辅助层各自匹配。DN 正索引复用 `get_dn_match_indices`，以图像及全局 GT 身份连接。padding、负 DN 不进入权重/回归均值。未获普通匹配的 GT 保留 w=1 并计数。所有原 DN decoder 层共用同一组 w，最终 DN 层继续使用原 CBR 修正框。分类 soft IoU 标签和 detach 规则原样保留。

w 在 [0.5,1.5] 内，每个 GT 的系数均值为 1；这只守恒系数和，不守恒实际 loss 或共享参数梯度范数。e≤5 无捕获、无残差计算，直接走原损失；e=6 的 ramp 为 1/15，e≥20 为 1。K=1、全部相似度相等/下溢到零或全部权重精确为 1 时调用原 DN 损失路径。元数据在应有 DN 时缺失则报错；非有限残差或非正宽高报错，不用 `nan_to_num` 掩盖。

## 实现位置

| 文件 | 职责 |
|---|---|
| `ultralytics-main/ultralytics/models/rtdetr/rmd_v1.py` | 固定公式、最终匹配复用、DN 框项替换、临时只读 hook、训练 model/trainer 与 epoch 同步 |
| `tools/rmd_v1_common.py` | 原配方 109 字段类型和值校验、数据/权重/源码身份、初始化复用、严格 JSON |
| `tools/rmd_v1_preflight.py` | 900 秒/最多 16 个真实 B16 训练 micro-batch 的共享预算、原生更新、恢复、生效门槛 |
| `tools/rmd_v1.py` / `.sh` | 操作入口、独立 tmux、真实退出码、训练/末尾评估分开记录 |
| `tools/rmd_v1_eval.py` | 母版修正评估器、全部 300 query 导出、val 锁、test、LIGHT/FULL 包 |
| `tools/check_rmd_v1.py` / `check_rmd_v1_ops.py` | 数学、真实前向/梯度/更新、原生新进程恢复与故障测试 |

hook 仅在单次 training predict 范围存在，`finally` 移除；捕获张量只放进本次 dn_meta 的局部副本，无永久 hook、全局 monkey-patch、跨步参考框缓存或第二次训练前向。criterion 保留的只有纯标量诊断。eval/predict 调用原推理路径。

`RMDTrainer.get_model` 实际调用母版构建和加载，再给同一个对象加无参数训练包装。隔离 RNG 中重建原生对照并逐项核对公共状态。正式开始时重新校验原生优化器分组、AMP、实际 args。检查和正式训练使用独立进程、模型、optimizer、scaler、RNG 和 dataloader。

全局唯一的小修复：`nn/autobackend.py` 的 warmup 输入从 `torch.empty` 改为相同 shape/dtype/device 的 `torch.zeros`。其他 empty 和模型计算不变。母版隔离对照使用同一修复；此项不计作损失创新。

## 固定配方与身份

权威 109 字段来自母版 [c2_args.yaml](../c19_lif_v1/c2_args.yaml)，实际服务器历史 C2 args 存在时逐类型、逐值核验。完整服务器配置样例见 [formal_train_args.yaml](formal_train_args.yaml)。`prepare` 生成实际 `formal_args.yaml`、逐字段 `recipe_diff.json` 和初始化证据。只允许模型/路径/run 身份变化及明确的新损失配置，禁止用通用 box/cls/dfl 参数替代 RT-DETR 的 VFL+5L1+2GIoU 权重。

正式初始化仍用公共 nc80 ImageNet backbone 源权重，经原 `init_c19_lif_v1.initialize` 组合初始化，再由原 Trainer nc1 映射。源 SHA256 固定为 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。已训练母版 best 只用于隔离生效探测。

| 身份 | 固定值 |
|---|---|
| 服务器主仓库 | `/root/autodl-tmp/projects/Crack_RTDETR` |
| 服务器 worktree | `/root/autodl-tmp/projects/Crack_RTDETR-rmd_v1` |
| Python | `/root/miniconda3/envs/rtdetr/bin/python` |
| tmux | `rmd-v1-training` |
| run | `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/rmd_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug` |
| 工程证据 | worktree 的 `outputs/rmd_v1/` |

数据保持 crack_det 划分：6048/45573、1728/12840、864/6663（图/GT）。prepare 对三份清单、每张图和标签计算 hash；生效探测只迭代原 train 增强 loader，不读取 test 做预测或调整参数。

## 生效门槛与有界预检

固定门槛在 `configs/rmd_v1.json` 中：至少 8 个有效原 B16/640 batch；按 GT 出现次数计的 K≥2 覆盖率至少 50%；已训练母版的全部有效正 DN 平均 abs(w−1) 至少 0.01。这是防止机制几乎不工作的工程门槛，不是准确率预测或显著性标准。阈值不能根据结果下调；同一身份已有 NOT_APPLICABLE 时不能反复探测直到碰到 PASS。

默认 preflight 的真实训练预算为：4 个 e20 原生 AMP micro-batch；新 Python 恢复完整原生状态后 4 个 micro-batch 和一次原生更新尝试；8 个已训练母版只读探测 batch。三者共享最多 16 个 micro-batch、900 秒，父进程有硬超时，进度持续报告。CPU 正确性夹具及一批真实 val 也计入时间预算，但不冒充 B16 训练容量证据。

原始 scale 没有完成更新时，仅隔离诊断 checkpoint 可以把 scaler 标记为 init_scale=128，后续新进程验证它的一次更新。正式训练仍从原生初始 scaler 开始。记录真实 optimizer post-step 回调、参数变化、overflow、scale、关键梯度和显存峰值。新进程恢复检查包含原 optimizer、EMA、scaler 逐项核对以及 epoch 前进，不能把 backward 或 scaler.step 被调用当作更新成功。

缺少母版已训练权重时记 APPLICABILITY_PENDING，保留容量与正确性结果；不能用随机初始化下的相似度宣布失效。默认只查精确母版 run 的 best.pt，可用 `probe --probe-weights` 提供原权重。模型拓扑、原 run 名、已训练状态、完整来源 SHA 和文件 hash 均要通过；来源 SHA 可来自 checkpoint 的 git 字段或原 run 的 source_record/plan，不接受另一个第三方案替代。

所有必要项目必须分别 PASS，且 APPLICABILITY_PASS，start 才放行。容量降低 batch/分辨率的夹具不具备放行资格。无 GPU/数据或超时保留 PENDING，错误保留 FAIL 和 traceback。

## 生命周期、评估与包

`prepare` 可重复核对并复用相同初始化，不覆盖正式 run；`status` 读取进程、dispatch、epoch、tmux、日志、best/last 和末尾评估；`start` 只派发一次独立 tmux；`resume` 只接受同一实验有效 last 的原 optimizer/EMA/scaler，重新同步 epoch；已有合法结束证据时拒绝 resume。

原生 final_eval 若失败，保存 `TRAINING_COMPLETED` 和 `FINAL_EVAL_FAILED` 及真实 Python exit1。恢复用 `val --recover-final-eval`，核对无活动 worker、结束证据和 best hash，只重跑评估，保持原异常和退出码。未来如需修复评估代码，`sync_rmd_v1.sh SHA --eval-only` 仅允许完成训练后更新评估器/warmup 与本目录文档；再用 `val --recover-final-eval --allow-eval-revision`。训练 SHA 和训练 fingerprint 永不重写。

独立 val/test 固定 FP32、B16/640、workers0、conf0.001、iou0.7、max_det300、augment=false、rect=false、seed42，复用母版 `corrected_sorted_conf_mask_v1`，按排序后的 confidence 同时筛选对应框。模型 best 仍由母版 val fitness 选择（当前权重为 mAP50–95 的 1.0）。独立 val 锁定 best hash、代码、数据和协议后，test 只读取相同 best。参照 52.454272219190514% / 52.20090191444802% 仅用于同协议最终比较，禁止根据 test 调参。

训练 CSV 保持原 giou_loss/cls_loss/l1_loss 三列含义；它们不是包含 aux/DN 的优化总 loss。额外的 `mechanism_batches.jsonl`/`mechanism_epochs.jsonl` 记录 r、K、覆盖率、s/w 分布、每层原/新 DN 框 loss、FP32 单位权重数值差异和普通定位项。统计不引入第二次训练前向。

`pack` 不隐式训练或评估，缺少 test 或发生故障仍可生成 LIGHT。包包含实际源码 snapshot/patch、配置与身份、退出/恢复记录、检查报告、训练日志/CSV/曲线及已有评估与 val 锁，MANIFEST 逐文件记录大小/SHA256 并读回核验。LIGHT 不包含数据集和权重；best/last 只记大小/hash。`pack --include-predictions` 纳入已经由 val/test 流式写出的压缩全 300 query 预测及 GT，含 query index、图像尺寸、坐标空间、checkpoint 和协议身份，绝不提前按 0.25 筛选。`pack --full` 是可选权重包，所有包/权重/训练产物均不提交 Git。

RMD 仍是未经长训验证的单一候选，不宣称保证提升或全球首创；相关方向包括任务规格列出的 DN-DETR、Adaptive Query Denoising 和 MonoDLGD，不扩展 embedding 对比、困难样本挖掘或其他第三模块。
