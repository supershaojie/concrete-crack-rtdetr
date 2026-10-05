# YOLO26m 可配置 COCO 比较实验

本轮交付工具，正式 `y26m_musgd_aug_x13_01` 为 **NOT_RUN**，启动时间由用户决定。
分支：`bench/yolo26m-configurable`。分支基点（YOLO11m 固定代码）为
`a08b072a4608dfaa92e7f30a5b3dd0c1301d2a87`。

服务器正式训练应检出实施代码 **`a25e6379be06970f1aa02fbb85d2c9a614a6436f`**，
其直接父提交为 `554f4533b1ac1ecf87fd7116cf9e33f8d7e128e9`。
后续分支 HEAD 可能只包含本说明和验收证据；不要用它改写已冻结 run 的代码身份。

本工具使用官方 YOLO26m detect、nc1、scale m、完整双分支、end2end=true、reg_max=1，
原始官方 COCO80 权重 `yolo26m.pt`。模型/损失/优化器固定为 Ultralytics v8.4.0，
`f2d3aed634a5b0e4828024718d4a61ab2f83fb19`，这是复现锁定版本。
实际下载资产已在交付环境重新核验：44,255,705 字节，
SHA256 `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`。
每个服务器复用/下载文件仍须重新校验，失败会停止。

默认配方的 41 个共同字段中，40 项数值沿用 `v8_aug_x13_01`，唯一显式改变的共同字段
是 optimizer=MuSGD。这是官方优化器算法加本项目训练配方；架构、匹配、损失公式和实际
参数组 LR/衰减不相同，不能称为所有模型超参数完全相同或官方完整 COCO 配方。

## 准备独立工作树和环境（不训练）

下面路径使用已确认的 AutoDL 项目布局。已有同名工作树时先检查身份和未提交文件；
以下命令保护已有目录，不执行覆盖、reset、清理或强推。

```bash
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-configurable
CODE_SHA=a25e6379be06970f1aa02fbb85d2c9a614a6436f
git -C "$MAIN" status --short
git -C "$MAIN" worktree list
git -C "$MAIN" fetch origin bench/yolo26m-configurable
git -C "$MAIN" cat-file -e "${CODE_SHA}^{commit}"
test ! -e "$WT"
git -C "$MAIN" worktree add --detach "$WT" "$CODE_SHA"
cd "$WT"
test "$(git rev-parse HEAD)" = "$CODE_SHA"
test -z "$(git status --porcelain --untracked-files=normal)"

bash scripts/autodl_yolo26m_configurable.sh bootstrap \
  --base-python /root/miniconda3/envs/rtdetr/bin/python
PY="$WT/.envs/yolo26m-configurable/bin/python"
test -x "$PY"
"$PY" -c 'import sys; print(sys.executable); print(sys.prefix)'
```

bootstrap 创建 `.envs/yolo26m-configurable`（venv --copies）、私有源码
`.vendor/yolo26m-configurable/ultralytics-v8.4.0`、运行配置与官方资产
`.runtime/yolo26m-configurable`。仅在私有 venv 安装锁定 overlay；读取并检查基础
Torch/TorchVision，不修改其他训练环境。若基础依赖不兼容，会明确失败，不能升级母环境
或退回 SGD/不同模型。可选 `--reuse-source /absolute/pinned/repository`
只复用核对过 SHA 的 Git 对象并重新检出；`--reuse-asset /absolute/yolo26m.pt`
只读核验后复制原始官方资产。bootstrap 的最终 check 会实际构造 MuSGD，但不训练。

bootstrap/base probe/check/config 都不启动正式训练。

## 查看默认配置，生成候选（不训练）

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-configurable
bash scripts/autodl_yolo26m_configurable.sh --help
bash scripts/autodl_yolo26m_configurable.sh check-config
bash scripts/autodl_yolo26m_configurable.sh check-config \
  --config "$PWD/benchmarks/comparison/yolo26m/configs/y26m_musgd_aug_x13_01.yaml"

bash scripts/autodl_yolo26m_configurable.sh config \
  --out "$PWD/runtime_configs/y26m_musgd_lr005_mix018.yaml" \
  --set lr0=0.005 --set momentum=0.9 --set mixup=0.18
bash scripts/autodl_yolo26m_configurable.sh check-config \
  --config "$PWD/runtime_configs/y26m_musgd_lr005_mix018.yaml"
```

完整 YAML 和合法部分 YAML 均可使用。优先级是提交的默认配置 < YAML/克隆配方 < --set。
bool/int/float/null 按类型解析；拒绝未知键、重复 YAML 键、重复 --set、NaN/Inf、
AutoBatch、optimizer=auto、非法概率和不支持组合。优化器支持 MuSGD/SGD/Adam/AdamW。
模型、初始化、类别、任务、end2end、reg_max、E2ELoss 结构和最终公共协议不开放覆盖。
目前 cache/rect/multi_scale 仅支持 false，copy_paste/bgr 仅支持 0，freeze/classes 必须 null。
Adam/AdamW 的 momentum 是 beta1；原生 beta2=.999、eps=1e-8 固定，warmup_momentum
在这两种优化器中不生效，因此拒绝修改它。检测中 erasing=0/auto_augment=null 不生效。

默认 epochs200、patience50、batch16、640、workers8、seed42、nbs64、AMP。
nbs 是累积参考 batch；物理 batch 为16。每个 run 的训练内验证随物理 batch，workers0。
正式训练 imgsz 固定640；64像素缩小预算仅用于内部 SMOKE_ONLY 验证。
完整字段及范围在 `benchmarks/comparison/yolo26m/config_schema.json`。

## 正式启动（下面各命令由用户择一执行）

默认 MuSGD 首轮：自动创建独立 tmux，会在交互终端进入会话；已在 tmux 中时切换客户端。
无交互终端会创建会话并返回。可加 `--detach` 显式返回。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-configurable
bash scripts/autodl_yolo26m_configurable.sh start --run-id y26m_musgd_aug_x13_01
```

使用服务器上刚生成的 YAML 启动另一 run；无需上传本地文件：

```bash
bash scripts/autodl_yolo26m_configurable.sh start \
  --run-id y26m_musgd_lr005_mix018_yaml \
  --config "$PWD/runtime_configs/y26m_musgd_lr005_mix018.yaml"
```

只用 --set 建立新 MuSGD 候选；与上面的 YAML run 使用不同 ID：

```bash
bash scripts/autodl_yolo26m_configurable.sh start \
  --run-id y26m_musgd_lr005_mix018_cli \
  --set lr0=0.005 --set momentum=0.9 --set mixup=0.18
```

单独 SGD 对照，以及 AdamW 候选。AdamW 命令改变首轮配方，没有更优或涨点证据：

```bash
bash scripts/autodl_yolo26m_configurable.sh start \
  --run-id y26m_sgd_aug_x13_01 --set optimizer=SGD

bash scripts/autodl_yolo26m_configurable.sh start \
  --run-id y26m_adamw_lr001_m09 \
  --set optimizer=AdamW --set lr0=0.001 --set momentum=0.9
```

可选仅克隆某次冻结配方：

```bash
bash scripts/autodl_yolo26m_configurable.sh start \
  --run-id y26m_musgd_clone_lr005 \
  --clone-config-from y26m_musgd_aug_x13_01 --set lr0=0.005
```

克隆不读取 best/last、不继承训练状态，新 UUID、新 run 仍从官方 COCO 初始化。
每次 start 在创建训练器前冻结输入 YAML、CLI、resolved config、完整 native args、命令、
代码/上游/补丁/环境/初始化身份。之后修改候选文件不影响 run。
输出为 `$PWD/outputs/yolo26m-configurable/<run-id>/`；重复 ID 被拒绝。

start 顺序为 preflight → train → 同一 best 的独立 FP32 val → 公共 CPU evaluate → summary。
test 保持 not_requested。早停按真实结果记录；不补轮数。
GPU0 可与其他模型共用；仅锁本 run/会话，不检查整卡空闲、不停止另一实验。
OOM/数值错误保留失败阶段与日志，不缩 batch、关闭 AMP、换尺寸/模型或自动换优化器。

## 状态、tmux 和合法中断恢复

```bash
RUN_ID=y26m_musgd_aug_x13_01
bash scripts/autodl_yolo26m_configurable.sh status --run-id "$RUN_ID"
tmux attach -t "comparison-yolo26m-$RUN_ID"
# 返回 SSH：依次按 Ctrl-b、d；这只脱离会话，训练继续。

# 查看合法中断/失败日志并确认有完整同一 run 的 last/best 后，显式恢复：
bash scripts/autodl_yolo26m_configurable.sh resume --run-id "$RUN_ID"
```

resume/finalize 只使用被冻结的配置、解释器、代码、数据和路径；不接受 --set/--config
等覆盖。恢复还核验类名、MuSGD 固定系数、组/3x LR/bias 预热身份、两套动量缓冲、
FP32 双分支模型与 EMA、scaler、scheduler、E2ELoss updates/o2m/o2o、RNG、早停、
待累积梯度、最近 optimizer-step、loader generator 和增强关闭状态。
已完成、到 epoch 上限或已到 patience 的 run 不能继续训练。
恢复限于 epoch 边界；worker RNG、预取内容、采样 cursor 和 Mosaic buffer 不保证逐比特重放。
恢复输出固定在原 `train/`，不会生成 `train2/`。

状态入口在进程退出后区分未启动、预检失败、训练中/失败/中断、合法完成、val/test 阶段。
日志包含原命令退出码和 tee 退出码，失败会话保留界面。

## 完成后 test、打包和归档

```bash
RUN_ID=y26m_musgd_aug_x13_01
bash scripts/autodl_yolo26m_configurable.sh status --run-id "$RUN_ID"
# 仅接受合法完成训练；复用已通过身份校验的同一 best 和 val：
bash scripts/autodl_yolo26m_configurable.sh finalize --run-id "$RUN_ID"
bash scripts/autodl_yolo26m_configurable.sh status --run-id "$RUN_ID"

# 默认审查包，不含 .pt；不是权重备份：
bash scripts/autodl_yolo26m_configurable.sh pack --run-id "$RUN_ID"
# 同时包含 best.pt/last.pt 及证据：
bash scripts/autodl_yolo26m_configurable.sh pack --run-id "$RUN_ID" --include-weights
```

pack 只读校验原 run；包的清单明确含权重与否、原路径及 SHA256。
排除数据集、虚拟环境、缓存。训练中仅可做标为不完整的审查快照，不能称为完成结果包。

Git 归档文件生成与 Git 提交/推送是两个步骤。下面将轻量归档写入另一工作树，保持原
训练代码 HEAD 和工作树干净；训练前可执行并记 NOT_RUN，训练后记录实际证据。
路径/分支已经存在时先检查，不能覆盖。

```bash
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-configurable
CODE_SHA=a25e6379be06970f1aa02fbb85d2c9a614a6436f
RUN_ID=y26m_musgd_aug_x13_01
STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
ARCH_WT="/root/autodl-tmp/projects/Crack_RTDETR-y26m-archive-$STAMP"
ARCH_BRANCH="archive/yolo26m-configurable/${RUN_ID}_${STAMP}"
test ! -e "$ARCH_WT"
git -C "$MAIN" worktree add -b "$ARCH_BRANCH" "$ARCH_WT" "$CODE_SHA"
cd "$WT"
bash scripts/autodl_yolo26m_configurable.sh archive-config --run-id "$RUN_ID" \
  --out "$ARCH_WT/docs/comparison/archives/yolo26m/${RUN_ID}_${STAMP}.json"

# 上一步只生成文件。以下步骤才同步到 GitHub：
git -C "$ARCH_WT" add "docs/comparison/archives/yolo26m/${RUN_ID}_${STAMP}.json"
git -C "$ARCH_WT" diff --cached --check
git -C "$ARCH_WT" commit -m "archive(bench): record YOLO26m $RUN_ID evidence"
git -C "$ARCH_WT" push -u origin "$ARCH_BRANCH"
git -C "$ARCH_WT" ls-remote origin "refs/heads/$ARCH_BRANCH"
```

## 实际实现与验收边界

COCO80→nc1 实测匹配 756/768 张量；12 个类别输出不匹配项分别来自 cv3/one2one_cv3。
迁移前后保护源 YAML，首次更新前逐项比较所有匹配张量，拒绝缺少训练分支的部署资产。
nc1 完整结构实测 21,774,430 参数；官方融合部署副本 20,350,223 参数（移除 one-to-many）。
CPU profiler 的已支持算子计数分别约 74.3386/68.0928 GFLOPs，是有明确漏算边界的部分计数，
不是完整 FLOPs；卷积/矩阵乘加按 MACs×2。独占硬件速度为 NOT_RUN。

MuSGD 固定 muon=.5/sgd=.5 是混合组两分量系数；BN/bias 等普通组不一律半速。
原生 regex `(?=.*23)(?=.*cv3)|proto\.semseg|flow_model` 匹配真实分类头张量，保留 3×LR：
基础 .01，匹配组 .03；空组也分别记录。bias 起点 .1，其余起点0，目标为各组 initial_lr×lf。
完整每组名称/形状/数量/use_muon/衰减/Nesterov/系数/源码 hash 及预热映射在实际 setup；
warmup_trace/epoch_trace 记录真实 LR、momentum、累积和步数。
reg_max1 的 dfl=1.5 权重作用于原生归一化 L1；不添加多 bin DFL。
E2ELoss 初始 .8/.2，final_o2m=.1，严格按官方 epoch-end update 保存/恢复。

增强复用参考 v8 实际数值类和 CutMix 语义：train square stretch，MotherHSV 加性 Hue，
Mosaic→CopyPaste(p0)→RandomPerspective→MixUp→检测 CutMix→NoAlbumentations→HSV→翻转→RGB Format。
close_mosaic5 在 epoch195（第196轮）关闭四个 mixer，重建 iterator/worker，保留 HSV/几何/翻转。
真实两 worker 的关闭检查已执行。伙伴仅从 train 的有限预变换读取。

实际数据轻量指纹为 `3401e485b40398ae096e8fd00101cae38b607ee1d6d1b5d5abba5c443e273483`，
train6048/45573框、val1728/12840框、test864/6663框，与参考一致。
算法沿用路径/label bytes/尺寸的 light_v1；不称为图像内容全哈希。
沿用历史划分和同源跨划分审计备注，本任务没有重划数据或解决历史同源泄漏。

最终 val/test 固定同一 best SHA，640/FP32/batch16/workers0/conf.001/max_det300/seed42，
rect=false、无TTA。使用 one-to-one 原生端到端输出，无额外 IoU NMS；公共 iou=.7/
agnostic_nms=false 字段不改变该路径，NMS_IoU=NOT_APPLICABLE。AP 匹配 IoU 阈值正常生效。
Attention 输入实际 FP32，并关闭 autocast/TF32；按真实整数 resize/pad、独立 x/y gain
反变换，保留 padding 外的正面积预测和空预测图，不重复裁剪/去重。
公共 evaluate/native_metrics/protocol 与既有 YOLO11/YOLOv8 模块均未修改。

本机 CUDA smoke 使用 RTX2060、2轮、batch2、64像素、workers0、合成图；人为在第1轮
checkpoint 后中断并从同一 last 恢复。它仅验证实现，不是正式训练或论文精度结果。
本模块49项检查中48项通过、1项POSIX信号检查在Windows跳过；公共准备检查5项通过。
Linux 实际 tmux、POSIX 进程信号、AutoDL 当前环境、全量 val/test 推理和正式200轮均为
NOT_RUN；本机已验证 Bash 语法、真实 wrapper CLI 和默认/首轮配置实际 MuSGD 构造。
详细证据见同分支 `docs/comparison/evidence/yolo26m_configurable_validation.json`。

在独立服务器环境可复查（不会启动正式 run）：

```bash
PY="$PWD/.envs/yolo26m-configurable/bin/python"
bash -n scripts/autodl_yolo26m_configurable.sh
"$PY" -m unittest discover -s benchmarks/comparison/yolo26m -p 'test_*.py' -v
"$PY" benchmarks/comparison/yolo26m/cuda_smoke.py \
  --out "$PWD/outputs/y26m_cuda_smoke_$(date +%s).json"
"$PY" benchmarks/comparison/yolo26m/run.py measure \
  --output "$PWD/outputs/y26m_complexity_$(date +%s).json"
```
