# YOLO11m COCO 可配置对比实验：交付与服务器命令

交付日期：2026-10-05（Asia/Shanghai）。分支：`bench/yolo11m-configurable`。
实施代码 SHA：`a08b072a4608dfaa92e7f30a5b3dd0c1301d2a87`。
真实直接父提交及分支基点：`a7b3d842df60adf28e5cf35c086c58ef752674f8`。
后续文档/证据提交只用于交付；服务器正式训练请检出上面的实施代码 SHA。
本次没有启动服务器正式训练，`y11_aug_x13_01` 是 **NOT_RUN 的计划配置**。

入口是 `scripts/autodl_yolo11m_configurable.sh`，实现位于 `benchmarks/comparison/yolo11m/`。
原 YOLOv8m 入口、RT-DETR、公共评估器和协议文件保持原内容。
默认配方与 `configs/y11_aug_x13_01.yaml` 采用用户文档明确列出的 `v8_aug_x13_01`
实际数值：200 轮、patience50、物理 batch16、640、workers8、GPU0、seed42、AMP，
SGD 0.01 / momentum0.937、cosine、warmup5，mixup0.234、cutmix0.065、close_mosaic5。
增强数值没有再乘 1.3。AdamW 示例仅说明可配置接口，不代表推荐更优或已验证涨点。

上游固定为官方 Ultralytics v8.3.20，commit `f4d8f7765a490f3920e2d14c592a2967e347f185`。
实测官方 `yolo11m.pt`：40,684,120 字节；SHA256
`d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`。
COCO nc80 模型未融合参数 20,114,688；crack nc1 模型 20,053,779；两者 state_dict 都是
649 个张量，643 个完全匹配并逐个验证相等，只有 `model.23.cv3.{0,1,2}.2.{weight,bias}`
六个类别输出张量因 nc80→nc1 改变。锁文件包含真实形状、来源、文件哈希和补丁哈希。
身份检查还验证 m 缩放、八个 C3k2/C3k 分支、C2PSA、非 legacy depthwise Detect、stride 和完整 YAML。
构建时复制 YAML，避免官方 nc 赋值原地修改源 COCO 模型的 YAML；没有修改检测头、损失或标签分配器。

## 1. 固定代码工作树与独立环境

以下代码块可在服务器 Bash 中依次执行。已存在的同名目录会被 Git 拒绝，不能覆盖旧工作树。

```bash
set -euo pipefail
REPO='/root/autodl-tmp/projects/Crack_RTDETR'
WT='/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-configurable'
CODE_SHA='a08b072a4608dfaa92e7f30a5b3dd0c1301d2a87'
git -C "$REPO" fetch origin bench/yolo11m-configurable
git -C "$REPO" cat-file -e "${CODE_SHA}^{commit}"
git -C "$REPO" worktree add --detach "$WT" "$CODE_SHA"
cd "$WT"
test "$(git rev-parse HEAD)" = "$CODE_SHA"
SCRIPT="$WT/scripts/autodl_yolo11m_configurable.sh"

# 只读使用兼容的母 Python；新 venv 与覆盖依赖都写入本工作树。
V8='/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-coco-b19-pilot'
BASE_PYTHON="$(command -v python3)"
if [[ -x "$V8/.envs/yolov8m-coco-b19-pilot/bin/python" ]]; then
    BASE_PYTHON="$V8/.envs/yolov8m-coco-b19-pilot/bin/python"
fi
BOOT_ARGS=(--base-python "$BASE_PYTHON")
V8_SOURCE="$V8/.vendor/yolov8m-coco-b19-pilot/ultralytics-v8.3.20"
if [[ -d "$V8_SOURCE/.git" ]]; then
    BOOT_ARGS+=(--reuse-source "$V8_SOURCE")
fi
bash "$SCRIPT" bootstrap "${BOOT_ARGS[@]}"
PY="$WT/.envs/yolo11m-configurable/bin/python"
"$PY" -c 'import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix)'
command -v tmux flock
```

bootstrap 只准备并校验 `.vendor/yolo11m-configurable/`、`.envs/yolo11m-configurable/` 和
`.runtime/yolo11m-configurable/`，不会训练。Linux venv 使用 `--copies`；解释器路径不会通过
`resolve()` 跟随软链接回母环境。只读源必须通过固定 SHA、补丁和内容哈希校验才能复制。
权重从锁定的官方 URL 下载；可以添加 `--reuse-asset /absolute/original/yolo11m.pt` 复制哈希一致的原始资产。
不能用 v8 best、scratch last 或其他 YOLO 尺寸。安装采用私有环境中的锁定 overlay 和 `--no-deps`，不升级母环境的 Torch。
母依赖不足会真实失败并保留 probe 日志，需要准备兼容的母 Python 后再次 bootstrap。

## 2. 查看和校验默认配置（不会训练）

后续代码块在上面创建的工作树执行；如换了 SSH 终端，先重设这四个变量。

```bash
WT='/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-configurable'
cd "$WT"
SCRIPT="$WT/scripts/autodl_yolo11m_configurable.sh"
PY="$WT/.envs/yolo11m-configurable/bin/python"
bash "$SCRIPT" --help
bash "$SCRIPT" check-config
bash "$SCRIPT" check-config --config "$WT/benchmarks/comparison/yolo11m/configs/y11_aug_x13_01.yaml"
bash "$SCRIPT" check --output "$WT/.runtime/yolo11m-configurable/manual_identity_check.json"
```

`check-config` 只解析配置；`check` 校验真实官方模型与迁移，不进行训练。
`check` 输出必须选一个尚不存在的文件。支持 YAML 完整配置或合法部分配置，以及重复使用的
`--set key=value`，优先级固定为 **默认 < YAML/克隆配方 < CLI 覆盖**。
bool/int/float/null 被明确解析；支持科学计数法。未知键、重复 YAML 键、重复同名 --set、NaN/Inf、
AutoBatch、optimizer=auto、非法概率/类型/组合、模型/类别/协议/初始化变更会被拒绝。
schema 见 `config_schema.json`。batch 允许 1–128，训练内验证跟随该物理 batch，始终 workers0。
imgsz640、device0、seed42、cache=false、rect=false、multi_scale=false、copy_paste=0、bgr=0 等固定；
非零 copy_paste 明确拒绝。amp/deterministic/cos_lr 可配置并冻结。分类 erasing/auto_augment 固定为无效值，不增加分类增强。

## 3. 默认配方正式启动（本次交付未执行）

```bash
bash "$SCRIPT" start --run-id y11_aug_x13_01
```

start 先校验配置，独占创建本 run 并冻结输入，然后新建 `comparison-yolo11m-y11_aug_x13_01`。
交互式普通 SSH 自动 attach；已有 tmux 内自动 switch-client；添加 `--detach` 可返回终端。
流程为 preflight → train → 独立 FP32 public val 导出 → 公共 CPU evaluate → summary。
test 保持 `not_requested`。输出独立落在 `outputs/yolo11m-configurable/y11_aug_x13_01/`。
与 v5/v8 共用 GPU0 被允许；没有 GPU 空闲门槛、整卡锁、自动停止其他任务或 OOM 改参重试。
默认数据路径沿用 `/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml` 与原数据根。
自定义输入路径只能在新的 start/prepare 时使用 `--data /absolute/data.yaml --data-root /absolute/root`。

每个 run 保存原 YAML（没有 YAML 则明确记录使用默认）、CLI JSON、resolved YAML、完整 native args、
各层配置哈希、完整命令、代码/上游/补丁/环境/初始化身份和数据清单。候选文件后续被修改不影响该 run。
禁止复用 run-id 调参；原输出和已存证据不会被覆盖。运行期间不要 checkout、pull、修改代码或更换依赖。

## 4. 服务器生成 YAML，再用 --config 启动新 run

不需要上传本地 YAML。

```bash
bash "$SCRIPT" config --out "$WT/runtime_configs/y11_yaml_lr005.yaml" \
    --set lr0=0.005 --set momentum=0.9 --set mixup=0.18
bash "$SCRIPT" check-config --config "$WT/runtime_configs/y11_yaml_lr005.yaml"
bash "$SCRIPT" start --run-id y11_yaml_lr005 \
    --config "$WT/runtime_configs/y11_yaml_lr005.yaml"
```

生成命令拒绝覆盖已有候选文件。也可编辑服务器上的候选 YAML；下一次更换 run-id 即可使用新配置。

## 5. 只用 --set 启动另一个 SGD 候选

```bash
bash "$SCRIPT" start --run-id y11_sgd_lr005_m09_mix018 \
    --set lr0=0.005 --set momentum=0.9 --set mixup=0.18
```

## 6. AdamW 候选（改变首轮共同优化器配方）

```bash
bash "$SCRIPT" check-config --set optimizer=AdamW --set lr0=0.001 --set momentum=0.9
bash "$SCRIPT" start --run-id y11_adamw_lr001_m09 \
    --set optimizer=AdamW --set lr0=0.001 --set momentum=0.9
```

支持 SGD/Adam/AdamW。Adam/AdamW 的 momentum 是原生 beta1，beta2=.999、eps=1e-8；
SGD 使用原生 Nesterov。参数组、有效 decay 缩放、初始/逐轮 LR、warmup iterations、梯度累积、
step 调用次数、真实 optimizer updates 和 AMP overflow skips 均有报告。
nbs64 是参考 batch；默认物理 batch16 的 warmup 后名义累积为4。
可选克隆入口示例：先对已有 run 的冻结配方检查，再以新 run-id 启动；模型始终重新从官方 COCO 初始化。

```bash
bash "$SCRIPT" check-config --clone-config-from y11_aug_x13_01 --set lr0=0.005
bash "$SCRIPT" start --run-id y11_clone_lr005 \
    --clone-config-from y11_aug_x13_01 --set lr0=0.005
```

## 7. 查看、离开 tmux 与显式恢复

```bash
bash "$SCRIPT" status --run-id y11_aug_x13_01
tmux list-sessions
tmux attach -t 'comparison-yolo11m-y11_aug_x13_01'
# 在训练窗口按 Ctrl-b 然后 d，只离开会话，训练继续。
# 已在另一个 tmux 时：
tmux switch-client -t 'comparison-yolo11m-y11_aug_x13_01'
# 经检查确认是合法中断、last/best 和身份完整后：
bash "$SCRIPT" resume --run-id y11_aug_x13_01
```

resume 拒绝 --config/--set/路径变化；只读取原配置、环境、数据和代码身份，以及同一 run 的 last。
恢复 FP32 live model/optimizer/EMA、scheduler、scaler、随机流、best/patience、关闭增强状态和共享计数。
CSV 与 epoch trace 从 checkpoint 恢复，保存原日志副本。已完成或已到 patience 的 checkpoint 禁止续训。
没有逐比特复现声明：prefetch worker 样本和跨轮未更新梯度没有重放。
status 区分未启动、preflight 失败/进行中、训练中、合法完成/早停、val/test 各阶段状态；
没有 completion 的已退出进程会显示失败。无法获取的退出码保留未知，stage/pipeline 日志单独保存真实退出码。
同 run 的活跃 session/锁会拒绝重复启动，其他模型的 session 不受影响。

## 8. 完成训练后的 test、两种 pack 与 Git 归档

```bash
bash "$SCRIPT" status --run-id y11_aug_x13_01
bash "$SCRIPT" finalize --run-id y11_aug_x13_01
bash "$SCRIPT" status --run-id y11_aug_x13_01
bash "$SCRIPT" pack --run-id y11_aug_x13_01 \
    --out "$WT/outputs/yolo11m-packages/y11_aug_x13_01_review.tar.gz"
bash "$SCRIPT" pack --run-id y11_aug_x13_01 --include-weights \
    --out "$WT/outputs/yolo11m-packages/y11_aug_x13_01_with_weights.tar.gz"
```

finalize 只接受有证据的 epoch-limit/patience 合法完成，校验 best/last 原哈希，复用已验证的同一 best/val，
补做 test；不训练、不下载、不更新代码。public val/test 固定 FP32、640、batch16、workers0、seed42、
conf.001、NMS IoU.7、max_det300、class-aware、rect=false、无 TTA；test 不参与选模。
原生 NMS 只执行一次并移除时间预算截断；按真实整数 resize/pad 与独立 x/y gain 反变换，
保留 padding-only false positives，每张原图都有稳定 image-id 记录，包括空预测。
训练 square stretch、MotherHSV、锁定检测 CutMix 与推理 letterbox 分别记录。
第196轮起关闭 Mosaic/MixUp/CutMix/CopyPaste并重建真实 worker；HSV/几何/翻转保留。
公共 AP 完全复用 `corrected_sorted_conf_mask_v1`，summary 区分训练选择指标与公共结果。

默认 pack 是 **排除 .pt 的审查包，不是权重备份**；include-weights 同时包含 best/last。
两者均只读原 run，校验配置/代码/源/数据/结果身份，附清单和 SHA256，记录原权重路径及是否包含权重。
不收集数据集、环境、缓存。训练中只允许标明不完整的审查快照；含权重模式要求训练合法完成。

`outputs/`、`runtime_configs/`、`.runtime/` 被忽略，**不会自动同步到 GitHub**。
请在单独归档工作树提交轻量记录，保留训练工作树的原始 SHA；以下“生成文件”和“提交/推送”是两个步骤。

```bash
REPO='/root/autodl-tmp/projects/Crack_RTDETR'
ARCHIVE_WT='/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-archives'
ARCHIVE_BRANCH='bench/yolo11m-run-archives'
git -C "$REPO" fetch origin bench/yolo11m-configurable
git -C "$REPO" worktree add -b "$ARCHIVE_BRANCH" "$ARCHIVE_WT" origin/bench/yolo11m-configurable
RID='y11_aug_x13_01'
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
REL="docs/comparison/archives/yolo11m/${RID}_${STAMP}.json"

# 生成真实冻结配置/args/身份/轻量结果；若训练 run 尚不存在，明确归档为 NOT_RUN。
bash "$SCRIPT" archive-config --run-id "$RID" --out "$ARCHIVE_WT/$REL"
# 下面才是 Git 归档与推送；上面的生成命令不自动执行这些操作。
git -C "$ARCHIVE_WT" add -- "$REL"
git -C "$ARCHIVE_WT" commit -m "archive(bench): record ${RID} frozen configuration and evidence"
git -C "$ARCHIVE_WT" push -u origin "$ARCHIVE_BRANCH"
git -C "$ARCHIVE_WT" ls-remote --heads origin "refs/heads/$ARCHIVE_BRANCH"
```

归档工作树/分支若已存在，使用原归档工作树和新的 STAMP，不能重建覆盖。
训练前的 NOT_RUN 配置可带 --set/--config；已有 run 的归档禁止覆盖，必须来自冻结证据。
已完成实验只归档实际有证据的状态，权重仅记录位置/哈希；大权重不进入普通 Git。

## 验证范围与复查指令

本机实测：34 项通过、1 项 POSIX signal 测试因 Windows 跳过；Bash语法、CLI help、配置生成/校验、
真实初始化、两种 pack、实际与 NOT_RUN 归档通过。真实本地数据指纹
`3401e485b40398ae096e8fd00101cae38b607ee1d6d1b5d5abba5c443e273483` 与 v8 参考一致，
train/val/test 图框数全部相符。采用参考 v8 light inventory：标签字节、相对路径、图片字节数/mtime和必要尺寸；
没有重新做全图片内容哈希、全图解码或 family 审计。
干净实施提交下的 CUDA SMOKE_ONLY 完成两轮、4 次真实更新、第1轮 checkpoint 中断/恢复、
两张 val 与两张 test 的独立 FP32/640 导出和公共 CPU 评估；未进行正式数据训练或正式全 split 推理。
真实数据 preflight 同时通过 loss/FP32 backward/AMP backward 和 RNG 保护检查，未创建正式 optimizer 更新。

本机环境为 Python3.9.25、Torch2.7.1+cu118、RTX2060 6GiB；上游发出 Python>=3.10 提示，
上述路径实际通过，版本原样记入环境报告；服务器以自身 probe 为准。
CUDA smoke 的显式短预算为 2轮/batch2/nbs2/64/workers0，以及验证脚本专用 unit loss scaling，
不把小预算结果当作正式200轮性能。formal trainer 仍采用原生 scaler/动态backoff。

**NOT_RUN：服务器 bootstrap、Linux 真实 tmux attach/switch、POSIX process-group SIGTERM、
200轮/合法早停的正式训练、真实 val/test 全 split 推理、独占硬件测速。**
原因：本轮在 Windows 本地实施，用户保留服务器正式启动时机；只验证了 Linux shell 语法和 tmux shim 参数/会话保护。

可在固定实施工作树中运行以下复查；输出文件必须不存在。前三组不会正式训练，最后一条只进行明确的合成短检查。

```bash
bash -n "$SCRIPT"
"$PY" "$WT/benchmarks/comparison/yolo11m/test_configurable.py"
"$PY" "$WT/benchmarks/comparison/yolo11m/test_b19.py"
"$PY" "$WT/benchmarks/comparison/yolo11m/test_initialization_guards.py"
"$PY" "$WT/benchmarks/comparison/yolo11m/test_lifecycle.py"
"$PY" "$WT/benchmarks/comparison/yolo11m/smoke.py" \
    --output "$WT/outputs/yolo11m-configurable-validation/server_cpu_smoke.json"
"$PY" "$WT/benchmarks/comparison/yolo11m/cuda_smoke.py" \
    --out "$WT/outputs/yolo11m-configurable-validation/server_cuda_SMOKE_ONLY.json"
# 可选 nc1 fused/unfused 复杂度（不测并行GPU论文速度）：
bash "$SCRIPT" measure --output "$WT/outputs/yolo11m-configurable-validation/nc1_complexity.json"
```

详细交付证据在 `docs/comparison/evidence/yolo11m_configurable_delivery.json`，
首轮计划归档在 `docs/comparison/archives/yolo11m/y11_aug_x13_01_NOT_RUN.json`。
官方来源：[YOLO11说明](https://docs.ultralytics.com/models/yolo11/)、
[固定上游代码](https://github.com/ultralytics/ultralytics/tree/f4d8f7765a490f3920e2d14c592a2967e347f185)、
[COCO源资产](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt)。
