# YOLOv8m COCO + B19 pilot：固定提交服务器命令

代码提交：`16a2e28686b3471ef1b7640b3ba697597335895c`。
代码实际父提交：`05c6b4c8aeaf04255b7e17f3391d2202e65d9ffe`。
发布分支：`bench/yolov8m-coco-b19-pilot`。本文随后续文档提交发布，下面始终检出上述固定代码 SHA。

本轮尚未登录服务器或启动正式训练。以下命令供用户随后执行。正式配方为官方 COCO 初始化、B19 在线增强、SGD、200 最大轮数、patience50、640、物理 batch16、nbs64、AMP、GPU0、workers8、seed42。原 scratch、原 COCO 和其他任务的工作树与会话独立保留。

## 1. 拉取固定代码并创建新工作树

```bash
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-coco-b19-pilot
CODE_SHA=16a2e28686b3471ef1b7640b3ba697597335895c
test "$(git -C "$MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
git -C "$MAIN" fetch origin refs/heads/bench/yolov8m-coco-b19-pilot:refs/remotes/origin/bench/yolov8m-coco-b19-pilot
git -C "$MAIN" cat-file -e "$CODE_SHA^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$CODE_SHA" refs/remotes/origin/bench/yolov8m-coco-b19-pilot
if test -e "$WORK"; then
  test "$(git -C "$WORK" rev-parse --show-toplevel)" = "$WORK"
  test "$(git -C "$WORK" rev-parse HEAD)" = "$CODE_SHA"
  test -z "$(git -C "$WORK" status --porcelain)"
else
  git -C "$MAIN" worktree add --detach "$WORK" "$CODE_SHA"
fi
cd "$WORK"
git log -1 --format='%H %P %s'
```

已存在的目录只有在正确 SHA 且干净时继续；检查失败后查看实际内容，不覆盖目录、不 reset。Detached worktree 固定训练版本，发布分支仍为上面的独立分支。

## 2. 独立环境、固定源码和原始权重

优先复用原 COCO 工作树的**原始**资产；`best.pt`、`last.pt`、scratch 或 YOLO26 权重均不可作此输入。若原始资产不在下述常见路径，可将 `ORIGINAL_COCO` 改为已下载原始 `yolov8m.pt` 的实际路径。找不到时从冻结 URL 下载。

```bash
BASE=/root/miniconda3/envs/rtdetr/bin/python
ORIGINAL_COCO=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m/.runtime/yolov8m/assets/yolov8m.pt
BOOTSTRAP=benchmarks/comparison/yolov8m/bootstrap.py
if test -f "$ORIGINAL_COCO"; then
  "$BASE" "$BOOTSTRAP" --base-python "$BASE" --reuse-asset "$ORIGINAL_COCO"
else
  "$BASE" "$BOOTSTRAP" --base-python "$BASE"
fi
PY="$WORK/.envs/yolov8m-coco-b19-pilot/bin/python"
ENTRY="$WORK/benchmarks/comparison/yolov8m/run.py"
test "$(cat "$WORK/.runtime/yolov8m-coco-b19-pilot/python_path.txt")" = "$PY"
"$PY" -c 'import sys; print("sys.executable:",sys.executable); print("sys.prefix:",sys.prefix); print("sys.base_prefix:",sys.base_prefix)'
cat "$WORK/.runtime/yolov8m-coco-b19-pilot/environment_probe.json"
sha256sum "$WORK/.runtime/yolov8m-coco-b19-pilot/assets/yolov8m.pt"
```

Bootstrap 使用 `venv --copies --system-site-packages`，先核对独立 `sys.prefix` 再安装冻结 overlay，不向母版环境 pip 安装。Python 路径不对 symlink 做 resolve。官方源码独立位于 `.vendor/yolov8m-coco-b19-pilot/ultralytics-v8.3.20`；每个模型入口和数据 worker 都核对真实 `ultralytics.__file__`、版本、commit、补丁和源码内容。

固定资产应为 52,136,884 bytes，SHA256：`5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`。固定模型源码为 v8.3.20 / `f4d8f7765a490f3920e2d14c592a2967e347f185`；固定 CutMix 摘录已经随本分支提交，来源 v8.4.0 / `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`，不会安装整个新版包。Bootstrap 的 `check_*.json` 记录 469/475 张量迁移和逐值校验。

## 3. 轻量 preflight

使用主项目的实际 YAML，图片与标签只读。下面只枚举已有公共准备结果目录的文件名，自动选择排序最后的完整 val/test COCO 目录；也可显式指定已有审计目录。适配器仍会逐项核对图片路径、尺寸、标签和类别，失败会保留差异。不存在已有 GT 时仅用标签和必要图像头生成 val/test GT。

```bash
DATA="$MAIN/configs/crack_autodl.yaml"
PUBLIC_COCO=''
if test -d "$MAIN/outputs/comparison_prepare"; then
  while IFS= read -r candidate; do
    if test -f "$candidate/test.json"; then PUBLIC_COCO="$candidate"; fi
  done < <(find "$MAIN/outputs/comparison_prepare" -maxdepth 4 -type f -path '*/coco/val.json' -printf '%h\n' | sort)
fi
printf 'Public GT reuse directory: %s\n' "${PUBLIC_COCO:-light generation from labels/headers}"
INPUTS=(--data "$DATA")
if test -n "$PUBLIC_COCO"; then INPUTS+=(--public-coco "$PUBLIC_COCO"); fi
CHECK_RUN="$WORK/outputs/yolov8m-coco-b19-pilot-preflight/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$RANDOM"
"$PY" "$ENTRY" preflight --run "$CHECK_RUN" "${INPUTS[@]}"
cat "$CHECK_RUN/preflight_status.json"
```

期望 train=6048/45573、val=1728/12840、test=864/6663（图片/框）；YOLO class0 → 公共 category1。此步骤复核冻结路径/标签清单及轻量身份，不全库解码、哈希或复制约 43 GB 图片。独立 cache/lock/status 位于各 run 内。这个检查目录不是随后训练的 run，正式启动会创建自己的 UUID 和清单。

## 4. 启动独立 tmux

```bash
RUN_ID="coco_b19_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$RANDOM"
RUN="$WORK/outputs/yolov8m-coco-b19-pilot/$RUN_ID"
START_ARGS=(--run-id "$RUN_ID" --data "$DATA" --base-python "$BASE")
if test -n "$PUBLIC_COCO"; then START_ARGS+=(--public-coco "$PUBLIC_COCO"); fi
bash scripts/autodl_yolov8m_coco_b19_pilot.sh start "${START_ARGS[@]}"
printf 'RUN_ID=%s\nRUN=%s\n' "$RUN_ID" "$RUN"
tmux attach -t comparison-yolov8m-coco-b19-pilot
```

`Ctrl-b d` 脱离会话。脚本只保护自身的同名 tmux 和既有 run，不停止或取消 `comparison-yolov8m-scratch`、YOLOv5m pilot 或任何其他会话。GPU0 可以并行共享，没有 GPU 必须空闲的门槛。物理 batch16/640/AMP 固定，OOM 明确失败，不 AutoBatch、不降 batch、不切优化器。

执行顺序：bootstrap → 本 run 的轻量 preflight → train → 同一 best 的公共 val 导出/CPU 评测 → 公共 test 导出/CPU 评测 → summary。新 tmux window 预设 `remain-on-exit`；失败保留窗口及真实退出码。

## 5. 日志、实际进度和退出码

```bash
LAUNCH="$WORK/outputs/yolov8m-coco-b19-pilot-launch/$RUN_ID"
tail -f "$LAUNCH/train.log"
cat "$LAUNCH/train.exit.json"
cat "$LAUNCH/pipeline_status.json"
cat "$RUN/training_progress.json"
cat "$RUN/actual_training_setup.json"
cat "$RUN/native_validation.json"
tail -n 3 "$RUN/epoch_trace.jsonl"
"$PY" "$ENTRY" summary --run "$RUN"
tmux list-panes -t comparison-yolov8m-coco-b19-pilot -F '#{pane_dead} #{pane_dead_status}'
```

文件尚未产生时 `cat`/`tail` 报缺失只表示该阶段尚未到达。`train.log` 保留 ANSI 和回车控制字符，原生进度包含实际 epoch、loss 和验证指标；`*.exit.json` 分别记录训练命令与 tee 的退出码。`epoch_trace.jsonl` 记录实际累积、各组 LR、最佳轮次和 worker-shared 增强累计计数。CutMix 的触发数、应用数、几何跳过数独立记录，配置概率是 0.03，不用应用比例替代它。

## 6. 训练自然结束后单独补跑公共评测

若训练已 completed，但后续导出/评测因外部原因失败，使用原来的 `RUN_ID`，不新建训练或换 best：

```bash
cd "$WORK"
RUN_ID='填写上面记录的同一个 RUN_ID'
RUN="$WORK/outputs/yolov8m-coco-b19-pilot/$RUN_ID"
PY="$WORK/.envs/yolov8m-coco-b19-pilot/bin/python"
ENTRY="$WORK/benchmarks/comparison/yolov8m/run.py"
for split in val test; do
  "$PY" "$ENTRY" export --run "$RUN" --split "$split"
  "$PY" "$ENTRY" evaluate --run "$RUN" --split "$split"
done
"$PY" "$ENTRY" summary --run "$RUN"
cat "$RUN/metrics/val.json"
cat "$RUN/metrics/test.json"
```

公共推理固定 FP32 参数/输入、方形 LetterBox640、配置 batch16、conf0.001、类内 NMS IoU0.7、max_det300、无 TTA；按实际 rounded resize gain(x/y) 和整数 padding 逆变换。NMS 后不追加裁框或删框，padding-only 假阳性也保留；空预测图保留完整记录。公共 AP 调用原 `corrected_sorted_conf_mask_v1`，不会重新实现 AP。

`metrics/*.json` 包含 P/R/AP50/AP75/mAP50–95 的 raw_0_1、display_percent、各类 AP、GT/预测/检查点身份；选定 best 轮次及实际完成轮数在 `train_status.json` 和 `summary.json`。原生训练 val 与公共结果分开。中断的训练须先按同一 run 恢复至合法停止，不能将 smoke 或未完成跑次标成 200 轮结果。

## 7. 同一 pilot run 的显式恢复

先查看日志确认是自身会话的死窗口；仍在运行的会话不要重复启动。以下清理仅处理本 pilot 已结束的死会话，其他任务不受影响。

```bash
cd "$WORK"
RUN_ID='填写需要恢复的同一个 RUN_ID'
RUN="$WORK/outputs/yolov8m-coco-b19-pilot/$RUN_ID"
test -f "$RUN/train/weights/last.pt"
if tmux has-session -t '=comparison-yolov8m-coco-b19-pilot' 2>/dev/null; then
  if test "$(tmux list-panes -t comparison-yolov8m-coco-b19-pilot -F '#{pane_dead}')" = 1; then
    tmux kill-session -t '=comparison-yolov8m-coco-b19-pilot'
  else
    printf '%s\n' 'Pilot session is still alive; refusing duplicate resume.' >&2
    exit 73
  fi
fi
bash scripts/autodl_yolov8m_coco_b19_pilot.sh start-resume \
  --run-id "$RUN_ID" --data "$MAIN/configs/crack_autodl.yaml" \
  --base-python /root/miniconda3/envs/rtdetr/bin/python
tmux attach -t comparison-yolov8m-coco-b19-pilot
```

恢复只接受本目录的 `last.pt`，核对同一 UUID、代码/增强/配方/数据/初始化身份、冻结原生超参数、best/last 成对一致、早停状态及完整 FP32 训练状态。拒绝其他 pilot run、旧 COCO、scratch、smoke 或其他模型的 checkpoint。保留初始化记录和先前 CSV/日志；每次恢复生成独立 launch 日志。仍采用原生 epoch 边界续训规则，未承诺预取 worker 批次和跨 epoch 待累计梯度的位级重放。

服务器 batch16/640 双任务显存、正式原生 val 精度和全量公共 val/test 指标须由真实服务器跑次记录。Linux SIGTERM 子进程组转发也尚待该平台运行验证。
