# YOLO11m scratch 服务器命令

2026-10-03（Asia/Shanghai）。本次没有登录服务器或启动正式实验。
代码固定为 **30ddef9139259df802587142f6d1ed46c99bb38a**，
父提交 **61c386bc722b11208453cab0808d8f6edb15385d**；
分支 bench/yolo11m-scratch。官方 v8.3.20 / f4d8f7765a490f3920e2d14c592a2967e347f185。
此文件由后续 docs-only 提交加入；分支顶端会多出文档，以下训练仍固定到上述真实代码 SHA。
代码提交已包含实现、增强说明和机器回执。

下一批可与 comparison-yolov13l-scratch 共用 GPU0，由用户安排启动时间。
这里只保护本模型会话和 run，不检查整卡空闲、不等待另一模型，也不管理当前 v5/v8 实验。

## 1. 核对远端、分支和独立工作树

在服务器终端执行，不切换主项目分支。现有同名目录或分支不一致时停止核对，不覆盖。

```bash
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-scratch
BRANCH=bench/yolo11m-scratch
CODE_SHA=30ddef9139259df802587142f6d1ed46c99bb38a
REMOTE=https://github.com/supershaojie/concrete-crack-rtdetr.git
test "$(git -C "$MAIN" remote get-url origin)" = "$REMOTE"
git -C "$MAIN" status --short
git -C "$MAIN" worktree list
git -C "$MAIN" fetch origin "$BRANCH"
git -C "$MAIN" cat-file -e "$CODE_SHA^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$CODE_SHA" FETCH_HEAD
if test -e "$WT"; then
    test -f "$WT/.git"
    test "$(git -C "$WT" rev-parse --show-toplevel)" = "$WT"
    test "$(git -C "$WT" rev-parse HEAD)" = "$CODE_SHA"
    test -z "$(git -C "$WT" status --porcelain)"
else
    if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
        test "$(git -C "$MAIN" rev-parse "$BRANCH")" = "$CODE_SHA"
        git -C "$MAIN" worktree add "$WT" "$BRANCH"
    else
        git -C "$MAIN" worktree add -b "$BRANCH" "$WT" "$CODE_SHA"
    fi
fi
cd "$WT"
test "$(git rev-parse HEAD)" = "$CODE_SHA"
git status --short
```

已有分支若指向后续 docs 提交，上述命令也会拒绝移动它。先检查其实际工作树用途；
不要对运行中的 checkout 执行 pull、reset、clean 或切换代码。恢复必须保持同一代码 SHA。

## 2. 仅准备环境与轻量预检

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-scratch
BASE_PY=/root/miniconda3/envs/rtdetr/bin/python
DATA=/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml
DATA_ROOT=/root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det
cd "$WT"
"$BASE_PY" benchmarks/comparison/yolo11m/bootstrap.py --base-python "$BASE_PY"
PY=$(<"$WT/.runtime/yolo11m-scratch/python_path.txt")
test "$PY" = "$WT/.envs/yolo11m-scratch/bin/python"
test ! -L "$PY"
"$PY" -c 'import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix); assert sys.prefix != sys.base_prefix'
"$PY" benchmarks/comparison/yolo11m/run.py preflight     --run "$WT/outputs/yolo11m-preflight/20261003_01"     --data "$DATA" --data-root "$DATA_ROOT"
cat "$WT/outputs/yolo11m-preflight/20261003_01/preflight_status.json"
```

bootstrap 只创建 --copies 私有 venv、固定源码并检查真实模型，**不会训练或加载外部预训练**。
pip 前验证解释器仍在专属环境；Torch 只读继承 base，仅私有环境安装固定 overlay。
依赖不兼容会失败，不在 rtdetr/v5/v8/YOLOv13 环境升级。首次 import 前已隔离 YOLO_CONFIG_DIR。
日志应指向本 WT/.vendor/yolo11m-scratch/ultralytics-v8.3.20/ultralytics/__init__.py，
版本8.3.20、固定 commit、M 结构与 0/649 外部迁移。保持 bootstrap.json 和环境检查回执。

预检应为 train6048/45573、val1728/12840、test864/6663（图/框），
冻结轻量数据身份为 3401e485b40398ae096e8fd00101cae38b607ee1d6d1b5d5abba5c443e273483。
只读原始数据、标签与必要图像头；不全量哈希/解码图片、不改划分、不转换全部 train。
缺少公共 GT 时生成 val/test；可用 --public-coco 显式提供已知目录，只有实际标签/尺寸/集合一致才复用。
这里不猜测服务器旧 GT 路径。预检目录已存在时保留并查看回执；要重做则另选明确的新编号。

## 3. 用户安排时间后正式启动

首次新 run 名称预设为 yolo11m_scratch_20261003_01，这不是已有结果。
流程为 bootstrap → 轻量 preflight → train → val/test FP32 export → 公共 CPU evaluate → summary。
重复 run、会话、输出均保护。不会自动 resume。

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-scratch
cd "$WT"
test "$(git rev-parse HEAD)" = 30ddef9139259df802587142f6d1ed46c99bb38a
test -z "$(git status --porcelain)"
bash scripts/autodl_yolo11m.sh start     --run-id yolo11m_scratch_20261003_01     --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml     --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det     --base-python /root/miniconda3/envs/rtdetr/bin/python
tmux list-windows -t '=comparison-yolo11m-scratch' -F '#{window_id} #{window_name} #{window_flags}'
WINDOW_ID=$(tmux display-message -p -t '=comparison-yolo11m-scratch' '#{window_id}')
tmux show-options -w -t "$WINDOW_ID" remain-on-exit
tmux attach-session -t '=comparison-yolo11m-scratch'
```

脚本先取得真实 window ID，设置 remain-on-exit，再释放 wait-for gate；失败界面保留。
终端保留逐轮回车刷新进度和验证摘要。脱离：Ctrl-b，松开后按 d。
SSH 重新连上服务器后，再执行上面的 attach-session 即可，不再次 start。
如果需要前台运行，用 run 替换 start，其他参数相同；不能同时运行同一个 run。
固定 GPU0/batch16/640/AMP。OOM 明确失败、保留日志，不自动改配方。
本机 batch1/640 与 batch2/64 smoke 不证明 batch16 双模型并行容量。

## 4. 进度、日志和真实退出码

```bash
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-scratch
RUN="$WT/outputs/yolo11m-scratch/yolo11m_scratch_20261003_01"
LAUNCH="$WT/outputs/yolo11m-scratch-launch/yolo11m_scratch_20261003_01"
test ! -f "$RUN/training_progress.json" || cat "$RUN/training_progress.json"
test ! -f "$RUN/train_status.json" || cat "$RUN/train_status.json"
test ! -f "$LAUNCH/pipeline_status.json" || cat "$LAUNCH/pipeline_status.json"
test ! -f "$LAUNCH/train.exit.json" || cat "$LAUNCH/train.exit.json"
tail -n 60 -f "$LAUNCH/train.log"
```

tail 的 Ctrl-C 只结束查看。尚未进入训练时查看 LAUNCH/bootstrap.log 或 preflight.log。
每个 stage 的 .exit.json 保存 command_exit、tee_exit、signal，tee 成功不会掩盖子进程失败。
epoch_trace.jsonl 保存实际 LR、累积、AMP 和增强阶段；初始化、身份、源码、环境、
expanded_train_args、actual_training_setup 均在 RUN。RUN/bootstrap 归档环境准备回执，
launch_directory.txt 指向本次原始完整日志。第191轮关闭 Mosaic/MixUp，其余增强继续。

## 5. 正常结束后的导出/评测恢复

一键流程会自动执行；训练已正常完成但导出/评测失败时，可在该 run 空闲后重试。
同一 best 分别处理 val/test，完整预测缓存会复用，不重新训练。

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-scratch
cd "$WT"
PY=$(<"$WT/.runtime/yolo11m-scratch/python_path.txt")
RUN="$WT/outputs/yolo11m-scratch/yolo11m_scratch_20261003_01"
for SPLIT in val test; do
    "$PY" benchmarks/comparison/yolo11m/run.py export --run "$RUN" --split "$SPLIT"
    "$PY" benchmarks/comparison/yolo11m/run.py evaluate --run "$RUN" --split "$SPLIT"
done
"$PY" benchmarks/comparison/yolo11m/run.py summary --run "$RUN"
```

已有有效预测、仅重算汇总/格式时优先用下面的纯 CPU 路径：

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-scratch
cd "$WT"
PY=$(<"$WT/.runtime/yolo11m-scratch/python_path.txt")
RUN="$WT/outputs/yolo11m-scratch/yolo11m_scratch_20261003_01"
for SPLIT in val test; do
    "$PY" benchmarks/comparison/yolo11m/run.py evaluate --run "$RUN" --split "$SPLIT"
done
"$PY" benchmarks/comparison/yolo11m/run.py summary --run "$RUN"
```

metrics/val.json 与 test.json 保存公共 corrected_sorted_conf_mask_v1 的
P/R/AP50/AP75/mAP50–95，含0–1原值及百分数；P/R采用最大F1工作点。
FP32导出：640、batch16、workers0、conf.001、NMS iou.7、max_det300、max_nms30000、
class-aware、rectFalse、无TTA、无NMS时间限制，原图浮点框和空预测都进入 jsonl.gz。
未正常结束、身份变更或缺图会报错；保留 .partial/不一致缓存供核查，不通过删除证据绕过校验。

## 6. 同一 run 显式恢复训练

只恢复中断且有完整 last/best 状态的本模型正式 run。先查看失败日志和 train_status。
代码、依赖、数据、recipe、原初始化和 UUID 必须一致；完成200轮或已到patience会拒绝恢复。
若旧 tmux 会话已结束，下面只在恰有一个 dead pane 时改名留存；仍运行或多个 pane 时停止，
不终止会话、不删除输出/锁/检查点。

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-scratch
RUN="$WT/outputs/yolo11m-scratch/yolo11m_scratch_20261003_01"
cd "$WT"
test "$(git rev-parse HEAD)" = 30ddef9139259df802587142f6d1ed46c99bb38a
test -z "$(git status --porcelain)"
test -f "$RUN/train/weights/last.pt"
test -f "$RUN/train/weights/best.pt"
cat "$RUN/train_status.json"
SESSION=comparison-yolo11m-scratch
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    DEAD=$(tmux list-panes -s -t "=$SESSION" -F '#{pane_dead}')
    test "$DEAD" = 1
    SAVED_SESSION="$SESSION-retained-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
    if tmux has-session -t "=$SAVED_SESSION" 2>/dev/null; then exit 73; fi
    tmux rename-session -t "=$SESSION" "$SAVED_SESSION"
fi
bash scripts/autodl_yolo11m.sh start-resume     --run-id yolo11m_scratch_20261003_01     --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml     --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det     --base-python /root/miniconda3/envs/rtdetr/bin/python
tmux attach-session -t '=comparison-yolo11m-scratch'
```

仅从该 RUN/train/weights/last.pt 恢复，核验 best/last、UUID/random/模型/源码/展开参数/
数据/环境/原初始化，再恢复 live model、optimizer、EMA、scaler、scheduler、最佳轮次和stopper。
COCO、其他模型/run、smoke、缺失状态均拒绝。原初始化和环境/参数回执保留，
resume_request/model/state/train_args 单独追加。原CSV留副本后回到checkpoint完成轮次。
本次独立 resume 日志目录会在终端打印，原会话及日志留存。
这是 epoch 边界恢复，不声称重放未完成 batch、预取 worker RNG 或部分累积。

## 7. 可选的后续独占硬件测速

仅在用户安排独占硬件时执行；不作为训练门槛。测 fused FP32/batch1/640 模型前向，
不含图片读取、预处理、NMS；共享GPU速度不能作为论文独占速度。

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo11m-scratch
cd "$WT"
PY=$(<"$WT/.runtime/yolo11m-scratch/python_path.txt")
RUN="$WT/outputs/yolo11m-scratch/yolo11m_scratch_20261003_01"
"$PY" benchmarks/comparison/yolo11m/speed.py --run "$RUN"     --warmup 20 --iterations 100     --output "$RUN/speed_fp32_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).json"
```

## 交付证据与待服务器验证

代码提交含 docs/comparison/evidence/yolo11m_scratch_validation.json、
yolo11m_expanded_args.json、YOLO11M_SCRATCH_IMPLEMENTATION.md、YOLO11M_AUGMENTATION.md。
实际数据轻量核对、模型/零外部权重、CPU真实worker、GPU训练→中断→恢复、
真实best FP32导出/公共评测、子进程/tee真实退出码已通过。
tmux目标/放行顺序用Windows Bash mock trace验证；
真实Linux tmux/POSIX信号、Linux --copies、正式batch16/640双模型容量仍待验证。
未登录服务器，未正式训练或全量推理，未停止现有实验，未修改母版、原数据及已有结果。
新YOLO11为random；已有RT-DETR为ImageNet骨干初始化，历史结果保留原条件。
