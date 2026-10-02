# Faster R-CNN (ResNet-50-FPN) scratch：服务器交付

交付日期：2026-10-03，北京时间。本文件提供命令，本次未登录服务器、未启动正式训练或全量推理，也未停止或修改其他实验。

## 已存在的版本锚点

| 项目 | 固定值 |
| --- | --- |
| GitHub 分支 | `bench/fasterrcnn-r50-fpn-scratch` |
| 实际训练代码提交 | `33f446b93e25c5ccdca930abafae41240d96bb4c` |
| 父提交：已验证 YOLOv8m scratch | `61c386bc722b11208453cab0808d8f6edb15385d` |
| 公共比较基点 | `529c456b9404f1d9ab66d82d2b2f9ec7e0c98545` |
| 母版锚点 | `a0459d6a652cb702699087c88fa39a3e4c4087ec` |
| TorchVision | v0.16.2 / `c6f39778e636ec40a69bdbc74386818c57a65af3` |
| 数据增强组件 | Ultralytics v8.3.20 / `f4d8f7765a490f3920e2d14c592a2967e347f185` |
| 目标运行环境 | torch 2.1.2 + torchvision 0.16.2，Python 3.9–3.11 |

本文件由**上述代码提交之后的单独文档提交**加入。服务器下面检出代码提交，正式运行的 `model_code_sha` 应为 `33f446b93e25c5ccdca930abafae41240d96bb4c`，不要用后续文档提交或其他正在开发分支代替。文档提交没有改变训练代码；其实际 SHA 可由交付分支上的 `git log -1 --format=%H -- docs/comparison/FASTERRCNN_R50_FPN_SCRATCH_SERVER_COMMANDS.md` 查询。

远端：[supershaojie/concrete-crack-rtdetr](https://github.com/supershaojie/concrete-crack-rtdetr)。本地独立工作树位于 `D:/MyProjects/Crack_RTDETR/outputs/worktrees/Crack_RTDETR-bench-fasterrcnn-r50-fpn-scratch`。所有下面的路径都是 Linux 服务器路径。

## 1. 建立并核对独立服务器工作树

此块只获取代码并建立工作树，不启动训练。不切换服务器主工作树的分支、不修改其未提交内容。

```bash
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-scratch
CODE_SHA=33f446b93e25c5ccdca930abafae41240d96bb4c
cd "$MAIN"
REMOTE=$(git remote get-url origin)
case "$REMOTE" in
  https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
  *) printf 'origin does not match the authorized repository\n' >&2; exit 1 ;;
esac
git status --short
git worktree list
git fetch origin refs/heads/bench/fasterrcnn-r50-fpn-scratch:refs/remotes/origin/bench/fasterrcnn-r50-fpn-scratch
git cat-file -e "$CODE_SHA^{commit}"
if [ -e "$WT" ]; then
  test -d "$WT"
  test "$(git -C "$WT" rev-parse HEAD)" = "$CODE_SHA"
  test -z "$(git -C "$WT" status --porcelain --untracked-files=normal)"
else
  git worktree add --detach "$WT" "$CODE_SHA"
fi
cd "$WT"
test "$(git rev-parse HEAD)" = "$CODE_SHA"
test -z "$(git status --porcelain --untracked-files=normal)"
git log -1 --format=fuller
```

若同名目录、分支、锁或输出不符，命令会停下供检查，不会覆盖、reset、强推或清理。使用 detached worktree 是为了固定已经验证的代码 SHA；GitHub 交付分支仍保留代码与文档两个提交。

## 2. 一键 tmux 流水线

这是正式启动命令，由用户在服务器执行。流程为 bootstrap → 轻量数据及独立模型预检 → CPU 资源统计 → train → 同一 best 的 val/test FP32 导出 → 公共 CPU evaluate → summary。

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-scratch
cd "$WT"
bash scripts/autodl_fasterrcnn_r50_fpn.sh start \
  --run-id frcnn_r50_fpn_scratch_20261003 \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det \
  --base-python /root/miniconda3/envs/rtdetr/bin/python
```

会话：`comparison-fasterrcnn-r50-fpn-scratch`。本批搭配会话是 `comparison-yolo26m-scratch`；两者均可使用 GPU0。启动器只检查本会话/本 run，显示 GPU 容量但不要求 GPU 空闲，不设整卡锁，也不等待其他模型完成。真实 OOM 明确失败，batch16、640 和 AMP 不会自动降低或关闭。

输出目录：`/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-scratch/outputs/fasterrcnn-r50-fpn-scratch/frcnn_r50_fpn_scratch_20261003`。

首次启动日志目录：`/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-scratch/outputs/fasterrcnn-r50-fpn-scratch-launch/frcnn_r50_fpn_scratch_20261003`。后续恢复使用独立的 `_resume_时间_随机数` 日志目录。已存在的 run 或日志不会被启动器删除或覆盖。

环境是本工作树的 `.envs/fasterrcnn-r50-fpn-scratch`，源码是 `.vendor/fasterrcnn-r50-fpn-scratch`，环境回执是 `.runtime/fasterrcnn-r50-fpn-scratch`。只有 base torch/vision 版本兼容时，私有 `--copies` venv 才只读继承它；否则在不继承包的私有 venv 安装官方 cu118 匹配组合，不升级/降级 `rtdetr`、v5/v8 或其他模型环境。CUDA wheel、驱动和算子是否兼容，以实际预检为准，不以本地 Windows 结果冒充。

不提供公共 COCO GT 路径时，预检从现有标签及 val/test 必要图片头生成公共 GT；不转换全部 train、不全量哈希或解码 43GB 图片、不修改源图片/标签。应核对 train `6048/45573`、val `1728/12840`、test `864/6663`，以及 frozen light identity `3401e485b40398ae096e8fd00101cae38b607ee1d6d1b5d5abba5c443e273483`。不符即失败并输出定位信息，不通过删图、重划分或改标签凑数。

## 3. 进入、脱离、重连与进度

```bash
tmux attach -t comparison-fasterrcnn-r50-fpn-scratch
```

脱离而不停止训练：先按 `Ctrl-b`，松开，再按 `d`。SSH 断开后重连，重新执行上面的 attach。不要用 `Ctrl-c` 代替脱离；`Ctrl-c` 会中断当前实验。

```bash
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-scratch
RUN="$WT/outputs/fasterrcnn-r50-fpn-scratch/frcnn_r50_fpn_scratch_20261003"
LAUNCH="$WT/outputs/fasterrcnn-r50-fpn-scratch-launch/frcnn_r50_fpn_scratch_20261003"
tmux list-sessions
tmux list-panes -t comparison-fasterrcnn-r50-fpn-scratch \
  -F 'window=#{window_id} pane=#{pane_id} dead=#{pane_dead} exit=#{pane_dead_status}'
test ! -f "$RUN/training_progress.json" || cat "$RUN/training_progress.json"
test ! -f "$RUN/train_status.json" || cat "$RUN/train_status.json"
test ! -f "$LAUNCH/pipeline_status.json" || cat "$LAUNCH/pipeline_status.json"
```

每轮显示进度、总 loss、LR 和 val 摘要；四项原生 loss、真实 optimizer step、实际增强调用、best epoch、patience 与 AMP scale 在 `epoch_trace.jsonl` 中。`train.log` 保留回车刷新语义，用 `tail -f` 可读，但实时观测优先 attach tmux。每阶段的 `*.exit.json` 分别保存 `command_exit`、`tee_exit` 和信号，不以 tee 成功代替训练成功。启动器先用真实 window ID 设置 `remain-on-exit`，再释放启动 gate，失败界面也会保留。

## 4. 可独立执行的环境/预检诊断

这些是排查入口，不需要在一键命令之前全部重复执行。独立预检使用另一个明确目录，避免占用正式 run 名称。

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-scratch
cd "$WT"
/root/miniconda3/envs/rtdetr/bin/python \
  benchmarks/comparison/fasterrcnn_r50_fpn/bootstrap.py \
  --base-python /root/miniconda3/envs/rtdetr/bin/python
PY=$(cat .runtime/fasterrcnn-r50-fpn-scratch/python_path.txt)
"$PY" -c 'import sys,torch,torchvision; print(sys.executable); print(sys.prefix); print(torch.__version__,torchvision.__version__,torch.version.cuda)'
"$PY" benchmarks/comparison/fasterrcnn_r50_fpn/run.py preflight \
  --run "$WT/outputs/fasterrcnn-preflight/preflight_20261003" \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det
```

Linux 不要对 `PY` 调用 `readlink -f` 或 Python `Path.resolve()`；它可能把 venv 解释器解析回母环境。程序检查 `sys.prefix`、`sys.executable` 和后续子进程路径。预检用独立进程和模型；正式训练重新设 seed 并重建 BN/optimizer/scaler/loader。检查通过不代表两个模型同时 batch16/640 的容量通过。

## 5. 完成后导出、仅 CPU 重算与汇总

一键流水线已自动执行以下步骤。若训练已完成而导出阶段中断，可在确认原流水线已退出后只补相应阶段。有效、身份一致的预测缓存会复用；不同身份的文件会被保护并拒绝覆盖。

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-scratch
cd "$WT"
PY=$(cat .runtime/fasterrcnn-r50-fpn-scratch/python_path.txt)
RUN="$WT/outputs/fasterrcnn-r50-fpn-scratch/frcnn_r50_fpn_scratch_20261003"
"$PY" benchmarks/comparison/fasterrcnn_r50_fpn/run.py export --run "$RUN" --split val
"$PY" benchmarks/comparison/fasterrcnn_r50_fpn/run.py export --run "$RUN" --split test
CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/comparison/fasterrcnn_r50_fpn/run.py evaluate --run "$RUN" --split val
CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/comparison/fasterrcnn_r50_fpn/run.py evaluate --run "$RUN" --split test
"$PY" benchmarks/comparison/fasterrcnn_r50_fpn/run.py summary --run "$RUN"
```

如果预测已有效生成、只是修正汇总或展示，只执行上面两条 `evaluate` 和一条 `summary`。CPU evaluate 只读取 GT/预测/身份回执，不扫描源图片、不构造检测器、不重跑推理。即使 GT 相同，其他 run、模型、配方或 best 的预测缓存也会被拒绝。

`metrics/val.json` 和 `metrics/test.json` 保存 P、R、AP50、AP75、mAP50–95 的 `raw_0_1`、`display_percent` 和单位；P/R 使用公共最大 F1 工作点。`predictions/*_complete.json` 必须有相同的 best SHA，所有图片（包括空预测）都有记录。epoch 选模用未舍入的公共 mAP50–95，相等值更新 best epoch；patience50，最多200轮，第191轮关闭 Mosaic/MixUp。test 不参与选模。

## 6. 显式、受控恢复

仅恢复**本实验同一 run**。训练已经完成时不要 resume，使用导出/评测入口。中断发生在某轮中间时，从上一个完整检查点重放该轮；保存恢复 Python/NumPy/CPU/CUDA RNG、loader generator、SGD、scaler、调度器、best epoch 和早停状态，但不承诺未经验证的逐位复现。

先检查原训练已经退出。以下只在本会话的 pane 已死亡且归档会话名不存在时重命名保留界面，不删除任何会话或日志：

```bash
set -euo pipefail
SESSION=comparison-fasterrcnn-r50-fpn-scratch
ARCHIVE=comparison-fasterrcnn-r50-fpn-scratch-before-resume-20261003
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  test "$(tmux display-message -p -t "=$SESSION" '#{pane_dead}')" = 1
  if tmux has-session -t "=$ARCHIVE" 2>/dev/null; then
    printf 'Protected existing archived session; inspect it before proceeding\n' >&2
    exit 1
  fi
  tmux rename-session -t "=$SESSION" "$ARCHIVE"
fi
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-scratch
cd "$WT"
test "$(git rev-parse HEAD)" = 33f446b93e25c5ccdca930abafae41240d96bb4c
bash scripts/autodl_fasterrcnn_r50_fpn.sh start-resume \
  --run-id frcnn_r50_fpn_scratch_20261003 \
  --base-python /root/miniconda3/envs/rtdetr/bin/python
```

恢复不自动选择“最新 run”，不接受 COCO、其他模型/其他 run 或 smoke 权重，不覆盖原始 `initialization.json`。完整 code/config/data/runtime ABI/UUID 身份和 best/last/index 的 SHA 必须吻合。如果信号恰好发生在几个原子文件写入之间，事务不一致会明确拒绝恢复并保留文件；不要手工改 epoch 或放宽身份校验。早停已达到或最终轮已保存但完成状态未落盘时，显式恢复可从完整一致检查点收尾，不追加训练。

## 7. 资源与后续独占测速

`resources.json` 在流水线内由单独 CPU 进程生成，记录单前景类别、640、未融合、实际提案数与完整模型参数量。Torch profiler 未计入的 NMS/RoIAlign/pooling 等算子逐项列出，`GFLOPs_total=null`；`GFLOPs_counted_2x_MACs` 是部分计数，不能当论文总 GFLOPs。训练和部署是相同未融合结构。

独占测速仅在用户确认其他 GPU 工作已自然结束后单独执行，绝不是训练门槛，也不用于报告并行训练速度：

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-fasterrcnn-r50-fpn-scratch
cd "$WT"
PY=$(cat .runtime/fasterrcnn-r50-fpn-scratch/python_path.txt)
RUN="$WT/outputs/fasterrcnn-r50-fpn-scratch/frcnn_r50_fpn_scratch_20261003"
"$PY" benchmarks/comparison/fasterrcnn_r50_fpn/run.py speed \
  --run "$RUN" --output "$RUN/speed_exclusive.json" --confirm-exclusive
```

测速对原 best 使用 FP32、batch1、640，20次预热、100次计时，保留原始毫秒值和计时边界。检测到其他 GPU compute 进程会拒绝测速，不停止它们。

## 验证回执与边界

- `docs/comparison/evidence/fasterrcnn_r50_fpn_scratch_validation.json`：完整本地验证、数据身份、环境、初始化、AMP 失败/修正证据、源文件/二进制信息和未验证项。
- `docs/comparison/evidence/fasterrcnn_r50_fpn_augmentation.json`：展开增强顺序、分布、关闭边界及差异。
- `docs/comparison/evidence/fasterrcnn_r50_fpn_smoke_export_identity.json`：真实合成 smoke best 的导出身份；明确是 SMOKE，不是正式结果。
- 模型实现及设计说明：`benchmarks/comparison/fasterrcnn_r50_fpn/README.md`。

本地：19 项检查通过，1 项真实 POSIX 信号检查跳过；真实双 worker 的 epoch190 关闭、类别和非方形坐标往返、空目标、缓存只读、零权重载入、学习率边界、错误身份拒绝、错误 child/tee 退出码和 tmux gate 顺序已测。RTX2060 上完成 batch2/64、三轮合成训练→首轮后中断→恢复→同一 best 的 val/test FP32 导出及公共 CPU 评测，并验证预检不改变父进程状态。还执行了单图真实 backbone640 检查。合成指标不进入论文主表。

本机运行版本为 torch 2.7.1+cu118 / torchvision 0.22.1+cu118，**不是目标版本运行通过**。官方 torchvision 0.16.2+cu118 Windows wheel 的185个 Python文件与固定 commit 全部静态一致，但该静态核对不验证 Linux CUDA 二进制。正式入口已验证会拒绝本机非目标版本。目标环境、真实 tmux/POSIX 信号、正式训练、全量 val/test、双模型 batch16/640 容量、独占速度仍待服务器验证。

**预先固定的数值补充：** 本机默认 GradScaler scale65536 导致合成反向梯度非有限，FP32 正常。训练前已据此明确加入 `amp_init_scale=128`、`amp_growth_interval=1000000000`，本地前向/损失/反向和恢复均通过。目标环境会独立复核；仍遇到非有限值即失败，不静默跳 batch，不降低 batch/尺寸或关闭 AMP。此选择发生在任何正式训练和排名之前。

本模型初始化为 random；既有 RT-DETR 为 ImageNet 骨干初始化。两者条件分别保留，不声称全表初始化相同。正式五项指标、正式最佳轮和完成轮当前均缺失，没有填入参考或合成数值。

固定官方来源：[模型 API](https://docs.pytorch.org/vision/0.16/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)、[模型源码](https://github.com/pytorch/vision/blob/c6f39778e636ec40a69bdbc74386818c57a65af3/torchvision/models/detection/faster_rcnn.py)、[训练参考](https://github.com/pytorch/vision/blob/c6f39778e636ec40a69bdbc74386818c57a65af3/references/detection/train.py)、[输入变换](https://github.com/pytorch/vision/blob/c6f39778e636ec40a69bdbc74386818c57a65af3/torchvision/models/detection/transform.py)。
