# YOLOv5m scratch：服务器完整命令（2026-10-03，北京时间）

分支 `bench/yolov5m-scratch`，实际父提交 `a4dcaa0b0b0c9356923c266ba351de67e033c540`。核对远端后未发现锚点之后的提交，原实现 worktree 干净；已有环境、数据、评测和启动器适配已继承。

**代码交付固定 SHA：`ea17605d538f28da5fa502ddaaf130f7a9ea8084`。** 本文档作为后续文档提交发布；下列命令锁定这个已验证代码提交。文档提交与它的训练代码相同。原分支、RT-DETR 母版和其他项目均未改动。

正式条件均为 epochs=200、patience=50、imgsz=640、batch=16、workers=8、seed=42、device=0、AMP=True、deterministic=True；SGD、nbs64、warmup5、cos_lr=True、close_mosaic10。YOLOv5 lrf=0.1，YOLOv8 lrf=0.01；其余优化器、损失、增强、锚框、EMA、原生验证设置保持原冻结实现。以原生 val 的完整精度 mAP50–95 选 best，相等值沿用原来的最新轮规则；test 不参与选模或早停。

新 YOLO=random、无外部预训练；旧 YOLO=COCO；既有 RT-DETR=ImageNet backbone，旧结果保持原样。可以比较这些具体训练设置下的表现并补充初始化敏感性研究，不得把整表描述成“全部从零训练”或“预训练条件相同”。不预设结果高低，不筛除真实指标。

本机已完成 CPU 检查及 RTX 2060 6GB 的合成 CUDA smoke：batch2、64像素、两轮，第一轮后中断，再用同一 scratch 检查点续训到第二轮，随后用真实 scratch best 做两张/每 split 的生产 FP32 640 导出与公共评测。Smoke 不进入正式训练。尚未运行 AutoDL 正式训练、完整数据 val/test 或 batch16/640 的双模型显存容量测试；Windows 上未实测真实 tmux/POSIX 信号。本次没有 SSH 或服务器操作。

## 1. 获取固定代码，创建独立 worktree

在 AutoDL 的 Bash 终端逐块执行；所有路径、分支与提交均已填写。已有目录只有 HEAD 正确且工作区干净才复用，不 reset、不覆盖。

```bash
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch
CODE_SHA=ea17605d538f28da5fa502ddaaf130f7a9ea8084
test "$(git -C "$MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
git -C "$MAIN" fetch origin refs/heads/bench/yolov5m-scratch:refs/remotes/origin/bench/yolov5m-scratch
git -C "$MAIN" cat-file -e "$CODE_SHA^{commit}"
if test -e "$WORK"; then
  test "$(git -C "$WORK" rev-parse --show-toplevel)" = "$WORK"
  test "$(git -C "$WORK" rev-parse HEAD)" = "$CODE_SHA"
  test -z "$(git -C "$WORK" status --porcelain)"
else
  git -C "$MAIN" worktree add --detach "$WORK" "$CODE_SHA"
fi
cd "$WORK"
git rev-parse HEAD
BASEPY=/root/miniconda3/envs/rtdetr/bin/python
test -x "$BASEPY"
"$BASEPY" -c 'import sys,torch,torchvision; print(sys.executable,sys.version); print(torch.__version__,torchvision.__version__,torch.version.cuda)'
```

## 2. 识别并定向停止旧的两组 COCO 正式训练

创建代码 worktree 不会启动训练。先检查旧 `comparison-yolov5m`、`comparison-yolov8m` 的 pane/PID/实际命令，再执行 stop。辅助脚本只匹配原 worktree 中的 `run.py train/pipeline` 或 `worker.py train`，并要求 `--run` 位于该模型原输出目录。它不会把 `yolov5m-view` 日志查看会话当成训练，也不会触碰新 scratch 会话或其他实验。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch
BASEPY=/root/miniconda3/envs/rtdetr/bin/python
AUDIT_ROOT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/legacy_stop_audit
for MODEL in yolov5m yolov8m; do
  "$BASEPY" scripts/legacy_yolo_jobs.py inspect --model "$MODEL" --audit-root "$AUDIT_ROOT"
done
```

核对屏幕上列出的两个旧作业后执行以下块。stop 先将稳定且 ZIP/CRC 完整的 best/last 复制到新的审计目录，再对已重新核实命令和进程启动标识的目标 PID 发送 SIGINT；不自动 SIGKILL、不清空 tmux。若发现未知原生 trainer 或 45 秒后仍未退出，将非零退出并保留审计，先处理具体报告再启动新组。

```bash
for MODEL in yolov5m yolov8m; do
  "$BASEPY" scripts/legacy_yolo_jobs.py stop --model "$MODEL" --audit-root "$AUDIT_ROOT"
  "$BASEPY" scripts/legacy_yolo_jobs.py assert-stopped --model "$MODEL" --audit-root "$AUDIT_ROOT"
done
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

原日志与检查点不覆盖；审计中的 `preserved_runs.json` 保留原生状态、CSV 行数及检查点序列化的真实 completed_epochs，CSV 可能领先于检查点，不能据此补写成 200 轮。旧作业已停止时，不伪造中断或完成轮数。已核验快照可应对旧 YOLOv5 非原子保存恰好被中断的情况，异常当前文件仍保留。

## 3. 独立环境与源码

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch
BASEPY=/root/miniconda3/envs/rtdetr/bin/python
VENV=/root/autodl-tmp/envs/comparison-yolov5m-scratch-v7
if ! test -e "$VENV"; then
  "$BASEPY" -m venv --system-site-packages "$VENV"
fi
test -x "$VENV/bin/python"
export YOLOV5_PYTHON="$VENV/bin/python"
export YOLOV5_SOURCE_PROJECT=/root/autodl-tmp/projects/Crack_RTDETR
"$YOLOV5_PYTHON" -m pip install -r benchmarks/comparison/yolov5m/requirements-server.txt
"$YOLOV5_PYTHON" -c 'import sys,torch,torchvision,numpy,IPython; print(sys.executable,torch.__version__,torchvision.__version__,numpy.__version__)'
"$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/assets.py --assets /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/assets --initialization random
```

网络只用于官方 YOLOv5 v7.0 源码 `915bbf294bb74c859f0b41f1c23bc395014ea679` 和缺少的运行依赖。源码应用本分支冻结补丁并核对完整 diff。scratch 路径不下载、不读取 `yolov5m.pt`；lock 中保留的 COCO 权重记录用于原模式核验，不能解读为本轮实际使用了 COCO。环境安装只写新 venv，Torch/torchvision 复用原服务器匹配组合。

## 4. 核对数据清单与随机初始化

```bash
"$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/run.py prepare \
  --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/preflight_random_20261003 \
  --source-project /root/autodl-tmp/projects/Crack_RTDETR \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/preflight_random_20261003/model_check.json
```

预期 `initialization_type=random`、`pretraining_source=null`、`pretrained_tensors_loaded=0`、`parameters_unfused=20871318`；resolved_options 中 weights 为空、cfg 为官方 `models/yolov5m.yaml`、resume=False。保留 depth=.67、width=.75、三尺度三锚框 Detect 与原生 BN/检测偏置初始化。

## 5. tmux 启动、查看与重连

确认另一份 YOLOv8m scratch 文档也完成旧作业停止后，可同时启动两组，两者通常均使用 GPU0。这里不等待 GPU 空闲；检查空闲显存并保留其他任务，内存不足明确失败，绝不自动缩小 batch、imgsz 或关闭 AMP。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch
export YOLOV5_PYTHON=/root/autodl-tmp/envs/comparison-yolov5m-scratch-v7/bin/python
export YOLOV5_SOURCE_PROJECT=/root/autodl-tmp/projects/Crack_RTDETR
bash scripts/autodl_yolov5m.sh yolov5m_random_e200_b16_s42_20261003
tmux attach -t comparison-yolov5m-scratch
```

Ctrl+B 后按 D 退出查看；重连与进度：

```bash
tmux attach -t comparison-yolov5m-scratch
tmux list-panes -t '=comparison-yolov5m-scratch' -F '#{pane_id} #{pane_pid} #{pane_current_command} #{pane_dead} #{pane_dead_status}'
tail -n 30 /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003/native/results.csv
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003/native/epoch_state.json
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003/status.json
```

二进制转发保留进度条原地刷新，完整原始日志在 `/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003.launcher.log`、run 内各阶段 `.log`；不要用把每个回车转成换行的查看器判断训练是否刷屏。launcher 的 `.launcher_exit.json` 与 run 的 status.json 分别记录外层 pipeline/tee 和真实子进程退出码。同名 session、输出、日志或 active.lock 均拒绝覆盖；remain-on-exit 使用创建时返回的 window ID。

## 6. 训练后统一评测、汇总、受控续训

启动器自动执行 prepare → train → 同一 best 的 val/test FP32 导出 → 公共 CPU 评测 → summary。已有成功结果不重写。查看：

```bash
/root/autodl-tmp/envs/comparison-yolov5m-scratch-v7/bin/python /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/benchmarks/comparison/yolov5m/run.py summary --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003/evaluation/val_unified_metrics.json
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003/evaluation/test_unified_metrics.json
```

仅在训练已完成、后续导出/评测尚未执行时，独立执行对应阶段；失败 stage 已登记时先查看其日志，不复用其名字覆盖证据：

```bash
PY=/root/autodl-tmp/envs/comparison-yolov5m-scratch-v7/bin/python
"$PY" /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/benchmarks/comparison/yolov5m/run.py export --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003 --source-project /root/autodl-tmp/projects/Crack_RTDETR
"$PY" /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/benchmarks/comparison/yolov5m/run.py evaluate --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003
"$PY" /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/benchmarks/comparison/yolov5m/run.py summary --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003
```

新实验**不要执行 resume**。只有本次同一 scratch run 中断、last/best 与 epoch_state 完整一致且未达到结束条件时，才手动执行：

```bash
/root/autodl-tmp/envs/comparison-yolov5m-scratch-v7/bin/python /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/benchmarks/comparison/yolov5m/run.py train --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-scratch/outputs/yolov5m-scratch/runs/yolov5m_random_e200_b16_s42_20261003 --source-project /root/autodl-tmp/projects/Crack_RTDETR --resume
```

只接受本 run 的 last.pt；核对 UUID、初始化、代码/配置/环境、数据清单、最佳轮与保存状态。COCO、其他 scratch run 和 smoke 检查点会被拒绝。正常完成的 run 禁止续训。保留原 initialization.json，恢复写独立记录。硬中断遗留 active.lock 时先核对 PID 已退出再人工处理，不自动删除锁。

## 7. 数据与结果解释

冻结数据为 train 6048/45573框、val 1728/12840框、test 864/6663框，单类 crack，YOLO0→公共1。preflight 除数量外还核对原交付的路径清单、标签清单与轻量数据身份；不一致时报告 expected/actual 并退出，不换数据。读取必要标签、文件大小，val/test 仅在必要时读图像头；不对约43GB图片做内容哈希，不复制或重新增强数据。YOLOv5 的轻量 identity 与 YOLOv8 算法字段不同，因此总 hash 不同，三组路径与标签清单 hash 完全一致。

预检是独立进程/模型；正式训练重新设 seed 并新建模型、optimizer、EMA、scaler 和 loaders。AMP 检查只在实际模型的副本上运行，比较 FP32/FP16 并恢复 Python/NumPy/CPU/CUDA RNG，不更新正式模型 BN、EMA 或优化器，不下载替代模型。

训练原生 val 指标与最终公共 val/test 分开报告。公共 FP32 评测的图像大小、阈值、NMS、max_det、类别和坐标映射沿用原实现；AP 为 0–1，百分数需乘100。提前停止则保留真实完成轮数与结束原因；未执行的指标不得补写参考值。
