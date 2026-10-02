# YOLOv8m scratch：服务器完整命令（2026-10-03，北京时间）

分支 `bench/yolov8m-scratch`，实际父提交 `74bc7d7168d4fdc199a54d1dc415c497dd0ddfce`。核对远端后未发现锚点之后的提交，原实现 worktree 干净；已有环境、数据、评测和启动器适配已继承。

**代码交付固定 SHA：`61c386bc722b11208453cab0808d8f6edb15385d`。** 本文档作为后续文档提交发布；下列命令锁定这个已验证代码提交。文档提交与它的训练代码相同。原分支、RT-DETR 母版和其他项目均未改动。

正式条件均为 epochs=200、patience=50、imgsz=640、batch=16、workers=8、seed=42、device=0、AMP=True、deterministic=True；SGD、nbs64、warmup5、cos_lr=True、close_mosaic10。YOLOv5 lrf=0.1，YOLOv8 lrf=0.01；其余优化器、损失、增强、锚框、EMA、原生验证设置保持原冻结实现。以原生 val 的完整精度 mAP50–95 选 best，相等值沿用原来的最新轮规则；test 不参与选模或早停。

新 YOLO=random、无外部预训练；旧 YOLO=COCO；既有 RT-DETR=ImageNet backbone，旧结果保持原样。可以比较这些具体训练设置下的表现并补充初始化敏感性研究，不得把整表描述成“全部从零训练”或“预训练条件相同”。不预设结果高低，不筛除真实指标。

本机已完成 CPU 检查及 RTX 2060 6GB 的合成 CUDA smoke：batch2、64像素、两轮，第一轮后中断，再用同一 scratch 检查点续训到第二轮，随后用真实 scratch best 做两张/每 split 的生产 FP32 640 导出与公共评测。Smoke 不进入正式训练。尚未运行 AutoDL 正式训练、完整数据 val/test 或 batch16/640 的双模型显存容量测试；Windows 上未实测真实 tmux/POSIX 信号。本次没有 SSH 或服务器操作。

## 1. 获取固定代码，创建独立 worktree

在 AutoDL 的 Bash 终端逐块执行；所有路径、分支与提交均已填写。已有目录只有 HEAD 正确且工作区干净才复用，不 reset、不覆盖。

```bash
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch
CODE_SHA=61c386bc722b11208453cab0808d8f6edb15385d
test "$(git -C "$MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
git -C "$MAIN" fetch origin refs/heads/bench/yolov8m-scratch:refs/remotes/origin/bench/yolov8m-scratch
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
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch
BASEPY=/root/miniconda3/envs/rtdetr/bin/python
AUDIT_ROOT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/legacy_stop_audit
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
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch
BASEPY=/root/miniconda3/envs/rtdetr/bin/python
"$BASEPY" benchmarks/comparison/yolov8m/bootstrap.py --base-python "$BASEPY"
PY=$(cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/.runtime/yolov8m-scratch/python_path.txt)
"$PY" -c 'import sys,torch,torchvision,numpy; print(sys.executable,torch.__version__,torchvision.__version__,numpy.__version__)'
```

网络只用于官方 Ultralytics v8.3.20 源码 `f4d8f7765a490f3920e2d14c592a2967e347f185` 和缺少的固定依赖。沿用原兼容补丁与源码身份校验；新环境是本 worktree 的 `.envs/yolov8m-scratch`，不会升级原 rtdetr 或另一模型环境。bootstrap 不下载、不读取 `yolov8m.pt`；官方 COCO 哈希校验能力保留在显式原模式。

## 4. 核对数据清单与随机初始化

```bash
PY=$(cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/.runtime/yolov8m-scratch/python_path.txt)
"$PY" benchmarks/comparison/yolov8m/run.py preflight \
  --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/preflight_random_20261003 \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det
"$PY" benchmarks/comparison/yolov8m/run.py check \
  --output /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/model_check_random_20261003.json
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/model_check_random_20261003.json
```

预期 `initialization_type=random`、`pretraining_source=null`、`pretrained_tensors_loaded=0`、`transferred_tensors=0`、`parameters_unfused=25856899`、scale=m、nc=1。从真实官方 `ultralytics/cfg/models/v8/yolov8.yaml` 字典明确设置 m 尺度并构建；正式 get_model 收到 weights=None，pretrained=False、resume=False。固定 16 元素 DFL 投影、BN 常量及检测偏置先验是原生初始化的一部分。

## 5. tmux 启动、查看与重连

确认另一份 YOLOv5m scratch 文档也完成旧作业停止后，可同时启动两组，两者通常均使用 GPU0。启动器显示显存和现有任务，不要求 GPU 空闲，不静默修改 batch16/640/AMP。两组总显存是否足够需要当前服务器实际运行确认。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch
bash scripts/autodl_yolov8m.sh start \
  --run-id yolov8m_random_e200_b16_s42_20261003 \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det \
  --base-python /root/miniconda3/envs/rtdetr/bin/python
tmux attach -t comparison-yolov8m-scratch
```

Ctrl+B 后按 D 退出查看；重连与进度：

```bash
tmux attach -t comparison-yolov8m-scratch
tmux list-panes -t '=comparison-yolov8m-scratch' -F '#{pane_id} #{pane_pid} #{pane_current_command} #{pane_dead} #{pane_dead_status}'
tail -n 30 /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/yolov8m_random_e200_b16_s42_20261003/train/results.csv
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/yolov8m_random_e200_b16_s42_20261003/training_progress.json
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/yolov8m_random_e200_b16_s42_20261003/train_status.json
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch-launch/yolov8m_random_e200_b16_s42_20261003/pipeline_status.json
```

独立日志目录 `/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch-launch/yolov8m_random_e200_b16_s42_20261003` 保存各阶段原始 log、命令与 command/tee 真实退出码。训练中尚无 pipeline_status.json 属正常现象，它在结束时生成。同名 session、run 或 launch 目录不会被删除覆盖；remain-on-exit 绑定实际 window ID，信号与失败不会记成成功。

## 6. 训练后统一评测、汇总、受控续训

启动器自动执行 bootstrap → 轻量 preflight → train → val FP32 导出/公共评测 → test FP32 导出/公共评测 → summary。查看：

```bash
PY=$(cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/.runtime/yolov8m-scratch/python_path.txt)
"$PY" /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/benchmarks/comparison/yolov8m/run.py summary --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/yolov8m_random_e200_b16_s42_20261003
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/yolov8m_random_e200_b16_s42_20261003/metrics/val.json
cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/yolov8m_random_e200_b16_s42_20261003/metrics/test.json
```

训练完成后，如需继续尚未完成的导出/评测，运行：

```bash
PY=$(cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/.runtime/yolov8m-scratch/python_path.txt)
for SPLIT in val test; do
  "$PY" /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/benchmarks/comparison/yolov8m/run.py export --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/yolov8m_random_e200_b16_s42_20261003 --split "$SPLIT"
  "$PY" /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/benchmarks/comparison/yolov8m/run.py evaluate --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/yolov8m_random_e200_b16_s42_20261003 --split "$SPLIT"
done
"$PY" /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/benchmarks/comparison/yolov8m/run.py summary --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/yolov8m_random_e200_b16_s42_20261003
```

已完成且身份相同的预测缓存会复用。统一评测是纯 CPU，不同结果保留为新文件。新实验**不要执行 resume**。仅当本次 scratch run 中断、未完成且 last/best 与最佳轮身份一致时，手动恢复同一个 run：

```bash
PY=$(cat /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/.runtime/yolov8m-scratch/python_path.txt)
"$PY" /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/benchmarks/comparison/yolov8m/run.py train --run /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov8m-scratch/outputs/yolov8m-scratch/yolov8m_random_e200_b16_s42_20261003 --resume
```

恢复先核对 UUID、初始化、模型、代码、数据与配置身份、optimizer、已完成轮数、best/last 配对以及 patience 状态。COCO、其他 scratch run 或 smoke 检查点不接受；正常完成状态不可恢复。原初始化记录不改，续训写 resume_model.json 与 resume_request.json，保留旧 CSV 副本。

## 7. 数据与结果解释

冻结数据为 train 6048/45573框、val 1728/12840框、test 864/6663框，单类 crack，YOLO0→公共1。preflight 除数量外还核对原交付的路径清单、标签清单与轻量数据身份；不一致时报告 expected/actual 并退出，不换数据。读取必要标签、文件大小，val/test 仅在必要时读图像头；不对约43GB图片做内容哈希，不复制或重新增强数据。YOLOv5 的轻量 identity 与 YOLOv8 算法字段不同，因此总 hash 不同，三组路径与标签清单 hash 完全一致。

预检是独立进程/模型；正式训练重新设 seed 并新建模型、optimizer、EMA、scaler 和 loaders。AMP 检查只在实际模型的副本上运行，比较 FP32/FP16 并恢复 Python/NumPy/CPU/CUDA RNG，不更新正式模型 BN、EMA 或优化器，不下载替代模型。

训练原生 val 指标与最终公共 val/test 分开报告。公共 FP32 评测的图像大小、阈值、NMS、max_det、类别和坐标映射沿用原实现；AP 为 0–1，百分数需乘100。提前停止则保留真实完成轮数与结束原因；未执行的指标不得补写参考值。
