# YOLOv5m COCO + b19 pilot：完整服务器命令

2026-10-03，北京时间。已验证代码提交 **`07d77c16ec168bd9547cd5aac04817c1d5d46e39`**，实际派生父提交 `b6d38d6b2d72661d03322257ebc796849eb4d2f3`，分支 `bench/yolov5m-coco-b19-pilot`。本命令文档在后续交付提交中发布，训练代码与上述固定代码提交相同；下面始终 checkout 这个已验证完整 SHA。

这是一组新增 pilot：原始官方 COCO 初始化 + b19 在线增强参数/CutMix、batch16、原 SGD 配方。父 v5 的预处理审计发现等比 resize + LetterBox，因此本 pilot 另外按要求修正 train square stretch，保留母版 HSV 算子。不能将结果变化归因于一个因素，也不声称完整复现 b19 数据/预处理。原 scratch、原COCO及母版结果均保留。

本轮未登录服务器、未启动正式训练、未停止现有 scratch 或其他任务。本机通过的是合成 CUDA batch2/64 两轮及生产 FP32/640 小样本导出；**AutoDL GPU0 两组 batch16/640 并发显存容量尚未验证**。以下命令由用户在 AutoDL Bash 终端按顺序执行，无 GPU 必须空闲的门槛，不自动取消另一组 `comparison-yolov8m-coco-b19-pilot`。

## 1. 拉取固定提交并创建新工作树

```bash
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-b19-pilot
CODE_SHA=07d77c16ec168bd9547cd5aac04817c1d5d46e39
test "$(git -C "$MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
git -C "$MAIN" fetch origin refs/heads/bench/yolov5m-coco-b19-pilot:refs/remotes/origin/bench/yolov5m-coco-b19-pilot
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

已有目录 HEAD/状态不一致时上述检查退出，先核实内容，不能 reset 或覆盖。在已冻结 run 上也不能切换代码或环境后继续恢复。无需停止旧 scratch：运行安排由用户管理，不运行继承文档里的 legacy stop 脚本。

## 2. 独立 --copies 环境，先核对再安装

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-b19-pilot
BASEPY=/root/miniconda3/envs/rtdetr/bin/python
VENV=/root/autodl-tmp/envs/comparison-yolov5m-coco-b19-pilot
if ! test -e "$VENV"; then
  "$BASEPY" -m venv --copies --system-site-packages "$VENV"
fi
test -x "$VENV/bin/python"
export YOLOV5_PYTHON="$VENV/bin/python"
export YOLOV5_SOURCE_PROJECT=/root/autodl-tmp/projects/Crack_RTDETR
"$YOLOV5_PYTHON" - "$VENV" <<'PY'
import pathlib,sys
expected=pathlib.Path(sys.argv[1]).resolve()
print('executable=',sys.executable,'prefix=',sys.prefix,'base_prefix=',sys.base_prefix)
assert pathlib.Path(sys.prefix).resolve()==expected
assert pathlib.Path(sys.executable).resolve().is_relative_to(expected), 'Executable resolves outside the new venv'
assert pathlib.Path(sys.base_prefix).resolve()!=expected
PY
"$YOLOV5_PYTHON" -m pip install -r benchmarks/comparison/yolov5m/requirements-server.txt
"$YOLOV5_PYTHON" - <<'PY'
import sys,torch,torchvision,numpy,PIL,cv2,pandas,IPython
print('executable=',sys.executable,'prefix=',sys.prefix)
for module in (torch,torchvision,numpy,PIL,cv2,pandas,IPython):
    print(module.__name__,getattr(module,'__version__',''),module.__file__)
assert torch.cuda.is_available()
print('GPU=',torch.cuda.get_device_name(0))
print('CUDA=',torch.version.cuda)
PY
```

Python至少3.9。只用新 venv 的 bin/python，不能将 resolve 后指向母版的解释器拿去 pip install。Torch/torchvision只读复用既有匹配安装，固定其余依赖叠加在新 venv；环境路径/分发版本将在 run 中冻结，母版包变动会让 guard 拒绝续训。不要安装 YOLO26 包；v5 进程检查真实 models/utils 路径并拒绝新版 Ultralytics API 导入。准备/网络进度照常显示。

## 3. 独立固定源码和原始官方 COCO 资产

```bash
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-b19-pilot
ASSETS="$WORK/outputs/yolov5m-coco-b19-pilot/assets"
ORIGINAL=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m/outputs/yolov5m/assets/yolov5m.pt
cd "$WORK"
if test -f "$ORIGINAL"; then
  "$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/assets.py --assets "$ASSETS" --initialization coco_detection_pretrained --weights-from "$ORIGINAL"
else
  "$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/assets.py --assets "$ASSETS" --initialization coco_detection_pretrained
fi
"$YOLOV5_PYTHON" benchmarks/comparison/yolov5m/assets.py --assets "$ASSETS" --verify-only
```

源码锁定 v7.0 的 `915bbf294bb74c859f0b41f1c23bc395014ea679`，应用并比对完整本分支补丁。资产必须是官方原文件，size42806829、SHA256 `61d933360ba5a7733a36764996c800287d973889d875227f5beedd2473a97a56`，复用时先核验，复制到新资产目录后再核验。若不在上述旧目录可直接使用官方下载路径；绝不能替换成旧 best/last、scratch或YOLO26权重。半成品下载保留，错误资产不覆盖。固定 CutMix/IOA/clip 原码和许可已随分支保存，不安装整个 v8.4.0 包。

## 4. 轻量 preflight，正式数据只读

```bash
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-b19-pilot
PREFLIGHT="$WORK/outputs/yolov5m-coco-b19-pilot/preflight_coco_b19_20261003"
"$YOLOV5_PYTHON" "$WORK/benchmarks/comparison/yolov5m/run.py" prepare \
  --run "$PREFLIGHT" --assets "$WORK/outputs/yolov5m-coco-b19-pilot/assets" \
  --source-project /root/autodl-tmp/projects/Crack_RTDETR \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det
"$YOLOV5_PYTHON" - "$PREFLIGHT" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1])
m=json.loads((r/'model_check.json').read_text())
o=json.loads((r/'resolved_options.json').read_text())
d=json.loads((r/'data/manifest.json').read_text())
assert m['initialization_type']=='coco_detection_pretrained'
assert m['loaded_tensors']==475 and m['total_tensors']==481
assert m['actual_value_equality_verified'] and m['all_compatible_tensors_loaded']
assert m['parameters_unfused']==20871318
assert o['batch_size']==16 and o['imgsz']==640 and o['workers']==8
assert o['optimizer']=='SGD' and o['freeze']==[0] and o['cache'] is False
assert o['resume'] is False and o['noplots'] is False
assert o['cfg']=='' and pathlib.Path(o['weights']).name=='yolov5m.pt'
print('model=',{k:m[k] for k in ('initialization_type','loaded_tensors','total_tensors','parameters_unfused')})
print('data=',d['splits'])
print('dataset_identity=',d['dataset_identity_sha256'])
print('environment=',m['environment'])
PY
```

预期 train6048/45573框、val1728/12840框、test864/6663框、crack YOLO0→公共1。同时核对冻结路径与标签 hash，数量一致也不能忽略差异。只读标签、清单和文件大小，无43GB图像内容哈希/全库解码、复制或离线增强；GT缺缓存时仅读必要图像头。清单/缓存/锁使用新 run 路径。preflight 为独立 CPU 模型，不能冒称已验证正式显存；正式训练重新 seed、新建模型/optimizer/EMA/scaler。

可选服务器合成检查使用专属新名字，不改正式 batch16；在用户安排的资源时段运行：

```bash
"$YOLOV5_PYTHON" "$WORK/benchmarks/comparison/yolov5m/worker.py" smoke --assets "$WORK/outputs/yolov5m-coco-b19-pilot/assets" --run "$WORK/outputs/yolov5m-coco-b19-pilot/server_cpu_smoke_20261003"
"$YOLOV5_PYTHON" "$WORK/benchmarks/comparison/yolov5m/test_coco_b19.py" --assets "$WORK/outputs/yolov5m-coco-b19-pilot/assets" --run "$WORK/outputs/yolov5m-coco-b19-pilot/server_cuda_smoke_20261003"
```

两条 smoke 均使用合成数据；CUDA smoke batch2/64两轮，不能复用其检查点、状态、环境freeze或 UUID 到正式 run。

## 5. 用户安排时间后启动独立 tmux

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-b19-pilot
export YOLOV5_PYTHON=/root/autodl-tmp/envs/comparison-yolov5m-coco-b19-pilot/bin/python
export YOLOV5_SOURCE_PROJECT=/root/autodl-tmp/projects/Crack_RTDETR
bash scripts/autodl_yolov5m_coco_b19_pilot.sh yolov5m_coco_b19_e200_b16_s42_20261003
tmux attach -t comparison-yolov5m-coco-b19-pilot
```

最大200轮、patience50、物理batch16、nbs64、AMP；warmup后accumulate4，不做autobatch/autooptimizer/降迁移比例/降输入。GPU0允许与另一 pilot 共用，nvidia-smi 仅显示信息，不检测空闲作为门槛。OOM或依赖错误真实非零退出并留日志，不伪造成功或改参数补偿。相同 session/output/log/active.lock 拒绝重启或覆盖。

启动器自动运行 prepare → train → 自然早停/达到上限后同一 best 的公共 val/test 导出 → 公共 CPU 评测 → summary。new-session 返回的 window ID 用于 remain-on-exit，因此失败窗口保留。Ctrl+B再按D只退出查看；重连：

```bash
tmux attach -t comparison-yolov5m-coco-b19-pilot
tmux list-panes -t '=comparison-yolov5m-coco-b19-pilot' -F '#{pane_id} #{pane_pid} #{pane_current_command} #{pane_dead} #{pane_dead_status}'
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-b19-pilot
RUN="$WORK/outputs/yolov5m-coco-b19-pilot/runs/yolov5m_coco_b19_e200_b16_s42_20261003"
tail -n 30 "$RUN/native/results.csv"
cat "$RUN/native/epoch_state.json"
cat "$RUN/native/initialization.json"
cat "$RUN/native/autoanchor.json"
cat "$RUN/native/effective_training.json"
tail -n 2 "$RUN/native/native_validation_calls.jsonl"
cat "$RUN/status.json"
test ! -f "$RUN.launcher_exit.json" || cat "$RUN.launcher_exit.json"
```

原始控制字符保留在 `$RUN.launcher.log` 和 run 各阶段 `.log`。每轮显示原生epoch/损失/指标，JSON记录完整浮点 native val 与实际调用条件。`augmentation_epoch_190.json` 核对关闭状态、hyp和计数；trigger/apply/skip分开记录，worker预取计数不是消费样本数。launcher_exit 与 status.json 保存真实pipeline/tee/子进程退出码。旧 scratch 会话和输出不会被脚本操作。

## 6. 自然停止后的公共评估和汇总

如果启动器已经完成，查看结果即可：

```bash
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-b19-pilot
RUN="$WORK/outputs/yolov5m-coco-b19-pilot/runs/yolov5m_coco_b19_e200_b16_s42_20261003"
PY=/root/autodl-tmp/envs/comparison-yolov5m-coco-b19-pilot/bin/python
"$PY" "$WORK/benchmarks/comparison/yolov5m/run.py" summary --run "$RUN"
cat "$RUN/selected_best.json"
cat "$RUN/evaluation/val_unified_metrics.json"
cat "$RUN/evaluation/test_unified_metrics.json"
cat "$RUN/summary.json"
```

只有 training_complete.json 已生成、后续阶段尚未执行时，再单独补以下步骤。若已有失败/半成品 stage，先检查其具体日志，不清空文件或覆盖证据：

```bash
"$PY" "$WORK/benchmarks/comparison/yolov5m/run.py" export --run "$RUN" --source-project /root/autodl-tmp/projects/Crack_RTDETR
"$PY" "$WORK/benchmarks/comparison/yolov5m/run.py" evaluate --run "$RUN"
"$PY" "$WORK/benchmarks/comparison/yolov5m/run.py" summary --run "$RUN"
```

export 自动尝试核验复用已有公共GT缓存，或通过图像头生成新GT。已知现有 COCO 缓存目录可另传 `--gt-cache`；不需要重做图片内容哈希。两个 split 锁定同一 selected_best hash；公共配置 FP32/640/batch16/conf.001/NMS.7/maxdet300/无TTA，保留空预测与原浮点坐标。原生 val 的 rect=True/NMS.6/AMP dtype 和最终公共指标分开报告。P/R/AP50/AP75/mAP、原始值/百分数、各类 AP、best epoch、真实轮数/结束原因和各项身份均在 JSON 中。未完成项不会填母版或旧组数值，不覆盖旧表。

## 7. 只恢复同一个 pilot run

新 run 不用 resume。已经中断的本 pilot 仅可在同一固定工作树、环境、数据与run路径恢复；禁止旧COCO/scratch、另一个UUID、smoke、其他模型或完成的run。检查 last/best 与 epoch_state、一致的原初始化审计。Ctrl+C中断不生成自然完成记录，不能将它补写为200轮；完成评测需要先恢复到真实早停/上限。

在一个新的查看会话内运行恢复命令，避免把旧失败 session 清空：

```bash
tmux new-session -s comparison-yolov5m-coco-b19-pilot-resume-01
```

进入此新会话的 Bash 后粘贴：

```bash
set -uo pipefail
tmux set-option -w -t "$(tmux display-message -p '#{window_id}')" remain-on-exit on
WORK=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov5m-coco-b19-pilot
RUN="$WORK/outputs/yolov5m-coco-b19-pilot/runs/yolov5m_coco_b19_e200_b16_s42_20261003"
PY=/root/autodl-tmp/envs/comparison-yolov5m-coco-b19-pilot/bin/python
LOG="$RUN.resume_01.console.log"
(set -o noclobber; : > "$LOG") || { printf '%s\n' 'Existing resume log protected'; exit 2; }
"$PY" "$WORK/benchmarks/comparison/yolov5m/run.py" train \
  --run "$RUN" --source-project /root/autodl-tmp/projects/Crack_RTDETR --resume 2>&1 | tee -a "$LOG"
CODES=("${PIPESTATUS[@]}")
printf '{"train_exit_code":%d,"tee_exit_code":%d}\n' "${CODES[0]}" "${CODES[1]}" > "$RUN.resume_01.exit.json"
cat "$RUN.resume_01.exit.json"
```

恢复后训练自然完成，按第6节执行 export/evaluate/summary。重连 `tmux attach -t comparison-yolov5m-coco-b19-pilot-resume-01`。再次恢复使用新序号会话和日志，旧记录保留。run.py 自身也为每次恢复建立新的 train_resume_N stage，保留实际退出码。

原始初始化审计不会覆盖，恢复审计另存。原生half checkpoint模型/EMA、optimizer动量、AMP scaler、scheduler、父进程 RNG、loader generator、累计/上次更新位置和早停状态保留；worker预取随机流重启，不承诺bitwise连续训练等价。中断 summary 中的 end_reason 是最后已保存 checkpoint 的状态，实际中断/失败原因见 latest_train_attempt.status/error，completion 明确为 incomplete。残留active.lock需先核实其中PID已退出，再人工处理，不能在活动训练上删锁或并行恢复。同名会话、运行锁和身份错误会明确拒绝。
