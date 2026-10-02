# YOLOv13l scratch 服务器命令

日期：2026-10-03（北京时间）。此文件交付命令，不代表已经登录服务器或运行训练。
下一批启动时间由用户安排。当前 YOLOv5m/YOLOv8m 不需停止；
本任务和 comparison-yolo11m-scratch 可同时使用 GPU0，无整卡锁或等待依赖。
显存不足会报错保留现场，不自动缩 batch/尺寸、关闭 AMP 或换模型。

代码固定为 **38b49c547ad1fd93d7b85ef9917fa812b1568eeb**，
直接父提交61c386bc722b11208453cab0808d8f6edb15385d。
本文件所在后续提交只补交付文档，不改变上述代码。
服务器工作树刻意停在代码提交，不随分支最新文档或上游 main 浮动。

官方 commit：73289949533efac82bb5f72ec19b746618656bd2。
补丁 SHA256：50c80a68320eda8f72918c5015692d4ddc99da759c9d2cc6e2fd4b654724af12。
补丁后源树 SHA256：9dcb09fc059147501051bb985e0538bd7f42270d41c25174c63b317400b26e7f。
新 YOLO=random；既有 RT-DETR=ImageNet 骨干，历史条件不改写。

## 1. 获取固定工作树

在服务器已有终端执行。本段只做 Git 准备，保留母项目未提交内容。
已有目标目录必须为指定提交且干净，否则停止检查，不覆盖。

~~~bash
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-scratch
CODE=38b49c547ad1fd93d7b85ef9917fa812b1568eeb

test "$(git -C "$MAIN" remote get-url origin)" = \
  https://github.com/supershaojie/concrete-crack-rtdetr.git
git -C "$MAIN" status --short
git -C "$MAIN" worktree list
git -C "$MAIN" fetch origin \
  refs/heads/bench/yolov13l-scratch:refs/remotes/origin/bench/yolov13l-scratch
git -C "$MAIN" cat-file -e "$CODE^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$CODE" origin/bench/yolov13l-scratch

if test -e "$WT"; then
  test "$(git -C "$WT" rev-parse HEAD)" = "$CODE"
  test -z "$(git -C "$WT" status --porcelain)"
else
  git -C "$MAIN" worktree add --detach "$WT" "$CODE"
fi
cd "$WT"
git show -s --format='%H %P %s'
~~~

## 2. 环境和轻量预检（不训练）

默认私有环境 .envs/yolov13l-scratch，源码 .vendor/yolov13l-scratch/yolov13，
设置/环境记录 .runtime/yolov13l-scratch。首次 import 前已隔离 Ultralytics 设置。
bootstrap 只读探测母环境，创建 --copies 私有 venv，安装固定小型依赖，
验证源码、模型、初始化和 CUDA 可用时的副本 AMP 数值，不下载外部权重。

~~~bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-scratch
BASE=/root/miniconda3/envs/rtdetr/bin/python
"$BASE" benchmarks/comparison/yolov13l/bootstrap.py --base-python "$BASE"
PY="$(cat .runtime/yolov13l-scratch/python_path.txt)"
"$PY" -c 'import sys; print("executable:",sys.executable); print("prefix:",sys.prefix); print("base:",sys.base_prefix)'
cat .runtime/yolov13l-scratch/bootstrap.json

"$PY" benchmarks/comparison/yolov13l/run.py preflight \
  --run "$PWD/outputs/yolov13l-preflight/20261003_seed42" \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det
~~~

此预检目录只作检查，正式一键流程会创建自己的新 run。
已有预检目录不会删除/覆盖，可直接查看其中 manifest.json。
匹配冻结清单必须得到6048/1728/864张及45573/12840/6663框。
数量或身份不符时从manifest定位路径/标签原因；不改标签、不重划分、不做43GB图片全量哈希。

如默认 bootstrap 因实际基础 Torch 组合不兼容失败，且还没有成功冻结环境或正式 run，
可显式选择全新目录安装 Torch2.2.2/TorchVision0.17.2 cu121；原失败目录保留。
这条替代路径需要服务器实际验证，不声称已在本地跑过该组合。

~~~bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-scratch
test ! -e .runtime/yolov13l-scratch/environment_identity.json
/root/miniconda3/envs/rtdetr/bin/python benchmarks/comparison/yolov13l/bootstrap.py \
  --base-python /root/miniconda3/envs/rtdetr/bin/python \
  --fresh-torch --env-dir "$PWD/.envs/yolov13l-scratch-torch222"
PY="$(cat .runtime/yolov13l-scratch/python_path.txt)"
~~~

成功冻结后，一键流程自动沿用记录中的解释器目录。
不向母环境安装包，不盲装作者 Flash wheel、ONNX/Gradio 或整份 requirements。
失败诊断位于 .runtime/yolov13l-scratch/*environment_probe.json。

## 3. 用户安排时间后正式一键启动

以下命令才会启动正式训练。固定 run-id 便于追踪；重跑时会保护已有输出。
bootstrap→轻量preflight→train→val/test FP32 export→公共CPU evaluate→summary 自动顺序执行。

~~~bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-scratch
bash scripts/autodl_yolov13l.sh start \
  --run-id 20261003_yolov13l_scratch_seed42 \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det \
  --base-python /root/miniconda3/envs/rtdetr/bin/python

tmux attach -t '=comparison-yolov13l-scratch'
~~~

脱离而继续运行：按 Ctrl-b，松开，再按 d。
断线后重新打开原服务器终端，执行同一条 tmux attach 即可重连。
脚本只保护 comparison-yolov13l-scratch；不操作 v5/v8/yolo11 会话。
remain-on-exit 以实际 window ID 设置，并先设置后放行；失败窗格保留。

固定配方：640、batch16、workers8、epochs200、patience50、seed42、AMP、SGD、nbs64。
训练内 val 为batch16/workers0；独立导出为FP32。详情见 recipe.yaml 和运行展开回执。

~~~bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-scratch
RUN="$PWD/outputs/yolov13l-scratch/20261003_yolov13l_scratch_seed42"
LAUNCH="$PWD/outputs/yolov13l-scratch-launch/20261003_yolov13l_scratch_seed42"
PY="$(cat .runtime/yolov13l-scratch/python_path.txt)"

tmux list-panes -t '=comparison-yolov13l-scratch' \
  -F '#{window_id} #{pane_dead} #{pane_dead_status} #{pane_current_command}'
tail -F "$LAUNCH/train.log"
~~~

退出 tail 用 Ctrl-c，仅退出查看。
另开终端可执行以下只读检查：

~~~bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-scratch
RUN="$PWD/outputs/yolov13l-scratch/20261003_yolov13l_scratch_seed42"
LAUNCH="$PWD/outputs/yolov13l-scratch-launch/20261003_yolov13l_scratch_seed42"
ls -lh "$RUN/train/weights"
cat "$RUN/training_progress.json"
cat "$RUN/initialization.json"
cat "$LAUNCH/pipeline_status.json"
cat "$LAUNCH/train.exit.json"
~~~

文件在相应阶段完成后出现。train.exit.json 分别保存 command_exit、tee_exit 和 signal。
过程进度和逐轮验证摘要保留；完整日志在 LAUNCH。run/launch_directories.txt 记录启动尝试目录。
200是上限，真实完成轮、最佳轮、停止原因见 train_status.json 和 summary.json。

## 4. 训练完成后补做导出或仅 CPU 重算

正常一键流程已做这些步骤。只有训练状态 completed、同一已选 best 和身份一致时才允许导出。
已有完整预测缓存校验通过会复用；不删除不完整缓存来掩盖失败。

~~~bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-scratch
PY="$(cat .runtime/yolov13l-scratch/python_path.txt)"
ENTRY=benchmarks/comparison/yolov13l/run.py
RUN="$PWD/outputs/yolov13l-scratch/20261003_yolov13l_scratch_seed42"

"$PY" "$ENTRY" export --run "$RUN" --split val
"$PY" "$ENTRY" export --run "$RUN" --split test
~~~

有效预测缓存已形成时，指标/汇总格式调整只运行：

~~~bash
"$PY" "$ENTRY" evaluate --run "$RUN" --split val
"$PY" "$ENTRY" evaluate --run "$RUN" --split test
"$PY" "$ENTRY" summary --run "$RUN"
~~~

结果：predictions/val.jsonl.gz、test.jsonl.gz；metrics/val.json、test.json；summary.json。
空预测照样有记录，类别0→1，原图浮点坐标不重复反变换。
*_fp32_arithmetic.json 给出内部运算dtype；指标同时含0–1和百分数。
P/R取公共最大F1点，训练原生fitness不代替公共主表指标。
若仅导出失败，不再调用train --resume，直接补导出/评测。

## 5. 显式受控恢复同一训练 run

仅用于未完成训练。先查看失败日志、last/best和checkpoint_pair.json。
不自动找latest，不恢复smoke/其他模型/COCO/其他UUID，不调整配方或换提交。
检查点不完整、配对哈希/初始化/环境不符、已到200轮或patience时会拒绝，需先定位原因。

如原本模型tmux窗格已经退出，先保留它并改名；活跃窗格会使此段停止。
不删除既有session、输出、锁或检查点。

~~~bash
set -euo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-scratch
RUN="$PWD/outputs/yolov13l-scratch/20261003_yolov13l_scratch_seed42"
cat "$RUN/train_status.json"
cat "$RUN/checkpoint_pair.json"

if tmux has-session -t '=comparison-yolov13l-scratch' 2>/dev/null; then
  test "$(tmux list-panes -t '=comparison-yolov13l-scratch' -F '#{pane_dead}')" = 1
  tmux rename-session -t '=comparison-yolov13l-scratch' \
    "comparison-yolov13l-scratch-archive-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
bash scripts/autodl_yolov13l.sh start-resume \
  --run-id 20261003_yolov13l_scratch_seed42 \
  --base-python /root/miniconda3/envs/rtdetr/bin/python
tmux attach -t '=comparison-yolov13l-scratch'
~~~

恢复记录另存resume_request_*、resume_model_*、resume_audit_*，原initialization不覆盖。
旧CSV保留备份。last/best包含训练参数、optimizer、EMA、scaler、scheduler、RNG和最佳/早停状态；
恢复按epoch检查点语义，不宣称worker预取和批内累积逐bit连续。

## 6. 可选有限验证、资源与后续测速

不作为正式训练门槛；这些命令不会用真实数据做全量推理或调用外部预训练。

~~~bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-scratch
PY="$(cat .runtime/yolov13l-scratch/python_path.txt)"

"$PY" benchmarks/comparison/yolov13l/run.py check \
  --output "$PWD/outputs/yolov13l-validation/server_check_20261003.json"
"$PY" benchmarks/comparison/yolov13l/probe.py --batch 1 \
  --output "$PWD/outputs/yolov13l-validation/server_batch1_640_20261003.json"
"$PY" benchmarks/comparison/yolov13l/test_scratch.py \
  --run "$PWD/outputs/yolov13l-validation/server_synthetic_smoke_20261003"
"$PY" benchmarks/comparison/yolov13l/run.py measure \
  --output "$PWD/outputs/yolov13l-validation/server_resources_20261003.json"
"$PY" -m unittest discover -s benchmarks/comparison/yolov13l -p test_lifecycle.py -v
"$PY" -m unittest discover -s benchmarks/comparison/yolov13l -p test_guards.py -v
~~~

measure的Conv/Linear/mm/bmm计数包含AAttn/HyperACE，但仍排除BN、激活、softmax、
逐元素等，不能作为“完整FLOPs”。单张640及64小smoke均不证明batch16双模型显存通过。
未来独占硬件时，可在真实训练完成后运行网络测速；不把并行速度填论文主表：

~~~bash
RUN="$PWD/outputs/yolov13l-scratch/20261003_yolov13l_scratch_seed42"
"$PY" benchmarks/comparison/yolov13l/speed.py --run "$RUN" --batch 1 \
  --warmup 10 --iterations 50 --output "$RUN/speed_exclusive_fp32_network.json"
~~~

本地完成19项测试、2项因Windows条件跳过；另有CUDA生命周期、640前后向、轻量数据及补丁重放回执。
待服务器验证：实际环境/驱动/GPU、Linux软链接路径、真实tmux/POSIX信号、正式双模型容量、
可用时Flash比较、正式指标和独占速度。详见 YOLOV13L_SCRATCH_INTEGRATION.md。
