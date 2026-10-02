# YOLO26m scratch 服务器命令

日期：2026-10-03（北京时间）。这些是交付给用户执行的命令，本次未登录服务器、未启动正式训练。

## 交付身份

- 分支：`bench/yolo26m-scratch`
- 直接父提交：`61c386bc722b11208453cab0808d8f6edb15385d`
- 实际训练代码 SHA：`6466680f8d1848b2817a6fe6a88764b1800179c6`
- 官方：Ultralytics `v8.4.0` / `f2d3aed634a5b0e4828024718d4a61ab2f83fb19`
- 补丁：`benchmarks/comparison/yolo26m/patches/0001-local-path-and-resume-continuity.patch`
- 补丁 SHA256：`22593d4637d41cfb1f906150d73574a2bc963824a9850f9c7312e107e76a9940`
- 受控源码 SHA256：`3597a1eae9ad21cd08666e2be31e4f811c27abea82e57e647abce77cb0da06d4`
- 最终 adapter SHA256：`10a51358aaa1bfe8d379a3368609b2ed8c68619a50c36403be4c7305ed28a8ec`

代码先提交，服务器说明和验证回执在后续**纯文档提交**。因此分支 HEAD 可以晚于训练代码 SHA；程序用最后影响 benchmarks/comparison 或本启动器的提交记录训练代码身份，同时校验内容hash。纯文档提交不改变训练/恢复身份。

## 1. 创建或核实本模型工作树

在服务器已有主项目环境执行整段；不切换主项目分支，不动其他工作树。已有同名路径/分支会先核实，非快进或脏工作树立即退出。

```bash
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
BRANCH=bench/yolo26m-scratch
CODE=6466680f8d1848b2817a6fe6a88764b1800179c6

test "$(git -C "$MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
git -C "$MAIN" status --short
git -C "$MAIN" worktree list
git -C "$MAIN" fetch origin refs/heads/bench/yolo26m-scratch:refs/remotes/origin/bench/yolo26m-scratch
if [[ -e "$WT" ]]; then
    test "$(git -C "$WT" rev-parse --show-toplevel)" = "$WT"
    test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
else
    if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git -C "$MAIN" worktree add "$WT" "$BRANCH"
    else
        git -C "$MAIN" worktree add -b "$BRANCH" "$WT" "origin/$BRANCH"
    fi
fi
cd "$WT"
test -z "$(git status --porcelain)"
git merge --ff-only "origin/$BRANCH"
test "$(git log -1 --format=%H -- benchmarks/comparison scripts/autodl_yolo26m.sh)" = "$CODE"
git log -2 --oneline
```

## 2. Bootstrap 和独立轻量预检（不训练）

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
BASE=/root/miniconda3/envs/rtdetr/bin/python
cd "$WT"
"$BASE" benchmarks/comparison/yolo26m/bootstrap.py --base-python "$BASE"
PY="$WT/.envs/yolo26m-scratch/bin/python"
test "$(cat .runtime/yolo26m-scratch/python_path.txt)" = "$PY"
"$PY" -c 'import sys,torch; print(sys.executable,sys.prefix,torch.__version__,torch.cuda.is_available())'
AUDIT_RUN="$WT/outputs/yolo26m-preflight/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$RANDOM"
"$PY" benchmarks/comparison/yolo26m/run.py preflight \
  --run "$AUDIT_RUN" \
  --data /root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml \
  --data-root /root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det
cat "$AUDIT_RUN/preflight_status.json"
```

新venv使用--copies，overlay只在本环境安装，基础Torch/TorchVision只读继承。若基础依赖不兼容，bootstrap明确失败并保留 `.runtime/yolo26m-scratch/*probe.json`；不得在rtdetr/v5/v8/其他模型环境升级。只有另行准备了独立兼容Python后才通过--base-python指定它，不自动改变配方。

这里只检查文件、标签、尺寸头及冻结身份，不等待公共全量审计，不生成train COCO，不重分数据或离线增强。正式run稍后建立自己的清单和缓存；此预检目录不会被拿来训练。

服务器可补跑本地Windows未覆盖的POSIX测试：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
.envs/yolo26m-scratch/bin/python benchmarks/comparison/yolo26m/test_lifecycle.py
```

## 3. 一键启动本批 YOLO26m

本交付使用明确run-id `yolo26m_scratch_20261003`，重复执行会保护已有run，不自动覆盖。用户按两组一批安排；配对会话为comparison-fasterrcnn-r50-fpn-scratch。两组均可用GPU0，本启动器不要求GPU空闲、不等待另一个实验完成、不取得整卡排他锁。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
bash scripts/autodl_yolo26m.sh start --run-id yolo26m_scratch_20261003
```

顺序：bootstrap → 轻量preflight → 从零train → 同best的val FP32导出/公共CPU评估 → test FP32导出/公共CPU评估 → summary。正式batch16、640、AMP、200轮/patience50固定。显示GPU容量；真实OOM明确失败，不自动减batch/尺寸或关闭AMP。

```bash
tmux attach -t comparison-yolo26m-scratch
```

脱离按 `Ctrl-b`，松开后按 `d`。断线后执行同一attach命令重连。完成/失败后保留界面；任务放行之前按真实window ID设置remain-on-exit。

```bash
tmux list-panes -t comparison-yolo26m-scratch -F '#{session_name} #{window_id} #{pane_id} dead=#{pane_dead} exit=#{pane_dead_status}'
```

## 4. 进度、日志和产物

```bash
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
RUN="$WT/outputs/yolo26m-scratch/yolo26m_scratch_20261003"
LAUNCH="$WT/outputs/yolo26m-scratch-launch/yolo26m_scratch_20261003"
tail -f "$LAUNCH/train.log"
```

每轮显示训练损失、进度和val摘要。日志保留回车刷新字节；每个stage都有`.command.json`、`.log`、`.exit.json`，记录子进程和tee各自退出码。`pipeline_status.json`给出最终状态；失败/中断不伪装完成。

```bash
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
RUN="$WT/outputs/yolo26m-scratch/yolo26m_scratch_20261003"
"$WT/.envs/yolo26m-scratch/bin/python" "$WT/benchmarks/comparison/yolo26m/run.py" summary --run "$RUN"
```

主要产物：

- manifest.json / input_checksums.json / data.yaml / train,val,test.txt / gt/*.json
- run_id.json / identity.json / initialization.json / source_identity.json / environment.json / pip_freeze.txt
- expanded_train_args.json / frozen_recipe.json / augmentation_mapping.json / actual_training_setup.json
- epoch_trace.jsonl / training_progress.json / checkpoint_state.json / train_status.json
- train/weights/last.pt、best.pt（完整可审计检查点，原子写入）
- predictions/val,test.jsonl.gz及complete.json、metrics/val,test.json、summary.json

本模型random，既有RT-DETR为ImageNet骨干；保留各自条件。正式指标未执行时为null，合成smoke数值不得填写论文主表。

## 5. 导出/评估阶段失败后的续做

训练已completed时不要resume训练。以下命令复用已选定best及已有正确预测缓存；合法缓存不会重新推理。保持工作树和run身份不变。

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
PY="$WT/.envs/yolo26m-scratch/bin/python"
ENTRY="$WT/benchmarks/comparison/yolo26m/run.py"
RUN="$WT/outputs/yolo26m-scratch/yolo26m_scratch_20261003"
for SPLIT in val test; do
    "$PY" "$ENTRY" export --run "$RUN" --split "$SPLIT"
    "$PY" "$ENTRY" evaluate --run "$RUN" --split "$SPLIT"
done
"$PY" "$ENTRY" summary --run "$RUN"
```

仅修复汇总或格式时，已有有效预测可跳过上面的export，只执行evaluate和summary；公共评估只用CPU。

## 6. 显式受控恢复同一训练

仅用于已检查过的中断/失败、具有完整last/best配对的未完成run。程序拒绝完整结束、达到patience、不同run/模型/初始化/源码/配方/数据和smoke身份，以及缺失optimizer/scaler/loss状态的检查点。恢复原200轮计划；不追加轮数，不自动找最新run。

先查看状态，不停止任何现有进程：

```bash
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
RUN="$WT/outputs/yolo26m-scratch/yolo26m_scratch_20261003"
cat "$RUN/train_status.json"
cat "$RUN/checkpoint_state.json"
tmux list-panes -t comparison-yolo26m-scratch -F '#{window_id} #{pane_id} dead=#{pane_dead} exit=#{pane_dead_status}'
```

原会话保留时，新建一个带门控的resume窗口，保留旧窗口和所有日志：

```bash
set -euo pipefail
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
RUN_ID=yolo26m_scratch_20261003
GATE="yolo26m-controlled-resume-$(date +%s)-$RANDOM"
printf -v CMD '%q ' bash "$WT/scripts/autodl_yolo26m.sh" resume --run-id "$RUN_ID"
WINDOW=$(tmux new-window -d -P -F '#{window_id}' -t comparison-yolo26m-scratch -n resume-yolo26m "tmux wait-for '$GATE'; exec $CMD")
tmux set-option -w -t "$WINDOW" remain-on-exit on
tmux wait-for -S "$GATE"
tmux select-window -t "$WINDOW"
tmux attach -t comparison-yolo26m-scratch
```

如果本模型会话已经不存在，可执行：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
bash scripts/autodl_yolo26m.sh start-resume --run-id yolo26m_scratch_20261003
```

恢复尝试产生独立launch目录（run-id加resume时间随机后缀）、resume_request/model/state记录；原初始化不覆盖。恢复训练权重、EMA、SGD状态、scaler、scheduler、E2ELoss更新数/权重、early-stop、累积梯度/step及主进程RNG。worker RNG/预取批次/Mosaic buffer/sampler游标不能逐位复原，详见接入说明。没有checkpoint的首轮失败不能伪装resume，应保留失败run并显式安排新run。

## 7. 资源与后续独占测速（不作为训练门槛）

已提供实际单类训练/部署参数和640输入的有边界算子计数：evidence/yolo26m_resources.json。重新测量无需GPU空闲：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
.envs/yolo26m-scratch/bin/python benchmarks/comparison/yolo26m/run.py measure \
  --output "outputs/yolo26m_resources_$(date +%Y%m%d_%H%M%S)_$RANDOM.json"
```

只有用户确认GPU独占且正式训练已完成后，才执行下列入口；它不会终止其他任务或自动等待GPU空闲：

```bash
WT=/root/autodl-tmp/projects/Crack_RTDETR-bench-yolo26m-scratch
"$WT/.envs/yolo26m-scratch/bin/python" "$WT/benchmarks/comparison/yolo26m/run.py" speed \
  --run "$WT/outputs/yolo26m-scratch/yolo26m_scratch_20261003" \
  --batch 1 --exclusive-gpu-confirmed \
  --output "$WT/outputs/yolo26m_exclusive_speed_$(date +%Y%m%d_%H%M%S)_$RANDOM.json"
```

测速使用同best、FP32官方融合部署结构，预热30批/测100批，记录GPU、进程、median/p95、吞吐和范围。包含模型前向和原生top-k，排除读图/预处理/传输；不是端到端数据管线时延。未进行本次论文独占测速。

## 已验证与待验证

本地最终：CPU集成9项通过；启动器7项中6通过、1项POSIX信号测试跳过；RTX2060合成batch2/64的3轮计划，首轮后中断恢复到3轮，通过SGD/EMA/scaler/scheduler/early-stop/损失状态核对及同best的640 FP32 val/test导出和公共评估。真实数据轻量身份一致。正式640/batch16与两模型同GPU容量、服务器环境、真实Linux tmux/信号以及正式指标仍待服务器执行。

完整回执见 [yolo26m_scratch_validation.json](evidence/yolo26m_scratch_validation.json)，设计和局限见 [接入说明](YOLO26M_SCRATCH_INTEGRATION.md)。本次未登录服务器，未停止或覆盖任何已有训练、其他开发工作树、原数据或结果。
