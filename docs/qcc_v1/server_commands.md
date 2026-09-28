# QCC v1 服务器执行命令

以下使用用户给定的历史服务器路径；本轮未登录服务器，脚本会检查路径、origin、基座、公共权重、配置和导入来源，不会擅自移动已有目录。固定解释器：`/root/miniconda3/envs/rtdetr/bin/python`。正式 batch16 / 640 / AMP / AdamW / 累积4 不变。

本次 CLI 修复代码锚点：`e1c6e874d0d168d5bdfdedc15d42cc48d1f68455`。修复了 GPU 预检入口两个 `add_qccument` 误写，新增 CLI 参数契约检查及旧 worktree 安全快进更新；公式、母版、配方和 GPU 资格不变。后续交付提交只更新文档。下面先 fetch，取得交付完整 SHA，验证代码与锚点相同，再同步。最终答复另提供固定到实际推送 SHA 的命令。

## 1. 同步独立 worktree

单独执行这一段。它不 prepare、不 preflight、更不启动训练。

```bash
set -euo pipefail
QCC_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
test "$(git -C "$QCC_MAIN" rev-parse --is-inside-work-tree)" = true
test "$(git -C "$QCC_MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
timeout 120s git -C "$QCC_MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin exp-rtdetr-r18-lite-qcc-v1
QCC_DELIVERY_SHA=$(git -C "$QCC_MAIN" rev-parse FETCH_HEAD)
git -C "$QCC_MAIN" merge-base --is-ancestor e1c6e874d0d168d5bdfdedc15d42cc48d1f68455 "$QCC_DELIVERY_SHA"
git -C "$QCC_MAIN" diff --exit-code e1c6e874d0d168d5bdfdedc15d42cc48d1f68455 "$QCC_DELIVERY_SHA" -- tools ultralytics-main configs docs/c19_lif_v1
printf 'Verified delivery SHA: %s\n' "$QCC_DELIVERY_SHA"
QCC_SYNC=$(mktemp /tmp/qcc_v1_sync.XXXXXX.sh)
git -C "$QCC_MAIN" show "$QCC_DELIVERY_SHA:tools/sync_qcc_v1.sh" > "$QCC_SYNC"
bash "$QCC_SYNC" "$QCC_DELIVERY_SHA"
```

目标是 `/root/autodl-tmp/projects/Crack_RTDETR-qcc_v1`，分支 `exp-rtdetr-r18-lite-qcc-v1`。必须从 **Git 中的新提交提取同步脚本**，不能直接运行旧 worktree 的旧版脚本。已有旧版目录属于同一仓库、分支正确且干净时，以 `merge --ff-only --no-autostash --no-overwrite-ignore` 更新。未提交改动、未跟踪文件、被忽略文件冲突、分叉历史、活动 QCC 进程/锁/tmux，或已经正式派发/存在正式 run，都会拒绝更新并保留原内容；不自动 stash/reset/clean。原主工作区和 ARG worktree 不动。

更新前的 `sync.json` 保存在 `outputs/qcc_v1/history/sync_<旧完整SHA>.json`。更新后的 `sync.json` 记录实际 checkout SHA；如果快进完成后身份写入被中断，重执行同一同步命令可以安全续完。

## 2. CLI 检查与必要 prepare（复用既有数据身份）

```bash
set -euo pipefail
QCC_WORK=/root/autodl-tmp/projects/Crack_RTDETR-qcc_v1
QCC_PY=/root/miniconda3/envs/rtdetr/bin/python
"$QCC_PY" "$QCC_WORK/tools/qcc_v1_preflight.py" --help
"$QCC_PY" "$QCC_WORK/tools/check_qcc_v1_cli.py"
test -s "$QCC_WORK/outputs/qcc_v1/prepare.json"
test -s "$QCC_WORK/outputs/qcc_v1/data_identity.json"
test -s "$QCC_WORK/outputs/qcc_v1/data_manifest.jsonl.gz"
test -s "$QCC_WORK/outputs/qcc_v1/qcc_v1_init.pt"
QCC_DATA_GUARD=$(mktemp /tmp/qcc_v1_data.XXXXXX.sha256)
sha256sum "$QCC_WORK/outputs/qcc_v1/data_identity.json" "$QCC_WORK/outputs/qcc_v1/data_manifest.jsonl.gz" > "$QCC_DATA_GUARD"
bash "$QCC_WORK/tools/qcc_v1.sh" prepare
sha256sum -c "$QCC_DATA_GUARD"
```

这段用于已经 prepare 完成、仅预检 CLI 失败的服务器。`--help` 必须退出 0；CLI 检查只验证帮助、参数解析和子进程参数契约，不执行 GPU 工作或提供训练资格。先检查已有身份存在，避免缺失时意外进入首次 inventory。只校验已保存的两个身份文件摘要，不重新扫描 train/val/test，也不使用 `--recheck-data`。身份/配置/目录时间不一致会拒绝继续。

代码 SHA 改变后必须重新 prepare：既有流程把旧记录写入 `history/prepare_<旧完整SHA>.json`，生成新源码快照并绑定真实代码 SHA；复用已有初始化及固定数据身份，完整配方保持。旧失败预检的详细记录留在原 `preflights/<时间>/`，不能作为新代码的预检资格。公共源仍是主工作区的 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，必须满足 DELIVERY 中 SHA256。

## 3. 有界 preflight（独立执行）

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh preflight --seconds 900 --micro-batches 16
```

最多总计16个训练 micro-batch、900秒；必须是固定 Linux Python、CUDA 和足够显存。诊断从独立状态在 e=20 覆盖 QCC，正式训练不会继承它。CLI 退出2意味着尚不具备启动资格；具体原因在 outputs/qcc_v1/preflight.json。不要用本地 B1/B2 成功代替此项。

## 4. 查看资格，再由用户单独 start

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh status
```

确认预检已具备资格后，另行执行下列命令。**这一步才启动正式最多200轮/patience50训练**，不会因为 preflight 完成而自动执行。

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh start
```

会创建 `qcc-v1-training` tmux，会话中运行真实训练 Python；记录 worker PID/创建时间、命令、run identity、日志和 `PIPESTATUS[0]`。run 为 `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/qcc_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。公共初始化/e=0，全新 optimizer/scaler/RNG。活动 worker/run、身份漂移或必要预检失败均拒绝启动。

## 5. 查看进度

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh status
tmux capture-pane -p -t qcc-v1-training -S -80
```

或交互查看：

```bash
tmux attach -t qcc-v1-training
```

status 包含真实 worker/退出码、日志路径、epoch、有效更新、AMP skips、best/last 与完成阶段。tmux 消失或 best.pt 存在不能证明训练完成。机制日志在 `/root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/outputs/qcc_v1/epochs.jsonl`，仅前4个batch采样。

## 6. 正确 resume

仅用于训练确实中断且本实验 last 仍含合法 optimizer/scaler/epoch 状态：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh status
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh resume
```

不从 best 或诊断 checkpoint 恢复，不重置 QCC 日程，不扩大预算。若训练已完成，或 checkpoint 已被 native strip，resume 会拒绝；仅评估失败用下一节恢复。

## 7. 训练结束，一次 finish

此命令包含尚未完成的正式 test，由用户在训练结束后主动执行：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh finish
```

顺序为训练完成与 best 身份 → 复用完整 formal FP32 val（优先训练 final_eval 已有的）或首次 val → 同 best 的首次 test → 离线汇总 → 一个 `QCC_v1_ANALYSIS_*.tar.gz`。成功重入不重新推理、不重选 best、不重复扫描数据。包位置和 SHA256 写入 finish.json/last_package.json。

## 8. 仅评估故障恢复

训练完成但 final_eval 失败，仅恢复 val：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh val
```

然后仍用同一条 finish 完成尚缺阶段：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh finish
```

完整成功锁会复用。只有当工具明确报告“成功旧锁缺导出/导出损坏”时，才显式允许同 best 重做该 split 的评估导出：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh val --recover-export
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh test --recover-export
```

两行是对应 split 的独立恢复选项，不应无故一起执行。新工具首次正式评估已自动完整导出，正常流程无需此选项。恢复保留原锁历史、失败记录与退出码，不伪装一次性成功。

## 9. 只打包现有结果

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh pack
```

不创建模型、不建立原始 DataLoader、不推理、不 inventory；只读取现有产物，可离线读取导出生成阈值统计，重复 pack 可复用该离线分析。材料不足输出 INCOMPLETE 诊断包；正式完整包由 finish 统一生成。默认包含已有的全部 query/GT，不包含大权重本体和原图全集。
