# RMD v1：固定 SHA 的分段服务器命令

实施 SHA：`614249029759fa58f8847b33bcd87f7a174b04b6`。分支：`exp-rtdetr-r18-lite-rmd-v1`。后续交付文档提交不改变实施代码，所有下列训练/检查命令绑定上述实施 SHA。

下面是用户在 AutoDL 服务器执行的命令。本次 Codex 没有启动这些服务器检查或正式长训。每段独立执行，等待上一步结束并检查状态，不把 preflight 与 200 epoch 训练无条件串联。

## 1. 同步；不切换主工作区分支

```bash
bash <<'RMD_SYNC'
set -euo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR
RMD_SHA=614249029759fa58f8847b33bcd87f7a174b04b6
timeout 180 git -c http.connectTimeout=20 -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin refs/heads/exp-rtdetr-r18-lite-rmd-v1:refs/remotes/origin/exp-rtdetr-r18-lite-rmd-v1
RMD_BOOTSTRAP=$(mktemp /tmp/sync_rmd_v1.XXXXXXXX.sh)
git show "$RMD_SHA:tools/sync_rmd_v1.sh" > "$RMD_BOOTSTRAP"
bash "$RMD_BOOTSTRAP" "$RMD_SHA"
RMD_SYNC
```

创建或安全复用 `/root/autodl-tmp/projects/Crack_RTDETR-rmd_v1`，核对完整 commit。已有脏工作区、其他仓库、正式 dispatch/训练现场、活动 tmux 等均保留。网络有总超时和低速边界，失败退出，不循环重试；不用 raw.githubusercontent.com 下载脚本。

## 2. prepare；等待完成

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh --help
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh prepare
```

入口固定 `/root/miniconda3/envs/rtdetr/bin/python`，设置当前 worktree 的 PYTHONPATH、`PYTHONUNBUFFERED=1`、`YOLO_AUTOINSTALL=false`。检查 `/root/autodl-tmp/projects/Crack_RTDETR/configs/crack_autodl.yaml`、crack_det 全部清单及公共初始化源权重。正式初始化保存到 worktree 的 `outputs/rmd_v1/rmd_v1_controlled_init.pt`，已经存在时必须通过 hash/来源校验才能复用。

原生离线 AMP 检查需要 `yolo26n.pt` 与 `ultralytics/assets/bus.jpg`。工具只复用当前 worktree 或主目录已有资源并记录 hash，不下载未知资源、不跳过原 AMP 检查、不静默关闭 AMP。

## 3. preflight；最多 900 秒 / 16 个真实训练 micro-batch

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh preflight
```

每 15 秒报告阶段/时间/剩余边界；完整日志位于 `outputs/rmd_v1/checks/` 下此次唯一目录，`outputs/rmd_v1/preflight.json` 指向完整报告。检查原生 AMP、B16/640、原累积规则、实际参数更新、新进程完整恢复、FP32 warmup→真实 val，以及隔离已训练母版的生效性。

至少 8 个有效 probe batch 后，K≥2 的 GT 覆盖率须 ≥50%，全部有效正 DN 的 mean(abs(w−1)) 须 ≥0.01。缺失原母版权重时为 APPLICABILITY_PENDING；K/权重差异不足时为 NOT_APPLICABLE；不得修改 tau、num_dn、batch、增强或门槛去凑 PASS。

如预检明确缺少已训练母版权重，补齐原 checkpoint 与其真实来源记录后，单独执行以下 probe；它也有自己的 900 秒/最多 16 batch 上限，不会长训。此命令不能用来覆盖同一身份已经确定的 NOT_APPLICABLE。

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh probe --probe-weights /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt
```

这份 best.pt 只供隔离诊断，绝不作为正式初始化。checkpoint 缺 git SHA 时，需要原 run 目录内的真实 `source_record.json`/`plan.json`，或原 C19 worktree 的 `outputs/c19_lif_v1/` 记录；不能填写猜测 SHA。

## 4. status；确认所有必要门槛分别 PASS

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh status
```

正确性、容量、恢复、实际 val 生命周期必须全部 PASS，且 APPLICABILITY_PASS 才执行下一段。PENDING/FAIL/NOT_APPLICABLE 时保留证据，可先 pack；start 自身也会拒绝，不会白跑 200 epoch。

## 5. 正式 start；仅用户在上一步确认后执行

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh start
```

只启动独立 tmux `rmd-v1-training`。正式 run 为 `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/rmd_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。原 epochs=200、patience=50，允许原规则提前停止。已存在 run 不覆盖，正式训练不继承任何诊断 optimizer/scaler/RNG/增强序列。

## 6. 日志与进度

```bash
tmux attach -t rmd-v1-training
```

按 `Ctrl-b` 再按 `d` 脱离 tmux；关闭用户电脑不影响服务器训练。

```bash
tail -n 80 -f /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/outputs/rmd_v1/console.log
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh status
```

真实 Python 与 tee 退出码分别保存在当前 attempt 中。tmux 存在或 tee 返回 0 不等于训练成功。机制逐 batch/epoch 统计保存为 `mechanism_batches.jsonl` 和 `mechanism_epochs.jsonl`。

只有训练未完成、无活动 worker，且本实验 last.pt 保留完整 optimizer/EMA/scaler 时才能恢复：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh resume
```

## 7. 等待训练正常结束，再独立 val

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh val
```

固定 FP32、640、B16、workers0、conf0.001、iou0.7、max_det300、augment=false、rect=false、seed42，使用母版 corrected_sorted_conf_mask_v1。成功生成 `outputs/rmd_v1/val_lock.json`，锁定原 val fitness 选出的 best、源码、数据及协议。

若 status 是 `TRAINING_COMPLETED` + `FINAL_EVAL_FAILED`，不要 resume 重训，改用：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh val --recover-final-eval
```

原训练退出码/异常不修改，单独生成恢复记录和 val 锁。如果后续确需评估代码修复，另行审查新的真实提交并使用同步脚本的 `--eval-only`；仅评估器/warmup/文档差异允许进入 `val --recover-final-eval --allow-eval-revision`，训练 SHA 永久保留。

## 8. val 锁成功后，测试同一个 best

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh test
```

不重新选择权重或扫描阈值。独立 test 保存相同配置和十个 IoU 阈值的 AP；不能把训练 CSV 峰值当成独立评估。

## 9. 打包下载；失败或缺 test 时也可运行

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh pack
```

需要包含已有 val/test 的压缩全部原始 query 预测与 GT 时：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-rmd_v1/tools/rmd_v1.sh pack --include-predictions
```

包路径和 SHA256 会打印，默认位于 `outputs/rmd_v1/packages/`。LIGHT 记录 best/last 大小与 hash，不含权重/数据集；缺失证据明确保留，pack 不自动运行 val/test。可选 `pack --full --include-predictions` 另生成权重包。所有包、权重和日志保留在服务器/下载目录，不加入 Git。
