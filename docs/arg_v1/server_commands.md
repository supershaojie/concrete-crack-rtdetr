# ARG v1 服务器分段命令

固定训练代码提交：`e6d6ce11783e57cdd3377f6b5a103665218df065`。
分支：`exp-rtdetr-r18-lite-arg-v1`。母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
代码提交已普通推送并核对远端。后续交付文档提交只记录此代码提交的证据和命令；服务器保持固定代码SHA，不执行泛化的git pull。

这些命令由用户在服务器执行。本轮未连接服务器、未派发正式训练。每一段完成后再执行下一段；preflight不会自动启动200epoch训练。

## 1. 同步（首条服务器命令）

完整复制此块。fetch和脚本内部fetch都有120秒上限、30秒低速超时；失败立即退出，不循环重试。

```bash
bash <<'ARG_SYNC'
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
SHA=e6d6ce11783e57cdd3377f6b5a103665218df065
timeout 120s git -C "$MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin exp-rtdetr-r18-lite-arg-v1
git -C "$MAIN" cat-file -e "$SHA^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$SHA" FETCH_HEAD
SCRIPT=$(mktemp /tmp/sync_arg_v1.XXXXXX.sh)
trap 'rm -f -- "$SCRIPT"' EXIT
git -C "$MAIN" show "$SHA:tools/sync_arg_v1.sh" > "$SCRIPT"
bash "$SCRIPT" "$SHA"
ARG_SYNC
```

成功创建/复用 `/root/autodl-tmp/projects/Crack_RTDETR-arg_v1`。主工作区不切分支。
已有同名分支、不同SHA的worktree或未提交变化会保留并拒绝同步；不要reset/clean覆盖它们。

## 2. prepare（必须等同步成功）

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh prepare
```

须已存在主仓库的 `configs/crack_autodl.yaml`、完整crack_det划分和 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`。
数据逐文件SHA读取可能需要数分钟，进度按1000图输出。
AMP自检还要求主仓库的 `yolo26n.pt`（或weights同名文件）和 `ultralytics-main/ultralytics/assets/bus.jpg`；缺失时短检报错，不会联网绕过或关闭AMP。

prepare建立 `outputs/arg_v1/arg_v1_init.pt`、`prepare.json`、`train_args.yaml`、`recipe_diff.json`、数据清单与源码快照。

## 3. 一次有界 preflight（必须等prepare成功）

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh preflight --seconds 900 --micro-batches 16
```

最多16个实际训练micro-batch、总900秒；诊断e=20，正式训练仍从e=0。
结果在 `outputs/arg_v1/preflight.json`。CUDA B16/640、有效参数更新、ARG-only梯度、新进程val和resume必须分别PASS。
若原生初始scale的有界尝试仅出现overflow、隔离scale128成功，原生适应项仍明确PENDING；其余必需项均PASS后才可能 `start_eligible=true`，正式scaler不改。
缺资源/容量/时间或真实异常时，不把其他PASS替代缺项。不要把preflight和start无条件串联。

可选的CPU机制复核入口（不能替代preflight）：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh probe
```

## 4. 查看状态

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh status
```

检查preflight每项状态、`start_eligible`、是否已有worker/tmux/run。该命令不会启动或恢复训练。

## 5. 正式 start（必须等必需短检通过）

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh start
```

入口固定使用 `/root/miniconda3/envs/rtdetr/bin/python`，设置当前worktree的PYTHONPATH、`PYTHONUNBUFFERED=1`、`YOLO_AUTOINSTALL=false`。
正式worker只在独立tmux `arg-v1-training` 中运行，关掉用户电脑不影响服务器训练。
run固定为 `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/arg_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。
保持原200epoch上限与patience50；正常早停不强行补训。

## 6. 日志与进度（start派发后）

```bash
tmux attach -t arg-v1-training
```

按 `Ctrl-b` 后按 `d` 脱离，不结束训练。也可独立执行：

```bash
tail -n 80 -F /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/outputs/arg_v1/dispatches/*/console.log
```

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh status
tail -n 3 /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/outputs/arg_v1/epochs.jsonl
```

dispatch目录包含worker精确命令、PID/创建时间、开始时间、console.log、exit.json、真实Python和tee退出码。不要把tmux存在当作成功。

## 7. 异常中断时的resume（仅训练尚未正常结束）

只有本实验有效 `last.pt` 含原epoch/optimizer/EMA/scaler且无活动worker时执行：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh resume
```

不从best、诊断或另一实验权重恢复。resume继续原epoch与ARG ramp，不重启warmup。
若状态是 `TRAINING_COMPLETED+FINAL_EVAL_FAILED`，跳到下面val恢复评估，不使用resume。

## 8. 训练后独立val / final_eval失败恢复

必须有原生训练结束证据、best和无活动worker。命令只评估原fitness选出的best：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh val
```

完整FP32协议：640/B16/workers0/conf.001/iou.7/max_det300/seed42，无augment/rect。
保存checkpoint hash、训练/eval SHA、数据身份、实际配置、指标、十阈值AP和 `outputs/arg_v1/val_lock.json`。
若在恢复final_eval，会保留原异常及exit1，另写 `evaluation_recovery.json`，不会重写训练fingerprint。

需要在这次val同时导出全300个query和GT，可在本段改用：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh val --include-predictions
```

两种val命令选择一种执行。

## 9. 锁定同一best的最终test（必须等val成功）

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh test
```

入口核对同一best hash、val锁、数据与eval源码，不重新选权重或扫阈值；已有最终test锁时拒绝重测。
如需要test预测，在首次执行本段时用 `test --include-predictions`，文件带独立test名称和身份，不混入val。

## 10. LIGHT交付包（失败时也可执行）

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh pack
```

输出路径会打印为 `/root/autodl-tmp/projects/Crack_RTDETR-arg_v1/outputs/arg_v1/ARG_v1_LIGHT_时间戳.tar.gz`，目录下可直接下载实际生成的文件。
默认无数据集和best/last权重；manifest含每个成员的大小及SHA256。缺少test仍打包故障证据，缺项不会标PASS。

需要val原始预测与GT的分析包：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v1/tools/arg_v1.sh pack --include-predictions
```

若val锁存在但未导出预测，会对锁定best/配置重新执行只读val导出；不会运行test或重新选权重。
