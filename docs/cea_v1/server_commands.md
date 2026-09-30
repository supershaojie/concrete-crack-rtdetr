# CEA v1 服务器命令

功能提交：`56e9ef56f3079ea18453b4ed7b477da65388ad04`。
分支：`exp-rtdetr-r18-lite-cea-v1`。
仓库：[supershaojie/concrete-crack-rtdetr](https://github.com/supershaojie/concrete-crack-rtdetr)。
母版：`a0459d6a652cb702699087c88fa39a3e4c4087ec`。
功能文件摘要：`1d985baedcef6f7b19ecc2ef50d290800b923b70b2817dc0c2243377b8e37d29`。

本文由后续 docs-only 提交补入；下列同步固定实际功能提交，因此不依赖文档提交的自引用 SHA。
这些命令不会把 prepare/diagnose/preflight 自动串成正式训练。允许其它实验同时使用 GPU0。

## 1. 同步独立工作树

在服务器终端复制整个代码块。先从实际仓库有界 fetch，再读取固定提交中的脚本；不下载未固定版本脚本执行。
现有目标若脏、有运行任务、分支不符或不能快进，会明确退出并保留现场。

```bash
(
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
SHA=56e9ef56f3079ea18453b4ed7b477da65388ad04
test "$(git -C "$MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
timeout 120 git -C "$MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin exp-rtdetr-r18-lite-cea-v1
SYNC_SCRIPT="$(mktemp /tmp/cea-v1-sync.XXXXXX.sh)"
git -C "$MAIN" show "${SHA}:tools/sync_cea_v1.sh" > "$SYNC_SCRIPT"
bash "$SYNC_SCRIPT" "$SHA"
)
```

成功后工作树为 `/root/autodl-tmp/projects/Crack_RTDETR-cea_v1`。不会切换主工作区。
后续每个代码块可单独复制执行。

## 2. prepare

验证实际源码 import、母版全部配方字段/类型、数据清单和指纹、公共源 SHA，生成本实验独立初值。
母版 args 与公共权重从主项目读取；不要求旧母版 alias 目录存在。
AMP 检查资源只复用服务器已有母版 bus.jpg/yolo26n.pt，缺失会在报告中明确标注，不升级环境。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py prepare
```

## 3. diagnose 与 preflight

diagnose 最多 64 张固定 val 图和 8 个原增强 B16 train batch、900 秒，不读 test、不更新参数。
母版 best 默认读取主项目原 run，强制核对
`24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa`。
缺母版 best 时报告 PENDING，仍可独立执行 preflight。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py diagnose
```

preflight 最多 16 个真实 microbatch、900 秒，采用 B16/640/AMP/AdamW/nbs64 和原增强。
只将 CEA 隔离上下文设 e=20，optimizer/warmup 仍按真实 epoch0。
包含 CPU/CUDA 契约检查及真实 CUDA GradScaler checkpoint 重建；有效更新不足或超时为 PENDING。
两个命令都不会自动长训。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py preflight
```

## 4. 查看报告与状态

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py status
/root/miniconda3/envs/rtdetr/bin/python -m json.tool outputs/cea_v1/diagnose.json
/root/miniconda3/envs/rtdetr/bin/python -m json.tool outputs/cea_v1/preflight.json
```

技术预检必须是同一身份的 TECHNICAL_PASS 才允许 start。
诊断的覆盖率、原始量级和梯度比例决定是否值得长训；有少量非零信号不等于性能有效。
本地 8 张 B1/FP32 val 的探针不能替代这里的完整增强训练诊断。

## 5. 单独 start

只有你决定开始正式实验时才执行。它从公共受控初值开始，绝不使用预检权重或母版 best。
展示诊断结论后直接创建独立 tmux，无二次确认，无空卡/显存阈值闸门。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py start
```

## 6. 查看 tmux，离开与断线重连

```bash
tmux attach -t cea-v1-training
```

按 **Ctrl+B，松开后按 D** 离开而不停止任务。SSH/网页断开或关闭本地电脑不停止服务器 worker。
任务成功/失败都会保留该窗口和最后指标；保留的 pane 不表示 worker 仍在训练。

查看最新本实验窗口（也适用于后续 resume/test）：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py status
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py attach
```

## 7. 同一未完成 run 的 resume

只在中断后且 last 仍含 optimizer/scaler/EMA/epoch、身份一致时使用。
已经完成并 strip optimizer 的 checkpoint 会被拒绝；评估失败应重跑评估，不重启训练。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py resume
```

它创建自己的 resume 会话并打印 attach 命令；也可使用上面的 `attach` 操作。
恢复精度是母版原有 epoch 粒度与 FP16 checkpoint 序列化精度，不是任意 microbatch 无损恢复。

## 8. 训练结束先看 test

best 按原训练 val mAP50-95 选择；入口先打印路径、原选择 epoch、SHA 并锁定。
独立 test 使用 FP32/640/B16/workers0/conf0.001/iou0.7/max_det300/seed42 和已修正排序 mask，
保存一次推理的全部预测/GT/十阈值统计。不会默认拿 last，不会自动 pack。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py test
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py attach
```

保留页面会显示 P、R、F1、AP50、AP75、mAP50-95、相对母版的百分点变化和 best SHA。
P/R/F1 使用与母版相同的各模型最大 F1 报告点口径，不用 test 选择模型或部署阈值。
完整同身份结果已存在时 REUSED，不重复推理。

需要独立 FP32 val 时单独执行；它不参与重新选择 best。完整离线包需要既有 val/test 两份结果。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py val
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py attach
```

可选 `finish` 只补必要 test 并展示指标，同样不 pack、不下载：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py finish
```

## 9. 决定保留后，单独 pack

纯离线读取现有结果，不做网络推理、不补评估、不下载。缺项明确标 INCOMPLETE；
不纳入原始图片和权重本体，权重保留服务器。产出清单与 SHA256，并完整读回验证归档。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-cea_v1
/root/miniconda3/envs/rtdetr/bin/python tools/cea_v1.py pack
```

## 10. 结果位置

| 内容 | 服务器位置 |
|---|---|
| RUN | `/root/autodl-tmp/projects/Crack_RTDETR-cea_v1/runs/c_series/cea_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug` |
| OUT | `/root/autodl-tmp/projects/Crack_RTDETR-cea_v1/outputs/cea_v1` |
| 公共受控初值 | 工作树 `weights/cea_v1_controlled_init.pt` |
| 正式训练日志 | RUN 的 `results.csv`、`cea_batches.jsonl`、`args.yaml` |
| 真实 Python/tee 退出码和终端日志 | OUT 的 `workers/*/exit_codes.json`、`console.log` |
| best/last | RUN 的 `weights/best.pt`、`weights/last.pt` |
| 选择 epoch、早停、best 哈希 | RUN 的 `best_selection.json`、`training_complete.json`，OUT 的 `best_lock.json` |
| 预检/诊断 | OUT 的 `preflight.json`、`diagnose.json`，详细数组在报告中的 worker 目录 |
| 独立评估指标入口 | OUT 的 `evaluation_val.json`、`evaluation_test.json` |
| 同次预测/GT/统计 | 指标报告指向的 `evaluation_*` 目录：`predictions_gt.jsonl.gz`、`ap_pr_stats.npz`、`curves.npz` |
| 离线包 | OUT 的 `packages/cea_v1_COMPLETE_*.tar.gz` 或 `cea_v1_INCOMPLETE_*.tar.gz`，旁边有 `.sha256` |

本次不自动下载任何服务器结果。开发侧测试和待执行项见 [VALIDATION.md](VALIDATION.md)。
