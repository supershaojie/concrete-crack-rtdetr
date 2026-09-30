# GPC v1 服务器命令

已推送并以远端 refs 核实的功能提交：**`b8d010b4514c7e28213f0fa9748ca67bcdcbc693`**。
分支：`exp-rtdetr-r18-lite-gpc-v1`。
实际仓库：<https://github.com/supershaojie/concrete-crack-rtdetr.git>。
此文件在后续 docs-only 提交中加入；以下命令固定到上述功能提交，功能摘要为 `0cadc7115e720363da7da27052c8c86d50408aaca263e6b7d9eaca76f84a218f`。

以下命令在服务器执行，保持现有 `/root/miniconda3/envs/rtdetr` 环境。允许 GPU0 与另一实验并发；不等待空卡、不自动减 batch。prepare、diagnose、preflight 都不会启动正式 200 轮训练。本地工程检查通过不等于服务器 TECHNICAL_PASS 或精度提升。

## 1. 同步到独立工作树

不会切换主工作区。fetch 每次最多 90 秒；现有工作树脏、身份不同或仍有该工作树的 Python 进程时明确退出。不会 reset、clean、强推或终止其他实验。

```bash
bash <<'GPC_SYNC'
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
GPC_COMMIT=b8d010b4514c7e28213f0fa9748ca67bcdcbc693
test "$(git -C "$MAIN" remote get-url origin)" = "https://github.com/supershaojie/concrete-crack-rtdetr.git"
timeout 90 git -C "$MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin exp-rtdetr-r18-lite-gpc-v1
GPC_SYNC_SCRIPT=$(mktemp /tmp/gpc-v1-sync.XXXXXX)
git -C "$MAIN" show "$GPC_COMMIT:tools/sync_gpc_v1.sh" > "$GPC_SYNC_SCRIPT"
bash "$GPC_SYNC_SCRIPT" "$GPC_COMMIT"
GPC_SYNC
```

目标：`/root/autodl-tmp/projects/Crack_RTDETR-gpc_v1`。若已有分支比固定提交更新，脚本保护现场并退出；查看真实 HEAD/脏文件/进程，不手工 hard reset。

## 2. prepare

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/tools/gpc_v1.py prepare
```

需要服务器公共初值、母版完整 args、真实数据 YAML/划分。输出严格映射、nc1 转换、配方比较和数据指纹，准备自己的未训练初值。缺项会明确 PENDING；不会使用母版 best 代替公共初值。

## 3. 有界 diagnose 与 preflight

先运行母版 best 现象诊断；最多 32 张 val 图、4 个原增强 B16 train batch、900 秒，无 optimizer step，不用 test。缺母版 best 时诊断 PENDING，可继续交付和技术预检。

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/tools/gpc_v1.py diagnose
```

独立运行技术预检；总计最多 900 秒、至多 16 个真实训练 microbatch，固定 B16/640/AMP/AdamW/nbs64，并执行配对 B8。只将 GPC 的 epoch 上下文设为 20，原 warmup/累积不变。有效更新不足为 PENDING，真实 OOM 为 RESOURCE_ERROR，不自动重试扩容或长训。

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/tools/gpc_v1.py preflight
```

两条命令各在独立子进程内执行，日志分别为 `outputs/gpc_v1/diagnose.log`、`preflight.log`。命令返回后查看 JSON；其成功不会自动调用 start。

## 4. 查看报告与状态

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/tools/gpc_v1.py status
cat /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/outputs/gpc_v1/diagnose.json
cat /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/outputs/gpc_v1/preflight.json
```

技术结果必须是同功能身份的 TECHNICAL_PASS。机制诊断与技术结果独立；完整诊断无作用时不建议长训，少量非零损失不证明 AP 提升。源码/算法/配方/环境版本/数据/初值改变后不能冒用旧预检。tmux pane 存在但 worker 已退出属于正常保留页面状态。

## 5. 用户独立启动正式训练

确认看过上述报告后，主动执行下列命令即表示运行意图；无重复确认弹窗。start 展示已有诊断，要求有效技术预检，从本实验公共受控初值 epoch0 开始。**此命令不要串到 prepare/preflight 的自动流水线。**

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/tools/gpc_v1.py start
```

## 6. 查看/离开/重新连接 tmux

```bash
tmux attach -t gpc-v1-training
```

离开但继续运行：先按 `Ctrl+b`，松开后按 `d`。之后可以关闭 SSH、网页或自己的电脑；任务继续在服务器运行。重新连接仍执行上面的 attach，或运行 status。结束页保留最后指标和 Python/tee 退出码。

## 7. 恢复同一未完成 run

只支持有效完整 epoch checkpoint；已完成或 strip optimizer 的 checkpoint 拒绝 resume。恢复在本实验 session 新窗口运行，旧页保留。不同算法/数据/配方/初值的 run 不混用。

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/tools/gpc_v1.py resume
tmux attach -t gpc-v1-training
```

若 200 轮或早停已完成，只是最终评估失败，直接补下一节评估，不重启训练。

## 8. 训练结束后先看独立 test

test 使用训练 val 选出的 best，先显示其路径、zero-based epoch、SHA256 并锁定身份，不使用 last；独立 FP32/640/B16/完整 split。结束后突出 P/R/F1/AP50/AP75/mAP50–95、相对母版百分点差和报告阈值口径。结果保留在单独 tmux 页面，**到指标展示为止，不打包或下载**。

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/tools/gpc_v1.py test
tmux attach -t gpc-v1-test
```

需要独立 val 时单独运行，与 test 锁定同一个 best；不会因 test 调整模型/权重/阈值。

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/tools/gpc_v1.py val
tmux attach -t gpc-v1-val
```

可选 `finish` 仅补缺少的独立 val/test，成功产物身份与校验均一致时显示 REUSED。它也不打包。

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/tools/gpc_v1.py finish
tmux attach -t gpc-v1-finish
```

## 9. 决定保留结果后，单独离线 pack

```bash
/root/miniconda3/envs/rtdetr/bin/python /root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/tools/gpc_v1.py pack
```

纯离线整理已有源码差异、配置、身份、诊断、预检、训练日志和预测统计；不触发评估、不下载，默认不含原始数据或权重本体。缺必要结果标 INCOMPLETE 并列出缺项；不会为了补包再次推理。包附逐文件清单和归档 SHA256，可读性与内容哈希检查后交付。

## 10. 位置

```text
工作树：/root/autodl-tmp/projects/Crack_RTDETR-gpc_v1
受控初值：/root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/weights/gpc_v1_controlled_init.pt
正式 run：/root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/runs/c_series/gpc_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug
权重：上述 run/weights/best.pt、last.pt（留服务器）
训练指标：上述 run/results.csv、gpc_metrics.jsonl、best_selection.json、training_finished.json
元数据/诊断/预检：/root/autodl-tmp/projects/Crack_RTDETR-gpc_v1/outputs/gpc_v1
正式任务日志与退出码：上述 outputs/gpc_v1/workers/<动作_时间>/console.log、exit.json
评估：上述 outputs/gpc_v1/evaluation/{val,test}/latest.json 与 attempt_*/
逐图数据：评估 attempt_*/predictions_gt.jsonl.gz
AP/PR 原始统计：评估 attempt_*/ap_pr_inputs.npz、curves.npz、metrics.json
离线包：上述 outputs/gpc_v1/packages/gpc_v1_{COMPLETE或INCOMPLETE}_*.tar.gz
```

开发交付未连接或修改服务器，正式数据/母版 best/B16+B8 容量与正式精度仍 PENDING。没有自动下载旧 DTR/GNR 结果，没有为 GPC 宣称涨点。
