# GEO v1 服务器命令

功能代码固定为 **3bc172c6f22470a00abb2ab6a434e22663406e78**，分支
`exp-rtdetr-r18-lite-geo-v1`。后续纯文档提交不改变这个训练代码身份。
以下路径来自母版历史归档，本地无法核验服务器实时状态；脚本会检查仓库、导入、
公共初始化、数据、原环境和预检，缺失或漂移时明确报错，不自动改配方。

- 主仓库：`/root/autodl-tmp/projects/Crack_RTDETR`
- 独立 worktree：`/root/autodl-tmp/projects/Crack_RTDETR-geo_v1`
- Python：`/root/miniconda3/envs/rtdetr/bin/python`
- run：`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/geo_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`
- tmux：`geo-v1-training`

## 1. 同步独立 worktree

本命令不改主仓库工作区，也不碰其他实验。已有 GEO worktree 若 SHA 或内容不同则保留并拒绝覆盖。
整个块只同步，不训练。

```bash
set -Eeuo pipefail
GEO_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
GEO_SHA=3bc172c6f22470a00abb2ab6a434e22663406e78
test "$(git -C "$GEO_MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
timeout 120s git -C "$GEO_MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin exp-rtdetr-r18-lite-geo-v1
git -C "$GEO_MAIN" cat-file -e "$GEO_SHA^{commit}"
git -C "$GEO_MAIN" show "$GEO_SHA:tools/sync_geo_v1.sh" > /tmp/sync_geo_v1_3bc172c.sh
bash /tmp/sync_geo_v1_3bc172c.sh "$GEO_SHA"
```

核验导入路径：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-geo_v1
PYTHONPATH="$PWD/ultralytics-main:$PWD/tools" /root/miniconda3/envs/rtdetr/bin/python -c 'import sys,torch,ultralytics; print(sys.executable); print(torch.__version__,torch.version.cuda); print(ultralytics.__file__)'
bash tools/geo_v1.sh --help
bash tools/geo_v1.sh preflight --help
bash tools/geo_v1.sh finish --help
bash tools/geo_v1.sh pack --help
```

ultralytics.__file__ 必须来自此 GEO worktree。原环境要求 Python3.10、torch2.1.2+cu121、
NumPy1.26.4；不为预检升级依赖。公共初始化在主仓库 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`。
离线 AMP 资源使用主仓库已有 bus.jpg/yolo26n.pt；缺资源时先恢复原资源，不跳过原 AMP 检查。

## 2. prepare

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh prepare
```

建立公共初始化与完整 recipe 身份，复用已确认的数据快照；找不到可复用快照才首次全量扫描。
已有快照没有变化证据时重复 prepare 不重新读取全套图片/标签。
仅当明确改过数据、且尚未 dispatch 时，才显式运行：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh prepare --refresh-data
```

## 3. 有界 preflight（与长训分开执行）

先等待 GPU0 上其他实验自然结束。预检不会停止其进程，也不会减 B16。

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh preflight --seconds 900 --micro-batches 16
```

上限包含初始化、原 AMP 检查、最多16个 micro-batch、checkpoint、新进程 resume、
有限 FP32 warmup 和一批真实 val。e20 只用于独立诊断。正式原 GradScaler 初始 scale
适应后仍未获得有效更新则 PENDING/非零退出，不自动降低 scale、不自动追加短训。
查看 `outputs/geo_v1/preflight.json` 和该次 preflights 子目录；需要后续诊断时只针对
已记录的具体失败原因，原 recipe 不变。本命令不会自动接 200 轮训练。

## 4. status 与独立 tmux start

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh status
```

仅在有效 preflight 为 PASS 且代码/数据/配方/初始化绑定一致时，单独执行：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh start
```

自动创建 tmux `geo-v1-training`，从公共初始化和 e0 重新开始，epochs200/patience50。
记录源码快照、run identity、PID、命令、开始/结束时间、日志、Python/tee 独立退出码。
已有活动 GEO run/session 会拒绝重复 dispatch；其他实验完全保留。

## 5. 查看进度

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh status
tmux attach-session -t geo-v1-training
```

离开 tmux 查看界面使用 Ctrl-b 后按 d。也可只读最新日志：

```bash
GEO_LOG=$(find /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/outputs/geo_v1/dispatches -name console.log -type f | sort | tail -n 1)
test -n "$GEO_LOG"
tail -n 80 "$GEO_LOG"
```

tmux 存在、tee 返回0、best.pt 存在均不等于训练完成；以 training_completed.json 和 worker
真实退出记录为准。机制日志 `outputs/geo_v1/mechanism.jsonl` 的统计只覆盖每轮前4个 batch。

## 6. 原实验恢复

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh status
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh resume
```

只接受本实验相同身份且具备 epoch/optimizer/scaler/EMA 的 last.pt，原日程续接。
已 strip 的 checkpoint 或已训练完成状态不允许训练 resume；完成后缺评估执行下一节。
只发生在 worker 领取前的 dispatch 故障，且 run 还不存在时，可重新执行 start；
不会覆盖已有 run 或带 worker 历史的初始化。

## 7. 训练后一次 finish

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh finish
```

同一原生 best，完整 FP32 val/test 均默认同时导出全部普通 queries 和 GT。
完整同身份结果直接复用；缺锁但已有完整记录时先核验补锁；仅补尚未成功评估。
然后离线生成 AP/曲线/阈值统计并打一个 COMPLETE 分析包。位置为
`outputs/geo_v1/GEO_v1_COMPLETE_时间戳.tar.gz`，实际文件路径和 SHA256 会由命令打印。
不包含原始数据或大权重本体，包含权重身份与全部预测/GT。此步骤由用户在训练后执行。

## 8. 评估故障独立恢复

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh val
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh test
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh finish
```

以上已完成的同身份 split 只复用，不重复推理；test 必须已有同 best 的正式 val。
发生实际代码/权重/数据身份冲突时会明确拒绝，不覆盖旧成功锁。历史失败记录保留。
无需因评估失败重新训练。也可直接再次 finish，由它判断需补的 split。

## 9. 只打现有材料

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-geo_v1/tools/geo_v1.sh pack
```

pack/status 都不会发起推理、重扫原数据、重挑 best 或运行 test。缺项会列出并标记
INCOMPLETE；不会冒充 COMPLETE。

## 有限网络失败时的 bundle 备用入口

实现已正常 push。另提供经过 `git bundle verify` 的 `GEO_v1_delivery.bundle`，用于
服务器无法访问 GitHub 时。先把该文件上传到 `/root/autodl-tmp/projects/GEO_v1_delivery.bundle`，
再执行以下实际路径命令。bundle 只含代码/文档提交，需主仓库已有母版 a0459d6。

```bash
set -Eeuo pipefail
GEO_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
GEO_SHA=3bc172c6f22470a00abb2ab6a434e22663406e78
GEO_BUNDLE=/root/autodl-tmp/projects/GEO_v1_delivery.bundle
git -C "$GEO_MAIN" bundle verify "$GEO_BUNDLE"
timeout 120s git -C "$GEO_MAIN" fetch --no-tags "$GEO_BUNDLE" exp-rtdetr-r18-lite-geo-v1
git -C "$GEO_MAIN" show "$GEO_SHA:tools/sync_geo_v1.sh" > /tmp/sync_geo_v1_3bc172c.sh
bash /tmp/sync_geo_v1_3bc172c.sh "$GEO_SHA" --bundle "$GEO_BUNDLE"
```

然后从 prepare 顺序执行；不使用 raw.githubusercontent.com 拉脚本。
