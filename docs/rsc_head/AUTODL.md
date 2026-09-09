# AutoDL 操作手册

只在用户主动执行 `start-direct` 时启动正式训练。此交付没有执行正式训练、完整数据集 val/test 或长时间服务器预检。

## 同步身份

分支 `codex/rsc-head`，基线 `67c3078e54a657fd96d65fee657a75fbb1dae0d6`。使用最终交付回复中的**固定完整 SHA 同步代码块**；完整交付后的同样命令也保存在本地 `outputs/RSC_HEAD_DELIVERY.md`。该文件在提交后生成，避免 Git 提交哈希自引用，不提交生成的交付说明或 checkpoint。

同步入口参数为 `bash tools/sync_rsc_head.sh FULL_SHA [MAIN] [WORKTREE]`。工具核对 origin 为 supershaojie/concrete-crack-rtdetr，fetch `codex/rsc-head`，校验固定 SHA 属于分支历史并包含 C2 基线和 RSC 源码。远程前进也保持固定 SHA。新建 detached worktree，保存 `outputs/rsc_head_delivery.json`，记录提交、目录和关键源码校验和；已有目录不是同库 linked worktree、不是同一 SHA 或存在未提交文件时停止并保留现场，不 reset/clean/stash/覆盖。主 checkout 的 HEAD 和状态前后不变。启动器验证该记录和 HEAD，不依赖分支名，所以支持 detached HEAD。

默认目录：

| 内容 | 路径 |
|---|---|
| 主仓库 | `/root/autodl-tmp/projects/Crack_RTDETR` |
| RSC worktree | `/root/autodl-tmp/projects/Crack_RTDETR-rsc-head` |
| 训练输出 | 主仓库 `/runs/c_series/rsc_head_rtdetr_r18_lite_e200_b16_onlineaug` |
| 运行证据与 console.log | worktree `/outputs/rsc_head` |
| 独立 val/test | worktree `/outputs/rsc_head/evaluation_val`、`evaluation_test` |
| 完整归档 | 主仓库 `/downloads/rsc_head` |

同步及每次命令均允许显式 `RSC_HEAD_MAIN` 覆盖；同步第三参数可选独立 worktree 路径。已有 rtdetr Python 默认 `/root/miniconda3/envs/rtdetr/bin/python`，或每次使用 `RSC_HEAD_PYTHON` 指定已验证的解释器；脚本不会安装/升级依赖。默认命令互不依赖上个终端变量，不覆盖 HOME/CODEX_HOME。路径覆盖必须在同步和后续命令中保持一致。

## 启动（单独执行）

先完成最终回复中的固定 SHA 同步，再执行：

```bash
bash <<'RSC_START'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-rsc-head
bash tools/autodl_rsc_head.sh start-direct
RSC_START
```

严格错误处理在子 bash 内，不会因失败直接关闭 SSH 交互终端。start-direct 做工作树、交付记录、C2 args/data/权重、CUDA/解释器、输出占用、同名 tmux/进程和原子锁检查，执行统一初始化及逐项映射，然后在 `rsc-head-training` tmux 启动。无需 prepare/audit 通过标记；full_server_preflight=NOT_RUN。

已有本实验输出、初始化文件、launch 目录、锁或会话时停止，不覆盖、不自动续训、不杀其他进程。需要重新实验时先人工归档和明确处理本实验现场；工具不会自行清理。OOM 保留日志、原 batch16/imgsz640/AMP 配方并停止，原生自动降 batch 重试被禁用。

证据包括 `plan.json`（命令、cwd、SHA、会话、时间、配置/初始化哈希）、`train_args.yaml`、`actual_train_args.yaml`、`parameter_diff.json`、`initialization.json`、`nc1_loading.json`、`training_setup.json`（完整 AdamW 分组）、环境信息、源码快照/patch、启动/进程/退出状态。worker 的 Python 异常不吞掉；shell 另存真实进程退出码；外层 tee 保存 PIPESTATUS。

## 状态与日志（任意新终端）

```bash
bash <<'RSC_STATUS'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-rsc-head
bash tools/autodl_rsc_head.sh status
RSC_STATUS
```

```bash
tail -n 100 -F /root/autodl-tmp/projects/Crack_RTDETR-rsc-head/outputs/rsc_head/console.log
```

NOT_STARTED=未提交运行；INITIALIZING=初始化；DISPATCHED=已调度；WORKER_STARTING=worker 启动但未进入训练；RUNNING=训练状态与进程启动 token 对应；SUCCEEDED=Python/shell 均记录 0 且有必需产物；FAILED 系列=明确失败或失联。tmux active/dispatched 不代表成功，缺失退出码不会算成功。状态输出含证据和日志尾部。日志查看 Ctrl-C 只退出 tail。

## 训练成功后的独立 val、test（分别执行）

```bash
bash <<'RSC_VAL'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-rsc-head
bash tools/autodl_rsc_head.sh val
RSC_VAL
```

```bash
bash <<'RSC_TEST'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-rsc-head
bash tools/autodl_rsc_head.sh test
RSC_TEST
```

val 读取训练选出的 best.pt；test 必须先有独立 val 成功报告，并核对同一 checkpoint SHA256、源码提交、数据和评估口径。训练末尾自动 val 不能替代本次独立 val/test。已存在对应评估目录时停止，不覆盖。

复用 C24/C2 系列修正后的逐行排序/阈值 mask 实现及原 RTDETRValidator 指标。固定 FP32、imgsz640、batch16、workers0、conf=.001、iou=.7、max_det300、augment=false、rect=false。不引入额外 NMS、query 排序规则、分数调优或阈值搜索。排序后的框、分数、类别使用同一行索引和 mask。

同次评估保存 P/R/mAP50/mAP50–95/AP75、完整 AP 数组、PR/F1/P/R 曲线、混淆矩阵和常规可视化；`predictions_gt.jsonl.gz` 覆盖每张图全部 300 个预测（另标 used_for_metrics）及所有 GT，坐标反变换至原图连续像素，不取整或裁剪。报告核对图片/实例数、split 图片集合哈希、stream 哈希、配置、checkpoint SHA。只从分数变化不能推导 AP 提升。

## 完整打包（不会自动补跑评估）

```bash
bash <<'RSC_PACK'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-rsc-head
bash tools/autodl_rsc_head.sh pack-complete
RSC_PACK
```

时间戳独立 `.tar.gz` 及 `.sha256`、`.inventory.json`、`.verification.json`，无 20 MiB 上限，不覆盖旧包。活跃训练不打包。收集 training 全部实际产物、独立 val/test、命令日志、运行与初始化证据、必要本地验证、源码/模型 YAML/配置/环境与来源哈希，不含原始数据集或权重来源的大文件副本。所有归档成员逐项回读、比对大小和 SHA256，外部校验再次计算。

缺少权重、test、曲线、初始化/运行证据或必要验证时保存**诊断包**并列出缺失，不能标为完整通过；查看 `.verification.json` 的 `evidence_complete` 与 `evidence_status`，归档可读不等于实验材料完整。

服务器校验并列出下载文件：

```bash
bash <<'RSC_VERIFY'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR/downloads/rsc_head
find . -maxdepth 1 -name '*.tar.gz.sha256' -exec sha256sum -c {} \;
ls -lh -- *.tar.gz *.sha256 *.inventory.json *.verification.json
RSC_VERIFY
```

可用 AutoDL Jupyter 文件页下载该目录中的归档及三个 sidecar。也可在本机 PowerShell 交互输入 AutoDL 实际 SSH 主机与端口（项目没有这些账号信息）：

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $rscServer = Read-Host 'AutoDL SSH 主机名'
    $rscPort = Read-Host 'AutoDL SSH 端口'
    New-Item -ItemType Directory -Path './rsc_head_downloads' -Force | Out-Null
    scp -P $rscPort "root@${rscServer}:/root/autodl-tmp/projects/Crack_RTDETR/downloads/rsc_head/rsc_head_complete_*" './rsc_head_downloads/'
    if ($LASTEXITCODE -ne 0) { throw '下载失败' }
    Get-ChildItem -LiteralPath './rsc_head_downloads' -Filter '*.tar.gz.sha256' | ForEach-Object {
        $parts = (Get-Content -LiteralPath $_.FullName -Raw).Trim() -split '\s+', 2
        $archive = Join-Path $_.DirectoryName $parts[1]
        $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $parts[0]) { throw "SHA256 不一致: $archive" }
        Write-Host "SHA256 OK: $archive"
    }
}
```

这段需要真实主机/端口输入，不能编造服务器账号；其余目录和命令使用上述固定默认值。
