# AutoDL操作

默认主仓库 `/root/autodl-tmp/projects/Crack_RTDETR`，独立worktree为主仓库路径加`-lcr`。LCR_MAIN可覆盖主仓库；每个新终端需明确设置，不依赖前一终端变量。现有rtdetr环境，不安装升级依赖；LCR_PYTHON可指定已验证解释器。

## 固定提交同步

复制最终交付回复中含完整40位SHA的同步代码块，不取分支最新tip代替。接口 `bash tools/sync_lcr.sh SHA [MAIN_REPO] [LCR_WORKTREE]`。初次在主仓库fetch分支，从固定SHA通过git show导出sync脚本到mktemp后执行。正确HTTPS/SSH origin均支持，不改协议。

远程即使前进仍确认交付SHA属于codex/lcr-aifi历史，新worktree采用detached HEAD。交付SHA、主仓库/worktree路径、模型和入口哈希记录于 `outputs/lcr_delivery.json`。已有目标脏/不同SHA/其他仓库或非worktree时停止保留，不reset/clean/stash。主工作树不切分支。

## 启动（本次交付没有执行）

```bash
bash <<'LCR_START'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-lcr
bash tools/autodl_lcr.sh start-direct
LCR_START
```

核对主仓库真实C2 args：`runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml`，数据 `configs/crack_autodl.yaml` 和全部split路径；核对统一权重路径/SHA，逐字段配方审计。原生AMP探针bus.jpg/yolo26n.pt优先本地复用，缺失仅从官方URL获取，不用于RT-DETR初始化。

无完整prepare/audit门槛，full_server_preflight=NOT_RUN；初始化/配置检查报错停止。已有lcr-training tmux、LCR worker、同名结果、初始化或launch目录均保护。结果旁`.lcr.lock`原子持久占用，失败也保留，不自动删锁或续训。

输出 `outputs/lcr/`：train_args.yaml/actual_train_args.yaml、逐字段差异、权重映射、源码snapshot/patch、pip freeze、plan/process/exit、console.log。记录实际Python/torch/Ultralytics/LCRAIFI来源、工作目录/命令/SHA/配置和初始化权重SHA256。tmux会话lcr-training，捕获Python及shell退出码；OOM不改batch/AMP/imgsz、不自动重启。

## 状态/日志

```bash
bash <<'LCR_STATUS'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-lcr
bash tools/autodl_lcr.sh status
if test -f outputs/lcr/console.log; then tail -n 80 outputs/lcr/console.log; fi
LCR_STATUS
```

跟随日志：`tail -f /root/autodl-tmp/projects/Crack_RTDETR-lcr/outputs/lcr/console.log`。

NOT_STARTED未提交；INITIALIZING初始化；DISPATCHED已调度；WORKER_STARTING正在setup；RUNNING已进入train-start；FINISHING待shell退出证据；SUCCEEDED要求双退出码0且best/last/results存在。FAILED等明确失败。tmux active或缺失退出码不能解释为成功。

## 独立val，训练成功结束后单独执行

```bash
bash <<'LCR_VAL'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-lcr
bash tools/autodl_lcr.sh val
LCR_VAL
```

best.pt由训练选出；输出outputs/lcr/evaluation_val。保持640/batch16/FP32及统一阈值与corrected_sorted_conf_mask_v1口径。历史C2若不同口径，需要相同入口重新评估才能公平比较。

## 独立test，独立val完成后单独执行

```bash
bash <<'LCR_TEST'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-lcr
bash tools/autodl_lcr.sh test
LCR_TEST
```

必须与独立val的best SHA256、源码、数据、设置一致。输出outputs/lcr/evaluation_test。metrics.json与predictions_gt.jsonl.gz保存真实split、图像/实例数量、预测、GT、标识、尺寸、类别及配置。原生曲线/混淆矩阵/样图来自同次评估。

## 完整打包，单独执行，不补跑评估

```bash
bash <<'LCR_PACK'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR-lcr
bash tools/autodl_lcr.sh pack-complete
LCR_PACK
```

默认主仓库downloads/lcr，微秒时间戳 `lcr_complete_YYYYMMDD_HHMMSS_ffffff.tar.gz` 及 `.tar.gz.sha256`、`.tar.gz.inventory.json`、`.tar.gz.verification.json`，禁止覆盖。真实文件名由实际打包输出，本次无正式训练包，不编造未来文件名。

收集全部现存training（含best/last和所有图表/样图）、独立val/test、运行日志/状态/初始化和验证证据、源码/YAML/入口/环境、清单哈希。无20MiB限制，不含原始数据集、凭证、token或其他实验结果。活动运行拒绝打包；不完整证据仍可导出，但evidence_complete=false、列出missing_evidence。evidence_status分PASSED/FAILED/MISSING_EVIDENCE；archive_integrity=passed仅证明归档可读且清单/成员哈希一致。

服务器核验最新实际包并打印四个下载文件：

```bash
bash <<'LCR_VERIFY'
set -Eeuo pipefail
cd /root/autodl-tmp/projects/Crack_RTDETR/downloads/lcr
LCR_SUM="$(find . -maxdepth 1 -type f -name 'lcr_complete_*.tar.gz.sha256' | sort | tail -n 1)"
test -n "$LCR_SUM"
sha256sum -c "$LCR_SUM"
tar -tzf "${LCR_SUM%.sha256}" >/dev/null
cat "${LCR_SUM%.sha256}.verification.json"
printf '\nDownload these four matching files:\n'
printf '%s\n' "${LCR_SUM%.sha256}" "$LCR_SUM" "${LCR_SUM%.sha256}.inventory.json" "${LCR_SUM%.sha256}.verification.json"
LCR_VERIFY
```

用AutoDL文件面板下载打印的四个同名文件，或本地PowerShell运行以下交互式下载。SSH主机/端口本次未提供，执行时输入真实信息：

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $lcrServer = Read-Host 'AutoDL SSH地址（root@实际主机）'
    $lcrPort = Read-Host 'AutoDL SSH端口'
    $lcrRemote = & ssh -p $lcrPort $lcrServer "find /root/autodl-tmp/projects/Crack_RTDETR/downloads/lcr -maxdepth 1 -type f -name 'lcr_complete_*.tar.gz' | sort | tail -n 1"
    if ($LASTEXITCODE -ne 0 -or -not $lcrRemote) { throw '未找到实际归档或SSH失败' }
    $lcrRemote = $lcrRemote.Trim()
    foreach ($suffix in @('', '.sha256', '.inventory.json', '.verification.json')) {
        & scp -P $lcrPort "${lcrServer}:${lcrRemote}${suffix}" .
        if ($LASTEXITCODE -ne 0) { throw '下载失败' }
    }
    $lcrFile = Split-Path $lcrRemote -Leaf
    $expected = ((Get-Content -LiteralPath "$lcrFile.sha256" -Raw).Trim() -split '\s+')[0]
    if ((Get-FileHash -LiteralPath $lcrFile -Algorithm SHA256).Hash -ine $expected) { throw 'SHA256不一致' }
    Write-Host "SHA256通过：$lcrFile"
    Get-Content -LiteralPath "$lcrFile.verification.json"
}
```

有限本机工具check_lcr.py/check_lcr_data.py/check_lcr_ops.py/check_lcr_sync.py不等于正式训练或完整评估；运行使用新输出目录，保留既有证据。
