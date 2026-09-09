# EVC-Deform AutoDL 操作手册

只运行下面明确选择的命令。同步/status/pack-complete 均不会启动训练或评估；只有 start-direct 启动正式训练，val/test 各启动一次独立评估。不依赖 prepare/audit 标记，不运行 batch16 完整服务器预检。记录 `full_server_preflight=NOT_RUN`。

## 固定 SHA 同步

将交付消息中的最终完整 40 位 SHA 填入 `EVC_SHA`。脚本不允许短 SHA，不自动切换、reset、stash、覆盖其他版本或已有修改，也不会更新 DRA/旧实验工作树。主仓库仅 fetch 和登记新 worktree，原分支和工作区文件保留。

```bash
set -euo pipefail
EVC_SHA='填写交付消息中的完整40位SHA'
export EVC_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
EVC_ROOT=/root/autodl-tmp/projects/Crack_RTDETR-evc
[[ "$EVC_SHA" =~ ^[0-9a-f]{40}$ ]]
case "$(git -C "$EVC_MAIN" remote get-url origin)" in
  https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo 'origin 不匹配，停止'; exit 1 ;;
esac
git -C "$EVC_MAIN" status --short
git -C "$EVC_MAIN" rev-parse HEAD
git -C "$EVC_MAIN" fetch origin codex/evc-deform
test "$(git -C "$EVC_MAIN" rev-parse FETCH_HEAD)" = "$EVC_SHA"
EVC_SYNC=$(mktemp /tmp/evc-sync.XXXXXX.sh)
git -C "$EVC_MAIN" show "$EVC_SHA:tools/sync_evc.sh" > "$EVC_SYNC"
bash "$EVC_SYNC" "$EVC_SHA" "$EVC_MAIN" "$EVC_ROOT"
cd "$EVC_ROOT"
git status --short
git rev-parse HEAD
```

`sync_evc.sh SHA [MAIN] [WORKTREE]` 支持显式根路径。现有目标目录必须属于同一仓库、`codex/evc-deform` 分支、相同 SHA 且无修改；否则停止并保留现场。没有目标目录时创建分支/worktree。同名本地分支若 SHA 不同则停止。远端必须仍指向指定交付 SHA，不接受悄悄前进到其他代码版本。

## 直接训练与状态

前提：主仓库保留原 C2 公共初始化、原运行 args 和数据配置/分割目录；沿用现有 `rtdetr` conda 环境，不自动升级或安装依赖。PyTorch 2.1.2 的接口为目标兼容范围，本次本机验证版本见 README。默认入口会记录实际环境。

```bash
export EVC_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
cd /root/autodl-tmp/projects/Crack_RTDETR-evc
bash tools/autodl_evc.sh start-direct
bash tools/autodl_evc.sh status
tail -n 80 outputs/evc/console.log
tail -f outputs/evc/console.log
# 可选查看独立会话；Ctrl-b 然后 d 仅分离终端
tmux attach -t evc-training
```

只做路径、数据 nc/split、公共初始化 SHA/映射、109 项配方、输出目录检查；记录源码快照与 pip freeze。原生 AMP 检查仍沿用 C2，需要时复用主仓库的 bus.jpg/yolo26n.pt，缺失才取官方资源，不将 YOLO26 的模型/训练配方用作本实验。正式训练 batch16、imgsz640、AMP及原生梯度累积规则保持 C2；原生 AMP 检查若关闭 AMP，则停止。OOM 不自动减 batch/imgsz/累积、不重启旧训练；保留异常、日志与非零退出码。

独立身份：

```text
tmux: evc-training
run: MAIN/runs/c_series/evc_deform_rtdetr_r18_lite_e200_b16_onlineaug
reservation: 同级 .evc.lock 目录
initialization: WORKTREE/weights/evc_deform_controlled_init.pt
launch metadata/log: WORKTREE/outputs/evc/
command logs: WORKTREE/outputs/evc_command_logs/
```

预留目录原子创建；同一实验重复启动、已有 run/init/launch 均拒绝。失败预留与日志保留用于诊断，不自动删除或覆盖。修复后是否归档失败现场及创建新实验身份须人工处理，勿删除其他任务文件/进程。

status 的状态包括 NOT_STARTED / INITIALIZING / DISPATCHED / RUNNING / FINISHING / SUCCEEDED / FAILED / FAILED_INTERRUPTED / FAILED_MISSING_OUTPUT。RUNNING 核对 PID 与 Linux 进程启动 token，避免 PID 复用。成功要求 Python 和 shell 退出码都为 0，并存在 best/last/results.csv；tmux 存在不表示成功。

## 正式训练后，独立 val 再 test

```bash
bash tools/autodl_evc.sh status
bash tools/autodl_evc.sh val
bash tools/autodl_evc.sh test
```

两次评估必须在训练成功后显式运行，输出目录不能已存在。使用冻结的 `outputs/evc/data_config.yaml`、训练 SHA、同一个 best checkpoint SHA；test 校验前面的 val 已完成且 checkpoint/data/source/policy/settings 一致。评估期间再核对 checkpoint 和 data SHA。

固定口径：imgsz640、batch16、workers0、FP32、conf0.001、iou0.7、max_det300、augment=False、rect=False。复用项目 C24/C26 的 `corrected_sorted_conf_mask_v1` 独立评估口径（排序后按对应 confidence mask 筛选）。训练内置 val 保持原 C2 validator，包括历史排序/mask 行为，二者区别明确记录；与 C2/C17 等比较时应使用同一独立评估口径，不能混用历史指标。

导出只复用当前 forward：同次保存全部 300 个最终 query 预测、置信度、类别、连续原图像素坐标、是否用于 metrics、全部 GT 到 `predictions_gt.jsonl.gz`。不补跑推理、不更改评估选择、不截断/四舍五入/裁剪导出。`metrics.json` 包含 AP75（原生 AP 矩阵第 6 个 IoU 列）、AP50、AP50:95、P/R、完整 AP 矩阵、checkpoint/data SHA、源码/环境及计数。图、混淆矩阵、可视化在各评估目录 plots 下。

## 完整打包、下载与校验

```bash
bash tools/autodl_evc.sh pack-complete
# 可选显式唯一输出路径：
# bash tools/autodl_evc.sh pack-complete --output /root/autodl-tmp/projects/Crack_RTDETR/downloads/evc/my_unique_evc.tar.gz

cd /root/autodl-tmp/projects/Crack_RTDETR/downloads/evc
EVC_PACK=$(ls -1t evc_complete_*.tar.gz | head -n 1)
sha256sum -c "$EVC_PACK.sha256"
cat "$EVC_PACK.verification.json"
```

timestamp tar.gz 不限 20 MiB，包含已有全部训练输出、best/last、args/results、曲线/混淆矩阵/已有样例、日志、独立 val/test 图与指标、同次完整预测和 GT、配置、源码/YAML、环境、初始化映射、Git SHA。无整个原始数据集；无自动训练/评估。运行中拒绝打包。

每个包另带 `.sha256`、`.inventory.json`、`.verification.json`；包内 MANIFEST.json 逐文件记录大小/hash，生成后流式读回每个文件验证。`archive_integrity=passed` 只表示包字节一致；只有证据文件齐全且训练退出、best/source/data、评估覆盖/hash均一致，才标为 `evidence_complete=true`。缺失项列入 `missing_evidence`，不完整包仍可用于故障交接，不能当作完整实验结果。不会覆盖同名包/sidecar/partial。

本地 PowerShell 下载（将 SSH 地址/端口替换为 AutoDL 实例信息）：

```powershell
$EvcServer = 'root@你的SSH主机'
$EvcPort = '你的SSH端口'
New-Item -ItemType Directory -Path .\evc-downloads -Force | Out-Null
scp -P $EvcPort "${EvcServer}:/root/autodl-tmp/projects/Crack_RTDETR/downloads/evc/evc_complete_*.tar.gz*" .\evc-downloads\
Get-ChildItem .\evc-downloads\*.tar.gz | ForEach-Object {
    $EvcExpected = (Get-Content -LiteralPath ($_.FullName + '.sha256')).Split(' ')[0]
    $EvcActual = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()
    if ($EvcActual -ne $EvcExpected) { throw "SHA256 mismatch: $($_.Name)" }
    Write-Output "SHA256 OK: $($_.Name)"
    Get-Content -LiteralPath ($_.FullName + '.verification.json')
}
```

## 可选本地复验

不是 start-direct 门槛。输出路径必须尚不存在。该工具仅合成数据小验证，临时模型/optimizer 从不作为正式初始化。

```bash
conda activate rtdetr
python tools/check_evc.py \
  --source "$EVC_MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt" \
  --c2-args "$EVC_MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml" \
  --output outputs/evc_optional_local_check
python tools/check_evc_ops.py --output outputs/evc_optional_ops.json
```
