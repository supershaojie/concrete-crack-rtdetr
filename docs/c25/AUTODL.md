# AutoDL：C25 手动运行

仅用户在服务器手动执行以下命令。沿用已有 `rtdetr` conda环境（Python3.10、PyTorch2.1.2+cu121、RTX4090），没有安装或升级步骤。使用 `/root/autodl-tmp/projects/Crack_RTDETR-c25` 独立工作树；主仓库与原SCCA工作树保留。

## 同步固定提交

将 `C25_SHA` 填为交付回复中已核对远端的完整40位SHA。本段只同步源码，不启动训练。不要求主仓库清空历史未提交文件；新工作树以detached HEAD固定提交，避免占用或改写其他工作树的分支。已有目标目录必须属于同一仓库、HEAD完全相同且没有未提交修改，否则显示状态后退出，不覆盖。

```bash
(
set -euo pipefail
C25_SHA='填入交付回复的完整40位SHA'
C25_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
C25_WT=/root/autodl-tmp/projects/Crack_RTDETR-c25
[[ "$C25_SHA" =~ ^[0-9a-f]{40}$ ]]
case "$(git -C "$C25_MAIN" remote get-url origin)" in
  https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo '仓库origin不匹配，请保留现场检查'; exit 1 ;;
esac
git -C "$C25_MAIN" fetch --no-tags origin codex/c25-cscef-scca
git -C "$C25_MAIN" cat-file -e "$C25_SHA^{commit}"
git -C "$C25_MAIN" merge-base --is-ancestor "$C25_SHA" FETCH_HEAD
if [[ -e "$C25_WT" || -L "$C25_WT" ]]; then
  git -C "$C25_MAIN" worktree list --porcelain | grep -Fx "worktree $C25_WT"
  git -C "$C25_WT" status --short
  [[ "$(git -C "$C25_WT" rev-parse HEAD)" == "$C25_SHA" ]] || { echo '已有工作树版本不同，未覆盖'; exit 1; }
  [[ -z "$(git -C "$C25_WT" status --porcelain)" ]] || { echo '已有修改，未覆盖'; exit 1; }
else
  git -C "$C25_MAIN" worktree add --detach "$C25_WT" "$C25_SHA"
fi
git -C "$C25_WT" show -s --format=fuller HEAD
)
```

远端将来增加提交时仍使用上述固定SHA，不自动追随新HEAD。此处没有force push、reset、clean、合并或删除目录。

## start-direct c25

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c25
bash tools/autodl_scca.sh start-direct c25
```

这是唯一启动正式训练的命令。读取真实C2 args并核对109字段、类型和主仓库数据路径；校验初始化SHA与公共映射并生成全新受控初始化。目标run目录、tmux会话 `scca-c25` 与跨工作树共享 `.scca.lock` 保护防止重复运行。worker通过原生训练API重建nc1并记录加载、实际参数和双模块优化器记录。

不绑定prepare/audit、额外整网FP32/AMP对照或真实batch16 smoke；launch/plan/setup均记录 `full_server_preflight=NOT_RUN`。保留原生AMP必要资源/检查，不运行附加大显存诊断。OOM保留日志并退出，不自动降batch/imgsz/lr、关闭AMP或终止其他实验。失败保留锁和证据，先检查具体原因；不要照搬删除锁后自动重跑。

正式目录固定为：

```text
/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c25_rtdetr_r18_lite_cscef_v51_scca_e200_b16_onlineaug
```

## status c25

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c25
bash tools/autodl_scca.sh status c25
```

## 查看完整日志

```bash
less -R +G /root/autodl-tmp/projects/Crack_RTDETR-c25/outputs/scca/c25/console.log
```

按 `g` 回日志开头、`G` 到末尾、`F`持续跟随、`Ctrl-C`停止跟随、`q`退出阅读，不会终止训练。

## val c25

正式训练成功退出后，对验证集选出的best做独立完整val：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c25
bash tools/autodl_scca.sh val c25
```

输出 `outputs/scca/c25/evaluation_val/`。已有评估目录会拒绝覆盖。不要用 `--batch`、`--data` 等覆盖默认值做正式对照；这些选项用于显式隔离的工具验证/其他历史模型评估。

## test c25

固定配置和同一个best后运行。入口要求已完成的val，核对权重SHA、数据SHA、源码commit、imgsz/batch/workers/half/conf/iou/max_det/augment/rect一致；不根据test调参。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c25
bash tools/autodl_scca.sh test c25
```

输出 `outputs/scca/c25/evaluation_test/`。C2/C17/C24同入口重评可在另外的明确输出目录按需执行，不是启动C25的前置条件。

## pack-complete c25

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-c25
bash tools/autodl_scca.sh pack-complete c25
```

不运行任何训练/推理，未运行评估、缺失checkpoint或图像均如实标记。轻量包仍可单独使用 `bash tools/autodl_scca.sh pack c25`，保留旧20 MiB限制及图片/权重排除规则；完整包没有这一限制。

## 下载位置与校验

打包命令打印本次时间戳文件名。用AutoDL文件浏览器进入以下目录，下载该tar.gz和同名三个sidecar，不要下载整个数据集：

```text
/root/autodl-tmp/projects/Crack_RTDETR/downloads/scca/
  c25_complete_YYYYMMDD_HHMMSS_microseconds.tar.gz
  c25_complete_YYYYMMDD_HHMMSS_microseconds.tar.gz.sha256
  c25_complete_YYYYMMDD_HHMMSS_microseconds.tar.gz.inventory.json
  c25_complete_YYYYMMDD_HHMMSS_microseconds.tar.gz.verification.json
```

服务器或Linux下载端（填入本次包名）：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR/downloads/scca
C25_PACKAGE='填入刚打印的c25_complete_时间戳.tar.gz'
sha256sum -c "$C25_PACKAGE.sha256"
cat "$C25_PACKAGE.verification.json"
```

Windows下载端用PowerShell核对：

```powershell
$C25Package = 'C:\下载目录\c25_complete_时间戳.tar.gz'
$expected = ((Get-Content -LiteralPath ($C25Package + '.sha256')) -split '\s+')[0]
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $C25Package).Hash.ToLowerInvariant() -ne $expected) { throw 'SHA256不匹配' }
Get-Content -LiteralPath ($C25Package + '.verification.json')
```

verification中的 `status=passed` 表示包逐文件读回校验通过；同时检查 `evidence_status` 和 `missing_evidence`，不能把缺失证据当作已运行。完整包内MANIFEST提供逐文件字节数与SHA256；权重、预测图片、结果包仅留服务器/下载端，不push到Git。
