# AutoDL：独立固定提交与一次启动

此文不给任何立即增加训练进程的授权。用户确认单卡GPU0有容量时才运行start-direct。
Codex交付回复给出本次真实完整40位SHA；下方FULL_SHA必须替换为该值。
严格模式放在子shell，不影响SSH终端。

首次同步：目标工作树不需预先存在。只fetch本分支，从指定提交导出同步脚本。

```bash
(
  set -euo pipefail
  C24_SHA=FULL_SHA
  C24_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
  git -C "$C24_MAIN" fetch origin refs/heads/codex/rtdetr-c24-lif-v1:refs/remotes/origin/codex/rtdetr-c24-lif-v1
  C24_SYNC=$(mktemp /tmp/sync_c24_lif_v1.XXXXXX.sh)
  git -C "$C24_MAIN" show "$C24_SHA:tools/sync_c24_lif_v1.sh" > "$C24_SYNC"
  bash "$C24_SYNC" "$C24_SHA" "$C24_MAIN" /root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1
)
```

同步不改主仓库HEAD，不删除runs/downloads，不reset/clean/stash。已有同SHA且干净detached工作树只核验；
不同SHA、tracked dirty、无关目录均拒绝并保留。同步读取分支专属remote ref，避免并行fetch的FETCH_HEAD竞争。

只启动一次（内部预检通过后才派发tmux）：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1
  CUDA_VISIBLE_DEVICES=0 bash tools/autodl_c24_lif_v1.sh start-direct
)
```

不要先运行一次完整preflight-only后再运行start-direct。preflight-only仅供主动选择只检查时使用。
现有本地AMP/true-half为REQUIRES_REVIEW；如果服务器仍出现不被证据支持的集合漂移，脚本会停止派发。
不会把本地PASS直接当成4090/B16容量PASS。

真实状态、预检日志及正式日志：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1/tools/autodl_c24_lif_v1.sh status
tail -n 80 -f /root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1/outputs/c24_lif_v1/preflight.log
tail -n 80 -f /root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1/outputs/c24_lif_v1/console.log
```

console.log在派发worker后才生成；缺失表示未派发训练，不要对不存在的console反复tail。
status区分NOT_STARTED/CHECKING/DISPATCHED/RUNNING/SUCCESS/FAILED/REQUIRES_REVIEW。
RUNNING在完成首个真实训练batch后写入。tmux存在或DISPATCHED都不是训练成功证据。
worker内部重新source conda.sh/activate rtdetr，设置本工作树PYTHONPATH、PYTHONUNBUFFERED和YOLO_AUTOINSTALL。
Python和外层shell都记录真实退出码；不由tee掩盖。不同任务的锁和worktree完全独立。

故障小包可直接执行，不读权重、不用GPU、不再预检：

```bash
bash /root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1/tools/autodl_c24_lif_v1.sh pack-light
```

下载输出打印的绝对`.tar.gz`文件，目录固定为 `/root/autodl-tmp/projects/Crack_RTDETR/downloads/c24_lif_v1`。
每包有独立时间戳、SHA256、inventory和verification；故障包硬上限8,000,000字节。

正式训练成功后，同一个best依次独立val、test、打包：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/projects/Crack_RTDETR-c24-lif-v1
  CUDA_VISIBLE_DEVICES=0 bash tools/autodl_c24_lif_v1.sh val
  CUDA_VISIBLE_DEVICES=0 bash tools/autodl_c24_lif_v1.sh test
  bash tools/autodl_c24_lif_v1.sh pack-complete
)
```

评估固定640/B16/workers0/device0/half=False/conf=.001/iou=.7/max_det300/augment=False/rect=False/plots=True。
test核验val锁定的best、代码、data及全部effective settings；不按test选择epoch或调参。
pack-complete缺少val/test时明确失败，不自动补跑。不默认附加init/best/last、全预测或数据集；
best/last及全量预测保留服务器并记录哈希，分析包含完整指标/曲线/CSV/args与一份源码快照。
