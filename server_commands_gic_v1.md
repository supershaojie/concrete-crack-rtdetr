# GIC v1 服务器命令

远端已核对为 `https://github.com/supershaojie/concrete-crack-rtdetr.git`，分支 `exp-rtdetr-r18-lite-gic-v1`，母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。交付提交完整SHA见聊天交付；下面从本次fetch取得真实SHA并核对，运行manifest记录实际HEAD，不在源码中伪造“最终自身SHA”。

本地开发目录为 Windows 的 `C:/Users/o'v'o/.codex/worktrees/gic-v1/Crack_RTDETR`；下列命令只在服务器执行。本次未启动服务器正式训练或test。

## 1. 同步独立worktree并做一次有界预检

```bash
set -euo pipefail
GIC_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
GIC_WT=/root/autodl-tmp/projects/Crack_RTDETR-gic_v1
GIC_BRANCH=exp-rtdetr-r18-lite-gic-v1
test "$(git -C "$GIC_MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
git -C "$GIC_MAIN" fetch origin "$GIC_BRANCH"
GIC_SHA="$(git -C "$GIC_MAIN" rev-parse FETCH_HEAD)"
git -C "$GIC_MAIN" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$GIC_SHA"
if [ -e "$GIC_WT" ]; then
  test "$(git -C "$GIC_WT" rev-parse --show-toplevel)" = "$GIC_WT"
  test "$(git -C "$GIC_WT" rev-parse --path-format=absolute --git-common-dir)" = "$(git -C "$GIC_MAIN" rev-parse --path-format=absolute --git-common-dir)"
  test -z "$(git -C "$GIC_WT" status --porcelain)"
  test "$(git -C "$GIC_WT" rev-parse HEAD)" = "$GIC_SHA" || { echo '已有worktree提交不同，保留现场；请核对已有实验，不覆盖。'; exit 1; }
else
  if git -C "$GIC_MAIN" show-ref --verify --quiet "refs/heads/$GIC_BRANCH"; then
    test "$(git -C "$GIC_MAIN" rev-parse "$GIC_BRANCH")" = "$GIC_SHA"
    git -C "$GIC_MAIN" worktree add "$GIC_WT" "$GIC_BRANCH"
  else
    git -C "$GIC_MAIN" worktree add -b "$GIC_BRANCH" "$GIC_WT" "$GIC_SHA"
  fi
fi
test "$(git -C "$GIC_WT" rev-parse HEAD)" = "$GIC_SHA"
printf 'GIC delivery HEAD: %s\n' "$GIC_SHA"
cd "$GIC_WT"
bash tools/gic_v1.sh preflight
```

preflight默认总上限900秒，写 `outputs/gic_v1/preflight.json` 和相应日志。检查109字段配方、公共初始化SHA/加载规则、数据划分、必要单测以及最多两个临时真实B16/640数值步。相同身份成功记录直接复用。AMP数值冒烟用单位loss缩放，正式训练保留母版GradScaler。共享GPU0上有其他作业不会阻止启动；真实OOM原样报告，不降B16、不换卡、不自动重试。

Python固定 `/root/miniconda3/envs/rtdetr/bin/python`。原生AMP检查需要已有的主仓库 `yolo26n.pt`（或 `weights/yolo26n.pt`）和 `ultralytics-main/ultralytics/assets/bus.jpg`（或主仓库bus.jpg）；入口只复用现存文件，缺失会列具体路径，不下载大包。已核验的完整母版args随分支保存；若要核对服务器自己的原始文件，可用 `preflight --mother-args /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml`。

## 2. 预检成功后明确启动训练

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-gic_v1
bash tools/gic_v1.sh start
bash tools/gic_v1.sh status
bash tools/gic_v1.sh attach
```

训练tmux=`gic-v1-training`。退出attach但保留训练用 `Ctrl-b d`。正式产物固定在 `/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/gic_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug`。证据位于worktree `outputs/gic_v1`。status读取进程、退出码、日志尾和产物；session存在不代表仍运行。

仅对同身份的中断训练，且原last仍包含optimizer/scaler/EMA时，可显式执行 `bash tools/gic_v1.sh resume`；从checkpoint的真实epoch继续。已完成训练不再resume。已有存活本实验pane/进程拒绝重复启动，其他实验不受影响。

## 3. 训练成功结束后单独执行finish

下面这组命令与启动训练分开粘贴。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-gic_v1
bash tools/gic_v1.sh status
bash tools/gic_v1.sh finish
tmux attach -t gic-v1-finish
```

finish核验训练完成、Python/tee退出0和固定best SHA，然后补齐独立FP32 val/test、同期全部query/GT导出、离线分析，显示指标并打包。两种会话都保留退出页面。原始指标小数及百分数均保存；P=Precision。已有成功且同身份结果复用，成功评估缺导出会报缺项并停止，不悄悄重推理。阈值错误分析只由val选阈值。

## 4. 随时只读查看，或只用已有文件打包

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-gic_v1
bash tools/gic_v1.sh status
bash tools/gic_v1.sh pack
```

pack不训练、不推理、不扫描原始数据。缺文件生成 `GIC_v1_PARTIAL_<时间>.tar.gz` 并列缺项；完整才生成 `GIC_v1_COMPLETE_<时间>.tar.gz`，包含best.pt、代码/配置/测试/训练/评估/导出/诊断和逐文件清单，另附SHA256。同身份完整包可复用。包与权重留服务器，先看指标再决定是否下载。

若Git推送因网络失败，本地正常重试为：`git push origin exp-rtdetr-r18-lite-gic-v1`；不使用force、不改默认分支、不清理其他worktree。
