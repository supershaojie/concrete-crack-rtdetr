# NCR v1 服务器命令

已核实 origin：`https://github.com/supershaojie/concrete-crack-rtdetr.git`。实验分支 `exp-rtdetr-r18-lite-ncr-v1`，精确母版 `a0459d6a652cb702699087c88fa39a3e4c4087ec`。交付提交以最终回复和 `git ls-remote` 核验为准；下面在 fetch 时解析实际 SHA，不硬编码尚未存在的自身提交。

本文件只给服务器命令；本地 Windows 的 Codex worktree 不是 `/root/...` 服务器目录。本次代码交付没有自动启动 200 轮训练、正式 val/test 或下载大包。

## 1. 获取分支并安全建立独立 worktree

复制整个代码块。已有同名目录/分支若身份不同会停止并保留，不覆盖，不改变主仓库当前分支。已有 NCR 目录必须恰为此交付 SHA；有未提交修改或另一版本时先人工查看差异，不能对正在训练的 worktree 自动更新代码。

```bash
(
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-ncr_v1
BRANCH=exp-rtdetr-r18-lite-ncr-v1
cd "$MAIN"
test "$(git remote get-url origin)" = 'https://github.com/supershaojie/concrete-crack-rtdetr.git'
git fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
DELIVERY_SHA=$(git rev-parse "refs/remotes/origin/$BRANCH")
git merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$DELIVERY_SHA"
printf 'Fetched NCR delivery: %s\n' "$DELIVERY_SHA"
if [ -e "$WT" ]; then
  test -f "$WT/.git"
  test "$(git -C "$WT" branch --show-current)" = "$BRANCH"
  test "$(git -C "$WT" rev-parse HEAD)" = "$DELIVERY_SHA"
  test -z "$(git -C "$WT" status --porcelain --untracked-files=no)"
  test "$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)" = "$(git rev-parse --path-format=absolute --git-common-dir)"
elif git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  test "$(git rev-parse "$BRANCH")" = "$DELIVERY_SHA"
  git worktree add "$WT" "$BRANCH"
else
  git worktree add -b "$BRANCH" "$WT" "$DELIVERY_SHA"
fi
git -C "$WT" status --short
git -C "$WT" log -1 --format='%H %s'
)
```

## 2. 有界预检

现有环境固定 `/root/miniconda3/envs/rtdetr/bin/python`，Python3.10 / torch2.1.2+cu121 / NumPy1.26.4；实查写入报告，不升级。数据 YAML 必须已存在于主仓库 `configs/crack_autodl.yaml`，数据位于主仓库 `datasets/crack_det`。公共初始化必须为主仓库 `weights/rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256 `fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。

母版原生 AMP 检查还需要主仓库已有 `yolo26n.pt`（也可在主仓库 weights 下）和 `ultralytics-main/ultralytics/assets/bus.jpg`。入口仅复制已有资源；缺失时说明缺项，不自动下载。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-ncr_v1
bash tools/ncr_v1.sh preflight
```

最多 15 分钟、必要测试、同一真实 B16/640 批次的 FP32/AMP 各一步，临时模型丢弃；检查 GradScaler 初值 128 继承母版损失冒烟，正式训练不改 scaler。检查记录在 `outputs/ncr_v1/preflight`。成功记录只在代码/完整配方/初始化/数据身份/环境完全一致时复用。失败先查看 `console.log` 和 `smoke.json`，不循环重试或降低 batch。GPU0 上已有其他实验不构成拒绝条件。

## 3. 单独启动正式训练

此步骤才会启动完整 200 轮训练。运行前确认上述预检成功；正式模型重新从公共初始化构建，不带入预检更新。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-ncr_v1
bash tools/ncr_v1.sh start
bash tools/ncr_v1.sh status
```

查看训练 pane（退出显示但不终止训练：Ctrl-b 然后 d）：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-ncr_v1
bash tools/ncr_v1.sh attach
```

训练 tmux：`ncr-v1-training`；日志：worktree 的 `outputs/ncr_v1/train.log`。固定正式输出：

```text
/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/ncr_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug
```

若真实失败且本实验已有可续训、未 strip 的 last，检查错误后才显式恢复：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-ncr_v1
bash tools/ncr_v1.sh start --resume
```

仅恢复同代码/配方/数据/初始化身份。e 从 checkpoint 的真实 epoch 继续，不重新渐入。无 last 的初始化阶段失败不会被自动覆盖；先保留并审查失败证据。运行中的同实验不会重复启动，其他实验不受锁影响。

## 4. 训练成功后，单独执行 finish

不要把此块和首次 start 一起粘贴。`status` 必须显示训练成功且 shell exit=0；入口还会核验固定 best 的 SHA。finish 在 `ncr-v1-finish` 内完成尚缺的独立 FP32 val/test、同步 300 query/GT 导出、离线分析和打包。

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-ncr_v1
bash tools/ncr_v1.sh status
bash tools/ncr_v1.sh finish
tmux attach-session -t ncr-v1-finish
```

结束后 pane 保留，显示 Python、tee 和最终退出码；`outputs/ncr_v1/metrics.md` 是指标页，每个 evaluation 目录有原始精度 JSON、导出及离线分析。val/test 均使用同一个 best。不会以 test 重选 epoch/阈值，也不以 last 替代 best。报告的 P/R 是评估器原生最大 F1 PR 工作点，非新的部署阈值选择。

## 5. 只读状态 / 离线打包

这两条均不触发训练或推理：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-ncr_v1
bash tools/ncr_v1.sh status
bash tools/ncr_v1.sh pack
```

完整时生成 `outputs/ncr_v1/packages/NCR_v1_COMPLETE_<时间>.tar.gz` 和 SHA256/manifest；缺证据则明确 PARTIAL 并列缺项。相同内容包直接复用。包及 best 留服务器，先看指标，由用户决定是否下载。
