# AutoDL：直接训练

先同步至交付回复给出的完整 SHA，再执行单模块：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-scca
bash tools/autodl_scca.sh start-direct c24
bash tools/autodl_scca.sh status c24
```

`start-direct` 自行校验本工作树 import、rtdetr 环境/CUDA、原 C2 109字段、数据路径、统一初始化 SHA 和逐键映射，生成干净初始化和训练参数，然后创建 `scca-c24` tmux 会话。使用原 C2 AMP 训练器，不依赖任何 prepare/passed/audit 文件，也不运行额外完整服务器预检；记录 **完整服务器预检未执行**。

训练由前几批实际运行观察。首轮原训练器默认会 OOM 减 batch，本实验在每批前通过原生 callback 将原重试计数设到上限，使原 catch 直接抛出并记录退出；不修改公共训练器，不降 batch/imgsz/lr，不终止其他会话。若原生 AMP 辅助检查明确关闭 AMP，入口停止以避免改变正式配方。辅助 bus.jpg/yolo26n.pt 先复用主仓库可信副本，缺失才从官方 Ultralytics 地址获取，不升级环境。

运行位置：

| 实验 | 正式结果（主仓库，保留原 project） | 本 worktree 记录 |
|---|---|---|
| C24 | `runs/c_series/c24_rtdetr_r18_lite_scca_e200_b16_onlineaug` | `outputs/scca/c24` |
| C25 | `runs/c_series/c25_rtdetr_r18_lite_cscef_v51_scca_e200_b16_onlineaug` | `outputs/scca/c25` |

后台完整训练日志在对应 `outputs/scca/c24/console.log`。status 显示会话、PID记录、退出状态和日志尾部。训练异常保留 `.scca.lock`、初始化和失败记录，不会静默覆盖重启；排查后若需重试，先把本实验的失败记录和锁归档到新位置，再手动重试，勿删除其他实验文件。

## 同步（保护现有工作树和历史结果）

下面从远端读取完整 SHA 并核对；若需要固定本次版本，把 `SCCA_EXPECTED` 的赋值替换为交付回复中的完整 SHA。

```bash
bash <<'SCCA_SYNC'
set -Eeuo pipefail
SCCA_MAIN=/root/autodl-tmp/projects/Crack_RTDETR
SCCA_WT=/root/autodl-tmp/projects/Crack_RTDETR-scca
SCCA_BRANCH=codex/scca-aifi
test "$(git -C "$SCCA_MAIN" remote get-url origin)" = https://github.com/supershaojie/concrete-crack-rtdetr.git
SCCA_EXPECTED="$(git -C "$SCCA_MAIN" ls-remote origin "refs/heads/$SCCA_BRANCH" | cut -f1)"
test "${#SCCA_EXPECTED}" -eq 40
git -C "$SCCA_MAIN" fetch origin "$SCCA_BRANCH"
test "$(git -C "$SCCA_MAIN" rev-parse FETCH_HEAD)" = "$SCCA_EXPECTED"
if test -e "$SCCA_WT"; then
  test -e "$SCCA_WT/.git"
  test "$(git -C "$SCCA_WT" branch --show-current)" = "$SCCA_BRANCH"
  test -z "$(git -C "$SCCA_WT" status --porcelain)"
  git -C "$SCCA_WT" merge --ff-only "$SCCA_EXPECTED"
else
  if git -C "$SCCA_MAIN" show-ref --verify --quiet "refs/heads/$SCCA_BRANCH"; then
    git -C "$SCCA_MAIN" worktree add "$SCCA_WT" "$SCCA_BRANCH"
    git -C "$SCCA_WT" merge --ff-only "$SCCA_EXPECTED"
  else
    git -C "$SCCA_MAIN" worktree add -b "$SCCA_BRANCH" "$SCCA_WT" "$SCCA_EXPECTED"
  fi
fi
test "$(git -C "$SCCA_WT" rev-parse HEAD)" = "$SCCA_EXPECTED"
git -C "$SCCA_WT" log -1 --format='%H %s'
SCCA_SYNC
```

如果已有分支附在别的 worktree，Git 会拒绝新增；先识别内容，不强行 checkout、reset、clean 或删除。同步只更新独立实验 worktree，原主工作树和历史 runs 不动。

## 训练后操作

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR-scca
bash tools/autodl_scca.sh status c24
bash tools/autodl_scca.sh val c24
# 结构和选择固定后才运行 test：
bash tools/autodl_scca.sh test c24
bash tools/autodl_scca.sh pack c24
```

结果包位于 `/root/autodl-tmp/projects/Crack_RTDETR/downloads/scca/`，新时间戳命名。pack 可在未执行 test 时运行，并写 NOT_RUN，绝不为了打包自动测试。

C2/C17 同口径 val（将下列实际 best 路径替换为已存在的历史文件）：

```bash
bash tools/autodl_scca.sh val c24 --weights /实际/C2/weights/best.pt --output outputs/scca/reference_c2_val
bash tools/autodl_scca.sh val c25 --weights /实际/C17/weights/best.pt --output outputs/scca/reference_c17_val
```

可选本地/服务器诊断，不作为直接训练前置条件：

```bash
bash tools/autodl_scca.sh diagnose c24
bash tools/autodl_scca.sh check c24 --output outputs/scca_check_optional_01
```

check 使用 batch1/2；不是4090 batch16容量证明。正常训练无额外统计。

C25 仅在看过 C24 单模块完整 val、比较 C2/C17 后由用户决定排期。无额外强制 prepare：

```bash
bash tools/autodl_scca.sh start-direct c25
bash tools/autodl_scca.sh status c25
bash tools/autodl_scca.sh val c25
bash tools/autodl_scca.sh test c25
bash tools/autodl_scca.sh pack c25
```

没有远程执行 AutoDL；Python3.10/PyTorch2.1.2+cu121/RTX4090 的正式 batch16 仍待用户运行。本地 PyTorch2.7.1 验证不能替代该环境的真实运行。
