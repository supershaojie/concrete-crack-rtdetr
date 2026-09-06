# AutoDL：C19 可执行操作

下述路径与 `tools/autodl_c19.sh` 一致。`prepare` 会运行数步独立调试smoke，但不会启动正式200轮。只有第二段显式 `start` 才启动正式训练。已有分支、worktree、run、锁或退出记录冲突时停止，先查看原因，不删除旧结果。

## 获取代码、环境、初始化、审计、109字段计划

交付消息提供精确commit。以下以远端分支的单次fetch结果固定本次获取版本，并打印该commit供核对；如果交付后分支更新，请使用交付消息的精确commit验证。

```bash
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
C19=/root/autodl-tmp/projects/Crack_RTDETR_cbr
BRANCH=exp-rtdetr-r18-lite-cbr
git -C "$MAIN" fetch origin "$BRANCH"
REV="$(git -C "$MAIN" rev-parse FETCH_HEAD)"
printf 'Fetched C19 commit: %s\n' "$REV"
git -C "$MAIN" worktree list
if [ -e "$C19" ]; then
    test "$(git -C "$C19" rev-parse HEAD)" = "$REV"
    test "$(git -C "$C19" branch --show-current)" = "$BRANCH"
    test -z "$(git -C "$C19" status --porcelain)"
elif git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    test "$(git -C "$MAIN" rev-parse "$BRANCH")" = "$REV"
    git -C "$MAIN" worktree add "$C19" "$BRANCH"
else
    git -C "$MAIN" worktree add -b "$BRANCH" "$C19" "$REV"
fi
cd "$C19"
bash tools/autodl_c19.sh prepare
python -m json.tool outputs/cbr/launch_c19/launch_plan.json | less
```

脚本自行激活现有 `rtdetr`，固定 `PYTHONPATH` 并打印实际导入位置；不执行pip/conda安装。初始化源、C2args/data均从主项目绝对路径读取。服务器要求Torch2.1.2、CUDA；每次prepare实际执行审计，真实smoke每次独立目录。受控初始化禁止覆盖；已有初始化仍会重新审计。若prepare失败，查看该次audit控制台与`outputs/cbr/launch_c19/audit.json`具体失败项。

默认完整训练目录仍位于主项目：
`/root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c19_rtdetr_r18_lite_cbr_e200_b16_onlineaug`。

## 显式启动正式训练

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_cbr
bash tools/autodl_c19.sh start
bash tools/autodl_c19.sh status
```

启动器重新验证计划、权重、环境、代码、目录、进程、session和锁，使用受控初始化重新构建正式模型。不会读取smoke权重。200轮计划中继承C2的patience=50，不修改早停设置。

## 查看日志和退出状态

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_cbr
bash tools/autodl_c19.sh status
tail -n 80 outputs/cbr/launch_c19/bootstrap.log
tail -F outputs/cbr/launch_c19/console.log
```

`Ctrl-C`只停止tail。实际session固定为：

```bash
tmux attach -t cbr_c19_rtdetr_r18_lite_cbr_e200_b16_onlineaug
```

`Ctrl-b d`脱离session。训练结束后查询：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_cbr
cat outputs/cbr/launch_c19/exit_code.json
cat outputs/cbr/launch_c19/process_exit_code.json
cat outputs/cbr/launch_c19/preflight.json
tail -n 6 /root/autodl-tmp/projects/Crack_RTDETR/runs/c_series/c19_rtdetr_r18_lite_cbr_e200_b16_onlineaug/results.csv
```

尚未退出时exit文件可能不存在，不能据此认定完成；`status`同时返回session、进程、run和锁。系统强杀后没有退出文件也不能认定成功，需检查日志与结果。保留锁以阻止静默重复启动。

## 训练后的val复验与小包

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_cbr
bash tools/autodl_c19.sh val
bash tools/autodl_c19.sh pack
ls -lh outputs/cbr/c19_val_diagnostic.tar.gz*
```

打包要求正式训练的初始化/审计/preflight/tmux/退出报告齐全，以及eval与run的best.pt SHA一致。包不含权重或图片。独立val对照先决定后续；不要把调试val/CSV最佳行混作最终复验结果。

固定test需先根据val写出真实选择记录，例如 `outputs/cbr/val_decision.json`，字段是 `selected_c19_sha256` 和 `reason`，分别填写已选best.pt的SHA与实际val依据。该决定未发生前不提供虚构内容。完成选择后工具调用如下（所有Python脚本已在仓库）：

```bash
cd /root/autodl-tmp/projects/Crack_RTDETR_cbr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export PYTHONPATH="$PWD/ultralytics-main"
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
python tools/cbr_results.py evaluate \
  --c2 "$MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/weights/best.pt" \
  --c19 "$MAIN/runs/c_series/c19_rtdetr_r18_lite_cbr_e200_b16_onlineaug/weights/best.pt" \
  --data "$MAIN/configs/crack_autodl.yaml" --samples outputs/cbr/fixed_val_samples.json \
  --split test --val-decision outputs/cbr/val_decision.json --device 0 --batch 16 \
  --output outputs/cbr/evaluation_test
python tools/cbr_results.py pack \
  --run "$MAIN/runs/c_series/c19_rtdetr_r18_lite_cbr_e200_b16_onlineaug" \
  --launch outputs/cbr/launch_c19 --evaluation outputs/cbr/evaluation_test \
  --output outputs/cbr/c19_test_diagnostic.tar.gz
```
