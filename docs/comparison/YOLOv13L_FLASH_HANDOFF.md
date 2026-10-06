# YOLOv13-L Flash：服务器交付

2026-10-06（Asia/Shanghai）。**未启动服务器正式训练**。新首轮 `v13l_aug_x13_flash_01` 为 NOT_RUN。实现已推送，下面使用真实固定 SHA，随后补交付文档的 HEAD 不替换训练身份。

| 身份 | 实际值 |
|---|---|
| 新分支 | `bench/yolov13l-flash-configurable` |
| 实施代码完整 SHA | `6673ac43fa270c711a3bcf41a8f3c5bc88c80efb` |
| 真实父提交 / 原 native 实施 | `64f6639847a0b6a1c18f8f246ba27e274a4de3ad` |
| 原 native 文档提交 | `6d89d890b124da4d9c9c20700ab42e929fa92fc3`，没有作为此次代码基础 |
| 实施推送核验 | 实际 `git ls-remote origin refs/heads/bench/yolov13l-flash-configurable` 返回上述6673ac43完整SHA，之后才补本文档 |
| 新完整配置哈希 | `09311ff91a891babebbc45290d00ad235600ff24574e34bab8fff02712f36acd` |
| 新受控 patch SHA256 | `e7ab23b370ee8793c6db4ac4bffe80110275819b02ff79b7a6d4da66dc88deb7` |
| 新 patched source SHA256 | `159eb35b428625c445047d073b6202defaea66aadcf00aaa6747de266846c307` |
| 首轮请求 / 解析 | requested=flash；resolved 必须 flash，失败退出；eval=native |

完整数值配方见 [首轮 preset](../../benchmarks/comparison/yolov13l/configs/v13l_aug_x13_flash_01.yaml)，与原 native 首轮完整 resolver 输出仅训练后端不同。默认200/patience50/batch16/640/workers8/seed42/deterministic/AMP true/nbs64/SGD 及整份增强配方保留。[实施说明](YOLOv13L_FLASH_CONFIGURABLE.md) 说明每阶段 dtype、确定性和身份边界；[验证证据](evidence/yolov13l_flash_delivery_validation.json) 保存实际检查、未运行原因及哈希。

本机实测56项通过，1项POSIX信号测试因Windows跳过。官方模型迁移、保护gate、native完整模型/CPU反向、严格Flash失败、auto冻结、作用域异常恢复、后端/配方/环境收据漂移拒绝、生命周期/打包/归档和公共评估回归通过。另真实执行 RTX2060 native CUDA AMP 参考（原权重/BN/RNG不变）、两张真实val图的公共 native FP32 API（CPU640/batch2，Flash/half casts为0）。这些检查不证明4090/Flash/640-batch16可用；该项、Flash kernel/parity、Linux新环境和真实tmux仍NOT_RUN，必须执行下方启动前检查。

训练 forward/backward 使用原生AMP和作者Flash FP16 Q/K/V，deterministic backward=true；训练内val scoped native，保留作者实际半精度语义并记录dtype；公共val/test仍native真实FP32，autocast/TF32关闭，无Flash/内部half。Flash不是OOM成功保证：旧run的最后失败位置由用户记录为C3 torch.cat，本机未核验服务器日志。旧run无可恢复best/last，保留，不resume/覆盖/删除；不结束其他GPU0实验、不等待整卡空闲。

## 直接粘贴：隔离 worktree → 安装 → 检查 → 通过后 start

以下块与已提交的 [YOLOv13L_FLASH_START.server.sh](YOLOv13L_FLASH_START.server.sh) 内容相同，前后仅加本地 heredoc 写文件/执行；不需要上传本地文件。普通服务器 SSH Bash 粘贴整块。Git fetch固定分支，训练检出6673ac43；已有路径身份不匹配会停止，不覆盖。母Python必须3.10且有PyYAML。耗时安装/build/640检查放在 `setup-yolov13l-flash-v13l_aug_x13_flash_01`，断开SSH后继续；全部检查通过后才创建训练会话 `comparison-yolov13l-v13l_aug_x13_flash_01`。

```bash
cat > /tmp/YOLOv13L_FLASH_START.server.sh <<'YOLOV13L_FLASH_SERVER'
#!/usr/bin/env bash
# AutoDL launcher, fixed to the VERIFIED implementation commit. Run from SSH.
set -Eeuo pipefail
CODE_SHA='6673ac43fa270c711a3bcf41a8f3c5bc88c80efb'
REPO='/root/autodl-tmp/projects/Crack_RTDETR'
WT='/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-flash-configurable'
BASE_PY='/root/miniconda3/envs/rtdetr/bin/python'
RUN_ID='v13l_aug_x13_flash_01'

# The accelerator can refer to unset variables; restore all old shell options.
enable_network() {
    local old_flags=$- old_options rc=0
    old_options=$(set +o)
    set +eu
    if [[ -r /etc/network_turbo ]]; then source /etc/network_turbo; rc=$?; fi
    eval "$old_options"
    [[ "$old_flags" == *e* ]] && set -e
    [[ "$old_flags" == *u* ]] && set -u
    (( rc == 0 )) || return "$rc"
    local bypass='localhost,127.0.0.1,::1,pypi.org,.pypi.org,files.pythonhosted.org,.pythonhosted.org,pypi.python.org,download.pytorch.org,.pytorch.org'
    export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$bypass"
    export no_proxy="${no_proxy:+$no_proxy,}$bypass"
}
git_network() {
    local attempt
    for attempt in 1 2 3; do
        if git -c http.version=HTTP/1.1 "$@"; then return 0; fi
    done
    return 1
}
enable_network
git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null || { printf 'Mother repository missing: %s\n' "$REPO" >&2; exit 66; }
command -v tmux >/dev/null || { printf 'tmux missing\n' >&2; exit 69; }
"$BASE_PY" -c 'import sys,yaml;assert sys.version_info[:2]==(3,10); print("Bootstrap caller:",sys.executable,"PyYAML",yaml.__version__)'
git_network -C "$REPO" fetch origin bench/yolov13l-flash-configurable
git -C "$REPO" cat-file -e "${CODE_SHA}^{commit}"
if [[ ! -e "$WT" ]]; then
    git -C "$REPO" worktree add --detach "$WT" "$CODE_SHA"
fi
[[ "$(git -C "$WT" rev-parse HEAD)" == "$CODE_SHA" && -z "$(git -C "$WT" status --porcelain --untracked-files=normal)" ]] || { printf 'Protected existing path; requires clean fixed implementation: %s\n' "$WT" >&2; exit 65; }
[[ ! -e "$WT/outputs/yolov13l-configurable/$RUN_ID" ]] || { printf 'Protected existing Flash run; use its status/resume/finalize commands\n' >&2; exit 73; }
export YOLOV13L_CODE_SHA="$CODE_SHA" YOLOV13L_BASE_PYTHON="$BASE_PY" YOLOV13L_RUN_ID="$RUN_ID"
SESSION="setup-yolov13l-flash-$RUN_ID"
gate="yolov13l-flash-setup-$RUN_ID-$RANDOM"
printf -v worker '%q ' env "YOLOV13L_CODE_SHA=$CODE_SHA" "YOLOV13L_BASE_PYTHON=$BASE_PY" "YOLOV13L_RUN_ID=$RUN_ID" bash "$WT/scripts/autodl_yolov13l_flash_prepare.sh"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    if tmux list-panes -s -t "=$SESSION" -F '#{pane_dead}' | command grep -qx 0; then
        printf 'Protected live setup session: %s\n' "$SESSION" >&2; exit 73
    fi
    window=$(tmux new-window -d -P -F '#{window_id}' -t "=$SESSION" "tmux wait-for '$gate'; exec $worker")
else
    window=$(tmux new-session -d -P -F '#{window_id}' -s "$SESSION" "tmux wait-for '$gate'; exec $worker")
fi
tmux set-option -w -t "$window" remain-on-exit on
tmux select-window -t "$window"
tmux wait-for -S "$gate"
printf 'Setup: %s\nTraining after checks: comparison-yolov13l-%s\nFixed code: %s\n' "$SESSION" "$RUN_ID" "$CODE_SHA"
if [[ -t 0 && -t 1 ]]; then
    if [[ -n "${TMUX:-}" ]]; then tmux switch-client -t "=$SESSION"; else tmux attach -t "=$SESSION"; fi
fi
YOLOV13L_FLASH_SERVER
bash /tmp/YOLOv13L_FLASH_START.server.sh
```

setup依次执行新copies venv、正常索引的pip工具链、cu121索引的Torch/Vision wheel、正常索引的其余依赖、官方2.7.3实际ABI筛选/算子验证、环境冻结；生成服务器 `runtime_configs/v13l_aug_x13_flash_01.yaml`，比较同resolver的完整preset/default/candidate；真实640/batch16最少3个训练批次和optimizer/EMA更新、native val切换恢复、实际公共SquarePredictor的native FP32路径；成功后调用start。任何阶段失败保留command/tee退出码和日志，不减batch、改尺寸、关AMP/deterministic或切native重试。

setup日志在新WT的 `outputs/yolov13l-flash-preparation/<ID>_<stamp>_<random>/`；`flash_readiness.json` 记录阶段/峰值/并行进程/实际dtype/调用证据/错误。失败不会创建正式run；只有成功的完整配方/代码/环境/数据/报告哈希收据才能prepare。冻结后不继续pip install。源码与旧COCO资产只读复用时也重新校验新patch/源哈希和原权重SHA；旧patched vendor不直接复制使用。

## 新 SSH 登录：状态和返回 tmux

```bash
WT='/root/autodl-tmp/projects/Crack_RTDETR-bench-yolov13l-flash-configurable'
RUN_ID='v13l_aug_x13_flash_01'
cd "$WT"
tmux list-sessions
# 安装/检查尚在运行时：
tmux attach -t "setup-yolov13l-flash-$RUN_ID"
```

已在tmux时用 `tmux switch-client -t "setup-yolov13l-flash-$RUN_ID"`；Ctrl-b后d只脱离界面。bootstrap成功并开始训练后，用新的训练会话：

```bash
cd "$WT"
PY="$(cat "$WT/.runtime/yolov13l-configurable/python_path.txt")"
bash scripts/autodl_yolov13l_configurable.sh status --python "$PY" --run-id "$RUN_ID"
tmux attach -t "comparison-yolov13l-$RUN_ID"
```

当前已在tmux时使用 `tmux switch-client -t "comparison-yolov13l-$RUN_ID"`。训练日志在 `outputs/yolov13l-configurable/$RUN_ID/launch/`；实际后端和dtype见 `attention_backend_resolution.json`、`actual_training_setup.json`、`training_flash_evidence.json`、`native_validation.json`、`epoch_trace.jsonl`。不要attach旧 `v13l_aug_x13_01` 当成这次Flash run。

## 显式 resume、合法完成后 finalize/test

仅本次新Flash run已保存完整last且尚未合法完成时恢复；首轮forward OOM且无last无法resume。resume不接受新config/--set或别的checkpoint。

```bash
cd "$WT"
bash scripts/autodl_yolov13l_configurable.sh status --python "$PY" --run-id "$RUN_ID"
bash scripts/autodl_yolov13l_configurable.sh resume --python "$PY" --run-id "$RUN_ID"
```

start/resume只做所选best独立FP32 val，test仍not_requested。200上限或合法patience早停后执行下面的finalize；它复用同一best/val和冻结身份，不训练、安装或拉新代码：

```bash
cd "$WT"
bash scripts/autodl_yolov13l_configurable.sh status --python "$PY" --run-id "$RUN_ID"
bash scripts/autodl_yolov13l_configurable.sh finalize --python "$PY" --run-id "$RUN_ID"
```

finalize同样进入新Flash run的训练会话，独立进程使用native/FP32 test；任何身份/环境/后端漂移、best/last破坏或非法提前结束都会拒绝。原生恢复范围保持可验证状态，不声称prefetch/跨epoch未step梯度逐比特重放。

## 两种只读 pack，Git 归档用独立工作树

```bash
cd "$WT"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
bash scripts/autodl_yolov13l_configurable.sh pack --python "$PY" --run-id "$RUN_ID" \
  --out "$WT/outputs/yolov13l-packages/${RUN_ID}_${STAMP}_review.tar.gz"
bash scripts/autodl_yolov13l_configurable.sh pack --python "$PY" --run-id "$RUN_ID" \
  --include-weights --out "$WT/outputs/yolov13l-packages/${RUN_ID}_${STAMP}_weights.tar.gz"
```

第一包排除.pt，只是审查证据；第二包在合法完成后包含真实best/last及SHA，不包含整数据集/vendor/venv/重复缓存。训练中只能打incomplete审查快照。轻量配置/结果生成和Git提交/推送分开；以下步骤保持训练worktree的6673ac43与clean：

```bash
set -euo pipefail
REPO='/root/autodl-tmp/projects/Crack_RTDETR'
ARCHIVE_WT="/root/autodl-tmp/projects/Crack_RTDETR-yolov13l-flash-archive-${RUN_ID}-${STAMP}"
ARCHIVE_BRANCH="codex/yolov13l-flash-archive-${RUN_ID}-${STAMP}"
for attempt in 1 2 3; do
  if git -c http.version=HTTP/1.1 -C "$REPO" fetch origin bench/yolov13l-flash-configurable; then break; fi
  if [[ "$attempt" == 3 ]]; then exit 1; fi
done
test ! -e "$ARCHIVE_WT"
git -C "$REPO" worktree add -b "$ARCHIVE_BRANCH" "$ARCHIVE_WT" origin/bench/yolov13l-flash-configurable
cd "$WT"
bash scripts/autodl_yolov13l_configurable.sh archive-config --python "$PY" --run-id "$RUN_ID" \
  --out "$ARCHIVE_WT/docs/comparison/archives/yolov13l/${RUN_ID}_actual.json"
git -C "$ARCHIVE_WT" add -- "docs/comparison/archives/yolov13l/${RUN_ID}_actual.json"
git -C "$ARCHIVE_WT" commit -m "docs(bench): archive ${RUN_ID} actual Flash configuration and results"
for attempt in 1 2 3; do
  if git -c http.version=HTTP/1.1 -C "$ARCHIVE_WT" push --set-upstream origin "$ARCHIVE_BRANCH"; then break; fi
  if [[ "$attempt" == 3 ]]; then exit 1; fi
done
git -C "$ARCHIVE_WT" rev-parse HEAD
git -c http.version=HTTP/1.1 -C "$ARCHIVE_WT" ls-remote origin "refs/heads/$ARCHIVE_BRANCH"
```

最后两个实际SHA一致才算归档推送核验。`outputs/`和`runtime_configs/`被忽略，不会自动进入GitHub；归档不含.pt或secrets。训练worktree不pull/commit改变HEAD，归档分支只保存实际轻量证据。

## 后续新候选：YAML/--set，不改代码

下面示例改变原首轮SGD配方中的LR/momentum/mixup，用新ID。先生成/检查完整候选，再在tmux做该候选的640真实readiness，成功后启动；不要改原run的冻结文件。AdamW可显式用 `--set optimizer=AdamW --set lr0=0.001 --set momentum=0.9`，它改变首轮共同优化器配方，不是推荐更优或已验证涨点。

```bash
cd "$WT"
NEW_ID='v13l_flash_sgd_lr005_m09_mix018'
NEW_CONFIG="$WT/runtime_configs/$NEW_ID.yaml"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
bash scripts/autodl_yolov13l_configurable.sh config --python "$PY" --out "$NEW_CONFIG" \
  --set lr0=0.005 --set momentum=0.9 --set mixup=0.18
bash scripts/autodl_yolov13l_configurable.sh check-config --python "$PY" --config "$NEW_CONFIG"
JOB_DIR="$WT/outputs/yolov13l-flash-preparation/${NEW_ID}_${STAMP}"
mkdir -p "$JOB_DIR"
printf -v CHECK_COMMAND '%q ' "$PY" "$WT/benchmarks/comparison/yolov13l/stage.py" \
  --directory "$JOB_DIR" --name readiness -- "$PY" "$WT/benchmarks/comparison/yolov13l/flash_checks.py" \
  --config "$NEW_CONFIG" --out "$JOB_DIR/readiness.json"
printf -v START_COMMAND '%q ' "$PY" "$WT/benchmarks/comparison/yolov13l/stage.py" \
  --directory "$JOB_DIR" --name start -- bash "$WT/scripts/autodl_yolov13l_configurable.sh" start \
  --python "$PY" --run-id "$NEW_ID" --config "$NEW_CONFIG" --detach
printf -v NETWORK_SCRIPT '%q' "$WT/scripts/yolov13l_flash_network.sh"
printf -v JOB_COMMAND '%q ' bash -c "set -Eeuo pipefail; source $NETWORK_SCRIPT; yolov13l_flash_network; $CHECK_COMMAND && $START_COMMAND"
GATE="yolov13l-new-candidate-$RANDOM"
WINDOW=$(tmux new-session -d -P -F '#{window_id}' -s "setup-yolov13l-flash-$NEW_ID" "tmux wait-for '$GATE'; exec $JOB_COMMAND")
tmux set-option -w -t "$WINDOW" remain-on-exit on
tmux wait-for -S "$GATE"
tmux attach -t "setup-yolov13l-flash-$NEW_ID"
```

新候选640检查报告和stage.py的command/tee真实退出收据保留在JOB_DIR。新候选仍从官方COCO开始；auto需显式 `--set train_attention_backend=auto`，只在prepare解析一次并冻结，默认严格flash从不静默回退。完整网络/最终曲线与native逐比特等价未验证，并行计时不能用于论文速度。
