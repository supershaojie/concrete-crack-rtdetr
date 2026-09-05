# C18 AutoDL 完整启动命令

在 AutoDL Bash 终端粘贴以下**整段**。已有正式 C18 run、launch lock 或日志时会停止并保留结果；已有同名 worktree 必须干净且与远端提交一致。复用已生成的初始化/审计时，后续工具仍校验来源、代码、权重和环境。

该段不修改 C2 配方，不等待或终止 C17，不重装环境。前半段只初始化和审计；带 `--tmux` 的命令开始正式训练。任何审计失败立即停止并打印原因。`--require-torch 2.1.2 --require-cuda` 在服务器执行实际兼容性验证，本地 2.7.1 不能代替。

可选：先设置 `export C18_EXPECTED_COMMIT=<交付消息中的完整提交>`，防止远端分支随后变化时误运行不同版本。不设置则使用该分支当前提交并打印记录。正常维护的等价入口为 `tools/launch_c18_autodl.sh`，无需临时脚本上传。

```bash
bash <<'C18_LAUNCH'
#!/usr/bin/env bash
# C18 only. No package installation, global editable changes, C17 coordination, or recipe overrides.
set -Eeuo pipefail
trap 'printf "C18 stopped at line %s (exit %s). See the error above; the recipe was not changed.\n" "$LINENO" "$?" >&2' ERR

MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORKTREE=/root/autodl-tmp/projects/Crack_RTDETR_gsdr_aifi_v3
BRANCH=exp-rtdetr-r18-lite-gsdr-aifi-v3
BASE=30f2da6e4bb91dcae5f761bcafd5993e263197da
NAME=c18_rtdetr_r18_lite_gsdr_aifi_v3_e200_b16_onlineaug
REPORT="$WORKTREE/outputs/gsdr_aifi_v3/launch_c18"
SOURCE="$MAIN/weights/rtdetr_r18_lite_imagenet_backbone_init.pt"
INITIALIZED="$WORKTREE/weights/rtdetr_r18_lite_gsdr_aifi_v3_imagenet_backbone_init.pt"
C2_ARGS="$MAIN/runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml"
RUN="$MAIN/runs/c_series/$NAME"

git -C "$MAIN" fetch origin "$BRANCH:refs/remotes/origin/$BRANCH"
REMOTE_COMMIT=$(git -C "$MAIN" rev-parse "refs/remotes/origin/$BRANCH")
if [[ -n "${C18_EXPECTED_COMMIT:-}" && "$REMOTE_COMMIT" != "$C18_EXPECTED_COMMIT" ]]; then
    printf 'Remote branch differs from the delivered commit: %s\n' "$REMOTE_COMMIT" >&2
    exit 1
fi
if [[ -e "$WORKTREE/.git" ]]; then
    test "$(git -C "$WORKTREE" rev-parse --show-toplevel)" = "$WORKTREE"
    test "$(git -C "$WORKTREE" branch --show-current)" = "$BRANCH"
    test -z "$(git -C "$WORKTREE" status --porcelain)"
    test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$REMOTE_COMMIT"
elif [[ -e "$WORKTREE" ]]; then
    printf 'Existing non-worktree directory preserved: %s\n' "$WORKTREE" >&2
    exit 1
elif git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    test "$(git -C "$MAIN" rev-parse "$BRANCH")" = "$REMOTE_COMMIT"
    git -C "$MAIN" worktree add "$WORKTREE" "$BRANCH"
else
    git -C "$MAIN" worktree add -b "$BRANCH" "$WORKTREE" "refs/remotes/origin/$BRANCH"
fi
git -C "$WORKTREE" merge-base --is-ancestor "$BASE" HEAD
if [[ -e "$RUN" || -e "$RUN.c18.launch.lock" || -e "$REPORT/tmux.json" || -e "$REPORT/console.log" || -e "$REPORT/exit_code.json" ]]; then
    printf 'Existing C18 run/launch preserved. Inspect: %s and %s\n' "$RUN" "$REPORT" >&2
    exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE/ultralytics-main"
command -v tmux >/dev/null
test -f "$C2_ARGS"
test -f "$SOURCE"
mkdir -p "$REPORT"
python -c 'import torch, ultralytics; print("PyTorch:", torch.__version__); print("Ultralytics:", ultralytics.__file__)'

if [[ ! -e "$INITIALIZED" ]]; then
    python tools/init_rtdetr_r18_lite_gsdr_aifi_v3_controlled.py \
        --source "$SOURCE" --output "$INITIALIZED" --report "$REPORT/initialization.json"
fi
# On a rerun, retain the existing audit. Preparation verifies its code, checkpoint, version and environment.
if [[ ! -e "$REPORT/audit.json" ]]; then
    python tools/audit_rtdetr_r18_lite_gsdr_aifi_v3.py \
        --source "$SOURCE" --initialized "$INITIALIZED" --report "$REPORT/audit.json" \
        --require-torch 2.1.2 --require-cuda
fi
LAUNCH_ARGS=(--c2-args "$C2_ARGS" --initialized "$INITIALIZED" --audit-report "$REPORT/audit.json"
             --name "$NAME" --report-dir "$REPORT")
python tools/train_rtdetr_r18_lite_gsdr_aifi_v3.py "${LAUNCH_ARGS[@]}"
# This is the formal training launch. Earlier steps only initialize, audit and prepare.
python tools/train_rtdetr_r18_lite_gsdr_aifi_v3.py "${LAUNCH_ARGS[@]}" --tmux
python - "$REPORT" <<'PY'
import json, pathlib, shlex, sys
report = pathlib.Path(sys.argv[1])
session = json.loads((report / 'tmux.json').read_text())['session']
print('Actual tmux session:', session)
print('Console:', report / 'console.log')
print('Exit status (written when the process exits):', report / 'exit_code.json')
print('tail -F ' + shlex.quote(str(report / 'console.log')))
print('tmux attach -t ' + shlex.quote(session))
print('Bootstrap log:', report / 'bootstrap.log')
PY
C18_LAUNCH
```

启动器及末尾 Python 从实际 `tmux.json` 读取 session，并打印 console.log、exit_code.json、tail 和 attach 命令。`exit_code.json` 在退出时才出现；bootstrap/import 失败查看 bootstrap.log，supervisor 的最终状态为 process_exit_code.json。

之后查看日志（新终端可直接粘贴，无需激活 Python 环境）：

```bash
REPORT=/root/autodl-tmp/projects/Crack_RTDETR_gsdr_aifi_v3/outputs/gsdr_aifi_v3/launch_c18
tail -F "$REPORT/console.log"
```

`tail` 的 Ctrl-C 只退出日志查看。tmux attach 命令使用启动器实际打印的 session；进入 tmux 后按 Ctrl-B，松开后按 D 可脱离。退出状态查看：

```bash
cat /root/autodl-tmp/projects/Crack_RTDETR_gsdr_aifi_v3/outputs/gsdr_aifi_v3/launch_c18/exit_code.json
```

尚未本地执行 Linux tmux、服务器 2.1.2 或真实数据训练；Bash 脚本已做 `bash -n` 语法检查。完整 109 项由 launch_plan.json 的 field_comparison 列出，实际首批前 preflight.json 通过后才允许 optimizer 开始训练。
