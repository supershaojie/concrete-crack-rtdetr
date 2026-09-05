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
