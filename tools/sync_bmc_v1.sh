#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && "$1" =~ ^[0-9a-f]{40}$ ]] || { echo 'usage: sync_bmc_v1.sh FULL_40_HEX_SHA' >&2; exit 2; }
SHA="$1"
MAIN="${BMC_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}"
WT="${BMC_WORKTREE:-/root/autodl-tmp/projects/Crack_RTDETR-bmc_v1}"
PYTHON="${BMC_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}"
BRANCH=exp-rtdetr-r18-lite-bmc-v1
BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
[[ "$(git -C "$MAIN" remote get-url origin)" == 'https://github.com/supershaojie/concrete-crack-rtdetr.git' ]] || { echo 'Wrong repository origin' >&2; exit 1; }
git -C "$MAIN" fetch --no-tags origin "$BRANCH"
[[ "$(git -C "$MAIN" rev-parse FETCH_HEAD)" == "$SHA" ]] || { echo 'Remote experiment branch does not equal supplied delivery SHA' >&2; exit 1; }
git -C "$MAIN" merge-base --is-ancestor "$BASE" "$SHA"
if [[ -e "$WT" ]]; then
    [[ -f "$WT/.git" ]] || { echo 'Existing destination is not a linked worktree; preserved' >&2; exit 1; }
    [[ "$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$MAIN" rev-parse --path-format=absolute --git-common-dir)" ]] || { echo 'Wrong worktree ownership' >&2; exit 1; }
    [[ -z "$(git -C "$WT" status --porcelain)" ]] || { echo 'Existing BMC worktree has changes; preserved' >&2; exit 1; }
    [[ "$(git -C "$WT" branch --show-current)" == "$BRANCH" ]] || { echo 'Existing worktree belongs to a different branch; preserved' >&2; exit 1; }
    CURRENT="$(git -C "$WT" rev-parse HEAD)"
    if [[ "$CURRENT" != "$SHA" ]]; then
        if command -v tmux >/dev/null && tmux has-session -t bmc-v1-training 2>/dev/null; then
            echo 'BMC session active; do not change its source' >&2; exit 1
        fi
        # The experiment worker's identity is checked independently of tmux.
        if pgrep -af -- "$WT/tools/bmc_v1.py _worker" >/dev/null; then
            echo 'BMC worker active; do not change its source' >&2; exit 1
        fi
        git -C "$WT" merge-base --is-ancestor "$CURRENT" "$SHA"
        git -C "$WT" merge --ff-only "$SHA"
    fi
else
    if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
        [[ "$(git -C "$MAIN" rev-parse "$BRANCH")" == "$SHA" ]] || { echo 'Existing same-name branch has different ownership/state; preserved' >&2; exit 1; }
        git -C "$MAIN" worktree add "$WT" "$BRANCH"
    else
        git -C "$MAIN" worktree add -b "$BRANCH" "$WT" "$SHA"
    fi
fi
export BMC_MAIN="$MAIN" PYTHONPATH="$WT/ultralytics-main:$WT/tools" PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
cd "$WT"
"$PYTHON" "$WT/tools/bmc_v1.py" _delivery --sha "$SHA"
