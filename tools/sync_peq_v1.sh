#!/usr/bin/env bash
# Safe full-SHA sync for exactly this experiment; never changes the main checkout.
set -euo pipefail
SHA="${1:-}"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "Usage: bash sync_peq_v1.sh FULL_SHA" >&2; exit 2; }
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-peq_v1
BRANCH=exp-rtdetr-r18-lite-peq-v1
PYTHON=/root/miniconda3/envs/rtdetr/bin/python
REMOTE=https://github.com/supershaojie/concrete-crack-rtdetr.git
[[ "$(git -C "$MAIN" remote get-url origin)" == "$REMOTE" ]] || { echo "Wrong repository origin" >&2; exit 1; }
git -C "$MAIN" fetch origin "$BRANCH"
FETCHED="$(git -C "$MAIN" rev-parse FETCH_HEAD)"
[[ "$FETCHED" == "$SHA" ]] || { echo "Fetched experiment branch differs from requested FULL_SHA" >&2; exit 1; }
git -C "$MAIN" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$SHA"
if [[ -e "$WT" ]]; then
    [[ -f "$WT/.git" ]] || { echo "Existing path is not a linked worktree; preserved" >&2; exit 1; }
    [[ "$(git -C "$WT" branch --show-current)" == "$BRANCH" ]] || { echo "Existing worktree belongs to another branch" >&2; exit 1; }
    [[ "$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$MAIN" rev-parse --path-format=absolute --git-common-dir)" ]] || { echo "Different Git common directories" >&2; exit 1; }
    [[ -z "$(git -C "$WT" status --porcelain)" ]] || { echo "Existing worktree has edits; preserved" >&2; exit 1; }
    if command -v tmux >/dev/null && tmux has-session -t '=peq-v1-training' 2>/dev/null; then
        echo "PEQ tmux is active; finish this experiment before syncing" >&2; exit 1
    fi
    env PYTHONPATH="$WT/ultralytics-main:$WT/tools" YOLO_AUTOINSTALL=false "$PYTHON" -c 'from peq_v1_runtime import status; from peq_v1_common import require; require(not status().get("active"), "A PEQ worker is still active")'
    if [[ "$(git -C "$WT" rev-parse HEAD)" != "$SHA" ]]; then
        git -C "$WT" merge-base --is-ancestor HEAD "$SHA"
        git -C "$WT" merge --ff-only "$SHA"
    fi
else
    if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git -C "$MAIN" merge-base --is-ancestor "$BRANCH" "$SHA"
        git -C "$MAIN" worktree add "$WT" "$BRANCH"
        git -C "$WT" merge --ff-only "$SHA"
    else
        git -C "$MAIN" worktree add -b "$BRANCH" "$WT" "$SHA"
    fi
fi
[[ "$(git -C "$WT" rev-parse HEAD)" == "$SHA" ]] || exit 1
export PYTHONPATH="$WT/ultralytics-main:$WT/tools"
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
cd "$WT"
"$PYTHON" "$WT/tools/peq_v1.py" record-delivery --sha "$SHA"
echo "Synced $SHA to $WT"
