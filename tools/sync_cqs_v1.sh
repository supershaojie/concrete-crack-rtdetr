#!/usr/bin/env bash
# Protect the main checkout and every unrelated worktree. No reset/clean/force.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Usage: bash tools/sync_cqs_v1.sh FULL_40_CHARACTER_SHA' >&2
  exit 2
fi
SHA=$1
MAIN=${CQS_V1_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}
WT=${CQS_V1_WORKTREE:-/root/autodl-tmp/projects/Crack_RTDETR-cqs_v1}
BRANCH=exp-rtdetr-r18-lite-cqs-v1
BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
PYTHON=/root/miniconda3/envs/rtdetr/bin/python
[[ "$(git -C "$MAIN" remote get-url origin)" == https://github.com/supershaojie/concrete-crack-rtdetr.git ]] || { echo 'Wrong repository origin' >&2; exit 1; }
[[ "$(realpath -m "$MAIN")" != "$(realpath -m "$WT")" ]] || { echo 'Independent worktree required' >&2; exit 1; }
# Bootstrap may already have fetched the exact branch; do not repeat network work.
if [[ "${CQS_SYNC_FETCHED_SHA:-}" != "$SHA" ]]; then
  git -C "$MAIN" fetch --no-tags origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
fi
[[ "$(git -C "$MAIN" rev-parse "refs/remotes/origin/$BRANCH")" == "$SHA" ]] || { echo 'Remote experiment branch differs from requested delivery SHA' >&2; exit 1; }
[[ "$(git -C "$MAIN" rev-parse "$SHA^{commit}")" == "$SHA" ]] || exit 1
git -C "$MAIN" merge-base --is-ancestor "$BASE" "$SHA" || { echo 'Delivery is not descended from fixed mother' >&2; exit 1; }
if [[ -e "$WT" ]]; then
  [[ -f "$WT/.git" ]] || { echo 'Existing path is not a linked worktree; preserved' >&2; exit 1; }
  [[ "$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$MAIN" rev-parse --path-format=absolute --git-common-dir)" ]] || { echo 'Wrong worktree repository' >&2; exit 1; }
  [[ "$(git -C "$WT" branch --show-current)" == "$BRANCH" ]] || { echo 'Wrong worktree branch; preserved' >&2; exit 1; }
  [[ -z "$(git -C "$WT" status --porcelain --untracked-files=normal)" ]] || { echo 'CQS worktree has changes/conflicts; preserved' >&2; exit 1; }
  if command -v tmux >/dev/null && tmux has-session -t =cqs-v1-training 2>/dev/null; then
    echo 'CQS tmux is active; sync refused' >&2; exit 1
  fi
  PYTHONPATH="$WT/ultralytics-main:$WT/tools" YOLO_AUTOINSTALL=false "$PYTHON" -c 'from cqs_v1 import active_processes; assert not active_processes(), "Active CQS worker preserved"'
  git -C "$WT" merge-base --is-ancestor HEAD "$SHA" || { echo 'Existing CQS history is not a safe fast-forward; preserved' >&2; exit 1; }
  git -C "$WT" merge --ff-only "$SHA"
else
  if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    [[ "$(git -C "$MAIN" rev-parse "$BRANCH")" == "$SHA" ]] || { echo 'Existing same-name branch has another identity; preserved' >&2; exit 1; }
    git -C "$MAIN" worktree add "$WT" "$BRANCH"
  else
    git -C "$MAIN" worktree add -b "$BRANCH" "$WT" "$SHA"
  fi
fi
[[ "$(git -C "$WT" rev-parse HEAD)" == "$SHA" ]] || exit 1
export PYTHONPATH="$WT/ultralytics-main"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export CQS_V1_MAIN="$MAIN"
"$PYTHON" -u "$WT/tools/cqs_v1.py" _record-delivery "$SHA"
