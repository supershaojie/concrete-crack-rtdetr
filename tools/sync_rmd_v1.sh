#!/usr/bin/env bash
# Full-SHA, bounded-network synchronization; never checkout the user's main worktree.
set -euo pipefail
SHA=${1:?Usage: bash sync_rmd_v1.sh FULL_40_CHARACTER_SHA}
MODE=${2:-}
[[ -z "$MODE" || "$MODE" == --eval-only ]] || { echo 'Only optional --eval-only is supported' >&2; exit 2; }
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a real full 40-character SHA' >&2; exit 2; }
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-rmd_v1
BRANCH=exp-rtdetr-r18-lite-rmd-v1
ORIGIN=$(git -C "$MAIN" remote get-url origin)
[[ "${ORIGIN%.git}" == https://github.com/supershaojie/concrete-crack-rtdetr ]] || { echo 'Wrong origin' >&2; exit 1; }
timeout 180 git -C "$MAIN" -c http.connectTimeout=20 -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
git -C "$MAIN" cat-file -e "$SHA^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$SHA" "origin/$BRANCH"
git -C "$MAIN" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$SHA"
if [[ -e "$WT" ]]; then
  [[ -f "$WT/.git" ]] || { echo 'Existing destination is not a linked worktree; preserved' >&2; exit 1; }
  [[ "$(git -C "$MAIN" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)" ]] || { echo 'Different repository; preserved' >&2; exit 1; }
  [[ -z "$(git -C "$WT" status --porcelain)" ]] || { echo 'Dirty RMD worktree preserved' >&2; exit 1; }
  CURRENT=$(git -C "$WT" rev-parse HEAD)
  if [[ "$CURRENT" != "$SHA" ]]; then
    if [[ -f "$WT/outputs/rmd_v1/training_identity.json" ]]; then
      [[ "$MODE" == --eval-only ]] || { echo 'Dispatched training is pinned; only completed-training --eval-only recovery is permitted' >&2; exit 1; }
      /root/miniconda3/envs/rtdetr/bin/python - "$WT" "$CURRENT" "$SHA" <<'PY'
import json, subprocess, sys
from pathlib import Path
root, old, new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path[:0] = [str(root/'tools'), str(root/'ultralytics-main')]
from rmd_v1 import workers
assert not workers(), 'Active RMD worker protected'
state = json.loads((root/'outputs/rmd_v1/training_completed.json').read_text())
assert state['status'] == 'TRAINING_COMPLETED', 'Training is not completed'
allowed = {'tools/rmd_v1_eval.py', 'ultralytics-main/ultralytics/nn/autobackend.py'}
changes = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', old, new], text=True).splitlines()
assert all(p in allowed or p.startswith('docs/rmd_v1/') for p in changes), 'Revision changes training code'
print('Evaluation-only revision; original training identity and exit records preserved:', changes)
PY
    fi
    if command -v tmux >/dev/null && tmux has-session -t '=rmd-v1-training' 2>/dev/null; then
      echo 'RMD tmux exists; worktree preserved' >&2; exit 1
    fi
    git -C "$WT" merge-base --is-ancestor "$CURRENT" "$SHA"
    git -C "$WT" merge --ff-only "$SHA"
  fi
else
  # A detached, pinned checkout avoids changing any existing local experiment branch.
  git -C "$MAIN" worktree add --detach "$WT" "$SHA"
fi
[[ "$(git -C "$WT" rev-parse HEAD)" == "$SHA" ]]
mkdir -p "$WT/outputs/rmd_v1"
printf '%s\n' "$SHA" > "$WT/outputs/rmd_v1/synchronized_sha.txt"
printf 'RMD worktree: %s\nPinned commit: %s\nNext: bash %s/tools/rmd_v1.sh prepare\n' "$WT" "$SHA" "$WT"
