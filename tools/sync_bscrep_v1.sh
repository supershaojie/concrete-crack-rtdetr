#!/usr/bin/env bash
# Local server-side synchronization only; this script never starts training.
set -Eeuo pipefail
SHA=${1:?Usage: sync_bscrep_v1.sh FULL_40_CHAR_SHA [INCREMENTAL_BUNDLE]}
BUNDLE=${2:-}
MAIN=${BSCREP_V1_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}
WT=${BSCREP_V1_WORKTREE:-/root/autodl-tmp/projects/Crack_RTDETR-bscrep-v1}
BRANCH=exp-rtdetr-r18-lite-bscrep-v1
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected full 40-character SHA' >&2; exit 2; }
export GIT_TERMINAL_PROMPT=0
git -C "$MAIN" rev-parse --git-dir >/dev/null
if ! git -C "$MAIN" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  for attempt in 1 2 3 4 5; do
    if timeout 120 git -C "$MAIN" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
        fetch --progress --no-tags origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"; then
      git -C "$MAIN" cat-file -e "$SHA^{commit}" 2>/dev/null && break
    fi
    echo "Fetch attempt $attempt/5 did not supply $SHA" >&2
  done
fi
if ! git -C "$MAIN" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  [[ -n "$BUNDLE" && -f "$BUNDLE" ]] || { echo 'Network unavailable: provide delivered small Git bundle as argument 2' >&2; exit 3; }
  git -C "$MAIN" bundle verify "$BUNDLE"
  git -C "$MAIN" fetch --no-tags "$BUNDLE" "refs/heads/$BRANCH:refs/remotes/bscrep-bundle/$BRANCH"
fi
git -C "$MAIN" cat-file -e "$SHA^{commit}"
if [[ -e "$WT" ]]; then
  [[ -f "$WT/.git" && "$(git -C "$WT" rev-parse HEAD)" == "$SHA" ]] || { echo 'Existing worktree does not match; preserved' >&2; exit 4; }
  [[ "$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$MAIN" rev-parse --path-format=absolute --git-common-dir)" ]]
  [[ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ]] || { echo 'Worktree has source edits; preserved' >&2; exit 4; }
else
  git -C "$MAIN" worktree add --detach "$WT" "$SHA"
fi
[[ "$(git -C "$WT" rev-parse HEAD)" == "$SHA" ]]
# Only after independent worktree and HEAD are verified.
mkdir -p "$WT/weights" "$WT/outputs"
ENV_FILE="$WT/outputs/bscrep_env.sh"
if [[ ! -e "$ENV_FILE" ]]; then
  (
    set -o noclobber
    {
      echo 'set -Eeuo pipefail'
      printf 'export BSCREP_V1_MAIN=%q\n' "$MAIN"
      printf 'export BSCREP_V1_WORKTREE=%q\n' "$WT"
      printf 'export BSCREP_V1_SHA=%q\n' "$SHA"
      echo 'source /root/miniconda3/etc/profile.d/conda.sh'
      echo 'conda activate rtdetr'
      printf 'cd %q\n' "$WT"
      printf 'export PYTHONPATH=%q\n' "$WT/ultralytics-main"
      echo 'export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1'
      echo 'export CUBLAS_WORKSPACE_CONFIG=:4096:8'
      echo '[[ "$(git rev-parse HEAD)" == "$BSCREP_V1_SHA" ]]'
    } > "$ENV_FILE"
  )
fi
printf 'Verified HEAD=%s\nEnvironment=%s\nTraining=NOT_STARTED\n' "$SHA" "$ENV_FILE"
