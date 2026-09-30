#!/usr/bin/env bash
# Fetch once with a timeout, pin a verified complete commit, preserve all other runs.
set -euo pipefail
EXPECTED=${1:?Pass the full verified CEA functionality commit SHA}
[[ "$EXPECTED" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a full 40-character SHA' >&2; exit 2; }
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-cea_v1
REMOTE=https://github.com/supershaojie/concrete-crack-rtdetr.git
BRANCH=exp-rtdetr-r18-lite-cea-v1
BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
[[ "$(git -C "$MAIN" remote get-url origin)" == "$REMOTE" ]] || { echo 'Wrong source remote'; exit 2; }
[[ "$(git -C "$MAIN" remote get-url --push origin)" == "$REMOTE" ]] || { echo 'Wrong push remote'; exit 2; }
timeout 120 git -C "$MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin "$BRANCH"
git -C "$MAIN" cat-file -e "$EXPECTED^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$BASE" "$EXPECTED"
git -C "$MAIN" merge-base --is-ancestor "$EXPECTED" FETCH_HEAD
if [[ -e "$WT" ]]; then
  [[ "$(git -C "$WT" rev-parse --show-toplevel)" == "$WT" ]] || { echo 'Unknown existing target preserved'; exit 2; }
  [[ "$(git -C "$WT" branch --show-current)" == "$BRANCH" ]] || { echo 'Unexpected existing branch'; exit 2; }
  [[ -z "$(git -C "$WT" status --porcelain)" ]] || { echo 'Dirty existing worktree preserved'; exit 2; }
  for proc in /proc/[0-9]*; do
    cwd=$(readlink "$proc/cwd" 2>/dev/null || true)
    if [[ "$cwd" == "$WT" ]]; then
      args=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
      if [[ "$args" == *python* && ( "$args" == *cea_v1* || "$args" == *ultralytics* ) ]]; then
        echo "Active process in CEA worktree, preserved: ${proc##*/}"; exit 2
      fi
    fi
  done
  if [[ -f "$WT/outputs/cea_v1/run.lock" ]]; then
    exec 9>"$WT/outputs/cea_v1/run.lock"
    flock -n 9 || { echo 'This CEA run has an active writer'; exit 2; }
  fi
  git -C "$WT" merge-base --is-ancestor HEAD "$EXPECTED" || { echo 'Existing HEAD is not an ancestor; no reset'; exit 2; }
  git -C "$WT" merge --ff-only "$EXPECTED"
else
  if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    [[ "$(git -C "$MAIN" rev-parse "$BRANCH")" == "$EXPECTED" ]] || { echo 'Existing unattached branch differs; preserved'; exit 2; }
    git -C "$MAIN" worktree add "$WT" "$BRANCH"
  else
    git -C "$MAIN" worktree add -b "$BRANCH" "$WT" "$EXPECTED"
  fi
fi
[[ "$(git -C "$WT" rev-parse HEAD)" == "$EXPECTED" ]]
printf 'SYNCED %s\nWorktree: %s\nNo training started.\n' "$EXPECTED" "$WT"
