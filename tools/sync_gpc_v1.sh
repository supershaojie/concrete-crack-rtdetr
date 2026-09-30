#!/usr/bin/env bash
# Invoke this script from the exact fetched commit; never execute a floating URL.
set -euo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-gpc_v1
REMOTE=https://github.com/supershaojie/concrete-crack-rtdetr.git
BRANCH=exp-rtdetr-r18-lite-gpc-v1
BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
EXPECTED=${1:?Pass the full verified functional commit SHA from server_commands.md}
[[ "$EXPECTED" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full SHA is required'; exit 2; }
[[ "$(git -C "$MAIN" remote get-url origin)" == "$REMOTE" ]] || { echo 'Unexpected remote'; exit 2; }
timeout 90 git -C "$MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin "$BRANCH"
git -C "$MAIN" cat-file -e "$EXPECTED^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$BASE" "$EXPECTED"
git -C "$MAIN" merge-base --is-ancestor "$EXPECTED" "origin/$BRANCH"
# Protect this target worktree's running processes; other experiments/GPU processes never gate sync.
for process in /proc/[0-9]*; do
    cwd=$(readlink "$process/cwd" 2>/dev/null || true)
    if [[ "$cwd" == "$WT" || "$cwd" == "$WT/"* ]]; then
        command=$(tr '\0' ' ' < "$process/cmdline" 2>/dev/null || true)
        if [[ "$command" == *python* || "$command" == *gpc_v1.py* ]]; then
            printf 'Active process in target worktree preserved: %s %s\n' "$process" "$command"
            exit 2
        fi
    fi
done
if [[ -e "$WT" ]]; then
    [[ "$(git -C "$WT" rev-parse --show-toplevel)" == "$WT" ]] || { echo 'Unknown target directory'; exit 2; }
    [[ "$(git -C "$WT" branch --show-current)" == "$BRANCH" ]] || { echo 'Target branch differs'; exit 2; }
    [[ -z "$(git -C "$WT" status --porcelain)" ]] || { echo 'Dirty target worktree preserved'; exit 2; }
    if [[ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ]]; then
        git -C "$WT" merge-base --is-ancestor HEAD "$EXPECTED"
        git -C "$WT" merge --ff-only "$EXPECTED"
    fi
else
    if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
        [[ "$(git -C "$MAIN" rev-parse "$BRANCH")" == "$EXPECTED" ]] || { echo 'Existing branch differs; inspect before altering it'; exit 2; }
        git -C "$MAIN" worktree add "$WT" "$BRANCH"
    else
        git -C "$MAIN" worktree add -b "$BRANCH" "$WT" "$EXPECTED"
    fi
fi
[[ "$(git -C "$WT" rev-parse HEAD)" == "$EXPECTED" ]]
printf 'Synced GPC worktree: %s\nSHA: %s\n' "$WT" "$EXPECTED"
