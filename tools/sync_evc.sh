#!/usr/bin/env bash
# Run from anywhere: bash /path/to/this/script FULL_COMMIT_SHA [MAIN_REPO] [EVC_WORKTREE]
# Never reset, clean, remove, switch, or update an existing checkout.
set -Eeuo pipefail
EVC_SHA="${1:?Supply the pinned full EVC commit SHA}"
EVC_MAIN_REPO="${2:-/root/autodl-tmp/projects/Crack_RTDETR}"
EVC_WORKTREE="${3:-/root/autodl-tmp/projects/Crack_RTDETR-evc}"
EVC_MAIN_REPO="$(realpath -m -- "$EVC_MAIN_REPO")"
EVC_WORKTREE="$(realpath -m -- "$EVC_WORKTREE")"
EVC_BRANCH=codex/evc-deform
EVC_BASE=beedcfa307e250fb2de47587097c51f9c141123b
[[ "$EVC_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full lowercase 40-character SHA is required'; exit 2; }
EVC_REMOTE="$(git -C "$EVC_MAIN_REPO" remote get-url origin)"
case "$EVC_REMOTE" in
    https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
    *) echo "Unexpected origin: $EVC_REMOTE"; exit 1 ;;
esac
[[ "$(realpath -m -- "$EVC_MAIN_REPO")" != "$(realpath -m -- "$EVC_WORKTREE")" ]] || { echo 'A separate EVC worktree is required'; exit 1; }
git -C "$EVC_MAIN_REPO" status --short
git -C "$EVC_MAIN_REPO" rev-parse HEAD
git -C "$EVC_MAIN_REPO" fetch origin "$EVC_BRANCH"
EVC_REMOTE_HEAD="$(git -C "$EVC_MAIN_REPO" rev-parse FETCH_HEAD)"
[[ "$EVC_REMOTE_HEAD" == "$EVC_SHA" ]] || { echo 'Remote branch differs from pinned SHA; inspect, no checkout changed'; exit 1; }
git -C "$EVC_MAIN_REPO" cat-file -e "$EVC_SHA^{commit}" || git -C "$EVC_MAIN_REPO" fetch origin "$EVC_SHA"
git -C "$EVC_MAIN_REPO" merge-base --is-ancestor "$EVC_BASE" "$EVC_SHA"
if [[ -e "$EVC_WORKTREE" ]]; then
    [[ "$(git -C "$EVC_WORKTREE" rev-parse --show-toplevel)" == "$EVC_WORKTREE" ]] || { echo 'Existing directory is not the requested worktree'; exit 1; }
    EVC_COMMON="$(cd -- "$EVC_WORKTREE" && realpath -- "$(git rev-parse --git-common-dir)")"
    EVC_MAIN_COMMON="$(cd -- "$EVC_MAIN_REPO" && realpath -- "$(git rev-parse --git-common-dir)")"
    [[ "$EVC_COMMON" == "$EVC_MAIN_COMMON" ]] || {
        echo 'Existing worktree belongs to another repository'; exit 1;
    }
    git -C "$EVC_WORKTREE" status --short
    [[ -z "$(git -C "$EVC_WORKTREE" status --porcelain)" ]] || { echo 'Existing EVC changes preserved; inspect before synchronizing'; exit 1; }
    [[ "$(git -C "$EVC_WORKTREE" rev-parse HEAD)" == "$EVC_SHA" ]] || { echo 'Existing EVC commit differs; preserved without reset/switch'; exit 1; }
    [[ "$(git -C "$EVC_WORKTREE" branch --show-current)" == "$EVC_BRANCH" ]] || { echo 'Existing checkout belongs to another branch; preserved'; exit 1; }
else
    if git -C "$EVC_MAIN_REPO" show-ref --verify --quiet "refs/heads/$EVC_BRANCH"; then
        [[ "$(git -C "$EVC_MAIN_REPO" rev-parse "$EVC_BRANCH")" == "$EVC_SHA" ]] || { echo 'Existing local EVC branch differs; preserved'; exit 1; }
        git -C "$EVC_MAIN_REPO" worktree add "$EVC_WORKTREE" "$EVC_BRANCH"
    else
        git -C "$EVC_MAIN_REPO" worktree add -b "$EVC_BRANCH" "$EVC_WORKTREE" "$EVC_SHA"
    fi
fi
[[ "$(git -C "$EVC_WORKTREE" rev-parse HEAD)" == "$EVC_SHA" ]]
printf 'EVC ready: %s\nCommit: %s\nNo training started.\n' "$EVC_WORKTREE" "$EVC_SHA"
