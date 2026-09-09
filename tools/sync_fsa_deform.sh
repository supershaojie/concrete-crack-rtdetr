#!/usr/bin/env bash
# Run from anywhere: bash /path/to/this/script FULL_COMMIT_SHA [MAIN_REPO] [FSA_WORKTREE]
# Never reset, clean, remove, switch, or update an existing checkout.
set -Eeuo pipefail
FSA_SHA="${1:?Supply the pinned full FSA commit SHA}"
FSA_MAIN_REPO="${2:-/root/autodl-tmp/projects/Crack_RTDETR}"
FSA_WORKTREE="${3:-/root/autodl-tmp/projects/Crack_RTDETR-fsa-deform}"
FSA_MAIN_REPO="$(realpath -m -- "$FSA_MAIN_REPO")"
FSA_WORKTREE="$(realpath -m -- "$FSA_WORKTREE")"
FSA_BRANCH=codex/fsa-deform
FSA_BASE=67c3078e54a657fd96d65fee657a75fbb1dae0d6
[[ "$FSA_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full lowercase 40-character SHA is required'; exit 2; }
FSA_REMOTE="$(git -C "$FSA_MAIN_REPO" remote get-url origin)"
case "$FSA_REMOTE" in
    https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
    *) echo "Unexpected origin: $FSA_REMOTE"; exit 1 ;;
esac
[[ "$(realpath -m -- "$FSA_MAIN_REPO")" != "$(realpath -m -- "$FSA_WORKTREE")" ]] || { echo 'A separate FSA worktree is required'; exit 1; }
git -C "$FSA_MAIN_REPO" status --short
git -C "$FSA_MAIN_REPO" rev-parse HEAD
git -C "$FSA_MAIN_REPO" fetch origin "$FSA_BRANCH"
FSA_REMOTE_HEAD="$(git -C "$FSA_MAIN_REPO" rev-parse FETCH_HEAD)"
[[ "$FSA_REMOTE_HEAD" == "$FSA_SHA" ]] || { echo 'Remote branch differs from pinned SHA; inspect, no checkout changed'; exit 1; }
git -C "$FSA_MAIN_REPO" cat-file -e "$FSA_SHA^{commit}" || git -C "$FSA_MAIN_REPO" fetch origin "$FSA_SHA"
git -C "$FSA_MAIN_REPO" merge-base --is-ancestor "$FSA_BASE" "$FSA_SHA"
if [[ -e "$FSA_WORKTREE" ]]; then
    [[ "$(git -C "$FSA_WORKTREE" rev-parse --show-toplevel)" == "$FSA_WORKTREE" ]] || { echo 'Existing directory is not the requested worktree'; exit 1; }
    FSA_COMMON="$(cd -- "$FSA_WORKTREE" && realpath -- "$(git rev-parse --git-common-dir)")"
    FSA_MAIN_COMMON="$(cd -- "$FSA_MAIN_REPO" && realpath -- "$(git rev-parse --git-common-dir)")"
    [[ "$FSA_COMMON" == "$FSA_MAIN_COMMON" ]] || {
        echo 'Existing worktree belongs to another repository'; exit 1;
    }
    git -C "$FSA_WORKTREE" status --short
    [[ -z "$(git -C "$FSA_WORKTREE" status --porcelain)" ]] || { echo 'Existing FSA changes preserved; inspect before synchronizing'; exit 1; }
    [[ "$(git -C "$FSA_WORKTREE" rev-parse HEAD)" == "$FSA_SHA" ]] || { echo 'Existing FSA commit differs; preserved without reset/switch'; exit 1; }
else
    git -C "$FSA_MAIN_REPO" worktree add --detach "$FSA_WORKTREE" "$FSA_SHA"
fi
[[ "$(git -C "$FSA_WORKTREE" rev-parse HEAD)" == "$FSA_SHA" ]]
printf 'FSA ready: %s\nCommit: %s\nNo training started.\n' "$FSA_WORKTREE" "$FSA_SHA"
