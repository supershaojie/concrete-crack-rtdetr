#!/usr/bin/env bash
# Run from anywhere: bash /path/to/this/script FULL_COMMIT_SHA [MAIN_REPO] [C26_WORKTREE]
# Never reset, clean, remove, switch, or update an existing checkout.
set -Eeuo pipefail
C26_SHA="${1:?Supply the pinned full C26 commit SHA}"
C26_MAIN_REPO="${2:-/root/autodl-tmp/projects/Crack_RTDETR}"
C26_WORKTREE="${3:-/root/autodl-tmp/projects/Crack_RTDETR-c26}"
C26_BRANCH=codex/c26-cbr-scca
C26_BASE=f6e9dfda765046ae7691302cf5ec89d3f76cec5d
[[ "$C26_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full lowercase 40-character SHA is required'; exit 2; }
C26_REMOTE="$(git -C "$C26_MAIN_REPO" remote get-url origin)"
case "$C26_REMOTE" in
    https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
    *) echo "Unexpected origin: $C26_REMOTE"; exit 1 ;;
esac
git -C "$C26_MAIN_REPO" fetch origin "$C26_BRANCH"
git -C "$C26_MAIN_REPO" cat-file -e "$C26_SHA^{commit}" || git -C "$C26_MAIN_REPO" fetch origin "$C26_SHA"
git -C "$C26_MAIN_REPO" merge-base --is-ancestor "$C26_BASE" "$C26_SHA"
if [[ -e "$C26_WORKTREE" ]]; then
    [[ "$(git -C "$C26_WORKTREE" rev-parse --show-toplevel)" == "$C26_WORKTREE" ]] || { echo 'Existing directory is not the requested worktree'; exit 1; }
    C26_COMMON="$(cd -- "$C26_WORKTREE" && realpath -- "$(git rev-parse --git-common-dir)")"
    C26_MAIN_COMMON="$(cd -- "$C26_MAIN_REPO" && realpath -- "$(git rev-parse --git-common-dir)")"
    [[ "$C26_COMMON" == "$C26_MAIN_COMMON" ]] || {
        echo 'Existing worktree belongs to another repository'; exit 1;
    }
    git -C "$C26_WORKTREE" status --short
    [[ -z "$(git -C "$C26_WORKTREE" status --porcelain)" ]] || { echo 'Existing C26 changes preserved; inspect before synchronizing'; exit 1; }
    [[ "$(git -C "$C26_WORKTREE" rev-parse HEAD)" == "$C26_SHA" ]] || { echo 'Existing C26 commit differs; preserved without reset/switch'; exit 1; }
else
    if git -C "$C26_MAIN_REPO" show-ref --verify --quiet "refs/heads/$C26_BRANCH"; then
        [[ "$(git -C "$C26_MAIN_REPO" rev-parse "$C26_BRANCH")" == "$C26_SHA" ]] || { echo 'Existing local C26 branch differs; preserved'; exit 1; }
        git -C "$C26_MAIN_REPO" worktree add "$C26_WORKTREE" "$C26_BRANCH"
    else
        git -C "$C26_MAIN_REPO" worktree add -b "$C26_BRANCH" "$C26_WORKTREE" "$C26_SHA"
    fi
fi
[[ "$(git -C "$C26_WORKTREE" rev-parse HEAD)" == "$C26_SHA" ]]
printf 'C26 ready: %s\nCommit: %s\nNo training started.\n' "$C26_WORKTREE" "$C26_SHA"
