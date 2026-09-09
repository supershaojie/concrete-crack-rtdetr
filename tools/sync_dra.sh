#!/usr/bin/env bash
# Run from anywhere: bash /path/to/this/script FULL_COMMIT_SHA [MAIN_REPO] [DRA_WORKTREE]
# Never reset, clean, remove, switch, or update an existing checkout.
set -Eeuo pipefail
DRA_SHA="${1:?Supply the pinned full DRA commit SHA}"
DRA_MAIN_REPO="${2:-/root/autodl-tmp/projects/Crack_RTDETR}"
DRA_WORKTREE="${3:-/root/autodl-tmp/projects/Crack_RTDETR-dra}"
DRA_MAIN_REPO="$(realpath -- "$DRA_MAIN_REPO")"
DRA_WORKTREE="$(realpath -m -- "$DRA_WORKTREE")"
[[ "$DRA_MAIN_REPO" != "$DRA_WORKTREE" ]] || { echo "DRA requires a separate worktree"; exit 1; }
DRA_BRANCH=codex/dra-aifi
DRA_BASE=beedcfa307e250fb2de47587097c51f9c141123b
[[ "$DRA_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full lowercase 40-character SHA is required'; exit 2; }
DRA_REMOTE="$(git -C "$DRA_MAIN_REPO" remote get-url origin)"
case "$DRA_REMOTE" in
    https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
    *) echo "Unexpected origin: $DRA_REMOTE"; exit 1 ;;
esac
git -C "$DRA_MAIN_REPO" status --short
git -C "$DRA_MAIN_REPO" rev-parse HEAD
git -C "$DRA_MAIN_REPO" fetch origin "$DRA_BRANCH"
DRA_REMOTE_HEAD="$(git -C "$DRA_MAIN_REPO" rev-parse FETCH_HEAD)"
git -C "$DRA_MAIN_REPO" cat-file -e "$DRA_SHA^{commit}" || git -C "$DRA_MAIN_REPO" fetch origin "$DRA_SHA"
git -C "$DRA_MAIN_REPO" merge-base --is-ancestor "$DRA_SHA" "$DRA_REMOTE_HEAD"
git -C "$DRA_MAIN_REPO" merge-base --is-ancestor "$DRA_BASE" "$DRA_SHA"
git -C "$DRA_MAIN_REPO" cat-file -e "$DRA_SHA:tools/init_dra.py"
if [[ -e "$DRA_WORKTREE" ]]; then
    [[ "$(git -C "$DRA_WORKTREE" rev-parse --show-toplevel)" == "$DRA_WORKTREE" ]] || { echo 'Existing directory is not the requested worktree'; exit 1; }
    DRA_COMMON="$(cd -- "$DRA_WORKTREE" && realpath -- "$(git rev-parse --git-common-dir)")"
    DRA_MAIN_COMMON="$(cd -- "$DRA_MAIN_REPO" && realpath -- "$(git rev-parse --git-common-dir)")"
    [[ "$DRA_COMMON" == "$DRA_MAIN_COMMON" ]] || {
        echo 'Existing worktree belongs to another repository'; exit 1;
    }
    [[ "$(git -C "$DRA_WORKTREE" branch --show-current)" == "$DRA_BRANCH" ]] || { echo "Existing target belongs to another branch; preserved"; exit 1; }
    git -C "$DRA_WORKTREE" status --short
    [[ -z "$(git -C "$DRA_WORKTREE" status --porcelain)" ]] || { echo 'Existing DRA changes preserved; inspect before synchronizing'; exit 1; }
    [[ "$(git -C "$DRA_WORKTREE" rev-parse HEAD)" == "$DRA_SHA" ]] || { git -C "$DRA_WORKTREE" rev-parse HEAD; git -C "$DRA_WORKTREE" diff --stat HEAD "$DRA_SHA"; echo "Existing DRA commit differs; preserved without reset/switch"; exit 1; }
else
    if git -C "$DRA_MAIN_REPO" show-ref --verify --quiet "refs/heads/$DRA_BRANCH"; then
        [[ "$(git -C "$DRA_MAIN_REPO" rev-parse "$DRA_BRANCH")" == "$DRA_SHA" ]] || { git -C "$DRA_MAIN_REPO" rev-parse "$DRA_BRANCH"; git -C "$DRA_MAIN_REPO" diff --stat "$DRA_BRANCH" "$DRA_SHA"; echo 'Existing local DRA branch differs; preserved'; exit 1; }
        git -C "$DRA_MAIN_REPO" worktree add "$DRA_WORKTREE" "$DRA_BRANCH"
    else
        git -C "$DRA_MAIN_REPO" worktree add -b "$DRA_BRANCH" "$DRA_WORKTREE" "$DRA_SHA"
    fi
fi
[[ "$(git -C "$DRA_WORKTREE" rev-parse HEAD)" == "$DRA_SHA" ]]
printf 'DRA ready: %s\nCommit: %s\nNo training started.\n' "$DRA_WORKTREE" "$DRA_SHA"
