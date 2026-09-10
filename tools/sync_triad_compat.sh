#!/usr/bin/env bash
# Fetch and verify a fixed historical branch commit. Never update a different existing worktree.
set -Eeuo pipefail
TRIAD_SHA="${1:?Supply the full 40-character commit SHA}"
TRIAD_MAIN_REPO="${2:-/root/autodl-tmp/projects/Crack_RTDETR}"
TRIAD_WORKTREE="${3:-/root/autodl-tmp/projects/Crack_RTDETR-triad-compat}"
TRIAD_BRANCH=codex/rtdetr-triad-compat
TRIAD_BASE=67c3078e54a657fd96d65fee657a75fbb1dae0d6
[[ "$TRIAD_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full lowercase SHA required'; exit 2; }
TRIAD_MAIN_REPO="$(realpath -m -- "$TRIAD_MAIN_REPO")"
TRIAD_WORKTREE="$(realpath -m -- "$TRIAD_WORKTREE")"
[[ "$TRIAD_MAIN_REPO" != "$TRIAD_WORKTREE" ]] || { echo 'Separate worktree required'; exit 1; }
TRIAD_REMOTE="$(git -C "$TRIAD_MAIN_REPO" remote get-url origin)"
case "$TRIAD_REMOTE" in
  https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo "Unexpected origin: $TRIAD_REMOTE"; exit 1 ;;
esac
git -C "$TRIAD_MAIN_REPO" fetch origin "$TRIAD_BRANCH"
TRIAD_REMOTE_HEAD="$(git -C "$TRIAD_MAIN_REPO" rev-parse FETCH_HEAD)"
git -C "$TRIAD_MAIN_REPO" cat-file -e "$TRIAD_SHA^{commit}"
git -C "$TRIAD_MAIN_REPO" merge-base --is-ancestor "$TRIAD_SHA" "$TRIAD_REMOTE_HEAD" || { echo 'Pinned SHA is outside branch history'; exit 1; }
git -C "$TRIAD_MAIN_REPO" merge-base --is-ancestor "$TRIAD_BASE" "$TRIAD_SHA"
if [[ -e "$TRIAD_WORKTREE" ]]; then
    [[ "$(git -C "$TRIAD_WORKTREE" rev-parse --show-toplevel)" == "$TRIAD_WORKTREE" ]] || { echo 'Existing directory is not requested worktree'; exit 1; }
    TRIAD_COMMON="$(cd -- "$TRIAD_WORKTREE" && realpath -- "$(git rev-parse --git-common-dir)")"
    TRIAD_MAIN_COMMON="$(cd -- "$TRIAD_MAIN_REPO" && realpath -- "$(git rev-parse --git-common-dir)")"
    [[ "$TRIAD_COMMON" == "$TRIAD_MAIN_COMMON" ]] || { echo 'Worktree belongs to another repository'; exit 1; }
    [[ -z "$(git -C "$TRIAD_WORKTREE" status --porcelain)" ]] || { echo 'Existing changes preserved; synchronization stopped'; exit 1; }
    [[ "$(git -C "$TRIAD_WORKTREE" rev-parse HEAD)" == "$TRIAD_SHA" ]] || { echo 'Existing worktree SHA differs; preserved without checkout/reset'; exit 1; }
else
    git -C "$TRIAD_MAIN_REPO" worktree add --detach "$TRIAD_WORKTREE" "$TRIAD_SHA"
fi
[[ "$(git -C "$TRIAD_WORKTREE" rev-parse HEAD)" == "$TRIAD_SHA" ]]
mkdir -p "$TRIAD_WORKTREE/outputs"
TRIAD_PIN="$TRIAD_WORKTREE/outputs/triad_compat_sync.json"
TRIAD_PIN_CONTENT="{\"sha\":\"$TRIAD_SHA\"}"
if [[ -e "$TRIAD_PIN" ]]; then
    [[ "$(cat "$TRIAD_PIN")" == "$TRIAD_PIN_CONTENT" ]] || { echo 'Existing pin content differs; preserved'; exit 1; }
else
    (set -o noclobber; printf '%s\n' "$TRIAD_PIN_CONTENT" > "$TRIAD_PIN")
fi
printf 'Triad ready: %s\nCommit: %s\nNo training started.\n' "$TRIAD_WORKTREE" "$TRIAD_SHA"
