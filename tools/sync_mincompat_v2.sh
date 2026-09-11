#!/usr/bin/env bash
# Fetch and verify a fixed historical branch commit. Never update a different existing worktree.
set -Eeuo pipefail
MINCOMPAT_SHA="${1:?Supply the full 40-character commit SHA}"
MINCOMPAT_MAIN_REPO="${2:-/root/autodl-tmp/projects/Crack_RTDETR}"
MINCOMPAT_WORKTREE="${3:-/root/autodl-tmp/projects/Crack_RTDETR-mincompat-v2}"
MINCOMPAT_BRANCH=codex/rtdetr-mincompat-v2
MINCOMPAT_BASE=05e6de8b582f4a5f6c11cd50f8bc0406dacd7d0b
[[ "$MINCOMPAT_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full lowercase SHA required'; exit 2; }
MINCOMPAT_MAIN_REPO="$(realpath -m -- "$MINCOMPAT_MAIN_REPO")"
MINCOMPAT_WORKTREE="$(realpath -m -- "$MINCOMPAT_WORKTREE")"
[[ "$MINCOMPAT_MAIN_REPO" != "$MINCOMPAT_WORKTREE" ]] || { echo 'Separate worktree required'; exit 1; }
MINCOMPAT_REMOTE="$(git -C "$MINCOMPAT_MAIN_REPO" remote get-url origin)"
case "$MINCOMPAT_REMOTE" in
  https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo "Unexpected origin: $MINCOMPAT_REMOTE"; exit 1 ;;
esac
git -C "$MINCOMPAT_MAIN_REPO" fetch origin "$MINCOMPAT_BRANCH"
MINCOMPAT_REMOTE_HEAD="$(git -C "$MINCOMPAT_MAIN_REPO" rev-parse FETCH_HEAD)"
git -C "$MINCOMPAT_MAIN_REPO" cat-file -e "$MINCOMPAT_SHA^{commit}"
git -C "$MINCOMPAT_MAIN_REPO" merge-base --is-ancestor "$MINCOMPAT_SHA" "$MINCOMPAT_REMOTE_HEAD" || { echo 'Pinned SHA is outside branch history'; exit 1; }
git -C "$MINCOMPAT_MAIN_REPO" merge-base --is-ancestor "$MINCOMPAT_BASE" "$MINCOMPAT_SHA"
if [[ -e "$MINCOMPAT_WORKTREE" ]]; then
    [[ "$(git -C "$MINCOMPAT_WORKTREE" rev-parse --show-toplevel)" == "$MINCOMPAT_WORKTREE" ]] || { echo 'Existing directory is not requested worktree'; exit 1; }
    MINCOMPAT_COMMON="$(cd -- "$MINCOMPAT_WORKTREE" && realpath -- "$(git rev-parse --git-common-dir)")"
    MINCOMPAT_MAIN_COMMON="$(cd -- "$MINCOMPAT_MAIN_REPO" && realpath -- "$(git rev-parse --git-common-dir)")"
    [[ "$MINCOMPAT_COMMON" == "$MINCOMPAT_MAIN_COMMON" ]] || { echo 'Worktree belongs to another repository'; exit 1; }
    [[ -z "$(git -C "$MINCOMPAT_WORKTREE" status --porcelain)" ]] || { echo 'Existing changes preserved; synchronization stopped'; exit 1; }
    if git -C "$MINCOMPAT_WORKTREE" symbolic-ref -q HEAD >/dev/null; then
        echo 'Existing attached worktree preserved; detached worktree required'; exit 1
    fi
    [[ "$(git -C "$MINCOMPAT_WORKTREE" rev-parse HEAD)" == "$MINCOMPAT_SHA" ]] || { echo 'Existing worktree SHA differs; preserved without checkout/reset'; exit 1; }
else
    git -C "$MINCOMPAT_MAIN_REPO" worktree add --detach "$MINCOMPAT_WORKTREE" "$MINCOMPAT_SHA"
fi
[[ "$(git -C "$MINCOMPAT_WORKTREE" rev-parse HEAD)" == "$MINCOMPAT_SHA" ]]
mkdir -p "$MINCOMPAT_WORKTREE/outputs"
MINCOMPAT_PIN="$MINCOMPAT_WORKTREE/outputs/mincompat_v2_sync.json"
MINCOMPAT_PIN_CONTENT="{\"sha\":\"$MINCOMPAT_SHA\"}"
if [[ -e "$MINCOMPAT_PIN" ]]; then
    [[ "$(cat "$MINCOMPAT_PIN")" == "$MINCOMPAT_PIN_CONTENT" ]] || { echo 'Existing pin content differs; preserved'; exit 1; }
else
    (set -o noclobber; printf '%s\n' "$MINCOMPAT_PIN_CONTENT" > "$MINCOMPAT_PIN")
fi
printf 'MinCompat ready: %s\nCommit: %s\nNo training started.\n' "$MINCOMPAT_WORKTREE" "$MINCOMPAT_SHA"
