#!/usr/bin/env bash
# Fetch and verify a fixed historical branch commit. Never update a different existing worktree.
set -Eeuo pipefail
SCI_ADAPTER_SHA="${1:?Supply the full 40-character commit SHA}"
SCI_ADAPTER_MAIN_REPO="${2:-/root/autodl-tmp/projects/Crack_RTDETR}"
SCI_ADAPTER_WORKTREE="${3:-/root/autodl-tmp/projects/Crack_RTDETR-sci-adapter}"
SCI_ADAPTER_BRANCH=codex/rtdetr-sci-adapter
SCI_ADAPTER_BASE=33535ab2a5d9acee4e62d34ae382df8b98b9bbde
[[ "$SCI_ADAPTER_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full lowercase SHA required'; exit 2; }
SCI_ADAPTER_MAIN_REPO="$(realpath -m -- "$SCI_ADAPTER_MAIN_REPO")"
SCI_ADAPTER_WORKTREE="$(realpath -m -- "$SCI_ADAPTER_WORKTREE")"
[[ "$SCI_ADAPTER_MAIN_REPO" != "$SCI_ADAPTER_WORKTREE" ]] || { echo 'Separate worktree required'; exit 1; }
SCI_ADAPTER_REMOTE="$(git -C "$SCI_ADAPTER_MAIN_REPO" remote get-url origin)"
case "$SCI_ADAPTER_REMOTE" in
  https://github.com/supershaojie/concrete-crack-rtdetr.git|https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo "Unexpected origin: $SCI_ADAPTER_REMOTE"; exit 1 ;;
esac
git -C "$SCI_ADAPTER_MAIN_REPO" fetch origin "$SCI_ADAPTER_BRANCH"
SCI_ADAPTER_REMOTE_HEAD="$(git -C "$SCI_ADAPTER_MAIN_REPO" rev-parse FETCH_HEAD)"
git -C "$SCI_ADAPTER_MAIN_REPO" cat-file -e "$SCI_ADAPTER_SHA^{commit}"
git -C "$SCI_ADAPTER_MAIN_REPO" merge-base --is-ancestor "$SCI_ADAPTER_SHA" "$SCI_ADAPTER_REMOTE_HEAD" || { echo 'Pinned SHA is outside branch history'; exit 1; }
git -C "$SCI_ADAPTER_MAIN_REPO" merge-base --is-ancestor "$SCI_ADAPTER_BASE" "$SCI_ADAPTER_SHA"
if [[ -e "$SCI_ADAPTER_WORKTREE" ]]; then
    [[ "$(git -C "$SCI_ADAPTER_WORKTREE" rev-parse --show-toplevel)" == "$SCI_ADAPTER_WORKTREE" ]] || { echo 'Existing directory is not requested worktree'; exit 1; }
    SCI_ADAPTER_COMMON="$(cd -- "$SCI_ADAPTER_WORKTREE" && realpath -- "$(git rev-parse --git-common-dir)")"
    SCI_ADAPTER_MAIN_COMMON="$(cd -- "$SCI_ADAPTER_MAIN_REPO" && realpath -- "$(git rev-parse --git-common-dir)")"
    [[ "$SCI_ADAPTER_COMMON" == "$SCI_ADAPTER_MAIN_COMMON" ]] || { echo 'Worktree belongs to another repository'; exit 1; }
    [[ -z "$(git -C "$SCI_ADAPTER_WORKTREE" status --porcelain)" ]] || { echo 'Existing changes preserved; synchronization stopped'; exit 1; }
    if git -C "$SCI_ADAPTER_WORKTREE" symbolic-ref -q HEAD >/dev/null; then
        echo 'Existing attached worktree preserved; detached worktree required'; exit 1
    fi
    [[ "$(git -C "$SCI_ADAPTER_WORKTREE" rev-parse HEAD)" == "$SCI_ADAPTER_SHA" ]] || { echo 'Existing worktree SHA differs; preserved without checkout/reset'; exit 1; }
else
    git -C "$SCI_ADAPTER_MAIN_REPO" worktree add --detach "$SCI_ADAPTER_WORKTREE" "$SCI_ADAPTER_SHA"
fi
[[ "$(git -C "$SCI_ADAPTER_WORKTREE" rev-parse HEAD)" == "$SCI_ADAPTER_SHA" ]]
mkdir -p "$SCI_ADAPTER_WORKTREE/outputs"
SCI_ADAPTER_PIN="$SCI_ADAPTER_WORKTREE/outputs/sci_adapter_sync.json"
SCI_ADAPTER_PIN_CONTENT="{\"sha\":\"$SCI_ADAPTER_SHA\"}"
if [[ -e "$SCI_ADAPTER_PIN" ]]; then
    [[ "$(cat "$SCI_ADAPTER_PIN")" == "$SCI_ADAPTER_PIN_CONTENT" ]] || { echo 'Existing pin content differs; preserved'; exit 1; }
else
    (set -o noclobber; printf '%s\n' "$SCI_ADAPTER_PIN_CONTENT" > "$SCI_ADAPTER_PIN")
fi
printf 'SCI Adapter ready: %s\nCommit: %s\nNo training started.\n' "$SCI_ADAPTER_WORKTREE" "$SCI_ADAPTER_SHA"
