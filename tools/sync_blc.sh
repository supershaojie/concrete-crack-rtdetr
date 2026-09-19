#!/usr/bin/env bash
# Fixed SHA only; never switch the main worktree or reset/clean any directory.
set -Eeuo pipefail
BLC_SHA="${1:?Usage: sync_blc.sh FULL_SHA [MAIN] [WORKTREE]}"
BLC_MAIN_REPO="$(realpath -- "${2:-/root/autodl-tmp/projects/Crack_RTDETR}")"
BLC_TARGET="$(realpath -m -- "${3:-/root/autodl-tmp/projects/Crack_RTDETR-blc-v1}")"
BLC_BRANCH=exp-rtdetr-r18-lite-blc-v1
BLC_BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
[[ "$BLC_SHA" =~ ^[0-9a-f]{40}$ && $# -le 3 ]] || { echo 'Full 40-character SHA required'; exit 2; }
[[ "$BLC_MAIN_REPO" != "$BLC_TARGET" ]] || { echo 'Separate BLC worktree required'; exit 1; }
BLC_ORIGIN="$(git -C "$BLC_MAIN_REPO" remote get-url origin)"
case "${BLC_ORIGIN%.git}" in
  https://github.com/supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr|ssh://git@github.com/supershaojie/concrete-crack-rtdetr) ;;
  *) echo 'Unexpected origin; preserved'; exit 1 ;;
esac
BLC_MAIN_HEAD="$(git -C "$BLC_MAIN_REPO" rev-parse HEAD)"
BLC_MAIN_STATUS="$(git -C "$BLC_MAIN_REPO" status --porcelain)"
if ! git -C "$BLC_MAIN_REPO" cat-file -e "$BLC_SHA^{commit}" 2>/dev/null; then
  for BLC_ATTEMPT in 1 2 3 4 5; do
    echo "BLC fetch attempt $BLC_ATTEMPT/5, timeout=120s"
    set +e
    GIT_TERMINAL_PROMPT=0 timeout 120s git -C "$BLC_MAIN_REPO" \
      -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
      fetch --progress --no-tags --no-write-fetch-head origin \
      "refs/heads/$BLC_BRANCH:refs/remotes/origin/$BLC_BRANCH"
    BLC_RC=$?
    set -e
    echo "fetch exit=$BLC_RC"
    if [[ "$BLC_RC" -eq 0 ]] && git -C "$BLC_MAIN_REPO" cat-file -e "$BLC_SHA^{commit}" 2>/dev/null; then break; fi
  done
fi
git -C "$BLC_MAIN_REPO" cat-file -e "$BLC_SHA^{commit}"
git -C "$BLC_MAIN_REPO" merge-base --is-ancestor "$BLC_BASE" "$BLC_SHA"
git -C "$BLC_MAIN_REPO" cat-file -e "$BLC_SHA:tools/blc_server.sh"
if [[ -e "$BLC_TARGET" ]]; then
  [[ -f "$BLC_TARGET/.git" ]] || { echo 'Existing non-worktree preserved'; exit 1; }
  [[ "$(git -C "$BLC_TARGET" rev-parse --show-toplevel)" == "$BLC_TARGET" ]]
  [[ "$(git -C "$BLC_TARGET" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$BLC_MAIN_REPO" rev-parse --path-format=absolute --git-common-dir)" ]]
  [[ "$(git -C "$BLC_TARGET" rev-parse HEAD)" == "$BLC_SHA" ]] || { echo 'Different existing HEAD preserved; use a new BLC directory'; exit 1; }
  [[ -z "$(git -C "$BLC_TARGET" status --porcelain --untracked-files=no)" ]] || { echo 'Tracked modifications preserved'; exit 1; }
  git -C "$BLC_TARGET" status --short
else
  git -C "$BLC_MAIN_REPO" worktree add --detach "$BLC_TARGET" "$BLC_SHA"
fi
[[ "$(git -C "$BLC_MAIN_REPO" rev-parse HEAD)" == "$BLC_MAIN_HEAD" ]]
[[ "$(git -C "$BLC_MAIN_REPO" status --porcelain)" == "$BLC_MAIN_STATUS" ]]
printf 'Verified BLC worktree=%s\nHEAD=%s\nNo training started.\n' "$BLC_TARGET" "$(git -C "$BLC_TARGET" rev-parse HEAD)"
