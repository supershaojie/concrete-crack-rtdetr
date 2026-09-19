#!/usr/bin/env bash
# Execute this file from the delivered commit; never from an unknown old worktree.
set -Eeuo pipefail
DPR_SHA="${1:?Usage: sync_dpr.sh FULL_SHA [MAIN_REPO] [WORKTREE]}"
DPR_MAIN="$(realpath -- "${2:-/root/autodl-tmp/projects/Crack_RTDETR}")"
DPR_WORKTREE="$(realpath -m -- "${3:-/root/autodl-tmp/projects/Crack_RTDETR-dpr-v1}")"
DPR_BRANCH=exp-rtdetr-r18-lite-dpr-v1
DPR_BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
[[ $# -le 3 && "$DPR_SHA" =~ ^[0-9a-f]{40}$ && "$DPR_MAIN" != "$DPR_WORKTREE" ]] || exit 2
case "$(git -C "$DPR_MAIN" remote get-url origin)" in
 https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
 *) echo 'Unexpected origin identity; preserved without printing possible credentials.' >&2; exit 1 ;;
esac
DPR_ORIGINAL_HEAD="$(git -C "$DPR_MAIN" rev-parse HEAD)"
DPR_ORIGINAL_STATUS="$(git -C "$DPR_MAIN" status --porcelain)"
if ! git -C "$DPR_MAIN" cat-file -e "$DPR_SHA^{commit}" 2>/dev/null; then
 for DPR_ATTEMPT in 1 2 3 4 5; do
  echo "DPR fetch attempt $DPR_ATTEMPT/5"
  if GIT_TERMINAL_PROMPT=0 timeout 120s git -C "$DPR_MAIN" -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags --no-write-fetch-head \
      origin "refs/heads/$DPR_BRANCH:refs/remotes/origin/$DPR_BRANCH"; then
   git -C "$DPR_MAIN" cat-file -e "$DPR_SHA^{commit}" && break
  else
   DPR_RC=$?; echo "Fetch failed: exit $DPR_RC" >&2
  fi
  [[ "$DPR_ATTEMPT" == 5 ]] || sleep 3
 done
fi
git -C "$DPR_MAIN" cat-file -e "$DPR_SHA^{commit}"
git -C "$DPR_MAIN" merge-base --is-ancestor "$DPR_BASE" "$DPR_SHA"
git -C "$DPR_MAIN" cat-file -e "$DPR_SHA:tools/dpr_server.sh"
if [[ -e "$DPR_WORKTREE" ]]; then
 [[ -f "$DPR_WORKTREE/.git" ]] || { echo 'Target is not a linked worktree; preserved.' >&2; exit 1; }
 [[ "$(git -C "$DPR_WORKTREE" rev-parse --show-toplevel)" == "$DPR_WORKTREE" ]]
 [[ "$(git -C "$DPR_WORKTREE" rev-parse --path-format=absolute --git-common-dir)" == \
    "$(git -C "$DPR_MAIN" rev-parse --path-format=absolute --git-common-dir)" ]]
 [[ -z "$(git -C "$DPR_WORKTREE" status --porcelain --untracked-files=no)" ]] || { echo 'Tracked edits preserved; refusing update.' >&2; exit 1; }
 [[ "$(git -C "$DPR_WORKTREE" rev-parse HEAD)" == "$DPR_SHA" ]] || { echo 'Existing worktree has another SHA; preserved. Use another new directory explicitly.' >&2; exit 1; }
else
 git -C "$DPR_MAIN" worktree add --detach "$DPR_WORKTREE" "$DPR_SHA"
fi
[[ "$(git -C "$DPR_MAIN" rev-parse HEAD)" == "$DPR_ORIGINAL_HEAD" ]]
[[ "$(git -C "$DPR_MAIN" status --porcelain)" == "$DPR_ORIGINAL_STATUS" ]]
[[ "$(git -C "$DPR_WORKTREE" rev-parse HEAD)" == "$DPR_SHA" ]]
printf 'Verified DPR worktree: %s\nHEAD: %s\nFormal training NOT_STARTED by sync.\n' "$DPR_WORKTREE" "$DPR_SHA"
