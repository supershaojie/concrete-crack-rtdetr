#!/usr/bin/env bash
# Run the copy extracted from the requested full commit, never a moving branch copy.
set -Eeuo pipefail
main="${RDM_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}"
worktree="${RDM_WORKTREE:-/root/autodl-tmp/projects/Crack_RTDETR-rdm-v1}"
sha="${1:?Usage: bash sync_rdm.sh FULL_40_CHARACTER_SHA}"
branch=exp-rtdetr-r18-lite-rdm-v1
base=a0459d6a652cb702699087c88fa39a3e4c4087ec
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full SHA required' >&2; exit 2; }
main="$(cd "$main" && pwd -P)"
[[ "$(git -C "$main" rev-parse --show-toplevel)" == "$main" ]] || { echo 'Main must be repository root'; exit 2; }
origin="$(git -C "$main" remote get-url origin)"
case "$origin" in
  https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo "Unexpected origin: $origin" >&2; exit 2;;
esac
export GIT_TERMINAL_PROMPT=0
if ! git -C "$main" cat-file -e "$sha^{commit}" 2>/dev/null; then
  for attempt in 1 2 3 4 5; do
    printf 'RDM fetch attempt %s/5\n' "$attempt"
    code=0
    timeout 120 git -C "$main" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
      fetch --progress --no-tags origin "$branch" || code=$?
    printf 'RDM fetch exit=%s\n' "$code"
    if git -C "$main" cat-file -e "$sha^{commit}" 2>/dev/null; then break; fi
  done
fi
git -C "$main" cat-file -e "$sha^{commit}" || { echo 'Target SHA unavailable; returning to terminal'; exit 1; }
git -C "$main" merge-base --is-ancestor "$base" "$sha" || { echo 'Mother is not an ancestor'; exit 2; }
if [[ -e "$worktree" ]]; then
  [[ -f "$worktree/.git" ]] || { echo 'Existing directory is not a linked worktree; preserved'; exit 2; }
  [[ "$(git -C "$worktree" rev-parse HEAD)" == "$sha" ]] || { echo 'Existing worktree has another HEAD; preserved'; exit 2; }
else
  git -C "$main" worktree add --detach "$worktree" "$sha"
fi
common_main="$(git -C "$main" rev-parse --path-format=absolute --git-common-dir)"
common_worktree="$(git -C "$worktree" rev-parse --path-format=absolute --git-common-dir)"
[[ "$(realpath "$common_main")" == "$(realpath "$common_worktree")" ]] || { echo 'Wrong shared repository'; exit 2; }
git -C "$worktree" diff --quiet && git -C "$worktree" diff --cached --quiet || { echo 'Tracked edits present; preserved'; exit 2; }
git -C "$worktree" status --short
printf 'RDM verified HEAD=%s worktree=%s\n' "$(git -C "$worktree" rev-parse HEAD)" "$worktree"
