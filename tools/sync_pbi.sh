#!/usr/bin/env bash
# Fixed-commit sync only. No training, reset, clean, remote rewrite or installation.
set -Eeuo pipefail
if [[ "${1:---help}" == --help ]]; then
  echo 'Usage: bash tools/sync_pbi.sh FULL_SHA [MAIN_REPO] [PBI_WORKTREE]'
  exit 0
fi
[[ $# -ge 1 && $# -le 3 ]] || exit 2
target="${1,,}"
[[ "$target" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full 40-hex SHA required'; exit 2; }
base=a0459d6a652cb702699087c88fa39a3e4c4087ec
branch=exp-rtdetr-r18-lite-pbi-v1
main="${2:-/root/autodl-tmp/projects/Crack_RTDETR}"
worktree="${3:-/root/autodl-tmp/projects/Crack_RTDETR-pbi-v1}"
main="$(cd "$main" && pwd -P)"
worktree="$(realpath -m -- "$worktree")"
[[ "$main" != "$worktree" ]]
[[ "$(cd "$(git -C "$main" rev-parse --show-toplevel)" && pwd -P)" == "$main" ]]
origin="$(git -C "$main" remote get-url origin)"
origin="${origin,,}"
# Canonical HTTPS/SSH or proxy with the canonical GitHub target suffix.
# Never print an origin URL, which may contain credentials.
[[ "$origin" =~ (^|[:/@])github\.com([:/]|:[0-9]+/)supershaojie/concrete-crack-rtdetr(\.git)?/?$ ]] || {
  echo 'Origin identity is unverified; preserved without modification.' >&2; exit 2;
}
unset origin
if ! git -C "$main" cat-file -e "$target^{commit}" 2>/dev/null; then
  command -v timeout >/dev/null
  for attempt in 1 2 3 4 5; do
    echo "Fetch $attempt/5, timeout 120s: $branch"
    if timeout --signal=TERM --kill-after=5s 120s env GIT_TERMINAL_PROMPT=0 \
      git -C "$main" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
      fetch --progress --no-tags --no-write-fetch-head origin \
      "refs/heads/$branch:refs/remotes/origin/$branch"; then
      git -C "$main" cat-file -e "$target^{commit}" 2>/dev/null && break
      echo 'Fetch completed but fixed target remains absent.' >&2
    else
      rc=$?; echo "Fetch failed: exit $rc (124 = timeout)." >&2
    fi
  done
fi
git -C "$main" cat-file -e "$target^{commit}"
git -C "$main" merge-base --is-ancestor "$base" "$target"
for path in tools/pbi_server.sh tools/init_pbi.py tools/check_pbi.py ultralytics-main/ultralytics/nn/modules/pbi.py; do
  git -C "$main" cat-file -e "$target:$path"
done
if [[ -e "$worktree" || -L "$worktree" ]]; then
  [[ -d "$worktree" && -f "$worktree/.git" ]] || { echo 'Existing destination preserved: not a linked worktree.'; exit 2; }
  [[ "$(cd "$worktree" && pwd -P)" == "$(cd "$(git -C "$worktree" rev-parse --show-toplevel)" && pwd -P)" ]]
  common_main="$(cd "$main" && cd "$(git rev-parse --git-common-dir)" && pwd -P)"
  common_target="$(cd "$worktree" && cd "$(git rev-parse --git-common-dir)" && pwd -P)"
  [[ "$common_main" == "$common_target" ]] || { echo 'Different repository: preserved.'; exit 2; }
  current_branch="$(git -C "$worktree" symbolic-ref --quiet --short HEAD || true)"
  [[ -z "$current_branch" || "$current_branch" == "$branch" ]] || { echo 'Another experiment branch: preserved.'; exit 2; }
  [[ -z "$(git -C "$worktree" status --porcelain --untracked-files=no)" ]] || { echo 'Tracked edits: preserved; review before sync.'; exit 2; }
  current="$(git -C "$worktree" rev-parse HEAD)"
  git -C "$worktree" merge-base --is-ancestor "$current" "$target" || { echo 'Non-descendant target: preserved.'; exit 2; }
  if [[ "$current" != "$target" ]]; then
    git -C "$worktree" switch --detach --no-overwrite-ignore "$target"
  fi
else
  git -C "$main" worktree add --detach "$worktree" "$target"
fi
[[ "$(git -C "$worktree" rev-parse HEAD)" == "$target" ]]
[[ -z "$(git -C "$worktree" status --porcelain --untracked-files=no)" ]]
modules="$worktree/ultralytics-main/ultralytics/nn/modules"
[[ "$(sed 's/\r$//' "$modules/cbr.py" | sha256sum | cut -d ' ' -f1)" == d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787 ]]
[[ "$(sed 's/\r$//' "$modules/lif_down.py" | sha256sum | cut -d ' ' -f1)" == 26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7 ]]
printf 'Verified PBI worktree: %s\nHEAD: %s\nBase ancestor: %s\n' "$worktree" "$target" "$base"
echo 'Sync complete. Formal training NOT_STARTED; final test NOT_RUN.'
