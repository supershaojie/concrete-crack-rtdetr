#!/usr/bin/env bash
# Sync only an already-existing repository, using its existing origin.
# No training, environment installation, reset, force push, or directory removal.
set -Eeuo pipefail

if [[ "${1:---help}" == --help ]]; then
  printf '%s\n' 'Usage: bash tools/sync_dcc.sh TARGET_40_HEX_SHA [MAIN_REPO] [DCC_WORKTREE]'
  printf '%s\n' 'Defaults: /root/autodl-tmp/projects/Crack_RTDETR and its sibling Crack_RTDETR-dcc-v1'
  exit 0
fi
[[ $# -ge 1 && $# -le 3 ]] || { printf 'Expected 1 to 3 arguments.\n' >&2; exit 2; }
target="${1,,}"
[[ "$target" =~ ^[0-9a-f]{40}$ ]] || { printf 'Target must be a complete 40-hex commit SHA.\n' >&2; exit 2; }
base=a0459d6a652cb702699087c88fa39a3e4c4087ec
branch=exp-rtdetr-r18-lite-dcc-v1
main="${2:-/root/autodl-tmp/projects/Crack_RTDETR}"
worktree="${3:-/root/autodl-tmp/projects/Crack_RTDETR-dcc-v1}"
[[ -d "$main" ]] || { printf 'Existing main repository is missing. Nothing cloned.\n' >&2; exit 2; }
main="$(cd "$main" && pwd -P)"
worktree="$(realpath -m -- "$worktree")"
[[ "$worktree" != "$main" ]] || { printf 'Worktree must differ from the main repository.\n' >&2; exit 2; }
[[ "$(git -C "$main" rev-parse --show-toplevel)" == "$main" ]] || {
  printf 'MAIN_REPO is not the repository root.\n' >&2; exit 2;
}
origin="$(git -C "$main" remote get-url origin)"
# Accept HTTPS, SSH and a configured proxy retaining the canonical target URL.
# Never print the URL: it could contain credentials.
origin_lc="${origin,,}"
[[ "$origin_lc" =~ (^|[:/@])github\.com([:/]|:[0-9]+/)supershaojie/concrete-crack-rtdetr(\.git)?/?$ ]] || {
  printf 'Existing origin identity is not verifiable as supershaojie/concrete-crack-rtdetr. Origin was not changed.\n' >&2
  exit 2
}
unset origin origin_lc
if ! git -C "$main" cat-file -e "$target^{commit}" 2>/dev/null; then
  command -v timeout >/dev/null || { printf 'GNU timeout is required for bounded fetch.\n' >&2; exit 2; }
  for attempt in 1 2 3 4 5; do
    printf 'Fetch attempt %s/5 (120-second budget), branch %s\n' "$attempt" "$branch"
    if timeout --signal=TERM --kill-after=5s 120s env GIT_TERMINAL_PROMPT=0 \
      git -C "$main" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
      fetch --progress --no-tags origin "$branch"; then
      if git -C "$main" cat-file -e "$target^{commit}" 2>/dev/null; then
        break
      fi
      printf 'Fetch completed, but the fixed target commit is still missing.\n' >&2
    else
      rc=$?
      printf 'Fetch attempt %s failed with exit code %s (124 means timeout).\n' "$attempt" "$rc" >&2
    fi
  done
fi
git -C "$main" cat-file -e "$target^{commit}" || {
  printf 'Target remains unavailable after bounded fetch; returning to terminal.\n' >&2; exit 1;
}
git -C "$main" cat-file -e "$base^{commit}" || {
  printf 'Specified successful base is absent from the fetched target history.\n' >&2; exit 2;
}
git -C "$main" merge-base --is-ancestor "$base" "$target" || {
  printf 'Target does not descend from the specified successful base.\n' >&2; exit 2;
}

if [[ -e "$worktree" || -L "$worktree" ]]; then
  [[ -d "$worktree" && -f "$worktree/.git" ]] || {
    printf 'Existing destination is not an independent git worktree; preserved untouched.\n' >&2; exit 2;
  }
  common_main="$(cd "$main" && cd "$(git rev-parse --git-common-dir)" && pwd -P)"
  common_target="$(cd "$worktree" && cd "$(git rev-parse --git-common-dir)" && pwd -P)"
  [[ "$common_main" == "$common_target" ]] || {
    printf 'Existing worktree belongs to a different repository; preserved untouched.\n' >&2; exit 2;
  }
  [[ "$(git -C "$worktree" rev-parse HEAD)" == "$target" ]] || {
    printf 'Existing worktree HEAD differs from the fixed target; no checkout or reset performed.\n' >&2; exit 2;
  }
else
  git -C "$main" worktree add --detach "$worktree" "$target"
fi
[[ "$(git -C "$worktree" rev-parse HEAD)" == "$target" ]]
[[ -z "$(git -C "$worktree" status --porcelain --untracked-files=no)" ]] || {
  printf 'Worktree has tracked changes; source identity cannot be certified. Changes were preserved.\n' >&2; exit 2;
}
modules="$worktree/ultralytics-main/ultralytics/nn/modules"
[[ "$(sed 's/\r$//' "$modules/cbr.py" | sha256sum | cut -d ' ' -f1)" == d6f35673489dade3fab4d360a9ac570fedccd25744afcc138e6c06fee134f787 ]]
[[ "$(sed 's/\r$//' "$modules/lif_down.py" | sha256sum | cut -d ' ' -f1)" == 26d480016114b679732015fc4567c2719aa86d16675a33f0fdd27a8d2eea3ee7 ]]
[[ -f "$modules/dcc.py" && -f "$worktree/docs/dcc/environment.sh" ]]
printf 'Verified worktree: %s\nHEAD: %s\nBase ancestor: %s\n' "$worktree" "$target" "$base"
printf '%s\n' 'Sync complete. Formal training NOT_STARTED; final test NOT_RUN.'
