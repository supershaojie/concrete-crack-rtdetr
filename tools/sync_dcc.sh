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
# Exact server-side patch read from DCC_RESUME_DIAG_20260918T154901_136532Z.zip.
# Only this unstaged change at this old HEAD can be absorbed automatically.
hookfix_head=5049bb0010f91aa89cd0294024ea73f6b0396164
hookfix_path=tools/preflight_dcc.py
hookfix_before=03f17cfa41a2cca3434a5e9038d803afc90b796038b3d92c88779e4cc94565ff
hookfix_after=34e17dea109bbf9706e668384574f166dc9728bb9c3d59e93e8d2d38fca4a16c
hookfix_block=$'        # DCC_PREFLIGHT_DETACH_BEFORE_SAVE_V1\n        for handle in handles:\n            handle.remove()\n        handles.clear()'
lf_sha256() { sed 's/\r$//' | sha256sum | cut -d ' ' -f1; }
main="${2:-/root/autodl-tmp/projects/Crack_RTDETR}"
worktree="${3:-/root/autodl-tmp/projects/Crack_RTDETR-dcc-v1}"
[[ -d "$main" ]] || { printf 'Existing main repository is missing. Nothing cloned.\n' >&2; exit 2; }
main="$(cd "$main" && pwd -P)"
worktree="$(realpath -m -- "$worktree")"
[[ "$worktree" != "$main" ]] || { printf 'Worktree must differ from the main repository.\n' >&2; exit 2; }
[[ "$(cd "$(git -C "$main" rev-parse --show-toplevel)" && pwd -P)" == "$main" ]] || {
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
  current="$(git -C "$worktree" rev-parse HEAD)"
  current_branch="$(git -C "$worktree" symbolic-ref --quiet --short HEAD || true)"
  [[ -z "$current_branch" || "$current_branch" == "$branch" ]] || {
    printf 'Existing worktree is on another named branch; preserved untouched: %s\n' "$current_branch" >&2; exit 2;
  }
  git -C "$worktree" merge-base --is-ancestor "$current" "$target" || {
    printf 'Pinned target is not a fast-forward descendant of the existing HEAD; worktree preserved.\n' >&2; exit 2;
  }
  dirty="$(git -C "$worktree" status --porcelain --untracked-files=no)"
  backup=""
  if [[ -n "$dirty" ]]; then
    # Reject every other staged/unstaged change. Untracked outputs stay in place.
    [[ "$current" == "$hookfix_head" && "$dirty" == " M $hookfix_path" && ! -L "$worktree/$hookfix_path" ]] &&
      [[ -z "$(git -C "$worktree" diff --summary)" ]] &&
      [[ "$(git -C "$worktree" show "$current:$hookfix_path" | lf_sha256)" == "$hookfix_before" ]] &&
      [[ "$(lf_sha256 < "$worktree/$hookfix_path")" == "$hookfix_after" ]] &&
      [[ "$(git -C "$worktree" show "$target:$hookfix_path" | sed 's/\r$//' | sed -n '/# DCC_PREFLIGHT_DETACH_BEFORE_SAVE_V1$/,+3p')" == "$hookfix_block" ]] || {
      printf '%s\n' 'Tracked changes are not exactly the verified HEAD5049 preflight hook fix already included in the target.' \
        'No stash, restore, reset or checkout was performed. Preserve/review the other edits before syncing.' >&2
      exit 2
    }
    backup="$worktree/outputs/dcc/sync_backups/$(date -u +%Y%m%dT%H%M%SZ)_$$"
    mkdir -p -- "$(dirname "$backup")"
    mkdir -- "$backup"
    git -C "$worktree" diff --binary HEAD -- "$hookfix_path" > "$backup/known_hookfix.patch"
    cp -- "$worktree/$hookfix_path" "$backup/preflight_dcc.py.original"
    printf 'old_head=%s\ntarget=%s\npath=%s\nbefore_lf_sha256=%s\nafter_lf_sha256=%s\n' \
      "$current" "$target" "$hookfix_path" "$hookfix_before" "$hookfix_after" > "$backup/identity.txt"
    sha256sum "$backup/known_hookfix.patch" "$backup/preflight_dcc.py.original" > "$backup/SHA256SUMS"
    # Recheck immediately before reversing only the backed-up four-line patch.
    [[ "$(git -C "$worktree" rev-parse HEAD)" == "$current" &&
       "$(git -C "$worktree" status --porcelain --untracked-files=no)" == "$dirty" &&
       "$(lf_sha256 < "$worktree/$hookfix_path")" == "$hookfix_after" ]] || {
      printf 'Worktree changed during backup; no patch removed. Backup: %s\n' "$backup" >&2; exit 2;
    }
    git -C "$worktree" apply --reverse --check "$backup/known_hookfix.patch"
    git -C "$worktree" apply --reverse "$backup/known_hookfix.patch"
    printf 'Verified local hook fix backed up before absorption: %s\n' "$backup"
  fi
  if [[ "$current" != "$target" ]]; then
    # Detach at the fixed descendant without moving any other branch. Protect
    # ignored outputs too; an untracked/ignored collision must stop the checkout.
    if git -C "$worktree" switch --detach --no-overwrite-ignore "$target"; then
      :
    else
      rc=$?
      if [[ -n "$backup" && "$(git -C "$worktree" rev-parse HEAD)" == "$current" &&
            -z "$(git -C "$worktree" status --porcelain --untracked-files=no)" ]]; then
        if git -C "$worktree" apply "$backup/known_hookfix.patch"; then
          printf 'Checkout failed; the verified local hook fix was restored from its backup.\n' >&2
        else
          printf 'Checkout failed; manual recovery patch remains at %s/known_hookfix.patch\n' "$backup" >&2
        fi
      fi
      printf 'Pinned checkout failed; all output directories were preserved.\n' >&2
      exit "$rc"
    fi
  fi
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
