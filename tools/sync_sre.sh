#!/usr/bin/env bash
# Fixed-object synchronization only. No server process or environment is changed.
set -Eeuo pipefail
SRE_SHA="${1:?Usage: sync_sre.sh FULL_SHA [MAIN] [WORKTREE]}"
SRE_MAIN="$(realpath -- "${2:-/root/autodl-tmp/projects/Crack_RTDETR}")"
SRE_WORKTREE="$(realpath -m -- "${3:-/root/autodl-tmp/projects/Crack_RTDETR-sre-v1}")"
SRE_BRANCH=exp-rtdetr-r18-lite-sre-v1
[[ $# -le 3 && "$SRE_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$SRE_MAIN" != "$SRE_WORKTREE" ]]
case "$(git -C "$SRE_MAIN" remote get-url origin)" in
 https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
 *) echo 'Unexpected origin; stop without modifying checkout'; exit 1 ;;
esac
SRE_OLD_HEAD="$(git -C "$SRE_MAIN" rev-parse HEAD)"
SRE_OLD_STATUS="$(git -C "$SRE_MAIN" status --porcelain)"
if ! git -C "$SRE_MAIN" cat-file -e "$SRE_SHA^{commit}" 2>/dev/null; then
 for SRE_ATTEMPT in 1 2 3 4 5; do
  if GIT_TERMINAL_PROMPT=0 timeout 120s git -C "$SRE_MAIN" \
    -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
    fetch --progress --no-tags --no-write-fetch-head origin \
    "refs/heads/$SRE_BRANCH:refs/remotes/origin/$SRE_BRANCH"; then
   git -C "$SRE_MAIN" cat-file -e "$SRE_SHA^{commit}" 2>/dev/null && break
  fi
 done
fi
if ! git -C "$SRE_MAIN" cat-file -e "$SRE_SHA^{commit}" 2>/dev/null; then
 if [[ -n "${SRE_BUNDLE:-}" && -f "$SRE_BUNDLE" ]]; then
  git -C "$SRE_MAIN" bundle verify "$SRE_BUNDLE"
  git -C "$SRE_MAIN" fetch --no-tags --no-write-fetch-head "$SRE_BUNDLE" \
    "refs/heads/$SRE_BRANCH:refs/remotes/sre-bundle/$SRE_BRANCH"
 else
  echo 'PUSH_PENDING: object missing after bounded fetch; supply verified incremental SRE_BUNDLE. No checkout overwritten.'
  exit 1
 fi
fi
git -C "$SRE_MAIN" cat-file -e "$SRE_SHA^{commit}"
git -C "$SRE_MAIN" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$SRE_SHA"
git -C "$SRE_MAIN" cat-file -e "$SRE_SHA:ultralytics-main/ultralytics/nn/modules/sre.py"
if [[ -e "$SRE_WORKTREE" ]]; then
 [[ -f "$SRE_WORKTREE/.git" ]] || { echo 'Existing target is not linked worktree; preserved'; exit 1; }
 [[ "$(git -C "$SRE_WORKTREE" rev-parse --show-toplevel)" == "$SRE_WORKTREE" ]]
 [[ "$(git -C "$SRE_WORKTREE" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$SRE_MAIN" rev-parse --path-format=absolute --git-common-dir)" ]]
 [[ "$(git -C "$SRE_WORKTREE" rev-parse HEAD)" == "$SRE_SHA" ]] || { echo 'Existing target SHA differs; use a new worktree directory'; exit 1; }
 [[ -z "$(git -C "$SRE_WORKTREE" status --porcelain --untracked-files=no)" ]] || { echo 'Modified source preserved'; exit 1; }
else
 git -C "$SRE_MAIN" worktree add --detach "$SRE_WORKTREE" "$SRE_SHA"
fi
[[ "$(git -C "$SRE_WORKTREE" rev-parse HEAD)" == "$SRE_SHA" ]]
[[ "$(git -C "$SRE_MAIN" rev-parse HEAD)" == "$SRE_OLD_HEAD" ]]
[[ "$(git -C "$SRE_MAIN" status --porcelain)" == "$SRE_OLD_STATUS" ]]
mkdir -p "$SRE_WORKTREE/outputs/sre"
SRE_ENV="$SRE_WORKTREE/outputs/sre/environment-$SRE_SHA.sh"
SRE_ENV_TEMP="$(mktemp "$SRE_WORKTREE/outputs/sre/.environment.XXXXXX")"
trap 'rm -f -- "$SRE_ENV_TEMP"' EXIT
{
 printf 'source /root/miniconda3/etc/profile.d/conda.sh\nconda activate rtdetr\n'
 printf 'export SRE_MAIN=%q\nexport SRE_SHA=%q\nexport SRE_WORKTREE=%q\n' "$SRE_MAIN" "$SRE_SHA" "$SRE_WORKTREE"
 printf 'export PYTHONPATH=%q\nexport YOLO_AUTOINSTALL=false\nexport PYTHONUNBUFFERED=1\n' "$SRE_WORKTREE/ultralytics-main"
 printf 'cd %q\n' "$SRE_WORKTREE"
 printf '[[ "$(git rev-parse HEAD)" == "$SRE_SHA" ]] || { echo "Environment SHA mismatch"; return 1; }\n'
} > "$SRE_ENV_TEMP"
if [[ -e "$SRE_ENV" ]]; then
 cmp "$SRE_ENV_TEMP" "$SRE_ENV" || { echo 'Existing environment differs; preserved'; exit 1; }
else
 (set -C; cat "$SRE_ENV_TEMP" > "$SRE_ENV")
fi
printf 'SRE ready: %s\nHEAD: %s\nEnvironment: %s\nFormal training NOT_STARTED\n' "$SRE_WORKTREE" "$SRE_SHA" "$SRE_ENV"
