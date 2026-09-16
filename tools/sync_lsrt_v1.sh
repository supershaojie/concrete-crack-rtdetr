#!/usr/bin/env bash
# Object first; bounded branch fetch; never switch/reset/clean an existing checkout.
set -Eeuo pipefail
if [[ ${1:-} == --help ]]; then
  printf '%s\n' 'Usage: bash sync_lsrt_v1.sh FULL_DELIVERY_SHA [MAIN_REPO] [NEW_WORKTREE] [FALLBACK_BUNDLE]'
  exit 0
fi
[[ $# -ge 1 && $# -le 4 ]] || { echo 'Supply full delivered SHA; use --help'; exit 2; }
LSRT_SHA="$1"
LSRT_MAIN="$(realpath -- "${2:-/root/autodl-tmp/projects/Crack_RTDETR}")"
LSRT_WORKTREE="$(realpath -m -- "${3:-${LSRT_MAIN}-lsrt-v1}")"
LSRT_BUNDLE="${4:-}"
LSRT_BRANCH=exp-rtdetr-r18-lite-lsrt-v1
LSRT_BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
[[ "$LSRT_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Require full 40-character lowercase SHA'; exit 2; }
[[ "$LSRT_MAIN" != "$LSRT_WORKTREE" ]] || { echo 'Independent worktree required'; exit 1; }
LSRT_REMOTE="$(git -C "$LSRT_MAIN" remote get-url origin)"
case "$LSRT_REMOTE" in
 https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
 *) echo "Unexpected origin preserved: $LSRT_REMOTE"; exit 1 ;;
esac
LSRT_MAIN_HEAD="$(git -C "$LSRT_MAIN" rev-parse HEAD)"
LSRT_MAIN_STATUS="$(git -C "$LSRT_MAIN" status --porcelain)"
LSRT_IMPORT_REF="refs/lsrt-deliveries/$LSRT_SHA"
LSRT_FETCH_REF="refs/lsrt-branch-fetches/$LSRT_SHA"
if ! git -C "$LSRT_MAIN" cat-file -e "$LSRT_SHA^{commit}" 2>/dev/null; then
  for LSRT_ATTEMPT in 1 2 3 4 5; do
    printf 'Fetch LSRT branch, bounded attempt %s/5\n' "$LSRT_ATTEMPT"
    if GIT_TERMINAL_PROMPT=0 timeout 120s git -C "$LSRT_MAIN" \
      -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
      fetch --progress --no-tags --no-write-fetch-head origin \
      "refs/heads/$LSRT_BRANCH:$LSRT_FETCH_REF"; then
      if git -C "$LSRT_MAIN" cat-file -e "$LSRT_SHA^{commit}" 2>/dev/null; then break; fi
    fi
  done
fi
if ! git -C "$LSRT_MAIN" cat-file -e "$LSRT_SHA^{commit}" 2>/dev/null; then
  [[ -n "$LSRT_BUNDLE" && -f "$LSRT_BUNDLE" ]] || { echo 'Network failed after at most 5 attempts. Supply the small delivered Git bundle as fourth argument.'; exit 1; }
  git -C "$LSRT_MAIN" cat-file -e "$LSRT_BASE^{commit}"
  git -C "$LSRT_MAIN" bundle verify "$LSRT_BUNDLE"
  LSRT_BUNDLE_REF="$(git bundle list-heads "$LSRT_BUNDLE" | awk -v sha="$LSRT_SHA" '$1==sha {print $2; exit}')"
  [[ -n "$LSRT_BUNDLE_REF" ]] || { echo 'Bundle does not advertise the fixed delivered SHA'; exit 1; }
  git -C "$LSRT_MAIN" fetch --no-tags --no-write-fetch-head "$LSRT_BUNDLE" "$LSRT_BUNDLE_REF:$LSRT_IMPORT_REF"
fi
git -C "$LSRT_MAIN" cat-file -e "$LSRT_SHA^{commit}"
git -C "$LSRT_MAIN" merge-base --is-ancestor "$LSRT_BASE" "$LSRT_SHA"
for LSRT_FILE in tools/init_lsrt_v1.py tools/train_lsrt_v1.py tools/preflight_lsrt_v1.py \
                 tools/lsrt_v1_results.py ultralytics-main/ultralytics/nn/modules/lsrt.py; do
  git -C "$LSRT_MAIN" cat-file -e "$LSRT_SHA:$LSRT_FILE"
done
if [[ -e "$LSRT_WORKTREE" ]]; then
  [[ -f "$LSRT_WORKTREE/.git" ]] || { echo 'Existing non-worktree target preserved'; exit 1; }
  [[ "$(git -C "$LSRT_WORKTREE" rev-parse --show-toplevel)" == "$LSRT_WORKTREE" ]] || { echo 'Different target root preserved'; exit 1; }
  [[ "$(git -C "$LSRT_WORKTREE" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$LSRT_MAIN" rev-parse --path-format=absolute --git-common-dir)" ]] || { echo 'Other repository target preserved'; exit 1; }
  [[ -z "$(git -C "$LSRT_WORKTREE" status --porcelain --untracked-files=no)" ]] || { echo 'Modified worktree preserved'; exit 1; }
  [[ "$(git -C "$LSRT_WORKTREE" rev-parse HEAD)" == "$LSRT_SHA" ]] || { echo 'Different worktree HEAD preserved; choose a new path'; exit 1; }
else
  git -C "$LSRT_MAIN" worktree add --detach "$LSRT_WORKTREE" "$LSRT_SHA"
fi
[[ "$(git -C "$LSRT_WORKTREE" rev-parse HEAD)" == "$LSRT_SHA" ]]
[[ "$(git -C "$LSRT_MAIN" rev-parse HEAD)" == "$LSRT_MAIN_HEAD" ]]
[[ "$(git -C "$LSRT_MAIN" status --porcelain)" == "$LSRT_MAIN_STATUS" ]]
# Create weights/outputs only after the independent worktree and SHA exist.
mkdir -p -- "$LSRT_WORKTREE/weights" "$LSRT_WORKTREE/outputs/lsrt_v1"
LSRT_PYTHON="${LSRT_V1_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}"
[[ -x "$LSRT_PYTHON" ]] || { echo 'Set LSRT_V1_PYTHON to existing rtdetr Python executable'; exit 1; }
"$LSRT_PYTHON" - "$LSRT_WORKTREE" "$LSRT_MAIN" "$LSRT_SHA" "$LSRT_BRANCH" "$LSRT_BASE" <<'PY'
import json, pathlib, sys
root, main = map(pathlib.Path, sys.argv[1:3])
record = dict(commit=sys.argv[3], branch=sys.argv[4], base=sys.argv[5], worktree=str(root), main=str(main))
path = root/'outputs/lsrt_v1/delivery.json'
if path.exists():
    assert json.loads(path.read_text()) == record, 'Existing delivery record differs; preserved'
else:
    with path.open('x') as stream:
        json.dump(record, stream, indent=2)
print(json.dumps(record, indent=2))
PY
printf 'LSRT worktree ready: %s\nFixed SHA: %s\nTraining NOT_STARTED; final test NOT_RUN.\n' "$LSRT_WORKTREE" "$LSRT_SHA"
