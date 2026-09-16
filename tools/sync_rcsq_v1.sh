#!/usr/bin/env bash
# Import a fixed delivery object and preserve all pre-existing worktrees.
set -Eeuo pipefail
if [[ ${1:-} == --help ]]; then
  printf 'Usage: bash tools/sync_rcsq_v1.sh FULL_SHA [MAIN_REPO] [NEW_WORKTREE] [FALLBACK_BUNDLE]\n'
  exit 0
fi
[[ $# -ge 1 && $# -le 4 ]] || { echo 'Use --help for arguments' >&2; exit 2; }
RCSQ_SHA=$1
RCSQ_BRANCH=exp-rtdetr-r18-lite-rcsq-v1
RCSQ_BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
RCSQ_MAIN=$(realpath -- "${2:-/root/autodl-tmp/projects/Crack_RTDETR}")
RCSQ_WORKTREE=$(realpath -m -- "${3:-${RCSQ_MAIN}-rcsq-v1}")
RCSQ_BUNDLE=${4:-}
[[ $RCSQ_SHA =~ ^[0-9a-f]{40}$ ]] || { echo 'Full lowercase 40-character SHA required' >&2; exit 2; }
[[ $RCSQ_MAIN != "$RCSQ_WORKTREE" ]] || { echo 'Separate worktree required' >&2; exit 1; }
RCSQ_REMOTE=$(git -C "$RCSQ_MAIN" remote get-url origin)
case "$RCSQ_REMOTE" in
  https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo "Unexpected origin: $RCSQ_REMOTE" >&2; exit 1 ;;
esac
RCSQ_MAIN_HEAD=$(git -C "$RCSQ_MAIN" rev-parse HEAD)
RCSQ_MAIN_STATUS=$(git -C "$RCSQ_MAIN" status --porcelain)
RCSQ_LOG=$(mktemp "${TMPDIR:-/tmp}/rcsq-sync.XXXXXXXX.log")
printf 'Fixed delivery %s; base %s; branch %s\n' "$RCSQ_SHA" "$RCSQ_BASE" "$RCSQ_BRANCH" >> "$RCSQ_LOG"
printf 'Sync log: %s\n' "$RCSQ_LOG"

# Local objects first: an available fixed commit needs no network request.
if ! git -C "$RCSQ_MAIN" cat-file -e "$RCSQ_SHA^{commit}" 2>/dev/null; then
  command -v timeout >/dev/null || { echo 'GNU timeout is required for bounded fetch' >&2; exit 1; }
  for RCSQ_ATTEMPT in 1 2 3 4 5; do
    printf 'Fetch attempt %s/5\n' "$RCSQ_ATTEMPT" | tee -a "$RCSQ_LOG"
    if timeout --kill-after=5s 120s env GIT_TERMINAL_PROMPT=0 \
      git -C "$RCSQ_MAIN" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
      fetch --progress --no-tags --no-write-fetch-head origin \
      "refs/heads/$RCSQ_BRANCH:refs/rcsq-sync/$RCSQ_SHA" 2>&1 | tee -a "$RCSQ_LOG"; then
      printf 'Fetch exit 0\n' >> "$RCSQ_LOG"
    else
      RCSQ_FETCH_STATUS=$?
      printf 'Fetch failed exit=%s attempt=%s\n' "$RCSQ_FETCH_STATUS" "$RCSQ_ATTEMPT" | tee -a "$RCSQ_LOG"
    fi
    if git -C "$RCSQ_MAIN" cat-file -e "$RCSQ_SHA^{commit}" 2>/dev/null; then break; fi
  done
fi

if ! git -C "$RCSQ_MAIN" cat-file -e "$RCSQ_SHA^{commit}" 2>/dev/null; then
  [[ -n $RCSQ_BUNDLE && -f $RCSQ_BUNDLE ]] || { echo "Target unavailable; provide the delivery bundle as argument 4. Log: $RCSQ_LOG" >&2; exit 1; }
  git -C "$RCSQ_MAIN" cat-file -e "$RCSQ_BASE^{commit}" || { echo 'Bundle prerequisite base missing; obtain the verified base separately' >&2; exit 1; }
  git -C "$RCSQ_MAIN" bundle verify "$RCSQ_BUNDLE" 2>&1 | tee -a "$RCSQ_LOG"
  git -C "$RCSQ_MAIN" bundle list-heads "$RCSQ_BUNDLE" | awk -v sha="$RCSQ_SHA" '$1==sha{found=1} END{exit !found}' || { echo 'Bundle does not advertise the requested fixed SHA' >&2; exit 1; }
  git -C "$RCSQ_MAIN" bundle unbundle "$RCSQ_BUNDLE" 2>&1 | tee -a "$RCSQ_LOG"
fi
git -C "$RCSQ_MAIN" cat-file -e "$RCSQ_SHA^{commit}"
git -C "$RCSQ_MAIN" merge-base --is-ancestor "$RCSQ_BASE" "$RCSQ_SHA"
for RCSQ_FILE in tools/init_rcsq_v1.py tools/train_rcsq_v1.py tools/preflight_rcsq_v1.py \
  ultralytics-main/ultralytics/nn/modules/rcs_q.py \
  ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-rcsq.yaml; do
  git -C "$RCSQ_MAIN" cat-file -e "$RCSQ_SHA:$RCSQ_FILE"
done

if [[ -e $RCSQ_WORKTREE ]]; then
  [[ -f $RCSQ_WORKTREE/.git ]] || { echo 'Existing target is not a linked worktree; preserved' >&2; exit 1; }
  [[ $(git -C "$RCSQ_WORKTREE" rev-parse --show-toplevel) == "$RCSQ_WORKTREE" ]] || { echo 'Wrong target root; preserved' >&2; exit 1; }
  [[ $(git -C "$RCSQ_WORKTREE" rev-parse --path-format=absolute --git-common-dir) == "$(git -C "$RCSQ_MAIN" rev-parse --path-format=absolute --git-common-dir)" ]] || { echo 'Different repository; preserved' >&2; exit 1; }
  [[ -z $(git -C "$RCSQ_WORKTREE" status --porcelain) ]] || { echo 'Modified target preserved' >&2; exit 1; }
  [[ $(git -C "$RCSQ_WORKTREE" rev-parse HEAD) == "$RCSQ_SHA" ]] || { echo 'Different target SHA preserved; choose a new directory' >&2; exit 1; }
else
  git -C "$RCSQ_MAIN" worktree add --detach "$RCSQ_WORKTREE" "$RCSQ_SHA"
fi
[[ $(git -C "$RCSQ_WORKTREE" rev-parse HEAD) == "$RCSQ_SHA" ]]
[[ $(git -C "$RCSQ_MAIN" rev-parse HEAD) == "$RCSQ_MAIN_HEAD" ]]
[[ $(git -C "$RCSQ_MAIN" status --porcelain) == "$RCSQ_MAIN_STATUS" ]]
# Only after the target checkout and exact HEAD have been verified.
mkdir -p -- "$RCSQ_WORKTREE/weights" "$RCSQ_WORKTREE/outputs/rcsq_v1/sync"
cp -n -- "$RCSQ_LOG" "$RCSQ_WORKTREE/outputs/rcsq_v1/sync/$(basename "$RCSQ_LOG")"
printf 'READY: %s at %s. No initialization, training or evaluation started.\n' "$RCSQ_WORKTREE" "$RCSQ_SHA"
