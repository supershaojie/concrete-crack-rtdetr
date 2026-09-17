#!/usr/bin/env bash
# Run inside a child bash, never source into the user's interactive shell.
set -Eeuo pipefail
if [[ ${1:-} == --help || $# -lt 1 || $# -gt 2 ]]; then
  printf '%s\n' 'Usage: bash tools/sync_csr_p3.sh FULL_HEAD_SHA [VERIFIED_INCREMENTAL_BUNDLE]'
  [[ ${1:-} == --help ]] && exit 0 || exit 2
fi
SHA=$1
[[ $SHA =~ ^[0-9a-f]{40}$ ]] || { echo 'A full 40-character SHA is required' >&2; exit 2; }
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-csr-p3-v1
BRANCH=exp-rtdetr-r18-lite-csr-p3-v1
BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
cd "$MAIN"
REMOTE=$(git remote get-url origin)
case "$REMOTE" in
  https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git) ;;
  *) echo "Unexpected origin: $REMOTE" >&2; exit 1 ;;
esac
if ! git cat-file -e "$SHA^{commit}" 2>/dev/null; then
  for attempt in 1 2 3 4 5; do
    printf 'Fetch attempt %s/5\n' "$attempt"
    if timeout 120s env GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 \
        -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags origin \
        "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"; then
      git cat-file -e "$SHA^{commit}" 2>/dev/null && break
    fi
  done
fi
if ! git cat-file -e "$SHA^{commit}" 2>/dev/null; then
  [[ $# == 2 && -f $2 ]] || { echo 'Object missing after bounded fetch; supply the verified incremental bundle.' >&2; exit 1; }
  git bundle verify "$2"
  git fetch --no-tags "$2" "refs/heads/$BRANCH:refs/remotes/csr-bundle/$BRANCH"
fi
git cat-file -e "$SHA^{commit}"
git merge-base --is-ancestor "$BASE" "$SHA"
if [[ -e "$WORK" ]]; then
  [[ -f "$WORK/.git" ]] || { echo 'Existing directory is not a linked worktree; preserved.' >&2; exit 1; }
  [[ $(git -C "$WORK" rev-parse --path-format=absolute --git-common-dir) == $(git -C "$MAIN" rev-parse --path-format=absolute --git-common-dir) ]] || { echo 'Existing worktree belongs to another repository; preserved.' >&2; exit 1; }
  [[ $(git -C "$WORK" rev-parse --show-toplevel) == "$WORK" ]]
  [[ $(git -C "$WORK" rev-parse HEAD) == "$SHA" ]] || { echo 'Existing worktree HEAD differs; preserved without reset.' >&2; exit 1; }
  [[ -z $(git -C "$WORK" status --porcelain --untracked-files=no) ]] || { echo 'Existing tracked changes preserved.' >&2; exit 1; }
else
  git worktree add --detach "$WORK" "$SHA"
fi
[[ $(git -C "$WORK" rev-parse HEAD) == "$SHA" ]]
mkdir -p "$WORK/outputs/csr_p3"
ENV_FILE="$WORK/outputs/csr_p3/environment.sh"
EXPECTED=$(mktemp)
trap 'rm -f -- "$EXPECTED"' EXIT
cat > "$EXPECTED" <<EOF
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
export CSR_P3_MAIN='$MAIN'
export CSR_P3_WORK='$WORK'
export CSR_P3_SHA='$SHA'
export PYTHONPATH='$WORK/ultralytics-main'
export YOLO_AUTOINSTALL=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd '$WORK'
test "\$(git rev-parse HEAD)" = '$SHA'
EOF
if [[ -e "$ENV_FILE" ]]; then
  cmp --silent "$EXPECTED" "$ENV_FILE" || { echo 'Existing environment file differs; preserved.' >&2; exit 1; }
else
  cp --no-clobber "$EXPECTED" "$ENV_FILE"
fi
printf 'Verified CSR-P3 HEAD: %s\nEnvironment: %s\n' "$SHA" "$ENV_FILE"
