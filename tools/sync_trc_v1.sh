#!/usr/bin/env bash
# Pinned, bounded, non-destructive delivery; this script never starts training.
set -Eeuo pipefail
[[ $# -ge 1 && $# -le 4 ]] || { echo 'Usage: sync_trc_v1.sh FULL_SHA [MAIN_REPO] [WORKTREE] [BUNDLE]'; exit 2; }
TRC_SHA="$1"
TRC_MAIN="$(realpath -- "${2:-/root/autodl-tmp/projects/Crack_RTDETR}")"
TRC_ROOT="$(realpath -m -- "${3:-${TRC_MAIN}-trc-v1}")"
TRC_BUNDLE="${4:-}"
TRC_BRANCH=exp-rtdetr-r18-lite-trc-v1
TRC_BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
[[ "$TRC_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Require the delivered full 40-character SHA'; exit 2; }
[[ "$TRC_ROOT" != "$TRC_MAIN" ]] || { echo 'A separate worktree is required'; exit 1; }
TRC_REMOTE="$(git -C "$TRC_MAIN" remote get-url origin)"
case "$TRC_REMOTE" in
 https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
 *) echo "Unexpected origin: $TRC_REMOTE"; exit 1 ;;
esac
TRC_MAIN_HEAD="$(git -C "$TRC_MAIN" rev-parse HEAD)"
TRC_MAIN_STATUS="$(git -C "$TRC_MAIN" status --porcelain)"
export GIT_TERMINAL_PROMPT=0
# Existing exact objects need no network. Each attempt has a hard 120-second limit.
if ! git -C "$TRC_MAIN" cat-file -e "$TRC_SHA^{commit}" 2>/dev/null; then
 for TRC_ATTEMPT in 1 2 3 4 5; do
  echo "Fetch $TRC_BRANCH attempt $TRC_ATTEMPT/5"
  if timeout 120s git -C "$TRC_MAIN" -c http.version=HTTP/1.1 \
    -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags \
    --no-write-fetch-head origin "refs/heads/$TRC_BRANCH:refs/remotes/origin/$TRC_BRANCH"; then
   git -C "$TRC_MAIN" cat-file -e "$TRC_SHA^{commit}" 2>/dev/null && break
  fi
 done
fi
if ! git -C "$TRC_MAIN" cat-file -e "$TRC_SHA^{commit}" 2>/dev/null; then
 [[ -n "$TRC_BUNDLE" && -f "$TRC_BUNDLE" ]] || { echo 'Exact commit unavailable. Upload the delivered small bundle and pass it as argument 4.'; exit 1; }
 git -C "$TRC_MAIN" bundle verify "$TRC_BUNDLE"
 [[ "$(git bundle list-heads "$TRC_BUNDLE" "refs/heads/$TRC_BRANCH" | cut -d' ' -f1)" == "$TRC_SHA" ]] || { echo 'Bundle branch does not identify the delivered SHA'; exit 1; }
 git -C "$TRC_MAIN" fetch --no-tags --no-write-fetch-head "$TRC_BUNDLE" \
  "refs/heads/$TRC_BRANCH:refs/codex-delivery/trc-v1/bundle-$TRC_SHA"
fi
git -C "$TRC_MAIN" cat-file -e "$TRC_SHA^{commit}"
git -C "$TRC_MAIN" merge-base --is-ancestor "$TRC_BASE" "$TRC_SHA"
for TRC_FILE in tools/init_trc_v1.py tools/preflight_trc_v1.py tools/train_trc_v1.py tools/test_trc_v1.py tools/pack_trc_v1_light.py ultralytics-main/ultralytics/nn/modules/trc_aifi.py docs/trc_v1/cbr_lif_args.yaml; do
 git -C "$TRC_MAIN" cat-file -e "$TRC_SHA:$TRC_FILE"
done
if [[ -e "$TRC_ROOT" ]]; then
 [[ -f "$TRC_ROOT/.git" ]] || { echo 'Existing target is not a linked worktree; preserved'; exit 1; }
 [[ "$(realpath -- "$(git -C "$TRC_ROOT" rev-parse --show-toplevel)")" == "$TRC_ROOT" ]] || { echo 'Wrong target root; preserved'; exit 1; }
 [[ "$(realpath -- "$(git -C "$TRC_ROOT" rev-parse --path-format=absolute --git-common-dir)")" == "$(realpath -- "$(git -C "$TRC_MAIN" rev-parse --path-format=absolute --git-common-dir)")" ]] || { echo 'Different repository; preserved'; exit 1; }
 [[ "$(git -C "$TRC_ROOT" rev-parse HEAD)" == "$TRC_SHA" ]] || { echo 'Different target SHA; preserved. Choose a new directory.'; exit 1; }
 [[ -z "$(git -C "$TRC_ROOT" status --porcelain --untracked-files=no)" ]] || { echo 'Modified target; preserved'; exit 1; }
else
 # No mkdir at this path before worktree add: Git owns creation of the checkout.
 git -C "$TRC_MAIN" worktree add --detach "$TRC_ROOT" "$TRC_SHA"
fi
[[ "$(git -C "$TRC_ROOT" rev-parse HEAD)" == "$TRC_SHA" ]]
[[ "$(git -C "$TRC_MAIN" rev-parse HEAD)" == "$TRC_MAIN_HEAD" ]]
[[ "$(git -C "$TRC_MAIN" status --porcelain)" == "$TRC_MAIN_STATUS" ]]
mkdir -p "$TRC_ROOT/weights" "$TRC_ROOT/outputs/trc_v1"
"${TRC_V1_PYTHON:-python3}" - "$TRC_ROOT" "$TRC_MAIN" "$TRC_SHA" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root, main = map(pathlib.Path, sys.argv[1:3])
files = subprocess.check_output(['git', 'ls-files', '-z', 'tools', 'ultralytics-main/ultralytics', 'configs', 'docs/trc_v1'], cwd=root).decode().strip('\0').split('\0')
record = dict(commit=sys.argv[3], branch='exp-rtdetr-r18-lite-trc-v1', main=str(main), worktree=str(root),
              files={p: hashlib.sha256((root / p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in files})
target = root / 'outputs/trc_v1_delivery.json'
if target.exists():
    if json.loads(target.read_text()) != record:
        raise SystemExit('Existing delivery record differs; preserved')
else:
    with target.open('x') as stream:
        json.dump(record, stream, indent=2)
print(json.dumps(dict(commit=record['commit'], worktree=str(root), source_files=len(files))))
PY
printf 'CBR + LIF + TRC worktree ready at %s, pinned %s. Formal training NOT_STARTED; test NOT_RUN.\n' "$TRC_ROOT" "$TRC_SHA"
