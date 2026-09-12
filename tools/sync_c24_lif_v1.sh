#!/usr/bin/env bash
# Fixed detached worktree; no checkout/reset/clean in an existing checkout.
set -Eeuo pipefail
C24_SHA="${1:?Supply delivered full 40-character SHA}"
C24_MAIN="$(realpath -- "${2:-/root/autodl-tmp/projects/Crack_RTDETR}")"
C24_WORKTREE="$(realpath -m -- "${3:-${C24_MAIN}-c24-lif-v1}")"
C24_BRANCH=codex/rtdetr-c24-lif-v1
[[ "$C24_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full SHA required'; exit 2; }
[[ "$C24_WORKTREE" != "$C24_MAIN" ]] || { echo 'Separate worktree required'; exit 1; }
C24_REMOTE="$(git -C "$C24_MAIN" remote get-url origin)"
case "$C24_REMOTE" in
 https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
 *) echo 'Unexpected origin; preserved'; exit 1 ;;
esac
C24_MAIN_HEAD="$(git -C "$C24_MAIN" rev-parse HEAD)"
C24_MAIN_STATUS="$(git -C "$C24_MAIN" status --porcelain --untracked-files=no)"
# Read a branch-specific ref, never shared FETCH_HEAD (other tasks may fetch concurrently).
git -C "$C24_MAIN" fetch origin "refs/heads/$C24_BRANCH:refs/remotes/origin/$C24_BRANCH"
C24_REMOTE_HEAD="$(git -C "$C24_MAIN" rev-parse "refs/remotes/origin/$C24_BRANCH")"
git -C "$C24_MAIN" cat-file -e "$C24_SHA^{commit}"
git -C "$C24_MAIN" merge-base --is-ancestor "$C24_SHA" "$C24_REMOTE_HEAD"
git -C "$C24_MAIN" merge-base --is-ancestor 0e95bbade3558b0d2b77c5531483c60810391d88 "$C24_SHA"
for C24_FILE in tools/init_c24_lif_v1.py tools/train_c24_lif_v1.py tools/check_c24_lif_v1.py tools/autodl_c24_lif_v1.sh ultralytics-main/ultralytics/nn/modules/scca_aifi.py; do
 git -C "$C24_MAIN" cat-file -e "$C24_SHA:$C24_FILE"
done
if [[ -e "$C24_WORKTREE" ]]; then
 [[ -f "$C24_WORKTREE/.git" ]] || { echo 'Target is not linked worktree; preserved'; exit 1; }
 [[ "$(git -C "$C24_WORKTREE" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$C24_MAIN" rev-parse --path-format=absolute --git-common-dir)" ]] || { echo 'Different repository; preserved'; exit 1; }
 [[ -z "$(git -C "$C24_WORKTREE" status --porcelain --untracked-files=no)" ]] || { echo 'Tracked dirty target preserved'; exit 1; }
 [[ "$(git -C "$C24_WORKTREE" rev-parse HEAD)" == "$C24_SHA" ]] || { echo 'Different SHA preserved; no update performed'; exit 1; }
 [[ -z "$(git -C "$C24_WORKTREE" symbolic-ref -q HEAD || true)" ]] || { echo 'Target must be detached; preserved'; exit 1; }
else
 git -C "$C24_MAIN" worktree add --detach "$C24_WORKTREE" "$C24_SHA"
fi
[[ "$(git -C "$C24_MAIN" rev-parse HEAD)" == "$C24_MAIN_HEAD" ]]
[[ "$(git -C "$C24_MAIN" status --porcelain --untracked-files=no)" == "$C24_MAIN_STATUS" ]]
C24_PYTHON="${C24_LIF_V1_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}"
[[ -x "$C24_PYTHON" ]] || { echo 'Set C24_LIF_V1_PYTHON to existing environment Python'; exit 1; }
"$C24_PYTHON" - "$C24_WORKTREE" "$C24_MAIN" "$C24_SHA" <<'PY'
import json,pathlib,sys
root,main=map(pathlib.Path,sys.argv[1:3]);sys.path.insert(0,str(root/'tools'))
from c24_lif_v1_common import fingerprint,write_json
record=dict(commit=sys.argv[3],branch='codex/rtdetr-c24-lif-v1',main=str(main),worktree=str(root),fingerprint=fingerprint())
path=root/'outputs/c24_lif_v1_delivery.json'
if path.exists():
    assert json.loads(path.read_text())==record,'Existing delivery record differs; preserved'
else:write_json(path,record)
print('Fixed SHA verified:',sys.argv[3],'\nWorktree:',root,'\nNo training started.')
PY
