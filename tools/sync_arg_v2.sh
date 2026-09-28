#!/usr/bin/env bash
# Read-only main checkout; exact commit, bounded fetch, no reset/clean/force.
set -Eeuo pipefail
SHA=${1:?Usage: bash sync_arg_v2.sh FULL_40_CHARACTER_SHA}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected complete lowercase commit SHA' >&2; exit 2; }
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-arg_v2
BRANCH=exp-rtdetr-r18-lite-arg-v2
ORIGIN=https://github.com/supershaojie/concrete-crack-rtdetr.git
BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
START=ef9cb7e05e5557f7dd06c95cf2361998a284adc9
[[ $(git -C "$MAIN" remote get-url origin) == "$ORIGIN" ]] || { echo 'Unexpected origin' >&2; exit 2; }
timeout 120s git -C "$MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin "$BRANCH"
git -C "$MAIN" cat-file -e "$SHA^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$BASE" "$SHA"
git -C "$MAIN" merge-base --is-ancestor "$START" "$SHA"
git -C "$MAIN" merge-base --is-ancestor "$SHA" FETCH_HEAD
if [[ -e "$WORK" ]]; then
    [[ -f "$WORK/.git" ]] || { echo 'Existing path is not a linked worktree; preserved' >&2; exit 2; }
    a=$(cd "$WORK" && realpath "$(git rev-parse --git-common-dir)")
    b=$(cd "$MAIN" && realpath "$(git rev-parse --git-common-dir)")
    [[ "$a" == "$b" ]] || { echo 'Different repository; preserved' >&2; exit 2; }
    [[ $(git -C "$WORK" rev-parse HEAD) == "$SHA" ]] || { echo 'Existing experiment has another SHA; preserved. Inspect status and history.' >&2; exit 2; }
    [[ -z $(git -C "$WORK" status --porcelain) ]] || { echo 'Worktree changes preserved; sync refused' >&2; exit 2; }
else
    if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
        [[ $(git -C "$MAIN" rev-parse "$BRANCH") == "$SHA" ]] || { echo 'Existing branch differs; preserved' >&2; exit 2; }
        git -C "$MAIN" worktree add "$WORK" "$BRANCH"
    else
        git -C "$MAIN" worktree add -b "$BRANCH" "$WORK" "$SHA"
    fi
fi
[[ $(git -C "$WORK" rev-parse HEAD) == "$SHA" ]]
mkdir -p "$WORK/outputs/arg_v2"
/root/miniconda3/envs/rtdetr/bin/python - "$WORK" "$MAIN" "$SHA" <<'PY'
import json, pathlib, sys
work, main, sha = sys.argv[1:]
p = pathlib.Path(work) / 'outputs/arg_v2/sync.json'
value = dict(worktree=work, main=main, commit=sha, branch='exp-rtdetr-r18-lite-arg-v2')
if p.exists():
    assert json.loads(p.read_text()) == value, 'Existing sync identity differs; preserved'
else:
    with p.open('x') as f:
        json.dump(value, f, indent=2)
print(json.dumps(value, indent=2))
PY
echo 'Sync verified. Next: bash /root/autodl-tmp/projects/Crack_RTDETR-arg_v2/tools/arg_v2.sh prepare'
