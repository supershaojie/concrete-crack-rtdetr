#!/usr/bin/env bash
# Exact committed source only. No modifications to main/other worktrees/runs.
set -Eeuo pipefail
if [[ ${1:-} == --help ]]; then
  echo 'Usage: bash sync_geo_v1.sh FULL_40_CHARACTER_SHA [--bundle LOCAL_BUNDLE]'
  echo 'Safely create/reuse the fixed GEO worktree after a bounded git fetch.'
  exit 0
fi
SHA=${1:?Expected the complete delivered 40-character SHA}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected complete lowercase SHA' >&2; exit 2; }
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-geo_v1
BRANCH=exp-rtdetr-r18-lite-geo-v1
ORIGIN=https://github.com/supershaojie/concrete-crack-rtdetr.git
BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
[[ $(git -C "$MAIN" remote get-url origin) == "$ORIGIN" ]] || { echo 'Unexpected origin; preserved' >&2; exit 2; }
if [[ $# -gt 1 ]]; then
  [[ $# == 3 && $2 == --bundle && -f $3 ]] || { echo 'Expected --bundle with an existing local file' >&2; exit 2; }
  git -C "$MAIN" bundle verify "$3"
  timeout 120s git -C "$MAIN" fetch --no-tags "$3" "$BRANCH"
else
  timeout 120s git -C "$MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin "$BRANCH"
fi
git -C "$MAIN" cat-file -e "$SHA^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$BASE" "$SHA"
git -C "$MAIN" merge-base --is-ancestor "$SHA" FETCH_HEAD
if [[ -e "$WORK" ]]; then
  [[ -f "$WORK/.git" ]] || { echo 'Existing path is not a linked worktree; preserved' >&2; exit 2; }
  a=$(cd "$WORK" && realpath "$(git rev-parse --git-common-dir)")
  b=$(cd "$MAIN" && realpath "$(git rev-parse --git-common-dir)")
  [[ "$a" == "$b" ]] || { echo 'Different repository; preserved' >&2; exit 2; }
  [[ $(git -C "$WORK" rev-parse HEAD) == "$SHA" ]] || { echo 'Existing GEO SHA differs; inspect status/history; no overwrite' >&2; exit 2; }
  [[ -z $(git -C "$WORK" status --porcelain) ]] || { echo 'GEO worktree has changes; preserved' >&2; exit 2; }
else
  if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    [[ $(git -C "$MAIN" rev-parse "$BRANCH") == "$SHA" ]] || { echo 'Existing GEO branch differs; preserved' >&2; exit 2; }
    git -C "$MAIN" worktree add "$WORK" "$BRANCH"
  else
    git -C "$MAIN" worktree add -b "$BRANCH" "$WORK" "$SHA"
  fi
fi
[[ $(git -C "$WORK" rev-parse HEAD) == "$SHA" ]]
mkdir -p "$WORK/outputs/geo_v1"
/root/miniconda3/envs/rtdetr/bin/python - "$WORK" "$MAIN" "$SHA" <<'PY'
import json,pathlib,sys
work,main,sha=sys.argv[1:]
p=pathlib.Path(work)/'outputs/geo_v1/sync.json'
value=dict(worktree=work,main=main,commit=sha,branch='exp-rtdetr-r18-lite-geo-v1')
if p.exists():
    assert json.loads(p.read_text())==value,'Existing sync identity differs; preserved'
else:
    with p.open('x') as f: json.dump(value,f,indent=2)
print(json.dumps(value,indent=2))
PY
echo 'Sync verified. Run prepare, preflight and status separately before start.'
