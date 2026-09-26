#!/usr/bin/env bash
set -Eeuo pipefail
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-tcr_v1
BRANCH=exp-rtdetr-r18-lite-tcr-v1
BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
PYTHON=/root/miniconda3/envs/rtdetr/bin/python
SHA="${1:?Usage: sync_tcr_v1.sh FULL_40_CHARACTER_SHA}"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { printf '%s\n' 'A complete lowercase commit SHA is required'; exit 64; }
[[ -x "$PYTHON" ]] || { printf '%s\n' 'Existing rtdetr Python not found'; exit 1; }
remote="$(git -C "$MAIN" remote get-url origin)"
[[ "$remote" == https://github.com/supershaojie/concrete-crack-rtdetr.git || "$remote" == git@github.com:supershaojie/concrete-crack-rtdetr.git ]] || { printf '%s\n' 'Unexpected origin'; exit 1; }
if command -v tmux >/dev/null && tmux has-session -t tcr-v1-training 2>/dev/null; then
  printf '%s\n' 'Existing TCR tmux session protected; do not sync while dispatched or running'
  exit 1
fi
"$PYTHON" - "$WORK" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
    except (FileNotFoundError,PermissionError): continue
    if any(a in {str(root/'tools/tcr_v1.py'),str(root/'tools/tcr_v1.sh')} for a in argv) and any(a in argv for a in ('_worker','_dispatch')):
        raise SystemExit('Active TCR process protected: '+proc.name)
PY
git -C "$MAIN" fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
[[ "$(git -C "$MAIN" rev-parse "refs/remotes/origin/$BRANCH")" == "$SHA" ]] || { printf '%s\n' 'Remote branch differs from supplied delivery SHA; refusing drifting head'; exit 1; }
git -C "$MAIN" merge-base --is-ancestor "$BASE" "$SHA"
if [[ -e "$WORK" ]]; then
  [[ -f "$WORK/.git" ]] || { printf '%s\n' 'Existing non-worktree directory preserved'; exit 1; }
  "$PYTHON" - "$MAIN" "$WORK" <<'PY'
from pathlib import Path
import subprocess,sys
def common(root):
    value=subprocess.check_output(['git','-C',root,'rev-parse','--git-common-dir'],text=True).strip()
    return (Path(root)/value).resolve()
if common(sys.argv[1]) != common(sys.argv[2]): raise SystemExit('Existing worktree belongs to another repository; preserved')
PY
  [[ "$(git -C "$WORK" branch --show-current)" == "$BRANCH" ]] || { printf '%s\n' 'Existing worktree belongs to another branch'; exit 1; }
  [[ -z "$(git -C "$WORK" status --porcelain)" ]] || { printf '%s\n' 'Modified worktree preserved'; exit 1; }
  git -C "$WORK" merge-base --is-ancestor HEAD "$SHA"
  git -C "$WORK" merge --ff-only "$SHA"
else
  if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git -C "$MAIN" merge-base --is-ancestor "$BRANCH" "$SHA"
    git -C "$MAIN" worktree add "$WORK" "$BRANCH"
    git -C "$WORK" merge --ff-only "$SHA"
  else
    git -C "$MAIN" worktree add -b "$BRANCH" "$WORK" "$SHA"
  fi
fi
[[ "$(git -C "$WORK" rev-parse HEAD)" == "$SHA" ]]
"$PYTHON" - "$MAIN" "$WORK" "$BRANCH" "$SHA" <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,sys
main,work,branch,sha=sys.argv[1:]
out=Path(work)/'outputs/tcr_v1'
out.mkdir(parents=True,exist_ok=True)
record=dict(main=main,worktree=work,branch=branch,commit=sha,synced=datetime.now(timezone.utc).isoformat())
tmp=out/'delivery.json.tmp'
tmp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
tmp.replace(out/'delivery.json')
print(json.dumps(record,indent=2))
PY
env PYTHONPATH="$WORK/ultralytics-main" PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false "$PYTHON" -u "$WORK/tools/tcr_v1.py" commands
