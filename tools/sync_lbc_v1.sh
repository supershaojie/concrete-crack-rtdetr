#!/usr/bin/env bash
# Bootstrap is self-contained: it never calls code in a stale experiment checkout.
set -Eeuo pipefail
lbc_python="${LBC_PYTHON:-/root/miniconda3/envs/rtdetr/bin/python}"
"$lbc_python" - "$@" <<'PY'
import datetime, json, os, pathlib, re, subprocess, sys

def need(ok, message):
    if not ok: raise SystemExit(message)

need(2 <= len(sys.argv) <= 4, 'Usage: sync_lbc_v1.sh FULL_SHA [MAIN] [WORKTREE]')
sha = sys.argv[1]
need(re.fullmatch(r'[0-9a-f]{40}', sha), 'Full lowercase SHA required')
main = pathlib.Path(sys.argv[2] if len(sys.argv)>2 else os.environ.get('LBC_MAIN','/root/autodl-tmp/projects/Crack_RTDETR')).resolve()
target = pathlib.Path(sys.argv[3] if len(sys.argv)>3 else str(main)+'-lbc_v1').resolve()
need(main != target and main not in target.parents, 'Use a separate sibling server worktree')
branch = 'exp-rtdetr-r18-lite-lbc-v1'

def git(where, *args):
    return subprocess.check_output(['git','-C',str(where),*args], text=True).strip()

origin = git(main,'config','--get','remote.origin.url')
need(origin in ('https://github.com/supershaojie/concrete-crack-rtdetr.git',
                'https://github.com/supershaojie/concrete-crack-rtdetr',
                'git@github.com:supershaojie/concrete-crack-rtdetr.git',
                'ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git'), 'Wrong origin; preserved')
before_head = git(main,'rev-parse','HEAD')
before_status = git(main,'status','--porcelain')
registry = git(main,'worktree','list','--porcelain')
other_heads = {}
for block in registry.split('\n\n'):
    rows = dict(line.split(' ',1) for line in block.splitlines() if ' ' in line)
    if 'worktree' in rows and pathlib.Path(rows['worktree']).resolve() != target:
        other_heads[rows['worktree']] = rows.get('HEAD')
git(main,'fetch','--no-write-fetch-head','origin',f'refs/heads/{branch}:refs/remotes/origin/{branch}')
remote = git(main,'rev-parse',f'refs/remotes/origin/{branch}')
need(remote == sha, 'Remote branch no longer equals delivered SHA; inspect the newer delivery')
git(main,'merge-base','--is-ancestor','a0459d6a652cb702699087c88fa39a3e4c4087ec',sha)
for filename in ('tools/lbc_v1.py','tools/lbc_v1.sh','ultralytics-main/ultralytics/models/rtdetr/lbc.py'):
    git(main,'cat-file','-e',sha+':'+filename)
if target.exists():
    need((target/'.git').is_file(), 'Target is not a linked worktree; preserved')
    need(pathlib.Path(git(target,'rev-parse','--show-toplevel')).resolve() == target, 'Wrong target root')
    need(git(main,'rev-parse','--path-format=absolute','--git-common-dir') ==
         git(target,'rev-parse','--path-format=absolute','--git-common-dir'), 'Different repository; preserved')
    need(not git(target,'status','--porcelain','--untracked-files=no'), 'Tracked modifications preserved')
    old = git(target,'rev-parse','HEAD')
    old_record = target/'outputs/lbc_v1_delivery.json'
    head_branch = git(target,'branch','--show-current')
    need(head_branch == branch or old_record.exists(), 'Unrecognized existing experiment; preserved')
    if old_record.exists():
        saved = json.loads(old_record.read_text())
        need(saved['branch'] == branch and saved['commit'] == old, 'Existing delivery identity differs; preserved')
    git(target,'merge-base','--is-ancestor',old,sha)
    if old != sha:
        # Git refuses conflicting untracked AND ignored paths. No force/reset/clean.
        git(target,'checkout','--no-overwrite-ignore','--detach',sha)
else:
    git(main,'worktree','add','--detach',str(target),sha)
need(git(target,'rev-parse','HEAD') == sha, 'Target update failed')
need(git(main,'rev-parse','HEAD') == before_head and git(main,'status','--porcelain') == before_status, 'Main checkout changed')
for path, head in other_heads.items():
    need(git(path,'rev-parse','HEAD') == head, 'Another worktree HEAD changed')
record = dict(commit=sha,branch=branch,main=str(main),worktree=str(target),origin=origin,verified_remote_head=remote)
out = target/'outputs/lbc_v1_delivery.json';out.parent.mkdir(exist_ok=True)
if out.exists():
    if json.loads(out.read_text()) != record:
        history=out.parent/'lbc_v1_sync_history';history.mkdir(exist_ok=True)
        out.rename(history/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json'))
if not out.exists():
    with out.open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record,indent=2))
print('Synced; formal training NOT started.')
PY
