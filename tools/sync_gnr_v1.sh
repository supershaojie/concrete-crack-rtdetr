#!/usr/bin/env bash
# Bounded synchronization of the documented, independent GNR worktree.
set -euo pipefail
SHA=${1:?Usage: bash tools/sync_gnr_v1.sh FULL_DELIVERED_SHA}
PY=/root/miniconda3/envs/rtdetr/bin/python
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WT=/root/autodl-tmp/projects/Crack_RTDETR-gnr_v1
"$PY" - "$SHA" "$MAIN" "$WT" <<'PYCODE'
import json, os, pathlib, re, subprocess, sys
import psutil

sha, main, wt = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
branch = 'exp-rtdetr-r18-lite-gnr-v1'
remote = 'https://github.com/supershaojie/concrete-crack-rtdetr.git'
if not re.fullmatch(r'[0-9a-f]{40}', sha):
    raise SystemExit('A real full 40-character delivery SHA is required')
def run(*args, cwd=main):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True, timeout=120).strip()
if run('remote', 'get-url', 'origin') != remote:
    raise SystemExit('Unexpected origin; no repository URLs are guessed')
for attempt in range(3):
    try:
        run('fetch', '--no-tags', 'origin', branch + ':refs/remotes/origin/' + branch)
        break
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f'Fetch {attempt+1}/3 failed: {error}', flush=True)
else:
    raise SystemExit('Bounded fetch failed; no worktree was changed')
run('cat-file', '-e', sha + '^{commit}')
run('merge-base', '--is-ancestor', sha, 'refs/remotes/origin/' + branch)
if wt.exists():
    if not (wt / '.git').is_file():
        raise SystemExit(f'Existing directory is not a linked worktree: {wt}')
    if pathlib.Path(run('rev-parse', '--show-toplevel', cwd=wt)).resolve() != wt.resolve():
        raise SystemExit('Unexpected worktree root')
    if run('branch', '--show-current', cwd=wt) != branch:
        raise SystemExit('Existing worktree branch is incompatible; preserved')
    if run('status', '--porcelain', '--untracked-files=normal', cwd=wt):
        raise SystemExit('Existing worktree has unreviewed changes; preserved')
    for process in psutil.process_iter():
        if process.pid == os.getpid():
            continue
        try:
            cmd = process.cmdline() or []
            if not any('python' in pathlib.Path(s).name.lower() for s in cmd[:1]):
                continue
            if pathlib.Path(process.cwd()).resolve() == wt.resolve() and any('gnr_v1' in s or 'train' in s for s in cmd[1:]):
                raise SystemExit(f'Worktree process still active, PID {process.pid}: {cmd}')
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    current = run('rev-parse', 'HEAD', cwd=wt)
    if current != sha:
        run('merge-base', '--is-ancestor', current, sha, cwd=wt)
        run('merge', '--ff-only', sha, cwd=wt)
else:
    exists = subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+branch],cwd=main).returncode == 0
    if exists:
        if run('rev-parse', branch) != sha:
            raise SystemExit('Existing local GNR branch differs from requested delivery; preserved')
        run('worktree', 'add', str(wt), branch)
    else:
        run('worktree', 'add', '-b', branch, str(wt), sha)
if run('rev-parse', 'HEAD', cwd=wt) != sha:
    raise SystemExit('Synchronized HEAD mismatch')
print(json.dumps({'status':'SYNCED','sha':sha,'branch':branch,'worktree':str(wt),'origin':remote},indent=2))
PYCODE
