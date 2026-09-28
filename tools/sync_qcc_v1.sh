#!/usr/bin/env bash
# Read-only main checkout; exact commit, bounded fetch, no reset/clean/force.
set -Eeuo pipefail
SHA=${1:?Usage: bash sync_qcc_v1.sh FULL_40_CHARACTER_SHA}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected complete lowercase commit SHA' >&2; exit 2; }
MAIN=/root/autodl-tmp/projects/Crack_RTDETR
WORK=/root/autodl-tmp/projects/Crack_RTDETR-qcc_v1
BRANCH=exp-rtdetr-r18-lite-qcc-v1
ORIGIN=https://github.com/supershaojie/concrete-crack-rtdetr.git
BASE=a0459d6a652cb702699087c88fa39a3e4c4087ec
[[ $(git -C "$MAIN" remote get-url origin) == "$ORIGIN" ]] || { echo 'Unexpected origin' >&2; exit 2; }
timeout 120s git -C "$MAIN" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 fetch --no-tags origin "$BRANCH"
git -C "$MAIN" cat-file -e "$SHA^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$BASE" "$SHA"
git -C "$MAIN" merge-base --is-ancestor "$SHA" FETCH_HEAD
if [[ -e "$WORK" ]]; then
    [[ -f "$WORK/.git" ]] || { echo 'Existing path is not a linked worktree; preserved' >&2; exit 2; }
    a=$(cd "$WORK" && realpath "$(git rev-parse --git-common-dir)")
    b=$(cd "$MAIN" && realpath "$(git rev-parse --git-common-dir)")
    [[ "$a" == "$b" ]] || { echo 'Different repository; preserved' >&2; exit 2; }
    [[ -z $(git -C "$WORK" status --porcelain) ]] || { echo 'Worktree changes preserved; sync refused' >&2; exit 2; }
else
    if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git -C "$MAIN" merge-base --is-ancestor "$BRANCH" "$SHA" || { echo 'Existing branch is not an ancestor; preserved' >&2; exit 2; }
        git -C "$MAIN" worktree add "$WORK" "$BRANCH"
    else
        git -C "$MAIN" worktree add -b "$BRANCH" "$WORK" "$SHA"
    fi
fi
/root/miniconda3/envs/rtdetr/bin/python - "$WORK" "$MAIN" "$SHA" <<'PY'
import json
from pathlib import Path
import subprocess
import sys


def synchronize(work, main, sha):
    # Reuse the experiment's operation lock before touching its checkout.
    sys.path.insert(0, str(work / 'tools'))
    from qcc_v1 import operation_lock, active_workers, has_tmux
    from qcc_v1_common import read_json, write_json, require, RUN
    import psutil

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=work, text=True, timeout=30).strip()

    out = work / 'outputs/qcc_v1'
    branch = 'exp-rtdetr-r18-lite-qcc-v1'
    value = dict(worktree=str(work), main=str(main), commit=sha, branch=branch)
    with operation_lock():
        require(git('branch', '--show-current') == branch, 'Unexpected worktree branch; preserved')
        require(not git('status', '--porcelain', '--untracked-files=all'), 'Worktree changes preserved; sync refused')
        require(not active_workers() and not has_tmux(), 'Active QCC worker/tmux; preserved')
        # Also catch a directly launched or orphaned GPU preflight.
        scripts = {work / 'tools/qcc_v1.py', work / 'tools/qcc_v1_preflight.py'}
        for process in psutil.process_iter(['pid', 'cmdline']):
            for arg in process.info['cmdline'] or []:
                if Path(arg).name not in ('qcc_v1.py', 'qcc_v1_preflight.py'):
                    continue
                path = Path(arg)
                if not path.is_absolute():
                    try:
                        path = Path(process.cwd()) / path
                    except psutil.NoSuchProcess:
                        continue
                require(path.resolve() not in scripts, f'Active QCC process {process.pid}; preserved')

        old_sha = git('rev-parse', 'HEAD')
        subprocess.run(['git', 'merge-base', '--is-ancestor', old_sha, sha], cwd=work, check=True, timeout=30)
        previous = read_json(out / 'sync.json')
        if previous:
            require(all(previous.get(k) == value[k] for k in ('worktree', 'main', 'branch')),
                    'Existing sync location/branch differs; preserved')
            # Allow retry after a completed fast-forward but interrupted identity write.
            if previous.get('commit') != old_sha:
                require(old_sha == sha, 'Existing sync commit differs from checkout; preserved')
                subprocess.run(['git', 'merge-base', '--is-ancestor', previous['commit'], sha],
                               cwd=work, check=True, timeout=30)
        else:
            require(not (out / 'prepare.json').exists(), 'Prepared worktree lacks sync identity; preserved')
        if old_sha != sha:
            require(not (out / 'training_identity.json').exists() and not RUN.exists(),
                    'Code update after formal dispatch/run is forbidden; preserved')
        if previous and previous != value:
            history = out / 'history' / ('sync_' + previous['commit'] + '.json')
            if history.exists():
                require(read_json(history) == previous, 'Existing sync history differs; preserved')
            else:
                write_json(history, previous)
        if old_sha != sha:
            # Reject tracked/untracked/ignored collisions; never reset, clean or autostash.
            subprocess.run(['git', 'merge', '--ff-only', '--no-autostash', '--no-overwrite-ignore', sha],
                           cwd=work, check=True, timeout=60)
        require(git('rev-parse', 'HEAD') == sha, 'Sync did not reach the requested commit')
        write_json(out / 'sync.json', value)
        print(json.dumps(value, indent=2))


if __name__ == '__main__':
    synchronize(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3])
PY
echo 'Sync verified. Next: bash /root/autodl-tmp/projects/Crack_RTDETR-qcc_v1/tools/qcc_v1.sh prepare'
