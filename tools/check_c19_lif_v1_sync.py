"""Real local Git/worktree integration; remote fetch redirected to an isolated fixture, no training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

from init_c19_lif_v1 import ROOT, require, write_json
import train_c19_lif_v1 as train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bash', default='bash')
    args = parser.parse_args()
    folder = args.output.resolve().with_suffix('').with_name(args.output.stem + '_fixture')
    folder.mkdir(parents=True, exist_ok=False)
    realgit = shutil.which('git')
    def git(*argv, cwd=ROOT):
        return subprocess.check_output([realgit, *argv], cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    pin = git('rev-parse','HEAD')
    remote, main_repo, worktree = folder/'remote.git', folder/'main', folder/'worktree with space'
    git('clone','--bare','--shared','--single-branch','--branch','codex/rtdetr-c19-lif-v1',str(ROOT),str(remote))
    git('clone','--shared',str(remote),str(main_repo))
    git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--allow-empty','-m','fixture remote advancement',cwd=main_repo)
    advance=git('rev-parse','HEAD',cwd=main_repo)
    git('push','origin','HEAD:refs/heads/codex/rtdetr-c19-lif-v1',cwd=main_repo)
    expected_origin='https://github.com/supershaojie/concrete-crack-rtdetr.git'
    git('remote','set-url','origin',expected_origin,cwd=main_repo)
    readme=main_repo/'README.md';readme.write_bytes(readme.read_bytes()+b'\nfixture existing edit\n')
    before=(git('rev-parse','HEAD',cwd=main_repo),git('status','--porcelain',cwd=main_repo))
    wrapper=folder/'bin';wrapper.mkdir()
    # Only fetch destination differs from the actual server command.
    # Every fetch, object/ancestry check, worktree creation and cleanliness check is real Git.
    (wrapper/'git').write_bytes(b'''#!/usr/bin/env bash
if [[ "${3:-}" == fetch && "${5:-}" == origin ]]; then
 exec "$LIF_TEST_GIT" "$1" "$2" fetch "$4" "$LIF_TEST_REMOTE" "${@:6}"
fi
exec "$LIF_TEST_GIT" "$@"
''')
    if os.name != 'nt': (wrapper/'git').chmod(0o755)
    env=dict(os.environ, LIF_TEST_GIT=realgit, LIF_TEST_REMOTE=str(remote),LIF_TEST_WINDOWS=str(int(os.name=='nt')),
             C19_LIF_V1_PYTHON=sys.executable,PATH=str(wrapper)+os.pathsep+os.environ['PATH'])
    checks=[]
    def sync(name, sha=pin, target=worktree, success=True):
        shell = 'test_bin="$1"; shift; if command -v cygpath >/dev/null; then test_bin="$(cygpath -u "$test_bin")"; fi; export PATH="$test_bin:$PATH"; source "$@"'
        cp=subprocess.run([args.bash,'-c',shell,'lif-sync-test',str(wrapper),str(ROOT/'tools/sync_c19_lif_v1.sh'),sha,str(main_repo),str(target)],env=env,capture_output=True,text=True)
        require((cp.returncode==0)==success,f'{name}: {cp.stdout}\n{cp.stderr}')
        checks.append(dict(name=name,exit_code=cp.returncode,tail=(cp.stdout+cp.stderr)[-1500:]))
        require((git('rev-parse','HEAD',cwd=main_repo),git('status','--porcelain',cwd=main_repo))==before,'Main checkout changed')
    sync('remote_advanced_pin_original')
    require(git('rev-parse','refs/remotes/origin/codex/rtdetr-c19-lif-v1',cwd=main_repo)==advance, 'Fixture remote redirection/advancement not exercised')
    require(git('rev-parse','HEAD',cwd=worktree)==pin and not git('branch','--show-current',cwd=worktree),'Not pinned detached HEAD')
    sync('repeat_same_clean_pin')
    with patch.object(train,'ROOT',worktree),patch.object(train,'MAIN',main_repo):
        require(train.verify_delivery()['commit']==pin,'Launcher rejected real detached delivery')
    dirty=worktree/'README.md';original_readme=dirty.read_bytes();dirty.write_bytes(original_readme+b'\npreserve fixture\n')
    sync('dirty_target_rejected',success=False)
    require(dirty.read_bytes()==original_readme+b'\npreserve fixture\n','Dirty target overwritten')
    # This file belongs only to this test fixture, never an experiment checkout.
    dirty.write_bytes(original_readme)
    untracked=worktree/'downloads';untracked.mkdir();(untracked/'keep.txt').write_text('keep')
    sync('untracked_outputs_preserved')
    require((untracked/'keep.txt').read_text()=='keep','Untracked results altered')
    sync('different_sha_rejected',sha=advance,success=False)
    alien=folder/'unrelated';alien.mkdir();(alien/'sentinel').write_bytes(b'preserve')
    sync('non_worktree_rejected',target=alien,success=False)
    require((alien/'sentinel').read_bytes()==b'preserve','Unrelated directory overwritten')
    git('remote','set-url','origin','https://example.invalid/wrong.git',cwd=main_repo)
    sync('wrong_origin_rejected',success=False)
    write_json(args.output,dict(status='PASSED',scope='Real local Git/worktrees; only fetch destination redirected to isolated local bare repository',
               pinned_code_commit=pin,fixture_remote_advanced_commit=advance,detached_launcher_accepted=True,main_dirty_state_preserved=True,
               checks=checks,training='NOT_RUN',fixture_directory=str(folder)))


if __name__=='__main__': main()
