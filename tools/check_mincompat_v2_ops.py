"""Run focused unit checks and actual sync shell with explicit Git stubs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import torch
from mincompat_v2 import ROOT,BASE_COMMIT,require,write_json


def sync_checks(bash):
    with tempfile.TemporaryDirectory(prefix='triad_sync_fixture_',dir=ROOT/'outputs') as directory:
        root=Path(directory);(root/'bin').mkdir();(root/'main/.git').mkdir(parents=True)
        stub=r'''#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$FIXTURE_LOG"
repo="$PWD"
if [[ "${1:-}" == -C ]]; then repo="$2"; shift 2; fi
case "$*" in
 'remote get-url origin') echo "${FIXTURE_ORIGIN:-https://github.com/supershaojie/concrete-crack-rtdetr.git}" ;;
 'status --porcelain') printf '%s' "${FIXTURE_DIRTY:-}" ;;
 'rev-parse --show-toplevel') echo "$repo" ;;
 'rev-parse --git-common-dir') echo "$FIXTURE_COMMON" ;;
 'rev-parse FETCH_HEAD') echo "${FIXTURE_REMOTE:-$FIXTURE_SHA}" ;;
 'rev-parse HEAD') echo "${FIXTURE_HEAD:-$FIXTURE_SHA}" ;;
 'symbolic-ref -q HEAD') [[ "${FIXTURE_ATTACHED:-no}" == yes ]] ;;
 'fetch origin codex/rtdetr-mincompat-v2'|'cat-file -e '*) : ;;
 'merge-base --is-ancestor '*) [[ "${FIXTURE_ANCESTOR:-yes}" == yes ]] ;;
 'worktree add --detach '*) mkdir -p "$4" ;;
 *) echo "Unexpected Git operation: $*" >&2; exit 99 ;;
esac
'''
        (root/'bin/git').write_text(stub,encoding='utf-8');(root/'bin/git').chmod(0o755)
        command='''set -eu
fixture="$(cd "$1" && pwd)"; script="$2"; sha="$3"
export PATH="$fixture/bin:$PATH" FIXTURE_LOG="$fixture/git.log" FIXTURE_SHA="$sha" FIXTURE_COMMON="$fixture/main/.git"
bash "$script" "$sha" "$fixture/main" "$fixture/worktree"
'''
        def invoke(**extra):
            return subprocess.run([bash,'-c',command,'fixture',root.as_posix(),(ROOT/'tools/sync_mincompat_v2.sh').as_posix(),BASE_COMMIT],
                env=dict(os.environ,**extra),capture_output=True,text=True)
        for name,env,ok in [('new',{},True),('detached_reuse',{},True),('dirty',{'FIXTURE_DIRTY':' M protected.txt'},False),
            ('different_SHA',{'FIXTURE_HEAD':'1'*40},False),('historical_ancestor',{'FIXTURE_REMOTE':'2'*40},True),
            ('outside_history',{'FIXTURE_ANCESTOR':'no'},False),('attached',{'FIXTURE_ATTACHED':'yes'},False),
            ('wrong_origin',{'FIXTURE_ORIGIN':'https://example.org/wrong.git'},False)]:
            proc=invoke(**env);require((proc.returncode==0)==ok,f'Sync {name}: {proc.stdout} {proc.stderr}')
        pin=root/'worktree/outputs/mincompat_v2_sync.json';pin.write_text('user file')
        require(invoke().returncode!=0 and pin.read_text()=='user file','Sync overwrote existing pin')
        log=(root/'git.log').read_text();require('worktree add --detach' in log,'No detached worktree test')
        require(not any(' '+word+' ' in log for word in ('reset','clean','stash','kill')),'Destructive sync command')
        return dict(status='passed',scope='actual Bash script with explicit Git stub; no network/server/git-worktree mutation',
            new_detached=True,clean_reuse=True,dirty_preserved=True,different_SHA_preserved=True,
            pinned_ancestor_accepted=True,outside_history_rejected=True,attached_preserved=True,wrong_origin_rejected=True,pin_preserved=True)


def run(bash,output):
    loader=unittest.TestLoader();suite=unittest.TestSuite()
    for name in ('test_mincompat_v2_lifecycle.py',):
        suite.addTests(loader.discover(str(ROOT/'ultralytics-main/tests'),pattern=name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(),'Focused unit tests failed')
    for name in ('autodl_mincompat_v2.sh','sync_mincompat_v2.sh'):
        subprocess.run([bash,'-n',str(ROOT/'tools'/name)],check=True)
    write_json(output,dict(unit_tests=result.testsRun,status='passed',sync=sync_checks(bash),bash_syntax='passed'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bash',default='bash');p.add_argument('--report',required=True)
    a=p.parse_args();torch.set_num_threads(4);run(a.bash,a.report)
