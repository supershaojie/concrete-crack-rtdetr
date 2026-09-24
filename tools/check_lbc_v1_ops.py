"""Temporary Git fixtures exercise safe first sync and clean forward updates offline."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
BASE='a0459d6a652cb702699087c88fa39a3e4c4087ec'
BRANCH='exp-rtdetr-r18-lite-lbc-v1'
ORIGIN='https://github.com/supershaojie/concrete-crack-rtdetr.git'


def main():
    folder=ROOT/'outputs/lbc_v1'/('ops_fixture_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True)
    repo=folder/'main';remote=folder/'remote.git';target=folder/'experiment'
    env=dict(os.environ,LBC_PYTHON=sys.executable.replace('\\','/'),GIT_AUTHOR_NAME='LBC fixture',
             GIT_AUTHOR_EMAIL='fixture@example.invalid',GIT_COMMITTER_NAME='LBC fixture',GIT_COMMITTER_EMAIL='fixture@example.invalid')
    def git(where,*args):
        p=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','-C',str(where),*args],env=env,capture_output=True,text=True)
        if p.returncode:raise RuntimeError(p.stderr)
        return p.stdout.strip()
    # Exact per-process sandbox ownership exception, never a global wildcard.
    gitdir=git(ROOT,'rev-parse','--absolute-git-dir')
    env.update(GIT_CONFIG_COUNT='2',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=ROOT.as_posix(),
               GIT_CONFIG_KEY_1='safe.directory',GIT_CONFIG_VALUE_1=gitdir.replace('\\','/'))
    git(ROOT,'init',str(repo))
    common=Path(git(ROOT,'rev-parse','--path-format=absolute','--git-common-dir'))
    (repo/'.git/objects/info/alternates').write_bytes(((common/'objects').as_posix()+'\n').encode())
    git(repo,'checkout','--detach',BASE)
    git(repo,'switch','-c','lbc-ops-fixture-main')
    for name in ('tools/lbc_v1.py','tools/lbc_v1.sh','ultralytics-main/ultralytics/models/rtdetr/lbc.py'):
        dest=repo/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('fixture only\n')
        git(repo,'add',name)
    git(repo,'commit','-m','fixture initial LBC paths')
    first=git(repo,'rev-parse','HEAD')
    git(repo,'branch','-f',BRANCH,first)
    git(repo,'clone','--shared','--bare',str(repo),str(remote))
    git(repo,'remote','add','origin',ORIGIN)
    git(repo,'config',f'url.{remote.as_uri()}.insteadOf',ORIGIN)
    git(repo,'config','protocol.file.allow','always')
    bash=shutil.which('bash') or 'C:/Program Files/Git/bin/bash.exe'
    def sync(sha,expect=True):
        before=git(repo,'rev-parse','HEAD'),git(repo,'status','--porcelain')
        p=subprocess.run([bash,str(ROOT/'tools/sync_lbc_v1.sh'),sha,str(repo),str(target)],env=env,capture_output=True,text=True)
        if (p.returncode==0)!=expect:raise AssertionError(p.stdout+'\n'+p.stderr)
        assert before==(git(repo,'rev-parse','HEAD'),git(repo,'status','--porcelain'))
        return dict(exit_code=p.returncode,expected_success=expect)
    result=dict(first=sync(first),same_sha=sync(first))
    (repo/'future.txt').write_text('tracked incoming\n')
    (repo/'weights').mkdir(exist_ok=True)
    (repo/'weights/collision.pt').write_text('ignored incoming fixture\n')
    git(repo,'add','future.txt');git(repo,'add','-f','weights/collision.pt')
    git(repo,'commit','-m','fixture forward update')
    second=git(repo,'rev-parse','HEAD')
    git(remote,'update-ref',f'refs/heads/{BRANCH}',second)
    # Share objects through the fixture main; no source-branch mutation/network.
    protected=target/'future.txt';protected.write_text('user untracked\n')
    result['untracked_collision']=sync(second,False)
    assert protected.read_text()=='user untracked\n' and git(target,'rev-parse','HEAD')==first
    assert target.resolve() in protected.resolve().parents
    protected.unlink()
    protected=target/'weights/collision.pt';protected.parent.mkdir(exist_ok=True);protected.write_text('user ignored\n')
    result['ignored_collision']=sync(second,False)
    assert protected.read_text()=='user ignored\n' and git(target,'rev-parse','HEAD')==first
    assert target.resolve() in protected.resolve().parents
    protected.unlink()
    (target/'tools/lbc_v1.py').write_text('user tracked change\n')
    result['tracked_change']=sync(second,False)
    (target/'tools/lbc_v1.py').write_text('fixture only\n')
    result['forward_update']=sync(second)
    assert git(target,'rev-parse','HEAD')==second
    assert list((target/'outputs/lbc_v1_sync_history').glob('*.json'))
    result['stale_sha']=sync(first,False)
    result.update(status='PASSED',offline=True,source_checkout_untouched=True,history_backed_up=True)
    (folder/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(folder=str(folder),**result),indent=2))


if __name__=='__main__':main()
