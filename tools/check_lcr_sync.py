"""Isolated local Git integration checks; remote URL identity is mocked, all Git IO is local."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from init_lcr import ROOT, require, runtime, write_json


def run(output, bash):
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    git=shutil.which('git')
    require(git is not None,'Git required')
    report=dict(runtime=runtime(),scope='Local real Git/worktree fixtures; only remote get-url identity mocked. No network, tmux or training.',checks=[])
    with tempfile.TemporaryDirectory(prefix='lcr_sync_',dir=ROOT/'outputs') as td:
        folder=Path(td).resolve()
        require(folder.is_relative_to((ROOT/'outputs').resolve()),'Fixture root outside outputs')
        remote,main,work=folder/'remote.git',folder/'main',folder/'lcr'
        def g(*args,cwd=None):
            p=subprocess.run([git,*map(str,args)],cwd=cwd,text=True,capture_output=True)
            require(p.returncode==0,p.stderr)
            return p.stdout.strip()
        g('clone','--bare','--shared',ROOT,remote)
        g('clone','--shared',remote,main)
        g('config','user.name','LCR Fixture',cwd=main); g('config','user.email','fixture@example.invalid',cwd=main)
        # Advance the local test remote; delivered SHA must remain usable as an ancestor.
        g('checkout','-B','codex/lcr-aifi',sha,cwd=main)
        g('commit','--allow-empty','-m','fixture remote advanced',cwd=main)
        g('push','origin','codex/lcr-aifi',cwd=main)
        (main/'preserve-user-file.txt').write_text('preserve this user fixture')
        before=(g('rev-parse','HEAD',cwd=main),g('status','--porcelain',cwd=main))
        bindir=folder/'bin'; bindir.mkdir()
        # Mock only the remote identity; all remaining Git operations are real and local.
        wrapper='#!/usr/bin/env bash\nset -e\n'
        wrapper+='if [[ "$*" == *"remote get-url origin"* ]]; then printf "%s\\n" "${LCR_FIXTURE_ORIGIN:-https://github.com/supershaojie/concrete-crack-rtdetr.git}"; exit 0; fi\n'
        wrapper+='exec '+shlex.quote(Path(git).as_posix())+' "$@"\n'
        (bindir/'git').write_text(wrapper,encoding='utf-8'); (bindir/'git').chmod(0o755)
        def sync(expected, label, target=work, commit=sha, identity=None):
            env=dict(os.environ,LCR_PYTHON=Path(__import__('sys').executable).as_posix())
            if identity: env['LCR_FIXTURE_ORIGIN']=identity
            bash_bindir=('/'+bindir.drive[0].lower()+bindir.as_posix()[2:]) if os.name=='nt' else bindir.as_posix()
            command='export PATH='+shlex.quote(bash_bindir)+':"$PATH"; bash '+shlex.quote((ROOT/'tools/sync_lcr.sh').as_posix())+' '+shlex.join([commit,main.as_posix(),target.as_posix()])
            p=subprocess.run([bash,'-c',command],env=env,text=True,capture_output=True)
            require((p.returncode==0)==expected,label+': '+p.stdout+'\n'+p.stderr)
            require(before==(g('rev-parse','HEAD',cwd=main),g('status','--porcelain',cwd=main)),'Main changed')
            report['checks'].append(dict(case=label,exit_code=p.returncode,expected_accept=expected,main_unchanged=True))
        sync(True,'ancestor SHA creates detached worktree')
        require(g('rev-parse','HEAD',cwd=work)==sha and g('branch','--show-current',cwd=work)=='','Not pinned detached')
        sync(True,'idempotent existing detached worktree',identity='git@github.com:supershaojie/concrete-crack-rtdetr.git')
        (work/'user.txt').write_text('preserve')
        sync(False,'dirty target rejected')
        require((work/'user.txt').read_text()=='preserve','User file overwritten')
        # Only remove our exact fixture file, never a user checkout.
        (work/'user.txt').unlink()
        sync(False,'different delivered SHA rejected',commit=g('rev-parse','HEAD',cwd=main))
        directory=folder/'not-worktree'; directory.mkdir(); (directory/'evidence.txt').write_text('preserve')
        sync(False,'ordinary directory rejected',target=directory)
        sync(False,'wrong remote identity rejected',identity='https://github.com/other/repository.git')
        # Validate actual launch identity guard against this real detached checkout, no dispatch.
        import train_lcr as train
        from unittest.mock import patch
        with patch.object(train,'ROOT',work),patch.object(train,'MAIN',main):
            train.verify_delivery()
        report['detached_launch_identity']='passed'
    report['status']='passed'
    write_json(output,report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bash',default=shutil.which('bash'))
    a=p.parse_args(); require(a.bash,'Bash required'); run(a.output,a.bash)
