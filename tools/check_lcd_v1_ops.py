"""Disposable git fixtures and dispatch/gate tests; never launches a training worker."""
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
import sys

import lcd_v1 as ops
import lcd_v1_sync as sync
from init_lcd_v1 import ROOT, require
from c19_lif_v1_diagnostic import atomic_json
from ultralytics.utils import YAML


def expect_failure(fn):
    try:fn()
    except (AssertionError,RuntimeError,subprocess.CalledProcessError):return
    raise AssertionError('Unsafe action was unexpectedly accepted')


def main():
    folder=Path(tempfile.mkdtemp(prefix='ops-',dir=ROOT/'outputs/lcd_v1'))
    main=folder/'main';dest=folder/'worktree';main.mkdir()
    def git(*args):return subprocess.check_output(['git','-C',str(main),*args],text=True).strip()
    git('init');git('config','user.email','fixture@example.invalid');git('config','user.name','LCD disposable fixture')
    git('remote','add','origin',sync.ORIGIN)
    (main/'a.txt').write_text('one');(main/'.gitignore').write_text('ignored.txt\n')
    git('add','a.txt','.gitignore');git('commit','-m','fixture one');one=git('rev-parse','HEAD')
    (main/'a.txt').write_text('two');git('commit','-am','fixture two');two=git('rev-parse','HEAD')
    (main/'ignored.txt').write_text('tracked incoming');git('add','-f','ignored.txt');git('commit','-m','fixture three');three=git('rev-parse','HEAD')
    # Source main remains at three; FETCH_HEAD alone chooses the synchronized snapshot.
    git('update-ref','FETCH_HEAD',one)
    actual_run=subprocess.run
    def run(argv,*args,**kwargs):
        if argv[0]=='tmux':return subprocess.CompletedProcess(argv,1)
        return actual_run(argv,*args,**kwargs)
    with patch.object(sync,'MAIN',main),patch.object(sync,'DEST',dest),patch.object(subprocess,'run',run):
        sync.sync(one)
        (main/'user-untracked.txt').write_text('preserve')
        (dest/'a.txt').write_text('user edit');git('update-ref','FETCH_HEAD',two)
        expect_failure(lambda:sync.sync(two));(dest/'a.txt').write_text('one')
        sync.sync(two)
        (dest/'ignored.txt').write_text('do not overwrite');git('update-ref','FETCH_HEAD',three)
        expect_failure(lambda:sync.sync(three))
        require((dest/'ignored.txt').read_text()=='do not overwrite','Ignored file overwritten')
        # Explicit fixture rename, never erase a user's file.
        (dest/'ignored.txt').rename(dest/'fixture-preserved.txt');sync.sync(three)
        git('update-ref','FETCH_HEAD',one);expect_failure(lambda:sync.sync(one))
        require(git('rev-parse','HEAD')==three and (main/'user-untracked.txt').read_text()=='preserve','Main changed')
    report=dict(sync_first_create=True,sync_forward_update=True,tracked_edits_rejected=True,
                ignored_collision_rejected=True,rewind_rejected=True,main_head_and_untracked_preserved=True)
    gate=dict(status='PASSED',identity={'fixture':1},capacity=dict(status='PASSED',batch=16,shape=[16,3,640,640],
              AMP=True,effective_updates=2,batches=7,elapsed_seconds=20))
    from copy import deepcopy
    with patch.object(ops,'verify_delivery',lambda:{}),patch.object(ops,'identity',lambda:{'fixture':1}):
        with patch.object(ops,'read',lambda path:gate):ops.require_preflight()
        for key,value in [('AMP',False),('effective_updates',0),('effective_updates',3),('batches',17),('elapsed_seconds',901)]:
            invalid=deepcopy(gate);invalid['capacity'][key]=value
            with patch.object(ops,'read',lambda path:invalid):expect_failure(ops.require_preflight)
        invalid=deepcopy(gate);invalid['identity']={'fixture':2}
        with patch.object(ops,'read',lambda path:invalid):expect_failure(ops.require_preflight)
    report['stale_or_incomplete_preflight_rejected']=True
    out=folder/'dispatch';out.mkdir();config=out/'args.yaml'
    with patch.object(ops,'OUT',out),patch.object(ops,'CONFIG',config),patch.object(ops,'MAIN',main),\
         patch.object(ops,'require_preflight',lambda:{}),patch.object(ops,'identity',lambda:{'fixture':True}),\
         patch.object(ops,'git',lambda *args:three),patch.object(ops,'process_rows',lambda:[]),\
         patch.object(sys,'executable',ops.PYTHON),patch.object(ops.shutil,'which',lambda name:'/usr/bin/tmux'):
        YAML.save(config,dict(save_dir=str(main/'runs/c_series'/ops.RUN_NAME)))
        commands=[]
        def fake_run(argv,*args,**kwargs):
            commands.append(argv);return subprocess.CompletedProcess(argv,1 if 'has-session' in argv else 0)
        with patch.object(subprocess,'run',fake_run):
            ops.dispatch()
            body=next(out.glob('worker_*.sh')).read_text()
            require(ops.PYTHON in body and str(ROOT/'tools/lcd_v1.py') in body and 'worker' in body,'Wrong worker command')
            require('exit $?' in body and "trap 'rc=$?;" in body,'Lost real exit code')
            require(any('new-session' in c and ops.SESSION in c for c in commands),'Worker not inside tmux')
            expect_failure(ops.dispatch)
        report.update(dispatch_command_checked=True,unresolved_dispatch_lock_rejected=True,formal_training='NOT_STARTED',
                      tmux='MOCKED locally; real session start pending server',status='PASSED')
    atomic_json(ROOT/'docs/lcd_v1/ops_checks.json',report)
    print(report)


if __name__=='__main__':main()
