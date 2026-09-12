"""Read-only by default. --apply archives one verified, abandoned preflight lock.

Never kills processes, removes evidence, or touches another experiment's lock.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

NAME='c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug'
SESSION='c19_lif_v1-training'


def require(value,message):
    if not value:raise RuntimeError(message)


def hash_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_evidence(owner_pid,old,proc_root=Path('/proc')):
    require(sys.platform=='linux' and (proc_root/'self/stat').is_file(),'Recovery requires inspectable Linux /proc on the server')
    require(type(owner_pid) is int and owner_pid>0,'Missing/invalid owner PID; refuse to guess')
    try:os.kill(owner_pid,0)
    except ProcessLookupError:pass
    except OSError as error:raise RuntimeError('Owner PID identity uncertain; preserved: '+repr(error)) from error
    else:raise RuntimeError('Owner PID is alive (including possible PID reuse); lock preserved')
    require(not (proc_root/str(owner_pid)).exists(),'Owner PID appeared during inspection')
    matches=[]
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():continue
        try:argv=(process/'cmdline').read_bytes().decode('utf-8',errors='replace').split('\0')
        except FileNotFoundError:continue  # exited during read
        except OSError as error:raise RuntimeError('Cannot inspect process '+process.name) from error
        text=' '.join(argv)
        if ('train_c19_lif_v1.py' in text or 'check_c19_lif_v1.py' in text or
            str(old/'outputs/c19_lif_v1/worker.sh') in text):matches.append(dict(pid=int(process.name),argv=argv))
    require(not matches,'Experiment worker/preflight process exists: '+json.dumps(matches))
    try:result=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True,check=False)
    except OSError as error:raise RuntimeError('Cannot inspect tmux; lock preserved') from error
    if result.returncode:
        message=result.stderr.lower()
        require(result.returncode==1 and ('no server running' in message or ('error connecting' in message and 'no such file or directory' in message)),
                'tmux inspection failed/uncertain: '+result.stderr)
        sessions=[]
    else:sessions=result.stdout.splitlines()
    require(SESSION not in sessions,'Exact experiment tmux session is alive')
    return dict(owner_pid=owner_pid,owner_pid_absent=True,experiment_processes=matches,tmux_sessions=sessions,exact_session_absent=True)


def inspect(main,old,old_sha):
    main=main.resolve(strict=True);old=old.resolve(strict=True)
    require(re.fullmatch('[0-9a-f]{40}',old_sha),'A full old SHA is required')
    require(main!=old and (old/'.git').is_file(),'Expected separate linked old worktree')
    def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
    require(git(old,'rev-parse','HEAD')==old_sha,'Old worktree HEAD differs; evidence preserved')
    require(Path(git(main,'rev-parse','--path-format=absolute','--git-common-dir')).resolve()==
            Path(git(old,'rev-parse','--path-format=absolute','--git-common-dir')).resolve(),'Old worktree belongs to another repository')
    run=main/'runs/c_series'/NAME;lock=run.with_name(NAME+'.c19_lif_v1.lock')
    require(not run.exists() and not run.is_symlink(),'Formal run path exists; never recover a training run')
    require(lock.is_dir() and not lock.is_symlink(),'Exact old lock missing or symlinked')
    launch=old/'outputs/c19_lif_v1';check=old/'outputs/c19_lif_v1_preflight/checks.json'
    files=[lock/'owner.json',launch/'launch_state.json',check]
    for f in files:require(f.is_file() and not f.is_symlink(),'Required evidence missing/symlinked: '+str(f))
    owner,state,checks=[json.loads(f.read_text(encoding='utf-8')) for f in files]
    require(Path(owner.get('worktree','')).resolve()==old and owner.get('variant')=='c19_lif_v1','Owner worktree/variant mismatch')
    require(owner.get('runtime',{}).get('commit')==old_sha and checks.get('runtime',{}).get('commit')==old_sha,'Owner/checks commit mismatch')
    require(state.get('status')=='failed','Old launch is not failed')
    require(checks.get('status')=='FAILED' and checks.get('formal_training')=='NOT_RUN' and checks.get('full_val_test')=='NOT_RUN',
            'Old checks do not prove preflight-only failure')
    for name in ('plan.json','worker.sh','process.json','training_state.json','training_setup.json','console.log','exit_code.json','process_exit_code.txt'):
        require(not (launch/name).exists(),'Dispatch/training evidence exists: '+name)
    # Refuse unexpected contents rather than move a lock with unknown ownership.
    require(sorted(p.name for p in lock.iterdir())==['owner.json'],'Unexpected lock contents; manual review required')
    live=process_evidence(owner.get('pid'),old)
    return dict(status='SAFE_TO_ARCHIVE',action='READ_ONLY',main=str(main),old_worktree=str(old),old_sha=old_sha,
                original_lock=str(lock),owner=owner,processes=live,
                evidence_sha256={str(f):hash_file(f) for f in files},
                reason='Verified failed preflight, no formal run/dispatch, dead owner PID, no experiment processes/session')


def recover(main,old,old_sha,apply=False):
    report=inspect(main,old,old_sha)
    if not apply:return report
    lock=Path(report['original_lock']);guard=lock.with_name(lock.name+'.recovery.guard')
    # Serialize recovery only. start-direct still has its own atomic run reservation.
    with guard.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid())+'\n')
    try:
        fresh=inspect(main,old,old_sha)
        require(fresh['evidence_sha256']==report['evidence_sha256'],'Evidence changed during recovery; preserved')
        archive=lock.with_name(lock.name+'.archived_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex)
        archive.mkdir(exist_ok=False)
        report.update(action='APPLY',archive=str(archive),archived_lock=str(archive/'lock'),status='ARCHIVING')
        def save():
            # Exclusive archive directory; no existing evidence is overwritten.
            temporary=archive/'recovery.json.tmp'
            with temporary.open('x',encoding='utf-8') as stream:
                json.dump(report,stream,indent=2,ensure_ascii=False);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,archive/'recovery.json')
        save()
        require(hash_file(lock/'owner.json')==report['evidence_sha256'][str(lock/'owner.json')],'Owner changed before archival')
        lock.rename(archive/'lock')  # Preserves bytes; no deletion or overwrite.
        require(hash_file(archive/'lock/owner.json')==report['evidence_sha256'][str(lock/'owner.json')],'Archived owner hash mismatch')
        report['status']='ARCHIVED';save()
        return report
    finally:
        # Only this invocation's temporary guard is removed, never an experiment lock.
        guard.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main',type=Path,required=True)
    parser.add_argument('--old-worktree',type=Path,required=True)
    parser.add_argument('--old-sha',required=True)
    parser.add_argument('--apply',action='store_true',help='Explicitly archive the verified abandoned lock; default is read-only')
    args=parser.parse_args()
    try:result=recover(args.main,args.old_worktree,args.old_sha,args.apply)
    except BaseException as error:
        print(json.dumps(dict(status='REFUSED',error=repr(error),action='No lock removal or process termination'),indent=2));raise SystemExit(1)
    print(json.dumps(result,indent=2,ensure_ascii=False))
