"""Recovery fixture tests: actual preserving filesystem IO, mocked server identity/liveness."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from c19_lif_v1_diagnostic import atomic_json
import recover_c19_lif_v1_preflight as recovery


def rejects(action):
    try:action()
    except (RuntimeError,FileExistsError):return True
    raise AssertionError('Unsafe recovery accepted')


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    main=out/'main';old=out/'old';main.mkdir();old.mkdir();(old/'.git').write_text('fixture')
    main=main.resolve();old=old.resolve();sha='c27064ce6b06d0c32d38e644fcff64394a1a5992'
    lock=main/'runs/c_series'/(recovery.NAME+'.c19_lif_v1.lock');lock.mkdir(parents=True)
    launch=old/'outputs/c19_lif_v1';launch.mkdir(parents=True)
    check=old/'outputs/c19_lif_v1_preflight/checks.json'
    owner=dict(pid=98765,worktree=str(old),variant='c19_lif_v1',runtime=dict(commit=sha))
    state=dict(status='failed');checks=dict(status='FAILED',formal_training='NOT_RUN',full_val_test='NOT_RUN',runtime=dict(commit=sha))
    atomic_json(lock/'owner.json',owner);atomic_json(launch/'launch_state.json',state);atomic_json(check,checks)
    report=dict(status='FAILED',scope='fixture filesystem + mocked git/proc/tmux; no server recovery',tests={});rows=report['tests']
    def git(command,**kwargs):return sha if command[-1]=='HEAD' else str(main/'.git')
    try:
        with patch.object(recovery.subprocess,'check_output',side_effect=git),patch.object(recovery,'process_evidence',return_value={'mocked':True}):
            before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            result=recovery.recover(main,old,sha)
            assert result['action']=='READ_ONLY' and before=={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            rows['default_read_only']='PASSED: exact file bytes and paths unchanged'
            for label,value in [('worktree',str(main)),('variant','c17_lif_v1'),('runtime',dict(commit='f'*40))]:
                wrong=deepcopy(owner);wrong[label]=value;atomic_json(lock/'owner.json',wrong)
                rows['wrong_owner_'+label]=rejects(lambda:recovery.recover(main,old,sha,True));atomic_json(lock/'owner.json',owner)
            for status in ('dispatched','initializing','training'):
                atomic_json(launch/'launch_state.json',dict(status=status))
                rows['state_'+status]=rejects(lambda:recovery.recover(main,old,sha,True))
            atomic_json(launch/'launch_state.json',state)
            wrong=deepcopy(checks);wrong['formal_training']='UNKNOWN';atomic_json(check,wrong)
            rows['unknown_training']=rejects(lambda:recovery.recover(main,old,sha,True));atomic_json(check,checks)
            for name in ('worker.sh','plan.json','process.json','console.log','training_state.json'):
                file=launch/name;file.write_text('fixture')
                rows['dispatch_'+name]=rejects(lambda:recovery.recover(main,old,sha,True));file.unlink()
            runpath=main/'runs/c_series'/recovery.NAME;runpath.mkdir()
            rows['existing_formal_run']=rejects(lambda:recovery.recover(main,old,sha,True));runpath.rmdir()
            guard=lock.with_name(lock.name+'.recovery.guard');guard.write_text('concurrent recovery fixture')
            rows['concurrent_recovery']=rejects(lambda:recovery.recover(main,old,sha,True));guard.unlink()
            contents=(lock/'owner.json').read_bytes();statebytes=(launch/'launch_state.json').read_bytes();checkbytes=check.read_bytes()
            result=recovery.recover(main,old,sha,True)
            assert result['status']=='ARCHIVED' and not lock.exists() and Path(result['archived_lock'],'owner.json').read_bytes()==contents
            assert (launch/'launch_state.json').read_bytes()==statebytes and check.read_bytes()==checkbytes
            rows['preserving_archive']='PASSED: exact owner bytes verified, old launch/check unchanged, unique archive + manifest'
            rows['second_apply_rejected']=rejects(lambda:recovery.recover(main,old,sha,True))
            # A second failed reservation is independently archived; previous archive stays intact.
            lock.mkdir();atomic_json(lock/'owner.json',owner)
            second=recovery.recover(main,old,sha,True)
            assert second['archive']!=result['archive'] and Path(result['archive']).is_dir()
            rows['repeated_failure_preserved']='PASSED: two non-overlapping retained archives'
        proc=out/'proc';(proc/'self').mkdir(parents=True);(proc/'self/stat').write_text('fixture')
        process=proc/'1234';process.mkdir();(process/'cmdline').write_bytes(b'python\0train_c17_lif_v1.py\0worker\0')
        no_owner=ProcessLookupError()
        with patch.object(recovery.sys,'platform','linux'),patch.object(recovery.os,'kill',side_effect=no_owner),patch.object(recovery.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'c17_lif_v1-training\n','')):
            recovery.process_evidence(98765,old,proc);rows['other_experiment_untouched']='PASSED: C17 worker/session ignored'
            with patch.object(recovery.os,'kill',return_value=None):rows['live_or_reused_owner_pid']=rejects(lambda:recovery.process_evidence(98765,old,proc))
            with patch.object(recovery.os,'kill',side_effect=PermissionError()):rows['uncertain_owner_pid']=rejects(lambda:recovery.process_evidence(98765,old,proc))
            rows['missing_owner_pid']=rejects(lambda:recovery.process_evidence(None,old,proc))
            (process/'cmdline').write_bytes(b'python\0train_c19_lif_v1.py\0worker\0')
            rows['active_worker']=rejects(lambda:recovery.process_evidence(98765,old,proc))
            (process/'cmdline').write_bytes(b'python\0check_c19_lif_v1.py\0')
            rows['active_preflight']=rejects(lambda:recovery.process_evidence(98765,old,proc))
            (process/'cmdline').write_bytes(b'python\0unrelated.py\0')
            with patch.object(recovery.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'c19_lif_v1-training\n','')):
                rows['exact_tmux_session']=rejects(lambda:recovery.process_evidence(98765,old,proc))
            with patch.object(recovery.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','permission denied')):
                rows['uncertain_tmux']=rejects(lambda:recovery.process_evidence(98765,old,proc))
        report['status']='PASSED'
    finally:atomic_json(out/'tests.json',report)
    print(json.dumps(dict(status=report['status'],tests=len(rows)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
