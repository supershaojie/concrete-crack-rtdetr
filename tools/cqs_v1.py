"""CQS v1 lifecycle. Formal training and test run only on explicit start/resume/test."""
from __future__ import annotations

import argparse
from functools import partial
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
import uuid

from cqs_v1_common import *


def prepare(args):
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.local:
        verify_delivery()
    source=Path(args.source or MAIN/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt')
    parent_args=Path(args.parent_args or MAIN/'runs/c_series/c19_lif_v1_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml')
    data=Path(args.data or MAIN/'configs/crack_autodl.yaml')
    report=dict(status='PENDING',runtime=runtime(),source=str(source),parent_args=str(parent_args),data=str(data),
                local_only=args.local,pending=[],source_hashes=source_hashes(),formal_training='NOT_RUN',test='NOT_RUN')
    try:
        require(YAML.load(ROOT/'docs/cqs_v1/research.yaml')==CQS_CONFIG,'Research YAML differs from model configuration')
        if source.is_file():
            if INIT.exists():
                audit=read_json(OUT/'initialization.json')
                require(audit['output_sha256']==sha256(INIT) and audit['source_sha256']==sha256(source)==SOURCE_SHA256,
                        'Existing init/source differs; preserve and investigate')
                loaded=RTDETR(str(INIT)).model
                verify_model(loaded,require_nc1=False)
            else:
                audit=initialize(source,INIT)
                write_json(OUT/'initialization.json',audit)
            report['initialization_sha256']=sha256(INIT)
        else:
            report['pending'].append('public untrained source')
        if parent_args.is_file():
            train_args,diff=recipe(parent_args)
            if args.local:
                report['local_data_path_mapping']=dict(server=train_args['data'],local=str(data),
                    rule='Read-only local content audit against frozen mother path/label hashes; formal args unchanged')
            else:
                require(Path(train_args['data']).resolve()==data.resolve(),'Formal data path differs from authoritative parent args')
            YAML.save(OUT/'train_args.yaml',train_args)
            write_json(OUT/'args_diff.json',diff)
            shutil.copyfile(parent_args,OUT/'authoritative_parent_args.yaml')
        else:
            report['pending'].append('authoritative trained mother args.yaml')
        if data.is_file():
            report['data_inventory']=inventory(data,OUT/'data_identity')
            shutil.copyfile(data,OUT/'data_config.yaml')
        else:
            report['pending'].append('data config and crack_det dataset')
        # Existing parent resource helper preserves actual check_amp, without bypassing it.
        if not args.local:
            from train_c19_lif_v1 import ensure_amp_resources
            ensure_amp_resources(MAIN,OUT/'amp_resources.json')
        else:
            report['pending'].append('server native AMP resources/check and B16/640 preflight')
        if not report['pending']:
            report['fingerprint']=fingerprint(report)
            report['status']='PASS'
        report['module_hashes']=source_contract()
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(OUT/'prepare.json',report)
    return report


def preflight():
    verify_delivery()
    require(not active_processes() and not session_active(),'This experiment is already active')
    prepared=read_json(OUT/'prepare.json')
    require(prepared['status']=='PASS' and not prepared['local_only'],'Complete server prepare first')
    current=fingerprint(prepared)
    require(current['sha256']==prepared['fingerprint']['sha256'],'Prepare identity changed; rerun prepare and inspect differences')
    if (OUT/'preflight.json').is_file():
        old=read_json(OUT/'preflight.json')
        if old.get('status')=='PASS' and old.get('fingerprint')==current['sha256']:
            print('Existing valid preflight reused; no repeat capacity run.')
            return old
    folder=OUT/f'preflight_{utc()}'
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(status='FAIL',fingerprint=current['sha256'],folder=str(folder),runtime=runtime())
    try:
        from check_cqs_v1 import algorithm_checks,mapping_and_gradient,benchmark
        report['algorithm']=algorithm_checks('cpu')
        report['mapping_gradient']=mapping_and_gradient('cuda')
        # A separate process imposes a wall-time bound including setup and dataloaders.
        command=[sys.executable,'-u',str(ROOT/'tools/cqs_v1.py'),'_capacity',str(folder)]
        with (folder/'capacity_console.log').open('x',encoding='utf-8') as stream:
            subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
        capacity=read_json(folder/'capacity.json')
        require(capacity['status']=='PASS' and capacity['effective_updates']>=1,'Invalid bounded capacity result')
        report['capacity']=capacity
        report['timing']=benchmark(build())
        report['status']='PASS'
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        if (folder/'capacity.json').is_file():report['capacity']=read_json(folder/'capacity.json')
        raise
    finally:
        write_json(folder/'preflight.json',report)
        write_json(OUT/'preflight.json',report)
    return report


def active_processes():
    """Identity-qualified process inspection only; never signal any process."""
    import psutil
    rows=[]
    script=(ROOT/'tools/cqs_v1.py').resolve()
    for p in psutil.process_iter(['pid','ppid','cmdline','create_time']):
        try:
            cmd=p.info.get('cmdline') or []
            if p.pid!=os.getpid() and '_worker' in cmd and any(Path(v).resolve()==script for v in cmd if v.endswith('cqs_v1.py')):
                children=[dict(pid=c.pid,cmdline=c.cmdline()) for c in p.children(recursive=True)]
                rows.append(dict(**p.info,children=children))
        except (psutil.NoSuchProcess,psutil.AccessDenied,OSError):
            continue
    return rows


def session_active():
    return bool(shutil.which('tmux')) and subprocess.run(['tmux','has-session','-t','='+SESSION],capture_output=True).returncode==0


def run_status():
    state=read_json(OUT/'state.json') if (OUT/'state.json').is_file() else dict(status='NOT_STARTED',phase='NOT_STARTED')
    processes=active_processes();session=session_active()
    dispatch=read_json(OUT/'dispatch.json') if (OUT/'dispatch.json').is_file() else {}
    if dispatch:
        exit_file=OUT/(dispatch['id']+'.shell-exit.json')
        if exit_file.is_file():
            shell=read_json(exit_file)
            if shell['python_exit']!=0 or shell['tee_exit']!=0:
                state.update(status='FAILED',phase='WORKER_EXIT',shell_exit=shell)
        elif state.get('status') in ('DISPATCHED','SETTING_UP','RUNNING') and not processes and not session:
            state.update(status='FAILED',phase='WORKER_OR_TMUX_MISSING')
    state.update(processes=processes,tmux_active=session,session=SESSION,run=str(RUN))
    csv=RUN/'results.csv'
    if csv.is_file():
        rows=csv.read_text(encoding='utf-8').splitlines()
        state['results_rows']=max(0,len(rows)-1);state['latest_epoch_row']=rows[-1] if len(rows)>1 else None
    log=Path(dispatch['log']) if dispatch.get('log') else None
    if log and log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-8000))
            state['log_tail']=stream.read().decode('utf-8',errors='replace')
        state['log']=str(log)
    return state


def require_gate():
    verify_delivery()
    prepared=read_json(OUT/'prepare.json');gate=read_json(OUT/'preflight.json')
    require(prepared['status']=='PASS' and not prepared['local_only'],'Server prepare missing')
    require(gate['status']=='PASS' and gate['fingerprint']==fingerprint(prepared)['sha256'],
            'Missing/stale CQS engineering preflight; source/config/source weights/data/environment identity changed')
    require(gate['capacity']['shape']==[16,3,640,640] and gate['capacity']['effective_updates']>=1,'No valid B16/640 update evidence')
    require(torch.cuda.is_available(),'CUDA required for formal run')
    return prepared,gate


def validate_resume():
    checkpoint=RUN/'weights/last.pt'
    require(checkpoint.is_file(),'No valid last.pt; initialization failures need archive-failed-run, not resume')
    ckpt=torch_load(checkpoint,map_location='cpu')
    require(0<=ckpt.get('epoch',-1)<199 and ckpt.get('optimizer') is not None and ckpt.get('scaler') is not None and
            ckpt.get('ema') is not None,'last.pt has no resumable epoch/optimizer/scaler/EMA')
    verify_model(ckpt['ema'])
    saved=ckpt['train_args'];planned=YAML.load(OUT/'train_args.yaml')
    require(all(saved.get(k)==v for k,v in planned.items() if k not in {'model','resume'}),'Resume recipe differs')
    require(Path(saved['save_dir']).resolve()==RUN,'Resume would target another run')
    return checkpoint


def worker_script(dispatch):
    command=[sys.executable,'-u',str(ROOT/'tools/cqs_v1.py'),'_worker',dispatch['id']]
    exit_file=OUT/(dispatch['id']+'.shell-exit.json')
    env=dict(PYTHONPATH=str(ROOT/'ultralytics-main'),PYTHONUNBUFFERED='1',YOLO_AUTOINSTALL='false',CQS_V1_MAIN=str(MAIN))
    # tmux servers may retain another experiment's environment.
    for key in ('PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','OMP_NUM_THREADS','MKL_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG'):
        env[key]=os.environ.get(key)
    exports=''.join(('export '+k+'='+shlex.quote(v) if v is not None else 'unset '+k)+'\n' for k,v in env.items())
    return ('#!/usr/bin/env bash\nset -uo pipefail\n'+exports+'cd '+shlex.quote(str(ROOT))+' || exit 125\n'+
            ' '.join(map(shlex.quote,command))+' 2>&1 | tee -a '+shlex.quote(dispatch['log'])+'\n'+
            'codes=("${PIPESTATUS[@]}")\npy_rc=${codes[0]}\ntee_rc=${codes[1]}\n'+
            'printf \'{"dispatch":"'+dispatch['id']+'","python_exit":%s,"tee_exit":%s}\\n\' "$py_rc" "$tee_rc" > '+shlex.quote(str(exit_file))+'\n'+
            'if [ "$py_rc" -ne 0 ]; then exit "$py_rc"; fi\nexit "$tee_rc"\n')


def dispatch_training(resume=False):
    require_gate()
    require(shutil.which('tmux'),'tmux required')
    require(not active_processes() and not session_active(),'CQS experiment already active; duplicate dispatch refused')
    if resume:
        validate_resume()
    else:
        require(not RUN.exists(),'Existing formal output preserved; use real resume or explicit archive-failed-run')
    lock=OUT/'dispatch.lock'
    require(not lock.exists(),'Existing dispatch reservation preserved; inspect status and archive-failed-run if appropriate')
    lock.mkdir(exist_ok=False)
    item=dict(id=utc()+'_'+uuid.uuid4().hex[:8],resume=resume,source_sha=git('rev-parse','HEAD'),
              log=str(OUT/f'console_{utc()}.log'),run=str(RUN),session=SESSION,created=utc())
    script=OUT/(item['id']+'.worker.sh')
    write_json(lock/'owner.json',item)
    write_json(OUT/'dispatch.json',item)
    write_json(OUT/'state.json',dict(dispatch=item['id'],status='DISPATCHED',phase='DISPATCHED',created=utc()))
    with script.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(worker_script(item))
    try:
        subprocess.run(['tmux','new-session','-d','-s',SESSION,'bash '+shlex.quote(str(script))],check=True)
    except BaseException as error:
        write_json(OUT/'state.json',dict(dispatch=item['id'],status='FAILED',phase='DISPATCH_FAILED',error=repr(error)))
        raise
    return item


def worker(dispatch_id):
    item=read_json(OUT/'dispatch.json')
    require(item['id']==dispatch_id,'Obsolete dispatch identity')
    code=1;wrapper=None
    try:
        write_json(OUT/'state.json',dict(dispatch=dispatch_id,status='SETTING_UP',phase='SETTING_UP',pid=os.getpid(),started=utc()))
        require_gate()
        require(git('rev-parse','HEAD')==item['source_sha'],'Source changed since dispatch')
        from cqs_v1_training import CQSTrainer
        trainer=partial(CQSTrainer,cqs_folder=OUT,formal=True,dispatch=dispatch_id)
        if item['resume']:
            last=validate_resume();wrapper=RTDETR(str(last));wrapper.train(resume=True,trainer=trainer)
        else:
            require(not RUN.exists(),'Formal run appeared after dispatch')
            wrapper=RTDETR(str(INIT));wrapper.train(trainer=trainer,**YAML.load(OUT/'train_args.yaml'))
        code=0
        write_json(OUT/'state.json',dict(dispatch=dispatch_id,status='COMPLETED',phase='COMPLETED',pid=os.getpid(),
            actual_epoch=wrapper.trainer.epoch+1,epoch_limit=200,early_stopped=wrapper.trainer.epoch+1<200,finished=utc()))
    except BaseException as error:
        write_json(OUT/'state.json',dict(dispatch=dispatch_id,status='FAILED',phase='FAILED',pid=os.getpid(),
                   actual_epoch=getattr(getattr(wrapper,'trainer',None),'epoch',-1)+1,error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        write_json(OUT/(dispatch_id+'.exit.json'),dict(dispatch=dispatch_id,python_exit=code,finished=utc()))
        lock=OUT/'dispatch.lock'
        if (lock/'owner.json').is_file() and read_json(lock/'owner.json')['id']==dispatch_id:
            (lock/'owner.json').unlink();lock.rmdir()


def archive_failed_run():
    require(not active_processes() and not session_active(),'Active experiment preserved')
    if RUN.exists():
        require(not list(RUN.rglob('*.pt')),'Checkpoint exists; use real resume, do not archive as setup failure')
        csv=RUN/'results.csv'
        require(not csv.is_file() or len(csv.read_text().splitlines())<=1,'Results data exists; not an initialization failure')
        # Absolute resolved paths must remain within the intended run parent.
        require(RUN.resolve().parent== (MAIN/'runs/c_series').resolve() and RUN.name==RUN_NAME,'Unsafe archive target')
        dest=RUN.with_name(RUN.name+'_failed_setup_'+utc())
        RUN.rename(dest)
    else:
        dest=None
    lock=OUT/'dispatch.lock'
    if lock.exists():
        lock.rename(OUT/('dispatch.lock.archived_'+utc()))
    write_json(OUT/'last_failed_archive.json',dict(archived_run=str(dest) if dest else None,created=utc()))
    return dict(archived_run=str(dest) if dest else None,next_action='start')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='action',required=True)
    prep=s.add_parser('prepare');prep.add_argument('--source');prep.add_argument('--parent-args');prep.add_argument('--data');prep.add_argument('--local',action='store_true')
    for name in ('preflight','start','status','resume','val','test','archive-failed-run'):
        s.add_parser(name)
    probe=s.add_parser('probe');probe.add_argument('--parent-best');probe.add_argument('--data');probe.add_argument('--count',type=int,default=64);probe.add_argument('--device',choices=('cpu','cuda'))
    pack=s.add_parser('pack');pack.add_argument('--output')
    worker=s.add_parser('_worker');worker.add_argument('dispatch')
    cap=s.add_parser('_capacity');cap.add_argument('folder')
    load=s.add_parser('_load-check');load.add_argument('checkpoint');load.add_argument('data');load.add_argument('folder');load.add_argument('device',choices=('cpu','cuda'))
    delivery=s.add_parser('_record-delivery');delivery.add_argument('sha')
    return p


def main():
    args=parser().parse_args()
    torch.set_num_threads(4)
    if args.action=='prepare':result=prepare(args)
    elif args.action=='preflight':result=preflight()
    elif args.action=='start':result=dispatch_training(False)
    elif args.action=='resume':result=dispatch_training(True)
    elif args.action=='status':result=run_status()
    elif args.action=='archive-failed-run':result=archive_failed_run()
    elif args.action=='_worker':result=worker(args.dispatch)
    elif args.action=='_capacity':
        from cqs_v1_training import capacity_worker
        result=capacity_worker(args.folder)
    elif args.action=='_load-check':
        from cqs_v1_training import independent_load
        result=independent_load(args.checkpoint,args.data,args.folder,args.device)
    elif args.action=='_record-delivery':
        require(git('rev-parse','HEAD')==args.sha and len(args.sha)==40,'Pinned SHA mismatch')
        require(git('branch','--show-current')==BRANCH and git('remote','get-url','origin')==REMOTE,'Wrong branch/repository')
        result=dict(sha=args.sha,base_sha=BASE_SHA,branch=BRANCH,worktree=str(ROOT),main=str(MAIN),created=utc(),source_hashes=source_hashes())
        write_json(OUT/'delivery.json',result)
        from cqs_v1_delivery import write_commands
        write_commands(args.sha)
    else:
        from cqs_v1_results import evaluate,probe,package
        result=evaluate(args.action) if args.action in ('val','test') else probe(args.parent_best,args.data,args.count,args.device) if args.action=='probe' else package(args.output)
    if result is not None:
        # The full identities remain in JSON; keep terminal output bounded.
        print(json.dumps(json_safe({k:v for k,v in result.items() if k not in ('fingerprint','source_hashes','identity_details','data_inventory')}),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
