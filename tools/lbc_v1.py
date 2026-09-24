"""LBC v1 operation entry. Prepare/preflight never dispatch formal training."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import tarfile
import time

from init_c19_lif_v1 import ROOT, SOURCE_SHA256, runtime, require, sha256, write_json
from c19_lif_v1_diagnostic import atomic_json as write_json
from lbc_v1_training import initialize, LBC_CONFIG, LBCTrainer, validate_checkpoint, deploy
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.cfg import get_cfg
import torch

BASE='a0459d6a652cb702699087c88fa39a3e4c4087ec'
BRANCH='exp-rtdetr-r18-lite-lbc-v1'
SESSION='lbc-v1-training'
RUN='lbc_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug'
SERVER_PYTHON='/root/miniconda3/envs/rtdetr/bin/python'
MAIN=Path(os.environ.get('LBC_MAIN','/root/autodl-tmp/projects/Crack_RTDETR')).resolve()
OUT=ROOT/'outputs/lbc_v1'
INIT=ROOT/'weights/lbc_v1_controlled_init.pt'


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def git(*args):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT,encoding='utf-8').strip()


def archive_report(path):
    path=Path(path)
    if path.exists():
        backup=path.parent/'history'/f'{stamp()}_{path.name}'
        backup.parent.mkdir(parents=True,exist_ok=True)
        path.rename(backup)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def code_fingerprint():
    names=git('ls-files','tools','ultralytics-main/ultralytics').splitlines()
    # Include current untracked LBC files during local implementation verification.
    names=sorted(set(names)|{str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'tools').glob('*lbc_v1*')}|
                 {'ultralytics-main/ultralytics/models/rtdetr/lbc.py'})
    digest=hashlib.sha256()
    for name in names:
        path=ROOT/name
        if path.is_file() and path.suffix in ('.py','.sh','.yaml'):
            digest.update(name.encode());digest.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return digest.hexdigest()


def identity():
    return dict(code_sha256=code_fingerprint(),init_sha256=sha256(INIT),
                recipe_sha256=sha256(OUT/'train_args.yaml'),config=LBC_CONFIG,
                data_inventory_sha256=sha256(OUT/'dataset_inventory.json'),
                data_config_sha256=sha256(OUT/'data.yaml'))


def recipe():
    expected=YAML.load(ROOT/'docs/c19_lif_v1/c2_args.yaml')
    actual_path=MAIN/'runs/c_series/c2_rtdetr_r18_lite_e200_b16_onlineaug/args.yaml'
    if actual_path.exists():
        actual=YAML.load(actual_path)
        require(actual==expected,'Existing authoritative C2 args differ from the pinned archive')
    target=dict(expected)
    target.update(model=str(INIT),name=RUN,project=str(MAIN/'runs/c_series'),
                  save_dir=str(MAIN/'runs/c_series'/RUN),data=str(OUT/'data.yaml'))
    diff=[dict(field=k,baseline=expected[k],lbc=target[k],changed=expected[k]!=target[k]) for k in sorted(expected)]
    require({r['field'] for r in diff if r['changed']} <= {'model','name','project','save_dir','data'},'Recipe changed')
    resolved=vars(get_cfg(overrides=target))
    additions={k:v for k,v in resolved.items() if k not in target}
    return target,diff,dict(fields=len(target),resolved_fields=len(resolved),pinned_default_additions=additions,
                            authority=str(actual_path if actual_path.exists() else ROOT/'docs/c19_lif_v1/c2_args.yaml'))


def verify_dataset():
    from c19_lif_v1_data import dataset_inventory
    inventory=dataset_inventory(MAIN/'datasets/crack_det')
    expected=read(ROOT/'docs/c19_lif_v1/summary.json')['dataset']
    require(inventory==expected,'Dataset differs from verified physical split; no automatic reshuffle')
    return inventory


def server_environment():
    info=runtime()
    require(Path(sys.executable).resolve()==Path(SERVER_PYTHON).resolve(),'Use the fixed server interpreter')
    require(platform.python_version()=='3.10.13' and str(torch.__version__)=='2.1.2+cu121',
            'Server runtime differs from locked baseline; no automatic upgrades')
    require(torch.cuda.is_available(),'Server CUDA unavailable')
    require('4090' in torch.cuda.get_device_name(0),'Expected verified RTX4090; report mismatch')
    return info


def verify_delivery():
    record=read(ROOT/'outputs/lbc_v1_delivery.json')
    require(record['commit']==git('rev-parse','HEAD') and record['branch']==BRANCH,'Run sync FULL_SHA first')
    require(Path(record['worktree']).resolve()==ROOT and Path(record['main']).resolve()==MAIN,'Wrong delivery paths')
    require(not git('status','--porcelain','--untracked-files=no'),'Tracked source modifications invalidate delivery')
    return record


def prepare(local=False):
    OUT.mkdir(parents=True,exist_ok=True)
    if not local: verify_delivery();server_environment()
    source=MAIN/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt'
    require(source.is_file() and sha256(source)==SOURCE_SHA256,'Missing/wrong public initialization')
    inventory=verify_dataset()
    data=YAML.load(ROOT/'docs/c19_lif_v1/c2_data.yaml');data['path']=str(MAIN/'datasets/crack_det')
    for filename in ('prepare.json','recipe_diff.json','resolved_args.yaml','dataset_inventory.json','data.yaml','train_args.yaml'):
        archive_report(OUT/filename)
    YAML.save(OUT/'data.yaml',data)
    args,diff,recipe_info=recipe()
    YAML.save(OUT/'train_args.yaml',args);YAML.save(OUT/'resolved_args.yaml',vars(get_cfg(overrides=args)))
    write_json(OUT/'recipe_diff.json',dict(recipe=recipe_info,fields=diff))
    write_json(OUT/'dataset_inventory.json',inventory)
    if INIT.exists():
        previous=read(OUT/'initialization.json')
        require(previous['output_sha256']==sha256(INIT) and previous['source_sha256']==SOURCE_SHA256,
                'Existing initial state has different provenance; preserved')
        saved=torch_load(INIT,map_location='cpu')
        require(saved.get('lbc_config')==LBC_CONFIG and saved.get('epoch')==-1 and saved.get('optimizer') is None,
                'Existing initialization is not the fixed untrained LBC version')
    else:
        write_json(OUT/'initialization.json',initialize(source,INIT))
    import ultralytics.models.utils.loss as criterion
    report=dict(status='PREPARED',local=local,source_commit=git('rev-parse','HEAD'),identity=identity(),runtime=runtime(),
                criterion=criterion.__file__,recipe=recipe_info,formal_training='NOT_RUN',
                historical_data_limitation='Augmented sibling images can cross splits; no resplitting in this experiment')
    write_json(OUT/'prepare.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def require_preflight():
    verify_delivery()
    p=read(OUT/'preflight.json')
    require(p['status']=='PASSED' and p['identity']==identity(),'Missing/stale/failed server preflight')
    require(p['capacity']['status']=='PASSED' and p['capacity']['effective_updates'] in (1,2) and p['capacity']['amp'] and
            p['capacity']['actual_shape']==[16,3,640,640], 'No valid B16/640 native AMP capacity proof')
    return p


def preflight(local=False):
    if not local:verify_delivery();server_environment()
    prepared=read(OUT/'prepare.json')
    require(prepared['identity']==identity(),'Prepare identity changed; run prepare')
    require(verify_dataset()==read(OUT/'dataset_inventory.json'),'Dataset identity changed')
    folder=OUT/('preflight_'+stamp());folder.mkdir()
    report=dict(status='FAILED',local=local,source_commit=git('rev-parse','HEAD'),identity=identity(),folder=str(folder))
    archive_report(OUT/'preflight.json')
    try:
        from lbc_v1_checks import run_checks
        report['checks']=run_checks(folder/'checks',MAIN/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt')
        cmd=[sys.executable,str(ROOT/'tools/lbc_v1.py'),'_capacity','--folder',str(folder/'capacity')]
        if local:cmd.append('--local')
        with (folder/'capacity.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
            try:code=proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
                else:proc.terminate()
                proc.wait(timeout=30)
                raise RuntimeError('900-second capacity limit reached; inspect partial report/log')
        require(code==0,f'Capacity check failed ({code}); see {folder}/capacity.log')
        report['capacity']=read(folder/'capacity/bounded.json')
        report['status']='PASSED_LOCAL_SMALL_ONLY' if local else 'PASSED'
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        write_json(OUT/'preflight.json',report)
        print(f"Preflight {report['status']}: {OUT/'preflight.json'}")


def session_active():
    return subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0


def process_token(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):
        return None


def worker_alive(state):
    pid=state.get('pid')
    if not pid:return False
    try:os.kill(pid,0)
    except OSError:return False
    return state.get('process_token') is not None and process_token(pid)==state['process_token']


def dispatch(resume=False):
    require_preflight();server_environment()
    require(not session_active(),'This experiment already has an active tmux session')
    require(not (OUT/'state.json').exists() or not worker_alive(read(OUT/'state.json')),
            'This experiment still has a live training process')
    run=MAIN/'runs/c_series'/RUN
    if resume:
        ckpt=torch_load(run/'weights/last.pt',map_location='cpu');validate_checkpoint(ckpt)
        frozen=read(OUT/'training_identity.json')
        require(frozen['identity']==identity(),'Resume configuration/initialization/source identity changed')
        old=ckpt['train_args'];new=YAML.load(OUT/'train_args.yaml')
        require(all(old[k]==v for k,v in new.items() if k not in ('model','resume')), 'Resume recipe changed')
    else:
        require(not run.exists(),'Formal output already exists; use resume after inspection')
        require(not (OUT/'training_identity.json').exists(),'Existing training identity preserved; inspect status')
        write_json(OUT/'training_identity.json',dict(identity=identity(),source_commit=git('rev-parse','HEAD')))
    launch=stamp();log=OUT/f'console_{launch}.log'
    state=OUT/'state.json'
    archive_report(state)
    archive_report(OUT/'process_exit_code.txt')
    cmd=[SERVER_PYTHON,str(ROOT/'tools/lbc_v1.py'),'_worker','--log',str(log)]
    if resume:cmd.append('--resume-run')
    # The process running inside tmux owns Python, tee and exit status.
    runner=OUT/f'worker_{launch}.sh'
    command=' '.join(shlex.quote(x) for x in cmd)
    runner.write_text('#!/usr/bin/env bash\nset -uo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+
        command+' 2>&1 | tee '+shlex.quote(str(log))+'\n'+
        'lbc_exit=${PIPESTATUS[0]}\nprintf "%s\\n" "$lbc_exit" > '+shlex.quote(str(OUT/'process_exit_code.txt'))+
        '\nexit "$lbc_exit"\n',encoding='utf-8')
    write_json(state,dict(status='DISPATCHED',session=SESSION,log=str(log),resume=resume,created=stamp()))
    subprocess.run(['tmux','new-session','-d','-s',SESSION,'bash '+shlex.quote(str(runner))],check=True)
    print('已派发 / DISPATCHED: '+str(log),flush=True)


def worker(log,resume=False):
    args=YAML.load(OUT/'train_args.yaml');state=OUT/'state.json'
    report=dict(status='INITIALIZING',pid=os.getpid(),process_token=process_token(os.getpid()),
                session=SESSION,log=str(log),start=stamp(),resume=resume)
    write_json(state,report)
    code=1
    try:
        require_preflight()
        if resume:args['resume']=str(MAIN/'runs/c_series'/RUN/'weights/last.pt')
        trainer=LBCTrainer(overrides=args);trainer.lbc_output=OUT
        def started(t):
            require(t.amp and t.batch_size==16 and t.args.imgsz==640,'Formal recipe/runtime changed')
            write_json(OUT/'training_setup.json',dict(model=t.lbc_model_audit,adaptation=t.lbc_adaptation,
                                                     optimizer_groups=t.lbc_optimizer_groups))
            YAML.save(OUT/'actual_train_args.yaml',vars(t.args))
        def live(t):
            report.update(status='RUNNING',epoch=t.epoch,pid=os.getpid())
            write_json(state,report)
            print(f'LBC_TRAINING_BATCH epoch={t.epoch} loss={float(t.loss):.7g}',flush=True) if t.epoch==t.start_epoch and len(t.lbc_epoch_rows)==1 else None
        trainer.add_callback('on_pretrain_routine_end',started)
        trainer.add_callback('on_train_batch_end',live)
        trainer.train();code=0;report['status']='COMPLETED'
    except BaseException as error:
        report.update(status='FAILED',error=repr(error))
        raise
    finally:
        report.update(exit_code=code,end=stamp());write_json(state,report)


def status():
    state=read(OUT/'state.json') if (OUT/'state.json').exists() else dict(status='NOT_DISPATCHED')
    alive=worker_alive(state)
    pid=state.get('pid')
    if state['status'] in ('RUNNING','INITIALIZING') and not alive:state['status']='FAILED_OR_INTERRUPTED'
    if state['status']=='RUNNING':
        logfile=Path(state['log'])
        if not logfile.exists() or 'LBC_TRAINING_BATCH' not in logfile.read_text(errors='replace'):
            state['status']='PROCESS_ALIVE_LOG_PENDING'
    state['pid_alive']=alive
    if os.name!='nt':
        state['session_active']=session_active()
        if state['status']=='DISPATCHED' and not state['session_active']:
            state['status']='DISPATCH_FAILED_OR_EXITED'
        exit_file=OUT/'process_exit_code.txt'
        if exit_file.exists():state['python_exit_code']=exit_file.read_text().strip()
        if pid:
            p=subprocess.run(['ps','-o','pid,ppid,stat,etime,args','--ppid',str(pid)],capture_output=True,text=True)
            state['children']=p.stdout.strip()
    csv=MAIN/'runs/c_series'/RUN/'results.csv'
    if csv.exists():state['results_csv_tail']=csv.read_text().splitlines()[-2:]
    print(json.dumps(state,ensure_ascii=False,indent=2))


def evaluate(split):
    from c19_lif_v1_results import evaluate as mother_evaluate
    require(read(OUT/'state.json')['status']=='COMPLETED','Formal training must complete before independent evaluation')
    require(read(OUT/'training_identity.json')['identity']==identity(),'Training identity changed')
    source=MAIN/'runs/c_series'/RUN/'weights/best.pt'
    dest=OUT/'best_deploy.pt'
    if split=='val':
        require(not (OUT/'evaluation_val').exists(),'Existing validation is preserved')
        if not dest.exists():deploy(source,dest)
        require(read(dest.with_suffix('.json'))['source_sha256']==sha256(source),'Deployment is from another best checkpoint')
        write_json(OUT/'val_selection_lock.json',dict(best_sha256=sha256(source),deploy_sha256=sha256(dest),source_commit=git('rev-parse','HEAD')))
    else:
        lock=read(OUT/'val_selection_lock.json')
        require(lock['best_sha256']==sha256(source) and lock['deploy_sha256']==sha256(dest),'Val-locked weights changed')
        require(not (OUT/'evaluation_test').exists() and not (OUT/'test_attempt.json').exists(),'Test already attempted; preserved, no checkpoint scan')
        write_json(OUT/'test_attempt.json',dict(time=stamp(),**lock))
    require(verify_dataset()==read(OUT/'dataset_inventory.json'),'Dataset changed')
    mother_evaluate(dest,OUT/'data.yaml',split,OUT/('evaluation_'+split),device='0',
                   val_report=OUT/'evaluation_val/metrics.json' if split=='test' else None,
                   extra_metrics=working_points)


def working_points(metrics):
    """Use the same evaluated confidence curves; no extra inference/checkpoint selection."""
    import numpy as np
    from ultralytics.utils.metrics import smooth
    p=np.asarray(metrics.box.p_curve).mean(0)
    r=np.asarray(metrics.box.r_curve).mean(0)
    f1=np.asarray(metrics.box.f1_curve).mean(0)
    confidence=np.asarray(metrics.box.px)
    i=int(smooth(f1,0.1).argmax())
    fixed={}
    for value in (.80,.90,.95):
        mask=p>=value
        if mask.any():
            eligible=np.flatnonzero(mask);j=int(eligible[r[mask].argmax()])
            fixed[str(value)]=dict(recall=float(r[j]),precision=float(p[j]),confidence=float(confidence[j]))
        else:fixed[str(value)]=None
    return dict(max_F1_working_point=dict(confidence=float(confidence[i]),precision=float(p[i]),recall=float(r[i]),f1=float(f1[i])),
                recall_at_minimum_precision=fixed,scope='IoU0.5 confidence curves; auxiliary diagnostic only')


def pack():
    require(not (OUT/'state.json').exists() or read(OUT/'state.json')['status'] not in ('RUNNING','INITIALIZING','DISPATCHED'),
            'Wait for a stable run before packaging')
    destination=OUT/f'lbc_v1_LIGHT_{stamp()}.tar.gz'
    metadata=dict(source_commit=git('rev-parse','HEAD'),formal_training=read(OUT/'state.json') if (OUT/'state.json').exists() else 'NOT_RUN')
    inputs={}
    for path in list((ROOT/'docs/lbc_v1').glob('*'))+list(OUT.glob('*.json'))+list(OUT.glob('*.yaml'))+list(OUT.glob('*.jsonl')):
        if path.is_file() and path.stat().st_size<1500000:inputs[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    for split in ('val','test'):
        path=OUT/f'evaluation_{split}/metrics.json'
        if path.exists():inputs[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    csv=MAIN/'runs/c_series'/RUN/'results.csv'
    if csv.exists():inputs['training/results.csv']=csv.read_bytes()
    for log in sorted(OUT.glob('console_*.log'))[-2:]:
        inputs['log_tail/'+log.name]='\n'.join(log.read_text(errors='replace').splitlines()[-150:]).encode()
    inputs['source.patch']=subprocess.check_output(
        ['git','-c',f'safe.directory={ROOT.as_posix()}','diff','--binary',BASE,'HEAD'],cwd=ROOT)
    inputs['identity.json']=json.dumps(metadata,indent=2).encode()
    manifest={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in inputs.items()}
    inputs['MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    with tarfile.open(destination,'x:gz') as archive:
        for name,data in inputs.items():
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    require(destination.stat().st_size<8*1024*1024,'LIGHT exceeds 8MB')
    with tarfile.open(destination,'r:gz') as archive:
        require(all(hashlib.sha256(archive.extractfile(n).read()).hexdigest()==r['sha256'] for n,r in manifest.items()),'Archive integrity failed')
    print(str(destination)+' SHA256='+sha256(destination))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','start','status','resume','val','test','pack','deploy','_worker','_capacity'])
    parser.add_argument('--local',action='store_true');parser.add_argument('--folder',type=Path)
    parser.add_argument('--log',type=Path);parser.add_argument('--resume-run',action='store_true')
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output',type=Path)
    a=parser.parse_args();torch.set_num_threads(4)
    if a.action=='prepare':prepare(a.local)
    elif a.action=='preflight':preflight(a.local)
    elif a.action in ('start','resume'):dispatch(a.action=='resume')
    elif a.action=='status':status()
    elif a.action=='_worker':worker(a.log,a.resume_run)
    elif a.action=='_capacity':
        from lbc_v1_preflight import bounded_check
        bounded_check(YAML.load(OUT/'train_args.yaml'),a.folder,server=not a.local)
    elif a.action=='deploy':
        require(a.checkpoint and a.output,'deploy requires --checkpoint and --output');deploy(a.checkpoint,a.output)
    elif a.action in ('val','test'):evaluate(a.action)
    elif a.action=='pack':pack()


if __name__=='__main__':main()
