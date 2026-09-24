"""LCD v1 isolated lifecycle. prepare/preflight never dispatch formal training."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

import torch
from init_lcd_v1 import (ROOT, BASE, BRANCH, MODEL, RESEARCH, PREFIX, NEW_KEYS, SOURCE_SHA256,
                         require, sha256, runtime, controlled, initialize, verify_model, tensor_digest)
from c19_lif_v1_diagnostic import atomic_json as write_json, rng_state, restore_rng
from c19_lif_v1_data import dataset_inventory, real_batch
from ultralytics import RTDETR
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.data.utils import check_det_dataset

MAIN=Path('/root/autodl-tmp/projects/Crack_RTDETR')
PYTHON='/root/miniconda3/envs/rtdetr/bin/python'
SESSION='lcd-v1-training'
RUN_NAME='lcd_v1_rtdetr_r18_lite_cbr_lif_e200_b16_onlineaug'
OUT=ROOT/'outputs/lcd_v1'
INIT=OUT/'lcd_v1_controlled_init.pt'
CONFIG=OUT/'train_args.yaml'


def utc():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def backup(path):
    path=Path(path)
    if path.exists():
        folder=OUT/'history'/utc();folder.mkdir(parents=True,exist_ok=True)
        path.rename(folder/path.name)


def save(path,value):backup(path);write_json(path,value)


def source_signature():
    # Exact relevant source, independent of logs, checkpoints, and Markdown-only commits.
    files=list((ROOT/'tools').glob('*.py'))+list((ROOT/'tools').glob('*.sh'))
    files+=list((ROOT/'ultralytics-main/ultralytics').rglob('*.py'))
    files+=list((ROOT/'ultralytics-main/ultralytics/cfg').rglob('*.yaml'))
    h=hashlib.sha256()
    for p in sorted(files):
        h.update(p.relative_to(ROOT).as_posix().encode());h.update(p.read_bytes().replace(b'\r\n',b'\n'))
    return h.hexdigest()


def identity():
    args=YAML.load(CONFIG)
    return dict(source=source_signature(),init=sha256(INIT),recipe=sha256(CONFIG),
                data=sha256(args['data']),inventory=sha256(OUT/'dataset_inventory.json'),research=RESEARCH)


def verify_delivery():
    require(git('remote','get-url','origin')=='https://github.com/supershaojie/concrete-crack-rtdetr.git','Wrong origin')
    record=read(OUT/'delivery.json')
    require(record['commit']==git('rev-parse','HEAD') and record['branch']==BRANCH,'Run sync with delivered full SHA first')
    require(Path(record['worktree']).resolve()==ROOT and Path(record['main']).resolve()==MAIN,'Wrong server worktree')
    require(not git('status','--porcelain','--untracked-files=no'),'Tracked source edits protected')
    return record


def recipe(main,data):
    original=YAML.load(ROOT/'docs/lcd_v1/mother_args.yaml')
    c2=YAML.load(ROOT/'docs/c19_lif_v1/c2_args.yaml')
    require(set(original)==set(c2),'Incomplete archived recipe')
    allowed={'model','name','save_dir','project','data'}
    require(all(type(v) is type(c2[k]) and v==c2[k] for k,v in original.items() if k not in allowed),'Mother recipe drift')
    result=dict(original,model=str(INIT),data=str(data),project=str(main/'runs/c_series'),name=RUN_NAME,
                save_dir=str(main/'runs/c_series'/RUN_NAME))
    diff=[dict(field=k,mother=original[k],lcd=result[k],changed=original[k]!=result[k]) for k in original]
    require(all(r['field'] in allowed for r in diff if r['changed']),'Unexpected config changes')
    return result,diff


def prepare(main=MAIN,data=None,local=False):
    if not local:verify_delivery()
    OUT.mkdir(parents=True,exist_ok=True)
    source=main/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt'
    require(source.is_file() and sha256(source)==SOURCE_SHA256,'Unified source missing/hash mismatch')
    data=Path(data or main/'configs/crack_autodl.yaml').resolve()
    args,diff=recipe(main,data)
    resolved=check_det_dataset(str(data),autodownload=False)
    require(resolved['nc']==1,'Expected nc=1')
    require(all(Path(resolved[s]).resolve()==Path(resolved['path']).resolve()/'images'/s for s in ('train','val','test')),'Physical split mapping changed')
    inventory=dataset_inventory(Path(resolved['path']))
    expected=read(ROOT/'docs/lcd_v1/dataset_inventory.json')
    require(inventory==expected,'Physical data differs from successful mother fingerprint; files are not moved')
    if INIT.exists():
        old=read(OUT/'initialization.json')
        require(old['init_sha256']==sha256(INIT),'Existing init hash mismatch')
        verify_model(torch_load(INIT,map_location='cpu')['model'],zero=True)
    else:save(OUT/'initialization.json',initialize(source,INIT))
    if CONFIG.exists() and YAML.load(CONFIG)!=args:backup(CONFIG)
    YAML.save(CONFIG,args)
    # Preserve byte-stable identity when an unchanged prepare is repeated.
    if not (OUT/'dataset_inventory.json').exists() or read(OUT/'dataset_inventory.json')!=inventory:
        save(OUT/'dataset_inventory.json',inventory)
    save(OUT/'recipe_diff.json',diff);save(OUT/'research.json',RESEARCH)
    info=runtime()
    save(OUT/'prepare.json',dict(status='PASSED',runtime=info,identity=identity(),source=str(source),
         data_root=str(resolved['path']),local_only=local,formal_training='NOT_STARTED',
         data_limit='已有增强同源图跨 split；本任务沿用物理 split，不重新划分'))
    print('PREPARED',CONFIG,flush=True)


def lifecycle(model,trainer,folder):
    """Exercise actual experiment save/restore methods including pending scaled gradients."""
    from types import SimpleNamespace
    from lcd_v1_trainer import LCDTrainer
    from lcd_v1_checks import optimizer
    from ultralytics.utils.torch_utils import EarlyStopping, ModelEMA
    t=LCDTrainer.__new__(LCDTrainer)
    t.model=deepcopy(model).cpu().float(); t.model.args=YAML.load(ROOT/'docs/lcd_v1/mother_args.yaml')
    t.model.names={0:'crack'};t.model.task='detect'
    t.optimizer,_=optimizer(t.model);t.optimizer.load_state_dict(trainer.optimizer.state_dict())
    t.scaler=torch.cuda.amp.GradScaler(enabled=False);t.ema=ModelEMA(t.model);t.ema.updates=2
    t.scheduler=torch.optim.lr_scheduler.LambdaLR(t.optimizer,lambda epoch:1.0)
    t.epoch=1;t.best_fitness=t.fitness=.1;t.metrics={};t.args=SimpleNamespace(**t.model.args)
    t.wdir=folder/'weights';t.last=t.wdir/'last.pt';t.best=t.wdir/'best.pt';t.save_period=-1
    t.read_results_csv=lambda:{}
    t.stopper=EarlyStopping(patience=50);t.accumulate=4;t._lcd_last_opt_step=752
    t.lcd_effective_updates=2;t.lcd_overflows=0;t.lcd_consecutive_overflows=0
    t.lcd_identity=dict(test='disposable epoch state');t.model.model[5].lcd.O.weight.grad=torch.ones_like(t.model.model[5].lcd.O.weight)*.01
    t.save_model();ckpt=torch_load(t.last,map_location='cpu')
    u=LCDTrainer.__new__(LCDTrainer);u.model=deepcopy(t.model);u.model.zero_grad();u.optimizer,_=optimizer(u.model)
    u.scaler=torch.cuda.amp.GradScaler(enabled=False);u.ema=ModelEMA(u.model)
    u.scheduler=torch.optim.lr_scheduler.LambdaLR(u.optimizer,lambda epoch:1.0)
    u.stopper=EarlyStopping(patience=50);u.resume=True;u.args=t.args;u.epochs=200;u.lcd_identity=t.lcd_identity
    u.resume_training(ckpt);u._lcd_restore_scheduler(u);u._lcd_epoch_start(u)
    require(u.start_epoch==2 and u._lcd_last_opt_step==752 and u.accumulate==4,'Epoch/accumulation lost')
    require(tensor_digest(u.model)==tensor_digest(t.model) and tensor_digest(u.ema.ema)==tensor_digest(t.ema.ema),'Model/EMA lost')
    require(torch.equal(u.model.model[5].lcd.O.weight.grad,t.model.model[5].lcd.O.weight.grad),'Pending scaled gradients lost')
    for key,state in t.optimizer.state_dict()['state'].items():
        for n,v in state.items():
            actual=u.optimizer.state_dict()['state'][key][n]
            require(torch.equal(v.cpu(),actual.cpu()) if isinstance(v,torch.Tensor) else v==actual,'Optimizer state lost')
    require(u.scheduler.state_dict()==t.scheduler.state_dict() and u.scaler.state_dict()==t.scaler.state_dict(),'Scheduler/scaler lost')
    expected=tensor_digest(t.model)
    subprocess.run([sys.executable,str(ROOT/'tools/lcd_v1.py'),'verify-load','--checkpoint',str(t.last),'--expected',expected],cwd=ROOT,check=True)
    with torch.no_grad():
        m=RTDETR(str(t.last)).model.eval();out=m(torch.rand(1,3,160,160))[0]
    require(out.shape==(1,300,5) and torch.isfinite(out).all(),'Ordinary label-free eval failed')
    return dict(status='PASSED',fresh_process=True,model_ema_optimizer_scheduler_scaler=True,pending_scaled_gradients=True,
                next_epoch=2,last_opt_step=752,label_free_eval=True,nonzero_LCD_restored=True)


def local_checks(source,dataset=None,folder=None):
    from lcd_v1_checks import formula,routes,learning,fusion,latency
    folder=Path(folder or OUT/'checks'/utc());folder.mkdir(parents=True,exist_ok=False)
    report=dict(status='FAILED',runtime=runtime(),source_signature=source_signature(),formal_training='NOT_STARTED',
                server_B16_640='PENDING',full_val_test='NOT_RUN',export='SKIPPED: no requested export format/dependency verification')
    try:
        report['formula']=formula()
        pair,lcd,audit=controlled(source);report['initialization']=audit
        from lcd_v1_trainer import LCDTrainer
        probe=LCDTrainer.__new__(LCDTrainer);probe.data=dict(nc=1,channels=3);probe.resume=False
        rebuilt=probe.get_model(lcd.yaml,lcd,verbose=False)
        require(tensor_digest(rebuilt)==tensor_digest(lcd),'Actual LCDTrainer rebuild differs')
        report['actual_lcd_trainer_rebuild']='PASSED'
        report['routes']=routes(pair,lcd)
        # Native criterion needs >=300 encoder candidates; 160 meets this requirement.
        if dataset:
            batch,records=real_batch(Path(dataset),160,2);report['samples']=records
        else:
            batch=dict(img=torch.rand(2,3,160,160),bboxes=torch.tensor([[.4,.5,.2,.3],[.5,.6,.1,.4]]),
                       cls=torch.zeros(2,1),batch_idx=torch.arange(2));report['samples']='synthetic'
        updated,trainer,report['cpu_learning']=learning(lcd,batch)
        torch.save(dict(model=updated,batch=batch),folder/'updated_cpu.pt')
        try:report['cpu_fusion']=fusion(updated,batch['img'],folder/'cpu_fusion.json')
        except RuntimeError:report['cpu_fusion']=read(folder/'cpu_fusion.json')
        report['lifecycle']=lifecycle(updated,trainer,folder)
        if torch.cuda.is_available():
            updated_gpu,amp_trainer,report['cuda_amp_learning']=learning(lcd,batch,device='cuda',amp=True)
            torch.save(dict(model=updated_gpu,batch=batch),folder/'updated_cuda.pt')
            try:report['cuda_fusion']=fusion(updated_gpu.cuda(),batch['img'].cuda(),folder/'cuda_fusion.json')
            except RuntimeError:report['cuda_fusion']=read(folder/'cuda_fusion.json')
            # Actual AMP scaler state round trip, including its growth tracker.
            scaler=torch.cuda.amp.GradScaler();scaler.load_state_dict(amp_trainer.scaler.state_dict())
            require(scaler.state_dict()==amp_trainer.scaler.state_dict(),'CUDA scaler restore differs')
            report['cuda_scaler_reload']='PASSED'
        else:report['cuda_amp_learning']=dict(status='SKIPPED',reason='CUDA unavailable')
        report['latency']=latency(pair,lcd)
        report['status']='PASSED' if all(report[k]['status'] in ('PASSED','PASS_CANDIDATE_PERMUTATION')
                                         for k in ('cpu_fusion','cuda_fusion') if k in report) else 'FAILED_FUSION'
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        write_json(folder/'checks.json',report)
        save(OUT/'local_checks.json',report)
    print('CHECKS',folder,flush=True)
    return report


def capacity(folder):
    from lcd_v1_checks import learning
    folder=Path(folder);prepared=read(OUT/'prepare.json')
    report=dict(status='FAILED',batch=16,imgsz=640,AMP=True,formal_training='NOT_STARTED')
    try:
        require(torch.cuda.is_available(),'CUDA required')
        batch,samples=real_batch(Path(prepared['data_root']),640,16)
        model=torch_load(INIT,map_location='cpu')['model'];verify_model(model,zero=True)
        _,_,report=learning(model,batch,device='cuda',amp=True,max_batches=16,seconds=900)
        report['samples']=samples;report['scope']='real B16/640, repeated fixed training batch; capacity/learning only, no convergence claim'
    except BaseException as error:report['error']=repr(error);raise
    finally:write_json(folder/'capacity.json',report)


def preflight():
    verify_delivery();prepared=read(OUT/'prepare.json');require(prepared['identity']==identity(),'Prepare identity changed')
    info=runtime()
    require(platform.python_version()=='3.10.13' and info['torch']=='2.1.2+cu121','Server version differs; no environment upgrade attempted')
    require(info['gpu'] and '4090' in info['gpu'],'Expected successful-server RTX4090')
    require(dataset_inventory(Path(prepared['data_root']))==read(OUT/'dataset_inventory.json'),'Data fingerprint changed')
    folder=OUT/'preflight_runs'/utc();folder.mkdir(parents=True)
    result=dict(status='FAILED',identity=identity(),commit=git('rev-parse','HEAD'),runtime=info,folder=str(folder))
    try:
        result['checks']=local_checks(Path(prepared['source']),Path(prepared['data_root']),folder/'checks')
        with (folder/'capacity.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'tools/lcd_v1.py'),'capacity','--folder',str(folder)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        result['capacity']=read(folder/'capacity.json')
        require(result['capacity']['status']=='PASSED','Capacity failed')
        require(result['checks']['status']=='PASSED','Necessary formula/lifecycle/fusion checks did not all pass')
        result['status']='PASSED'
    except BaseException as error:
        result['error']=repr(error)
        if (folder/'capacity.json').exists():result['capacity']=read(folder/'capacity.json')
        raise
    finally:write_json(folder/'preflight.json',result);save(OUT/'preflight.json',result)
    print('PREFLIGHT PASSED; formal training NOT STARTED',flush=True)


def require_preflight():
    verify_delivery();r=read(OUT/'preflight.json')
    require(r['status']=='PASSED' and r['identity']==identity(),'Valid preflight for current source/init/recipe required')
    c=r['capacity']
    require(c['status']=='PASSED' and c['batch']==16 and c['shape']==[16,3,640,640] and c['AMP'] is True,'B16/640 AMP evidence missing')
    require(1<=c['effective_updates']<=2 and c['batches']<=16 and c['elapsed_seconds']<=900,'Bounded update evidence missing')
    return r


def process_rows():
    rows=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=[a.decode(errors='replace') for a in p.read_bytes().split(b'\0') if a]
            if str(ROOT/'tools/lcd_v1.py') in argv and 'worker' in argv:rows.append(dict(pid=int(p.parent.name),argv=argv))
        except (OSError,ValueError):pass
    return rows


def dispatch(resume=False):
    require_preflight();require(sys.executable==PYTHON,'Use fixed server interpreter')
    require(shutil.which('tmux'),'tmux required')
    require(subprocess.run(['tmux','has-session','-t','='+SESSION],capture_output=True).returncode!=0,'LCD session already exists')
    require(not process_rows(),'LCD worker already active')
    args=YAML.load(CONFIG);run=Path(args['save_dir'])
    require(run==MAIN/'runs/c_series'/RUN_NAME,'Run identity changed')
    if resume:
        ckpt=torch_load(run/'weights/last.pt',map_location='cpu')
        require(ckpt.get('lcd_state',{}).get('identity')==identity(),'Resume checkpoint identity differs')
        require(0<=ckpt['epoch']<199 and ckpt.get('optimizer') is not None,'No unfinished epoch checkpoint to resume')
        if (OUT/'exit.json').exists():require(read(OUT/'exit.json')['code']!=0,'Completed run is not resumed')
    else:require(not run.exists(),'Existing formal run protected; use resume for unfinished epoch state')
    stamp=utc();log=OUT/f'console_{stamp}.log';worker=OUT/f'worker_{stamp}.sh'
    command=[PYTHON,'-u',str(ROOT/'tools/lcd_v1.py'),'worker']+(['--resume'] if resume else [])
    # Atomically reserve the single experiment dispatch; concurrent starters cannot both pass.
    lock=OUT/'active.lock'
    if lock.exists():
        previous=read(lock)
        finished=(OUT/'exit.json').exists() and read(OUT/'exit.json').get('dispatch')==previous['dispatch']
        # Resume already verified a full last.pt plus absence of our session/process.
        # It can recover a machine/pane interruption with no Python finally record.
        require(resume or finished or (OUT/'shell_exit.json').exists(),'Unresolved dispatch lock; inspect status')
        backup(lock)
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(dispatch=stamp),f)
    for p in (OUT/'exit.json',OUT/'process.json',OUT/'shell_exit.json'):backup(p)
    plan=dict(status='DISPATCHED',dispatch=stamp,log=str(log),session=SESSION,resume=resume,
              identity=identity(),command=command,commit=git('rev-parse','HEAD'))
    save(OUT/'dispatch.json',plan)
    # Separate session environment from other experiments; shell trap records Python startup failures.
    body='#!/usr/bin/env bash\nset -uo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'
    body+='export PYTHONPATH='+shlex.quote(str(ROOT/'ultralytics-main'))+'\nexport YOLO_AUTOINSTALL=false\n'
    for name in ('PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUBLAS_WORKSPACE_CONFIG'):
        body+=('export '+name+'='+shlex.quote(os.environ[name]) if name in os.environ else 'unset '+name)+'\n'
    body+='trap \'rc=$?; printf "{\\"code\\":%s,\\"unix_time\\":%s}\\n" "$rc" "$(date +%s)" > '+shlex.quote(str(OUT/'shell_exit.json'))+'\' EXIT\n'
    body+=' '.join(map(shlex.quote,command))+' >> '+shlex.quote(str(log))+' 2>&1\nexit $?\n'
    worker.write_text(body,encoding='utf-8')
    try:subprocess.run(['tmux','new-session','-d','-s',SESSION,'bash '+shlex.quote(str(worker))],check=True)
    except BaseException:
        write_json(OUT/'exit.json',dict(code=1,dispatch=stamp,finished=utc(),reason='tmux dispatch failed'));raise
    print('已派发；尚未确认 RUNNING。日志：'+str(log),flush=True)


def worker(resume=False):
    from lcd_v1_trainer import LCDTrainer
    from lcd_v1_checks import optimizer
    plan=read(OUT/'dispatch.json');code=1
    try:
        require_preflight();require(plan['identity']==identity(),'Dispatch identity changed')
        prepared=read(OUT/'prepare.json')
        require(dataset_inventory(Path(prepared['data_root']))==read(OUT/'dataset_inventory.json'),'Dataset changed after preflight')
        args=YAML.load(CONFIG); expected=deepcopy(args)
        if resume:args['model']=args['resume']=str(Path(args['save_dir'])/'weights/last.pt')
        write_json(OUT/'process.json',dict(pid=os.getpid(),dispatch=plan['dispatch'],status='SETTING_UP',started=utc()))
        trainer=LCDTrainer(overrides=args);trainer.lcd_identity=identity()
        def setup(t):
            verify_model(t.model,zero=not resume)
            require(bool(t.amp) and t.batch_size==16,'Native AMP/B16 requirement failed')
            actual=vars(t.args)
            exempt={'model','resume'} if resume else set()
            differences={k:[v,actual.get(k)] for k,v in expected.items() if k not in exempt and (type(v)!=type(actual.get(k)) or v!=actual.get(k))}
            require(not differences,'Actual recipe differs: '+str(differences))
            require(type(t.optimizer) is torch.optim.AdamW,'Expected original AdamW')
            ids=[id(p) for g in t.optimizer.param_groups for p in g['params']]
            require(len(ids)==len(set(ids)) and set(ids)=={id(p) for p in t.model.parameters()},'Optimizer coverage differs')
            YAML.save(OUT/'actual_train_args.yaml',actual)
            write_json(OUT/'process.json',dict(pid=os.getpid(),dispatch=plan['dispatch'],status='RUNNING',started=utc()))
            print('LCD_TRAINING_RUNNING',flush=True)
        trainer.add_callback('on_train_start',setup)
        trainer.train();code=0
    except KeyboardInterrupt:code=130;raise
    finally:write_json(OUT/'exit.json',dict(code=code,dispatch=plan['dispatch'],finished=utc()))


def status():
    plan=read(OUT/'dispatch.json') if (OUT/'dispatch.json').exists() else {}
    processes=process_rows();state='NOT_DISPATCHED'
    exit_record=read(OUT/'exit.json') if (OUT/'exit.json').exists() else None
    shell=read(OUT/'shell_exit.json') if (OUT/'shell_exit.json').exists() else None
    log=Path(plan.get('log',OUT/'missing.log'));tail=''
    if log.is_file():
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-6000));tail=f.read().decode(errors='replace')
    if exit_record:state='COMPLETED' if exit_record['code']==0 else 'FAILED'
    elif shell:state='FAILED'  # shell exited without a Python completion record
    elif plan:
        proc=read(OUT/'process.json') if (OUT/'process.json').exists() else {}
        state='RUNNING' if processes and proc.get('status')=='RUNNING' and log.is_file() and log.stat().st_size else ('SETTING_UP' if processes else 'DISPATCHED_OR_INTERRUPTED')
    run=Path(YAML.load(CONFIG)['save_dir']) if CONFIG.exists() else MAIN/'runs/c_series'/RUN_NAME
    children=[]
    for p in processes:
        path=Path('/proc')/str(p['pid'])/'task'/str(p['pid'])/'children'
        if path.exists():children+=path.read_text().split()
    session_active=bool(shutil.which('tmux')) and subprocess.run(['tmux','has-session','-t','='+SESSION],capture_output=True).returncode==0
    csv_tail=''
    if (run/'results.csv').is_file():
        with (run/'results.csv').open('rb') as f:
            f.seek(max(0,(run/'results.csv').stat().st_size-4096));csv_tail='\n'.join(f.read().decode(errors='replace').splitlines()[-2:])
    result=dict(status=state,session=SESSION,session_active=session_active,processes=processes,children=children,log=str(log),exit=exit_record,shell_exit=shell,
                results_csv_tail=csv_tail,
                results_csv=str(run/'results.csv'),best_exists=(run/'weights/best.pt').exists(),last_exists=(run/'weights/last.pt').exists())
    print(json.dumps(result,ensure_ascii=False,indent=2));print(tail)
    return result


def evaluate(split):
    from c19_lif_v1_results import evaluate as mother_evaluate
    require_preflight();require(read(OUT/'exit.json')['code']==0,'Formal training has not completed')
    prepared=read(OUT/'prepare.json')
    require(dataset_inventory(Path(prepared['data_root']))==read(OUT/'dataset_inventory.json'),'Data changed')
    args=YAML.load(CONFIG);best=Path(args['save_dir'])/'weights/best.pt'
    return mother_evaluate(best,Path(args['data']),split,OUT/('evaluation_'+split),
                           val_report=OUT/'evaluation_val/metrics.json' if split=='test' else None,
                           model_verifier=verify_model,metric_summary=metric_summary)


def metric_summary(metrics):
    import numpy as np
    from ultralytics.utils.metrics import smooth
    box=metrics.box;i=int(smooth(box.f1_curve.mean(0),.1).argmax())
    p,r=np.asarray(box.p_curve)[0],np.asarray(box.r_curve)[0]
    return dict(working_point=dict(confidence=float(box.px[i]),precision=float(p[i]),recall=float(r[i]),
                                  f1=float(box.f1_curve[0,i]),policy='native smoothed maximum-F1 at IoU .5'),
                recall_at_precision={str(level):float(r[p>=level].max()) if np.any(p>=level) else None for level in (.8,.9,.95)},
                auxiliary_scope='native 1000-point confidence curves, IoU .5; no threshold or checkpoint tuning',
                historical_mother=dict(val_mAP50_95=.52454272,test_mAP50_95=.52200902,retested=False),
                performance_conclusion='Assess full val metrics first; test is locked to this val-selected checkpoint')


def pack():
    OUT.mkdir(parents=True,exist_ok=True)
    stage=OUT/'pack_stage'/utc();stage.mkdir(parents=True)
    (stage/'source.patch').write_bytes(subprocess.check_output(['git','diff',BASE,'HEAD'],cwd=ROOT))
    write_json(stage/'identity.json',dict(commit=git('rev-parse','HEAD'),branch=BRANCH,source=source_signature(),
                                        formal_training='NOT_RUN' if not (OUT/'dispatch.json').exists() else status()))
    files=[p for p in OUT.glob('*.json')]+[p for p in OUT.glob('*.yaml')]
    files+=list((ROOT/'docs/lcd_v1').glob('*'))
    files+=list((OUT/'diagnostics').glob('*.json'))
    for split in ('val','test'):
        files+=list((OUT/f'evaluation_{split}').glob('metrics.json'))
    if CONFIG.exists():
        run=Path(YAML.load(CONFIG)['save_dir'])
        files += [p for p in (run/'results.csv',run/'args.yaml') if p.exists()]
    for log in OUT.glob('console_*.log'):
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-32000));(stage/log.name).write_bytes(f.read())
    dest=OUT/f'lcd_v1_LIGHT_{utc()}.tar.gz'
    with tarfile.open(dest,'w:gz') as tar:
        for p in stage.iterdir():tar.add(p,arcname=p.name)
        for i,p in enumerate(files):
            if p.is_file() and p.stat().st_size<=1024*1024:tar.add(p,arcname=f'evidence/{i}_{p.name}')
    require(dest.stat().st_size<=8*1024*1024,'LIGHT exceeds 8MB; inspect archive, no automatic deletion')
    print(str(dest),sha256(dest),dest.stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','start','resume','status','worker','val','test','pack','local-check','capacity','verify-load'])
    parser.add_argument('--main',type=Path,default=MAIN);parser.add_argument('--data',type=Path)
    parser.add_argument('--local',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--folder',type=Path);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--expected')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.action=='prepare':prepare(args.main,args.data,args.local)
    elif args.action=='local-check':
        local_checks(args.main/'weights/rtdetr_r18_lite_imagenet_backbone_init.pt',args.data,args.folder)
    elif args.action=='preflight':preflight()
    elif args.action in ('start','resume'):dispatch(args.action=='resume')
    elif args.action=='status':status()
    elif args.action=='worker':worker(args.resume)
    elif args.action in ('val','test'):evaluate(args.action)
    elif args.action=='pack':pack()
    elif args.action=='capacity':capacity(args.folder)
    elif args.action=='verify-load':
        model=torch_load(args.checkpoint,map_location='cpu')['model'];verify_model(model)
        require(tensor_digest(model)==args.expected,'Fresh-process load differs')
        require(torch.count_nonzero(model.model[5].lcd.O.weight)>0,'Updated LCD was lost')
        print('FRESH_PROCESS_PASSED')


if __name__=='__main__':main()
