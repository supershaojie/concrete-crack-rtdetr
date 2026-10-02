"""Prepare, train, export, evaluate or run the frozen sequence. No implicit asset installs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone

from support import (HERE, PROJECT, atomic_json, environment, identity, read_json, sha256, write_json,
                     initialization_record)
from assets import verify
from data import inspect, save_check, make_gt

def utc():
    return datetime.now(timezone.utc).isoformat()

class ChildInterrupted(KeyboardInterrupt):
    def __init__(self, code):
        self.exit_code=128-code if code<0 else code
        super().__init__('Child interrupted: '+str(code))

def isolated_env():
    env=os.environ.copy()
    for name in ('PYTHONPATH','RANK','LOCAL_RANK','WORLD_SIZE'):
        env.pop(name,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONUTF8='1')
    return env

def stage(run, name, command):
    """Record real subprocess and Python tee exit codes separately, including signals/log failures."""
    state_path=run/'status.json'
    state=read_json(state_path) if state_path.exists() else {'stages':{}}
    if name in state['stages']:
        raise FileExistsError('Stage already recorded; no implicit overwrite: '+name)
    entry={'command':command,'started':utc(),'status':'running','process_exit_code':None,'tee_exit_code':None}
    state['stages'][name]=entry
    atomic_json(state_path,state)
    process=None
    tee_code=0
    try:
        with (run/(name+'.log')).open('xb') as log:
            process=subprocess.Popen(command,cwd=PROJECT,env=isolated_env(),stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
            while True:
                block=process.stdout.read1(65536)
                if not block:
                    break
                log.write(block)
                log.flush()
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
            rc=process.wait()
        entry['process_exit_code']=rc
        entry['status']='completed' if rc==0 else 'failed'
        if rc:
            if rc < 0 or rc in (130,143):
                raise ChildInterrupted(rc)
            raise subprocess.CalledProcessError(rc,command)
    except BaseException as e:
        if process and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        entry['process_exit_code']=process.returncode if process else None
        entry['status']='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed'
        entry['error']=repr(e)
        if isinstance(e,(OSError,BrokenPipeError)):
            tee_code=1
        raise
    finally:
        if process and process.stdout:
            process.stdout.close()
        entry['tee_exit_code']=tee_code
        entry['finished']=utc()
        atomic_json(state_path,state)

def worker(run, assets, action, *extra):
    return [sys.executable,str(HERE/'worker.py'),action,'--run',str(run),'--assets',str(assets),*extra]

def prepare(a):
    if a.run.exists():
        raise FileExistsError('Run output exists; choose a new name or explicit train --resume')
    assets=verify(a.assets)
    a.run.mkdir(parents=True,exist_ok=False)
    atomic_json(a.run/'status.json',{'created':utc(),'stages':{},'preparation':'running'})
    try:
        manifest=inspect(a.data,a.data_root,a.source_project)
        save_check(manifest,a.run/'data')
        recipe=read_json(HERE/'recipe.json')
        # Import the actual evaluator policy; no independent copy of its metric implementation.
        sys.path.insert(0,str(HERE.parent/'evaluation'))
        from evaluate import POLICY_SHA
        init=initialization_record(assets['upstream'])
        checkpoint_identity={'run_id':uuid.uuid4().hex, 'code':identity(),
                             'data_identity':manifest['dataset_identity_sha256'], 'initialization':init}
        frozen={'identity':identity(),'recipe':recipe,'hyp':__import__('yaml').safe_load((HERE/'hyp.yaml').read_text()),
                'initialization':init, 'checkpoint_identity':checkpoint_identity,
                'input_checksums':{p.name:sha256(p) for p in (a.run/'data').iterdir() if p.is_file()},
                'assets':assets,'environment':environment(),
                'data_identity':manifest['dataset_identity_sha256'],
                'prediction_identity':{
                    'model':recipe['model'],'model_code_sha':identity()['project_commit'],
                    'initialization_type':init['initialization_type'], 'pretraining_source':init['pretraining_source'],
                    'pretrained_tensors_loaded':init['pretrained_tensors_loaded'], 'run_id':checkpoint_identity['run_id'],
                    'dataset_identity_sha256':manifest['dataset_identity_sha256'],
                    'evaluation_config_sha256':POLICY_SHA,
                    'postprocessing':recipe['evaluation'],
                    'upstream_commit':assets['lock']['commit'],'patch_sha256':assets['patch_sha256']}}
        write_json(a.run/'frozen.json',frozen)
        print(json.dumps(frozen,indent=2),flush=True)
        stage(a.run,'check',worker(a.run,a.assets,'check'))
        state=read_json(a.run/'status.json');state['preparation']='completed';atomic_json(a.run/'status.json',state)
    except BaseException as e:
        state=read_json(a.run/'status.json')
        state['preparation']='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed'
        state['error']=repr(e);atomic_json(a.run/'status.json',state)
        raise

def guard(a):
    if read_json(a.run/'status.json').get('preparation') != 'completed':
        raise ValueError('Preparation/import check has not completed')
    frozen=read_json(a.run/'frozen.json')
    if frozen['identity']!=identity():
        raise ValueError('Code/config identity changed after freeze')
    if frozen['assets']!=verify(a.assets):
        raise ValueError('Asset identity/path changed')
    if frozen['environment']!=environment():
        raise ValueError('Environment changed after freeze')
    for name, expected in frozen['input_checksums'].items():
        if sha256(a.run/'data'/name)!=expected:
            raise ValueError('Frozen input artifact changed: '+name)
    return frozen

def train(a):
    guard(a)
    current=inspect(a.data,a.data_root,a.source_project)
    if current['dataset_identity_sha256']!=read_json(a.run/'frozen.json')['data_identity']:
        raise ValueError('Data list/labels/sizes changed')
    name='train'
    extra=[]
    if a.resume:
        if (a.run/'native/training_complete.json').exists():
            raise ValueError('Completed runs cannot resume')
        if not (a.run/'native/weights/last.pt').exists():
            raise FileNotFoundError('Explicit resume requires this run native/weights/last.pt')
        number=len([x for x in read_json(a.run/'status.json')['stages'] if x.startswith('train')])
        name='train_resume_'+str(number)
        extra=['--resume']
    stage(a.run,name,worker(a.run,a.assets,'train',*extra))

def export(a):
    guard(a)
    manifest=read_json(a.run/'data/manifest.json')
    # Lightweight recheck avoids exporting against labels changed during a long training run.
    current=inspect(a.data,a.data_root,a.source_project)
    if current['dataset_identity_sha256']!=manifest['dataset_identity_sha256']:
        raise ValueError('Data changed since preparation')
    selected=read_json(a.run/'native/training_complete.json')
    best=a.run/'native/weights/best.pt'
    selection=a.run/'selected_best.json'
    value={'checkpoint_sha256':sha256(best),**selected}
    if selection.exists() and read_json(selection)!=value:
        raise ValueError('Selected best changed between split exports')
    write_json(selection,value)
    out=a.run/'evaluation';out.mkdir(exist_ok=True)
    for split in ('val','test'):
        gt_path=out/(split+'_gt.json')
        if not gt_path.exists():
            if a.gt_cache:
                gt=make_gt(manifest,split,a.gt_cache/(split+'.json'))
            else:
                gt=None
                roots=[a.source_project/'outputs/comparison_prepare',
                       PROJECT.parent/'Crack_RTDETR-comparison-base/outputs/comparison_prepare']
                candidates=sorted({p for root in roots for p in root.glob('*/coco/'+split+'.json')})
                for cached in candidates:
                    try:
                        gt=make_gt(manifest,split,cached)
                        print('Reusing checked public GT: '+str(cached),flush=True)
                        break
                    except (ValueError,KeyError,OSError) as e:
                        print('Cached GT not usable: '+str(cached)+': '+str(e),flush=True)
                if gt is None:
                    gt=make_gt(manifest,split)
            write_json(gt_path,gt)
        if not (out/(split+'_export.json')).exists():
            stage(a.run,'export_'+split,worker(a.run,a.assets,'export','--split',split))

def evaluate(a):
    out=a.run/'evaluation'
    selected=read_json(a.run/'selected_best.json')
    for split in ('val','test'):
        receipt=read_json(out/(split+'_export.json'))
        predictions=out/(split+'_predictions.jsonl')
        if receipt['checkpoint_sha256']!=selected['checkpoint_sha256'] or receipt['predictions_sha256']!=sha256(predictions):
            raise ValueError('Export cache identity changed')
        target=out/(split+'_unified_metrics.json')
        if target.exists():
            raise FileExistsError('Metrics already exist; use evaluator CLI with a new output for CPU recomputation')
        stage(a.run,'evaluate_'+split,[sys.executable,str(HERE.parent/'evaluation/evaluate.py'),
              '--gt',str(out/(split+'_gt.json')),'--predictions',str(predictions),'--output',str(target)])
    summary(a.run)

def summary(run):
    native=read_json(run/'native/training_complete.json') if (run/'native/training_complete.json').exists() else None
    result={'training':native or 'NOT_EXECUTED','output':str(run),'units':'raw values 0-1; percent=raw*100'}
    for split in ('val','test'):
        p=run/'evaluation'/(split+'_unified_metrics.json')
        if p.exists():
            d=read_json(p)
            result[split]={k:d[k] for k in ('precision','recall','AP50','AP75','mAP50_95')}
        else:
            result[split]='NOT_EXECUTED'
    atomic_json(run/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','train','export','evaluate','pipeline','summary','resources'])
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=PROJECT/'outputs/yolov5m-scratch/assets')
    p.add_argument('--source-project',type=Path,default=Path('/root/autodl-tmp/projects/Crack_RTDETR'))
    p.add_argument('--data',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--gt-cache',type=Path,help='optional existing public COCO directory; checked against actual labels/IDs')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    a.run,a.assets,a.source_project=a.run.resolve(),a.assets.resolve(),a.source_project.resolve()
    a.data=(a.data or a.source_project/'configs/crack_autodl.yaml').resolve()
    a.data_root=(a.data_root or a.source_project/'datasets/crack_det').resolve()
    if a.gt_cache:a.gt_cache=a.gt_cache.resolve()
    if a.resume and a.action!='train':
        p.error('--resume is only allowed with explicit train action')
    signal_exit={'code':130}
    def interrupted(signum,frame):
        signal_exit['code']=128+signum
        raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    lock=a.run.with_name(a.run.name+'.active.lock')
    lock.parent.mkdir(parents=True,exist_ok=True)
    # Atomic lock guards concurrent invocations. Stale locks require explicit inspection.
    with lock.open('x',encoding='utf-8') as f:
        f.write(str(os.getpid()))
    owns_state=a.action not in ('prepare','pipeline') or not a.run.exists()
    outcome='failed'
    try:
        if a.action in ('prepare','pipeline'):prepare(a)
        if a.action in ('train','pipeline'):train(a)
        if a.action in ('export','pipeline'):export(a)
        if a.action in ('evaluate','pipeline'):evaluate(a)
        if a.action=='summary':summary(a.run)
        if a.action=='resources':
            guard(a)
            stage(a.run,'resources',worker(a.run,a.assets,'resources'))
        outcome='completed'
    except KeyboardInterrupt as exc:
        outcome='interrupted'
        raise SystemExit(getattr(exc,'exit_code',signal_exit['code']))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    finally:
        if owns_state and (a.run/'status.json').exists():
            state=read_json(a.run/'status.json')
            state['last_action']={'name':a.action,'status':outcome,'finished':utc()}
            atomic_json(a.run/'status.json',state)
        lock.unlink()

if __name__=='__main__':
    main()
