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
import shutil
from datetime import datetime, timezone

from support import (HERE, PROJECT, atomic_json, environment, identity, read_json, sha256, write_json,
                     initialization_record)
from assets import verify, prepare as prepare_assets, DEFAULT_CACHE
from data import inspect, save_check, make_gt, attach_dimensions
from config import read_yaml, resolve, freeze_config, snapshot, recipe, run_id
from support import canonical, digest

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

def prepare(a, small_sample=False):
    if a.run.exists():
        raise FileExistsError('Run output exists; choose a new run-id or use resume --run-id')
    resolve(read_yaml(a.config))  # report invalid candidates before creating any runtime/run
    assets=prepare_assets(a.assets,a.asset_cache)
    a.run.mkdir(parents=True,exist_ok=False)
    atomic_json(a.run/'status.json',{'created':utc(),'stages':{},'preparation':'running'})
    try:
        config=freeze_config(a.config,a.run)
        manifest=inspect(a.data,a.data_root,a.source_project,enforce_counts=not small_sample)
        attach_dimensions(manifest,a.source_project,a.dataset_cache,small_sample=small_sample)
        save_check(manifest,a.run/'data')
        resolved_recipe=recipe(config)
        # Import the actual evaluator policy; no independent copy of its metric implementation.
        sys.path.insert(0,str(HERE.parent/'evaluation'))
        from evaluate import POLICY_SHA
        init=initialization_record(assets['upstream'],config)
        config_hash=digest(canonical(config))
        checkpoint_identity={'run_id':a.run_id, 'run_uuid':uuid.uuid4().hex, 'code':identity(),
                             'config_sha256':config_hash, 'scope':'SMOKE_ONLY_SYNTHETIC' if small_sample else 'FORMAL_CANDIDATE',
                             'data_identity':manifest['dataset_identity_sha256'], 'initialization':init}
        frozen={'identity':identity(),'recipe':resolved_recipe,'hyp':read_yaml(a.run/'train_hyp.yaml'),
                'scope':checkpoint_identity['scope'], 'config_sha256':config_hash,
                'config_checksums':{name:sha256(a.run/name) for name in
                                    ('user_config.yaml','resolved_config.yaml','train_hyp.yaml')},
                'paths':{k:str(getattr(a,k)) if getattr(a,k) is not None else None for k in
                         ('assets','source_project','data','data_root','gt_cache','dataset_cache')},
                'initialization':init, 'checkpoint_identity':checkpoint_identity,
                'input_checksums':{p.name:sha256(p) for p in (a.run/'data').iterdir() if p.is_file()},
                'assets':assets,'environment':environment(),
                'data_identity':manifest['dataset_identity_sha256'],
                'prediction_identity':{
                    'model':config['model'],'model_code_sha':identity()['project_commit'],
                    'initialization_type':init['initialization_type'], 'pretraining_source':init['pretraining_source'],
                    'pretrained_tensors_loaded':init['pretrained_tensors_loaded'], 'run_id':checkpoint_identity['run_id'],
                    'dataset_identity_sha256':manifest['dataset_identity_sha256'],
                    'evaluation_config_sha256':POLICY_SHA,
                    'config_sha256':config_hash, 'postprocessing':config['evaluation'],
                    'upstream_commit':assets['lock']['commit'],'patch_sha256':assets['patch_sha256']}}
        write_json(a.run/'frozen.json',frozen)
        print(json.dumps({k:frozen[k] for k in ('scope','identity','config_sha256','paths')},indent=2),flush=True)
        stage(a.run,'check',worker(a.run,a.assets,'check'))
        model_check=read_json(a.run/'model_check.json')
        if model_check['transferred_tensors'] != init['pretrained_tensors_loaded']:
            raise ValueError('Actual official COCO compatible transfer differs from its pinned reference')
        frozen['model_check_sha256']=sha256(a.run/'model_check.json')
        frozen['resolved_options_sha256']=sha256(a.run/'resolved_options.json')
        write_json(a.run/'frozen.json',frozen)
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
    snapshot(a.run)
    if frozen['identity']!=identity():
        raise ValueError('Fixed code/defaults identity changed after freeze; keep this run at its training SHA')
    if frozen['assets']!=verify(a.assets):
        raise ValueError('Asset identity/path changed')
    if frozen['environment']!=environment():
        raise ValueError('Environment changed after freeze')
    for name, expected in frozen['input_checksums'].items():
        if sha256(a.run/'data'/name)!=expected:
            raise ValueError('Frozen input artifact changed: '+name)
    for name in ('model_check','resolved_options'):
        if sha256(a.run/(name+'.json')) != frozen[name+'_sha256']:
            raise ValueError('Frozen runtime check changed: '+name)
    current=inspect(a.data,a.data_root,a.source_project,enforce_counts=frozen['scope']!='SMOKE_ONLY_SYNTHETIC')
    manifest=read_json(a.run/'data/manifest.json')
    if current['dataset_identity_sha256']!=frozen['data_identity'] or current['data_yaml_sha256']!=manifest['data_yaml_sha256']:
        raise ValueError('Data list/labels/sizes or source data YAML changed')
    return frozen

def train(a):
    guard(a)
    name='train'
    extra=[]
    if a.resume_training:
        if (a.run/'native/training_complete.json').exists():
            raise ValueError('Completed runs cannot resume')
        if not (a.run/'native/weights/last.pt').exists():
            raise FileNotFoundError('Explicit resume requires this run native/weights/last.pt')
        number=len([x for x in read_json(a.run/'status.json')['stages'] if x.startswith('train')])
        name='train_resume_'+str(number)
        extra=['--resume']
    stage(a.run,name,worker(a.run,a.assets,'train',*extra))

def retry_stage(run,name,command):
    recorded=read_json(run/'status.json')['stages']
    if name in recorded:
        index=1
        while name+'_retry_'+str(index) in recorded:
            index+=1
        name=name+'_retry_'+str(index)
    stage(run,name,command)

def export(a, splits=('val',)):
    frozen=guard(a)
    manifest=read_json(a.run/'data/manifest.json')
    selected=read_json(a.run/'native/training_complete.json')
    best=a.run/'native/weights/best.pt'
    selection=a.run/'selected_best.json'
    value={'checkpoint_sha256':sha256(best),'config_sha256':frozen['config_sha256'],
           'checkpoint_identity':frozen['checkpoint_identity'],**selected}
    if selection.exists() and read_json(selection)!=value:
        raise ValueError('Selected best changed between split exports')
    if not selection.exists():
        write_json(selection,value)
    out=a.run/'evaluation';out.mkdir(exist_ok=True)
    for split in splits:
        gt_path=out/(split+'_gt.json')
        if not gt_path.exists():
            if a.gt_cache:
                gt=make_gt(manifest,split,a.gt_cache/(split+'.json'))
            else:
                gt=None
                roots=[a.source_project/'outputs/comparison_prepare',
                       a.source_project.parent/'Crack_RTDETR-comparison-base/outputs/comparison_prepare']
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
        cached_gt=read_json(gt_path)
        expected_gt=make_gt(manifest,split)  # frozen sizes/labels only; no image content reads
        if (cached_gt['info']['dataset_identity_sha256']!=manifest['dataset_identity_sha256'] or
            cached_gt['info']['split']!=split or any(cached_gt[k]!=expected_gt[k]
            for k in ('images','annotations','categories'))):
            raise ValueError('Public GT cache differs from frozen run labels/dimensions: '+split)
        if (out/(split+'_export.json')).exists():
            validate_cache(a.run,split)
        else:
            retry_stage(a.run,'export_'+split,worker(a.run,a.assets,'export','--split',split))
            validate_cache(a.run,split)

def validate_cache(run,split,metrics=False):
    frozen=read_json(run/'frozen.json')
    selected=read_json(run/'selected_best.json')
    out=run/'evaluation'
    receipt=read_json(out/(split+'_export.json'))
    predictions=out/(split+'_predictions.jsonl')
    gt=out/(split+'_gt.json')
    if (selected['checkpoint_sha256']!=sha256(run/'native/weights/best.pt') or
        receipt['status']!='completed' or receipt['checkpoint_sha256']!=selected['checkpoint_sha256'] or
        receipt['config_sha256']!=frozen['config_sha256'] or receipt['checkpoint_identity']!=frozen['checkpoint_identity'] or
        receipt['gt_sha256']!=sha256(gt) or receipt['predictions_sha256']!=sha256(predictions)):
        raise ValueError('Export cache identity changed: '+split)
    with predictions.open(encoding='utf-8') as stream:
        header=json.loads(stream.readline())
    expected={**frozen['prediction_identity'],'checkpoint_sha256':selected['checkpoint_sha256'],'split':split}
    if header.get('type')!='metadata' or any(header['identity'].get(k)!=v for k,v in expected.items()):
        raise ValueError('Prediction metadata differs from run snapshot: '+split)
    if receipt['images']!=len(read_json(gt)['images']):
        raise ValueError('Export coverage differs: '+split)
    if metrics:
        result=read_json(out/(split+'_unified_metrics.json'))
        if (receipt.get('metrics_sha256')!=sha256(out/(split+'_unified_metrics.json')) or
            result['policy_sha256']!=frozen['prediction_identity']['evaluation_config_sha256'] or
            result['identity']!=header['identity'] or result['gt_sha256']!=sha256(gt) or
            result.get('predictions_sha256')!=sha256(predictions) or result['images']!=receipt['images']):
            raise ValueError('Public metrics cache identity changed: '+split)
    return receipt

def evaluate(a, splits=('val',)):
    guard(a)
    out=a.run/'evaluation'
    for split in splits:
        validate_cache(a.run,split)
        predictions=out/(split+'_predictions.jsonl')
        target=out/(split+'_unified_metrics.json')
        if target.exists():
            validate_cache(a.run,split,metrics=True)
            print('Reusing identity-checked public '+split+' metrics',flush=True)
            continue
        retry_stage(a.run,'evaluate_'+split,[sys.executable,str(HERE.parent/'evaluation/evaluate.py'),
              '--gt',str(out/(split+'_gt.json')),'--predictions',str(predictions),'--output',str(target)])
        value=read_json(target)
        value['predictions_sha256']=sha256(predictions)
        atomic_json(target,value)
        receipt=read_json(out/(split+'_export.json'))
        receipt['metrics_sha256']=sha256(target)
        atomic_json(out/(split+'_export.json'),receipt)
        validate_cache(a.run,split,metrics=True)
    summary(a.run)

def summary(run):
    complete=run/'native/training_complete.json'
    epoch_state=run/'native/epoch_state.json'
    if complete.exists():
        native={**read_json(complete),'completion':'completed'}
    elif epoch_state.exists():
        native={**read_json(epoch_state),'completion':'incomplete'}
        if (run/'status.json').exists():
            stages=read_json(run/'status.json')['stages']
            attempts=[v for k,v in stages.items() if k.startswith('train')]
            if attempts:
                native['latest_train_attempt']=max(attempts,key=lambda v:v['started'])
    else:
        native='INCOMPLETE_NO_SAVED_EPOCH' if (run/'native').exists() else 'NOT_EXECUTED'
    result={'native_training_val':native,
            'selected_best':read_json(run/'selected_best.json') if (run/'selected_best.json').exists() else 'NOT_SELECTED',
            'training':native,'output':str(run),'units':'raw values 0-1; percent=raw*100'}
    if (run/'frozen.json').exists():
        frozen=read_json(run/'frozen.json')
        result.update(scope=frozen['scope'],run_id=frozen['checkpoint_identity']['run_id'],
                      training_code_sha=frozen['identity']['project_commit'],config_sha256=frozen['config_sha256'])
    for split in ('val','test'):
        p=run/'evaluation'/(split+'_unified_metrics.json')
        if p.exists():
            d=read_json(p)
            raw={k:d[k] for k in ('precision','recall','AP50','AP75','mAP50_95')}
            result[split]={'raw':raw,'percent':{k:100*v for k,v in raw.items()},
                'ap_by_class':d['ap_by_class'],'identity':d['identity'],'gt_sha256':d['gt_sha256'],
                'predictions_sha256':sha256(run/'evaluation'/(split+'_predictions.jsonl'))}
        else:
            result[split]='NOT_EXECUTED'
    atomic_json(run/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)

def finalize(a):
    guard(a)
    if not (a.run/'native/training_complete.json').is_file():
        raise ValueError('finalize requires a fully completed run; incomplete training cannot be finalized')
    validate_cache(a.run,'val',metrics=True)  # selection stage must already have completed
    export(a,('test',))
    evaluate(a,('test',))
    summary(a.run)

def archive(a):
    frozen=guard(a)
    for split in ('val','test'):
        validate_cache(a.run,split,metrics=True)
    if frozen['scope']!='FORMAL_CANDIDATE' or not (a.run/'native/training_complete.json').is_file():
        raise ValueError('Only a finalized formal run may be archived as an experiment')
    summary(a.run)
    target=a.archive_dir.resolve()
    target.mkdir(parents=True,exist_ok=False)
    names=['user_config.yaml','resolved_config.yaml','train_hyp.yaml','frozen.json','model_check.json',
           'selected_best.json','summary.json','status.json','native/training_complete.json',
           'native/effective_training.json','native/initialization.json','native/autoanchor.json',
           'native/results.csv','evaluation/val_unified_metrics.json','evaluation/test_unified_metrics.json']
    for name in names:
        path=a.run/name
        if path.is_file():
            dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    write_json(target/'archive_identity.json',{'training_code_sha':frozen['identity']['project_commit'],
               'run_id':a.run_id,'config_sha256':frozen['config_sha256'],
               'files':{name:sha256(target/name) for name in names if (target/name).is_file()}})
    print('Frozen configuration and small results archived: '+str(target),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','start','resume','finalize','export','evaluate','summary','resources','archive'])
    p.add_argument('--run-id',required=True,type=run_id)
    p.add_argument('--config',type=Path,help='new run YAML; never needed for resume/finalize')
    p.add_argument('--output-root',type=Path,default=PROJECT/'outputs/yolov5m-coco-native-ft-v1/runs')
    p.add_argument('--assets',type=Path,help='prepared native source runtime (frozen per run)')
    p.add_argument('--asset-cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--source-project',type=Path)
    p.add_argument('--data',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--gt-cache',type=Path,help='optional existing public COCO directory; checked against actual labels/IDs')
    p.add_argument('--dataset-cache',type=Path,help='existing dimension/label dataset_manifest.json; lightweight identity checked')
    p.add_argument('--archive-dir',type=Path)
    a=p.parse_args()
    a.run=(a.output_root.resolve()/a.run_id)
    a.resume_training=a.action=='resume'
    fresh=a.action in ('prepare','start')
    if fresh:
        if a.config is None:p.error('new runs require --config YAML')
        a.config=a.config.resolve()
        a.assets=(a.assets or PROJECT/'outputs/yolov5m-coco-native-ft-v1/assets').resolve()
        a.source_project=(a.source_project or Path('/root/autodl-tmp/projects/Crack_RTDETR')).resolve()
        a.data=(a.data or a.source_project/'configs/crack_autodl.yaml').resolve()
        a.data_root=(a.data_root or a.source_project/'datasets/crack_det').resolve()
        for key in ('gt_cache','dataset_cache'):
            if getattr(a,key):setattr(a,key,getattr(a,key).resolve())
    else:
        if a.config is not None:p.error('--config is only allowed for a new prepare/start run; resume uses its snapshot')
        frozen=read_json(a.run/'frozen.json')
        if frozen['checkpoint_identity']['run_id']!=a.run_id:p.error('Run directory/run-id identity mismatch')
        for key,value in frozen['paths'].items():
            supplied=getattr(a,key)
            if supplied is not None and str(supplied.resolve())!=value:
                p.error('Run path differs from frozen snapshot: '+key)
            setattr(a,key,Path(value) if value is not None else None)
    if (a.action=='archive') != (a.archive_dir is not None):
        p.error('archive requires --archive-dir; other actions do not accept it')
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
    owns_state=not fresh or not a.run.exists()
    outcome='failed'
    try:
        if fresh:prepare(a)
        if a.action=='start':train(a)
        if a.action=='resume':
            if not (a.run/'native/training_complete.json').exists():
                # Prepared-only runs can enter training without loading a foreign checkpoint.
                a.resume_training=(a.run/'native/weights/last.pt').exists()
                train(a)
            else:
                guard(a)
                print('Training complete; continuing only this run public val stages',flush=True)
        if a.action in ('export','start','resume'):export(a)
        if a.action in ('evaluate','start','resume'):evaluate(a)
        if a.action=='finalize':finalize(a)
        if a.action=='archive':archive(a)
        if a.action=='summary':
            guard(a);summary(a.run)
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
        try:
            if owns_state and (a.run/'status.json').exists():
                state=read_json(a.run/'status.json')
                state['last_action']={'name':a.action,'status':outcome,'finished':utc()}
                atomic_json(a.run/'status.json',state)
                if a.action in ('start','resume','finalize'):
                    summary(a.run)  # interrupted/failed runs retain their real saved epoch count
        finally:
            lock.unlink()

if __name__=='__main__':
    main()
