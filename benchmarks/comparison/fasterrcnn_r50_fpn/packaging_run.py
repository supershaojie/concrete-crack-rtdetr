"""Explicit core whitelist, excluded weights, and strict <100,000,000-byte archive."""
from __future__ import annotations
from datetime import datetime,timezone
import io
from pathlib import Path
import tarfile
import uuid
from configuration import frozen_config
from support import HERE,ROOT,atomic_bytes,canonical,read_json,sha256,write_json

LIMIT=100000000
ROOT_CORE=('user_config.yaml','resolved_config.yaml','cli_overrides.json','config_input.json','config_identity.json',
    'run_id.json','launch_command.json','identity.json','initialization_identity.json','initialization.json','environment.json',
    'source_correspondence.json','manifest.json','data.yaml','train.txt','val.txt','test.txt','input_checksums.json',
    'preflight_model.json','preflight_identity.json','preflight_status.json','actual_training_setup.json',
    'training_progress.json','train_status.json','epoch_trace.jsonl','summary.json','visualization_handoff.json','visualization_layers.json','tmux_context.json')

def check_results(run):
    from export import evaluate_split
    cfg=frozen_config(run)
    from data import verify_inputs
    manifest=verify_inputs(run,())
    if manifest['dataset_identity_sha256']!=read_json(run/'identity.json')['dataset_identity_sha256']:
        raise ValueError('Run manifest identity differs')
    state=read_json(run/'train_status.json')
    if state.get('status')!='completed': raise ValueError('Pack requires completed training and full val/test')
    if not cfg['early_stopping'] and state['completed_epoch']!=cfg['epochs']: raise ValueError('Incomplete epoch budget')
    if sha256(run/'checkpoints/best.pt')!=state['best_sha256']: raise ValueError('Selected best changed')
    from engine import checked_pair
    checked_pair(run,read_json(run/'identity.json'),cfg)
    # CPU-only cache validation: no source image IO or GPU inference.
    for split in ('val','test'):
        seal=read_json(run/'metrics'/(split+'_complete.json')); metrics=Path(seal['metrics_path'])
        done=read_json(run/'predictions'/(split+'_complete.json')); pred=run/'predictions'/(split+'.jsonl.gz')
        if seal['status']!='completed' or sha256(metrics)!=seal['sha256'] or sha256(pred)!=done['sha256']:
            raise ValueError('Result seal differs: '+split)
        value=read_json(metrics)
        from export import evaluator_api,prediction_identity
        settings=read_json(run/'predictions'/(split+'_inference_settings.json')) if (run/'predictions'/(split+'_inference_settings.json')).exists() else {}
        expected=prediction_identity(read_json(run/'identity.json'),state['best_sha256'],split,run/'gt'/(split+'.json'),{**cfg,**settings})
        identity,rows=evaluator_api().read_public(pred)
        if identity!=expected or value['identity']!=expected or done['checkpoint_sha256']!=state['best_sha256'] or value['predictions_sha256']!=done['sha256']:
            raise ValueError('Cache belongs to another best/run/config/data: '+split)
        verified=evaluator_api().evaluate(read_json(run/'gt'/(split+'.json')),rows,identity)
        if verified['images']!=done['images']: raise ValueError('Incomplete prediction receipt')
        for key in ('precision','recall','AP50','AP75','mAP50_95'):
            if value[key]!=verified[key]: raise ValueError('Cached metric replay differs: '+key)
    return cfg,state

def assert_allowed_file(path,root):
    path=Path(path); root=Path(root).absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent!=root.parent): raise ValueError('Symlink cannot enter package: '+str(path))
    if not path.resolve().is_relative_to(root.resolve()): raise ValueError('File outside whitelist root')
    if path.suffix in ('.pt','.pth','.ckpt','.safetensors','.npy'): raise ValueError('Weight/checkpoint/cache forbidden in light package')

def build_archive(destination,files,metadata,limit=LIMIT):
    destination=Path(destination).absolute(); destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): raise FileExistsError('Existing package protected')
    temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    inventory={name:{'bytes':path.stat().st_size,'sha256':sha256(path),'source':str(path)} for name,path in files.items()}
    metadata={**metadata,'files':inventory,'size_limit_exclusive':limit}
    with temporary.open('xb') as out, tarfile.open(fileobj=out,mode='w:gz') as archive:
        raw=canonical(metadata); info=tarfile.TarInfo('package_manifest.json'); info.size=len(raw); archive.addfile(info,io.BytesIO(raw))
        for name,path in sorted(files.items()):
            if sha256(path)!=inventory[name]['sha256']: raise ValueError('Evidence changed before packing: '+name)
            archive.add(path,arcname=name,recursive=False)
            if sha256(path)!=inventory[name]['sha256']: raise ValueError('Evidence changed during packing: '+name)
    if temporary.stat().st_size>=limit:
        # Keep partial archive and original core evidence for diagnosis; never deliver oversized package.
        raise ValueError(f'Package {temporary.stat().st_size} bytes exceeds exclusive cap {limit}; core artifacts retained: {temporary}')
    import os
    os.rename(temporary,destination)
    return {'path':str(destination),'bytes':destination.stat().st_size,'sha256':sha256(destination),'weights_included':False,'files':len(files)}

def pack(run,out=None):
    run=Path(run).absolute(); cfg,state=check_results(run)
    from run import summary
    from visualization import write_handoff
    summary(run); write_handoff(run)
    files={}
    required=('resolved_config.yaml','identity.json','initialization.json','environment.json','manifest.json','train_status.json',
        'train/results.csv','train/results.png','visualization_handoff.json')
    for name in required:
        if not (run/name).is_file(): raise FileNotFoundError('Missing required package core: '+name)
    for name in ROOT_CORE:
        if (run/name).is_file(): files['run/'+name]=run/name
    for split in ('val','test'):
        required_split=[f'predictions/{split}.jsonl.gz',f'predictions/{split}_complete.json',f'gt/{split}.json',
            f'metrics/{split}_complete.json',f'curves/{split}/curve_values.json']
        for name in required_split:
            if not (run/name).is_file(): raise FileNotFoundError('Missing package core '+name)
            files['run/'+name]=run/name
        metric=Path(read_json(run/'metrics'/(split+'_complete.json'))['metrics_path'])
        files['run/'+metric.relative_to(run).as_posix()]=metric
    for folder,patterns in {'predictions':('*_inference_settings.json',),'gt':('*_origin.json',),
        'train':('results.csv','results.png'),'curves':('*.png',),'native_auxiliary':('*.json','*.log'),
        'metrics':('*.log',),'stage_invocations':('*.json',),'epoch_metrics':('*.json',),'resumes':('*.json',),'transform_trees':('0001.json',f"{state['completed_epoch']:04d}.json")}.items():
        for pattern in patterns:
            for path in (run/folder).rglob(pattern):
                files['run/'+path.relative_to(run).as_posix()]=path
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py','.yaml','.json','.txt','.md'): files['implementation/'+path.name]=path
    for name in ('benchmarks/comparison/common.py','benchmarks/comparison/dataset.py','benchmarks/comparison/evaluation/evaluate.py',
        'benchmarks/comparison/evaluation/native_metrics.py','benchmarks/comparison/evaluation/native_provenance.json',
        'scripts/autodl_fasterrcnn_r50_fpn_configurable.sh','docs/comparison/FASTERRCNN_R50_FPN_CONFIGURABLE_SERVER_COMMANDS.md'):
        files['source/'+name]=ROOT/name
    for name,path in files.items(): assert_allowed_file(path,run if name.startswith('run/') else ROOT)
    destination=Path(out).absolute() if out else ROOT/'outputs/fasterrcnn-packages'/(run.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]+'.tar.gz')
    if destination.is_relative_to(run): raise ValueError('Package output must be outside original run')
    metadata={'scope':read_json(run/'run_id.json')['scope'],'best_epoch':state['best_epoch'],'actual_epochs':state['completed_epoch'],
        'best_sha256':state['best_sha256'],'weights_included':False,'weights_retained_on_server':str(run/'checkpoints'),
        'excluded':['COCO/best/last weights','optimizer/scaler states','dataset images','env/vendor/cache directories'],
        'optional_omissions':['per-batch detection images','confusion matrix (not computed)','intermediate transform trees'],
        'model_code_sha':read_json(run/'identity.json')['model_code_sha']}
    try: result=build_archive(destination,files,metadata)
    except ValueError as exc:
        if 'exceeds exclusive cap' not in str(exc): raise
        removed=[name for name in files if name.startswith('run/curves/') and name.endswith('.png') or name.startswith('run/native_auxiliary/') and name.endswith('.log')]
        if not removed: raise
        reduced={k:v for k,v in files.items() if k not in removed}
        result=build_archive(destination,reduced,{**metadata,'optional_omissions':metadata['optional_omissions']+removed})
    write_json(destination.with_suffix(destination.suffix+'.receipt.json'),result)
    print('PACKAGE '+str(result),flush=True); return result
