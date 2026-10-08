"""Explicit small-package whitelist; weights, raw images, environments and symlinks excluded."""
from __future__ import annotations
from pathlib import Path
import os
import shlex
import zipfile
from configuration import frozen_config,paths
from export import final_identity,show_summary,validate_cache
from support import HERE,ROOT,canonical,git,read_json,sha256,status,write_json
MAX_BYTES=100000000
RUN_PATTERNS=('*.json','*.yaml','structure/*.json','structure/*.yaml','structure/official_includes/**/*.yml',
    'data/*.json','data/*.yaml','data/*.txt','data/gt/*.json','predictions/*.jsonl.gz','predictions/*_complete.json',
    'metrics/*.json','curves/*.json','curves/*.png','train/results.csv','train/results.png',
    'train/val_metrics/*.json','augmentation/*.json','logs/*.log')

def safe_file(path,root):
    path=Path(path); root=Path(root).absolute()
    for item in (path,*path.parents):
        if item==root.parent: break
        if item.is_symlink(): raise ValueError('Package whitelist encountered symlink: '+str(item))
    if not path.resolve().is_relative_to(root.resolve()): raise ValueError('Package file escapes source root')
    if path.suffix in {'.pth','.pt','.ckpt','.safetensors'}: raise ValueError('Weights cannot enter small package')
    return path

def handoff(run,cfg):
    run=Path(run); identity=read_json(run/'identity.json'); receipt=read_json(run/'preflight_receipt.json')
    best=read_json(run/'train_status.json'); locations=paths(cfg)
    commands={'project':str(ROOT),'python':str(Path(os.sys.executable).absolute()),'code_sha':git('rev-parse','HEAD'),
        'commands':{stage:shlex.join([str(Path(os.sys.executable).absolute()),str(HERE/'run.py'),stage,'--run-dir',str(run)])
            for stage in ('preflight','train','final','evaluate','pack')},
        'explicit_resume':shlex.join([str(Path(os.sys.executable).absolute()),str(HERE/'run.py'),'train','--run-dir',str(run),
            '--resume',str(run/'train/weights/last.pth')]),'original_launch':read_json(run/'launch_command.json')}
    write_json(run/'server_commands.json',commands)
    info={'verification_status':'Synthetic local smoke only' if identity['scope']=='SMOKE_ONLY' else 'Observed on executing host; formal full val/test receipts required',
        'scope':identity['scope'],'server_project':str(ROOT),'interpreter':str(Path(os.sys.executable).absolute()),
        'sys_prefix':os.sys.prefix,'model_code_sha':identity['model_code_sha'],'upstream_commit':identity['upstream_commit'],
        'official_source':locations['source'],'config':str(run/'resolved_config.yaml'),'config_sha256':identity['config_sha256'],
        'checkpoint':str(run/'train/weights/best.pth'),'checkpoint_sha256':best['best_sha256'],
        'selected_source':best['selected_source'],'selected_key':best['selected_key'],
        'ema_training_state_key':'last.ema.module' if cfg['ema'] else None,
        'best_epoch':best['best_epoch'],'data_manifest':str(run/'data/manifest.json'),
        'dataset_root':read_json(run/'data/manifest.json')['data_root'],'dataset_identity_sha256':identity['dataset_identity_sha256'],
        'gt_paths':{s:str(run/f'data/gt/{s}.json') for s in ('val','test')},
        'predictions':{s:str(run/f'predictions/{s}.jsonl.gz') for s in ('val','test')},
        'metrics':{s:str(run/f'metrics/{s}.json') for s in ('val','test')},
        'load_interface':{'file':str(HERE/'model.py'),'call':'load_for_visualization(config, checkpoint, device) -> (native gradient-capable model, metadata)'},
        'layer_shapes':receipt['layer_shapes'],'candidate_layer_shapes':receipt.get('candidate_layer_shapes',{}),
        'forward':'eval dict pred_logits[B,300,1] raw logits, pred_boxes[B,300,4] normalized cxcywh',
        'input':'RGB float32 [0,1], fixed640 letterbox auto=False scaleup=True; per-record rounded gains/integer padding',
        'postprocessor':'Sizes are input640 WH, then inverse once; native0 -> public1; no NMS/clipping',
        'deploy_or_fusion':False,'grad_cam_implemented':False,'environment':read_json(run/'environment.json')}
    write_json(run/'visualization_handoff.json',info)

def package(run):
    run=Path(run).absolute(); cfg=frozen_config(run); manifest=read_json(run/'data/manifest.json')
    for split in ('val','test'):
        if not validate_cache(run,split,final_identity(run,split,cfg,manifest)): raise ValueError('Missing complete split')
        metric=run/f'metrics/{split}.json'; done=read_json(run/f'metrics/{split}_complete.json')
        if sha256(metric)!=done['sha256']: raise ValueError('Metrics checksum differs')
    from plotting import plot_training,plot_curves
    plot_training(run)
    for split in ('val','test'): plot_curves(run,split)
    show_summary(run); handoff(run,cfg); status(run,'package','running')
    files={}
    for pattern in RUN_PATTERNS:
        for path in run.glob(pattern):
            if path.is_file(): files['run/'+path.relative_to(run).as_posix()]=safe_file(path,run)
    # Snapshot only the committed small adapter and exact public evaluator components.
    for path in HERE.rglob('*'):
        if path.is_file() and path.suffix in {'.py','.yaml','.json','.txt','.md'}:
            files['code/dfine_m/'+path.relative_to(HERE).as_posix()]=safe_file(path,HERE)
    for rel in ('scripts/autodl_dfine_m_configurable.sh','docs/comparison/DFINE_M_CONFIGURABLE_SERVER_COMMANDS.md',
        'benchmarks/comparison/common.py','benchmarks/comparison/dataset.py',
        'benchmarks/comparison/evaluation/evaluate.py','benchmarks/comparison/evaluation/native_metrics.py',
        'benchmarks/comparison/evaluation/native_provenance.json'):
        path=ROOT/rel; files['code/'+rel]=safe_file(path,ROOT)
    required=('run/train/results.csv','run/train/results.png','run/visualization_handoff.json',
        'run/predictions/val.jsonl.gz','run/predictions/test.jsonl.gz','run/metrics/val.json','run/metrics/test.json')
    if not set(required)<=set(files): raise ValueError('Missing core whitelist artifact')
    listing={n:{'bytes':p.stat().st_size,'sha256':sha256(p)} for n,p in sorted(files.items())}
    destination=run.parent/(run.name+'_results.zip'); temporary=destination.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted(files.items()): archive.write(path,name)
        archive.writestr('FILE_MANIFEST.json',canonical({'files':listing,'excludes':'All weights/checkpoints/raw images/venvs/vendor',
            'GT_bytes_preserved':True,'max_bytes_exclusive':MAX_BYTES}))
    if temporary.stat().st_size>=MAX_BYTES:
        # Only reduce optional logs; core predictions/configs/metrics/GT stay untouched on server and in the archive.
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for name,path in sorted(files.items()):
                if name.startswith('run/logs/') and path.stat().st_size>1000000:
                    continue
                archive.write(path,name)
            kept={n:v for n,v in listing.items() if not (n.startswith('run/logs/') and v['bytes']>1000000)}
            archive.writestr('FILE_MANIFEST.json',canonical({'files':kept,'optional_omissions':[n for n in listing if n not in kept],
                'GT_bytes_preserved':True,'max_bytes_exclusive':MAX_BYTES}))
    if temporary.stat().st_size>=MAX_BYTES:
        temporary.unlink(); status(run,'package','failed',reason='Core package exceeds strict size limit')
        raise ValueError('Core package still >=100,000,000 bytes; server artifacts retained')
    with zipfile.ZipFile(temporary) as archive:
        bad=archive.testzip()
        if bad: raise ValueError('Zip checksum validation failed: '+bad)
        inventory=read_json_from_zip(archive)
        for name,record in inventory['files'].items():
            from support import digest
            if digest(archive.read(name))!=record['sha256']: raise ValueError('Archive content hash differs: '+name)
    temporary.replace(destination)
    done={'path':str(destination),'bytes':destination.stat().st_size,'sha256':sha256(destination),
        'strictly_under_100000000':True,'weights_included':False}
    write_json(run/'package_receipt.json',done); status(run,'package','completed',**done)
    print('PACKAGE completed '+str(destination)+' bytes='+str(done['bytes'])+' SHA256='+done['sha256'],flush=True)
    show_summary(run)
    return done

def read_json_from_zip(archive):
    import json
    return json.loads(archive.read('FILE_MANIFEST.json'))
