"""Synthetic local-only GPU training -> interrupted save -> resume -> FP32 val/test -> package."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import uuid
from support import ROOT,canonical,environment,read_json,run_identity,sha256,write_json
from configuration import freeze_config,resolve_config
from data import preflight
from model import build_model,prepare_weights,seed_all
from engine import train,checked_pair,load_checkpoint,validate_checkpoint
from export import export_split,evaluate_split
from test_core import synthetic_data

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path)
    args=parser.parse_args(); base=(args.output or ROOT/'outputs'/('SMOKE_ONLY_FRCNN_'+uuid.uuid4().hex)).absolute()
    base.mkdir(parents=True,exist_ok=False); run=base/'run'; data=synthetic_data(base/'data')
    cfg=resolve_config({'epochs':3,'warmup_epochs':1,'close_mosaic':1,'batch_size':2,'eval_batch_size':2,
        'train_workers':0,'eval_workers':0,'imgsz':64,'model_transform_min_size':64,'model_transform_max_size':64,
        'lr0':.0001,'data_yaml':str(data)})
    freeze_config(run,cfg,b'{}\n',{}, {'mode':'internal_synthetic_smoke'},scope='SMOKE_ONLY_SYNTHETIC',require_clean=False)
    manifest=preflight(data,run,enforce_counts=False)
    from probe import probe
    probe(run,cfg,run/'preflight_model.json',local_compat=True)
    env=environment(strict=False); init=prepare_weights(cfg)
    write_json(run/'initialization_identity.json',init)
    identity=run_identity(manifest,read_json(run/'run_id.json')['run_uuid'],cfg,env,init,scope='SMOKE_ONLY_SYNTHETIC',require_clean=False)
    try: train(run,manifest,identity,cfg,env,interrupt_after_epoch=1)
    except KeyboardInterrupt: pass
    else: raise ValueError('Synthetic interruption did not occur')
    last,index=checked_pair(run,identity,cfg)
    if last['completed_epoch']!=1: raise ValueError('Wrong interruption boundary')
    for label,other in [('other_run',{**identity,'run_uuid':'foreign'}),('scratch',{**identity,'initialization_type':'random'})]:
        try: validate_checkpoint(last,other,cfg,resume=True)
        except ValueError: pass
        else: raise ValueError('Checkpoint identity rejection failed: '+label)
    del last; gc.collect()
    result=train(run,manifest,identity,cfg,env,resume=True)
    if result['completed_epoch']!=3: raise ValueError('Smoke did not complete frozen epoch budget')
    for split in ('val','test'):
        export_split(run,split,manifest,identity,cfg)
        evaluate_split(run,split)
        # Same complete cache must return without reconstructing the detector.
        from unittest.mock import patch
        with patch('model.build_model',side_effect=AssertionError('Unexpected repeated GPU inference')):
            export_split(run,split,manifest,identity,cfg)
    from visualization import load_for_visualization
    restored=load_for_visualization(run,run/'checkpoints/best.pt','cuda:0')
    import torch
    if not torch.is_grad_enabled(): raise ValueError('Visualization globally disabled gradients')
    rgb=__import__('numpy').zeros((51,93,3),dtype='uint8'); image,geometry=restored['preprocess_rgb'](rgb)
    with torch.enable_grad():
        image.requires_grad_(); transformed,_=restored['model'].transform([image],None)
        feature=restored['model'].backbone(transformed.tensors)['3']; feature.sum().backward()
    if image.grad is None or not torch.isfinite(image.grad).all(): raise ValueError('Gradient-capable recovery failed')
    del restored; gc.collect(); torch.cuda.empty_cache()
    from packaging_run import pack
    package=pack(run,base/'smoke_results.tar.gz')
    evidence={'status':'PASSED_LOCAL_SYNTHETIC_GPU_SMOKE','scope':'SMOKE_ONLY_SYNTHETIC','run':str(run),
        'environment':env,'initialization_sha256':init['sha256'],'model_structure':read_json(run/'initialization.json')['structure'],
        'epochs':3,'optimizer_steps':result['optimizer_steps'],'saved_then_resumed_at_epoch':1,
        'public_splits':{s:read_json(run/'metrics'/(s+'.json'))['raw_0_1'] for s in ('val','test')},
        'prediction_cache_reuse_without_GPU':True,'gradient_capable_restore':True,'package':package,
        'formal_training_results':False,'server_batch16_640_capacity':'NOT_TESTED','target_torch2_1_2_vision0_16_2':'NOT_TESTED_LOCAL_COMPATIBILITY'}
    write_json(base/'smoke_validation.json',evidence); print(json.dumps({'evidence':str(base/'smoke_validation.json'),'package':package},indent=2))
if __name__=='__main__': main()
