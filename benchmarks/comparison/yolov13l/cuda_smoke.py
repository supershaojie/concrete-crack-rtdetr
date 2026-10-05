"""Explicit SMOKE_ONLY: synthetic data, 64 pixels, batch2, two epochs and same-run resume."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import uuid
from support import (ROOT, HERE, SOURCE, ASSET, configure, recipe, runtime_environment,
                     write_json, read_json, sha256, verify_checkpoint_files)
from configuration import freeze_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    import yaml
    source=configure(ROOT/'outputs/yolov13l-configurable-validation/cuda-runtime',SOURCE)
    import torch
    import numpy as np
    from PIL import Image
    from data import preflight, verify_inputs
    from run import train, load_initial_model, summary, guard
    from verification import preflight_model_probe
    from adapters import ComparisonTrainer, capture_rng
    from export import export_split, evaluate_split
    if not torch.cuda.is_available():
        write_json(args.out,{'status':'NOT_RUN','reason':'No CUDA device','formal_training_started':False})
        return
    torch.set_num_threads(2)
    run_id='SMOKE_ONLY_cuda_'+uuid.uuid4().hex[:8]
    run=ROOT/'outputs/yolov13l-configurable-validation'/run_id
    shared=run.parent/(run_id+'_data')
    for split in ('train','val','test'):
        (shared/'images'/split).mkdir(parents=True)
        (shared/'labels'/split).mkdir(parents=True)
        for i in range(4 if split=='train' else 2):
            Image.fromarray(np.random.default_rng(42+i).integers(0,256,(35+i,57+i,3),dtype=np.uint8)).save(shared/'images'/split/f'{i}.png')
            (shared/'labels'/split/f'{i}.txt').write_text('0 0.5 0.5 0.4 0.3\n' if i%2==0 else '')
    data_yaml=shared/'data.yaml'
    data_yaml.write_bytes(yaml.safe_dump({'path':str(shared),'train':'images/train','val':'images/val',
                                        'test':'images/test','names':{0:'crack'}}).encode())
    cfg={**recipe(),'epochs':2,'patience':0,'batch':2,'nbs':2,'workers':0,'imgsz':64,'warmup_epochs':0.,
         'close_mosaic':1,'plots':True}
    paths={'python':str(Path(os.sys.executable).absolute()),'sys_prefix':str(Path(os.sys.prefix).absolute()),
           'source':str(SOURCE),'weights':str(ASSET),'data':str(data_yaml),'data_root':str(shared),
           'public_coco':None,'reuse_run':None,'label_cache':None}
    freeze_config(run,yaml.safe_dump(cfg).encode(),paths,run_id,require_clean=False,smoke=True)
    write_json(run/'environment.json',runtime_environment())
    manifest=preflight(data_yaml,run,enforce_counts=False)
    initial,record=load_initial_model(config=cfg)
    rng=capture_rng()
    probe=preflight_model_probe(initial,cfg,run,manifest)
    if not torch.equal(rng['torch_cpu'],torch.get_rng_state()): raise ValueError('CPU RNG not restored by probe')
    del initial
    write_json(run/'preflight_model_probe.json',probe)
    write_json(run/'source_identity.json',source)
    write_json(run/'preflight_status.json',{'status':'completed','exit_code':0,'scope':'SMOKE_ONLY'})
    interrupted={'done':False}
    original_save=ComparisonTrainer.save_model
    def save_and_interrupt(t):
        original_save(t)
        if t.epoch==0 and not interrupted['done']:
            interrupted['done']=True
            raise KeyboardInterrupt('SMOKE_ONLY intentional interruption after atomic epoch1 checkpoint')
    # Exercises the real native trainer and checkpoint path, without completing first attempt.
    from unittest.mock import patch
    original_init=ComparisonTrainer.__init__
    def smoke_init(t,*a,**kw):
        original_init(t,*a,**kw)
        def unit_scale_for_short_smoke(trainer):
            if trainer.comparison_identity['scope'] != 'SMOKE_ONLY':
                raise ValueError('Smoke-only scaler adjustment cannot apply to formal training')
            if not trainer.resume:
                state=trainer.scaler.state_dict(); state['scale']=1.
                trainer.scaler.load_state_dict(state)
        t.add_callback('on_pretrain_routine_end',unit_scale_for_short_smoke)
    # Unit scaling makes this four-step synthetic test exercise real optimizer state;
    # the production adapter retains the native initial scale and automatic backoff.
    with patch.object(ComparisonTrainer,'__init__',smoke_init), patch.object(ComparisonTrainer,'save_model',save_and_interrupt):
        try:
            train(SimpleNamespace(run=run,resume=False),manifest,source)
        except KeyboardInterrupt:
            write_json(run/'train_status.json',{'status':'interrupted','exit_code':130,'scope':'SMOKE_ONLY'})
    if not interrupted['done']: raise ValueError('Smoke did not exercise checkpoint interruption')
    seals=verify_checkpoint_files(run)
    last=torch.load(run/'train/weights/last.pt',map_location='cpu',weights_only=False)
    for group in ('model','ema'):
        if any(v.is_floating_point() and v.dtype!=torch.float32 for v in last['comparison_training_state'][group].values()):
            raise ValueError('Resumable training state is not FP32')
    saved_steps=last['comparison_training_state']['optimizer_steps']
    del last
    guard(run,'resume')  # real launcher guard pins imports before trusted deserialization
    result=train(SimpleNamespace(run=run,resume=True),verify_inputs(run),source)
    if result['completed_epoch']!=2: raise ValueError('SMOKE_ONLY resumed budget changed')
    guard(run,'finalize')
    last=torch.load(run/'train/weights/last.pt',map_location='cpu',weights_only=False)
    actual_updates=last['comparison_training_state']['optimizer_updates']
    if actual_updates < 1 or not last['comparison_training_state']['optimizer']['state']:
        raise ValueError('Smoke did not exercise a real optimizer update/state')
    if [r['epoch'] for r in last['comparison_epoch_trace']] != [1,2]:
        raise ValueError('Resume did not preserve both checkpoint epoch traces')
    del last
    val=export_split(run,'val',manifest,source)
    evaluated_val=evaluate_split(run,'val',manifest,source)
    write_json(run/'export_val_status.json',{'status':'completed','exit_code':0})
    write_json(run/'evaluate_val_status.json',{'status':'completed','exit_code':0,'metrics':evaluated_val['metrics']})
    # Test never participated in selection; exercise identical selected best independently.
    test=export_split(run,'test',manifest,source)
    evaluated_test=evaluate_split(run,'test',manifest,source)
    if val['checkpoint_sha256']!=test['checkpoint_sha256']: raise ValueError('Smoke val/test selected different best')
    write_json(run/'export_test_status.json',{'status':'completed','exit_code':0})
    write_json(run/'evaluate_test_status.json',{'status':'completed','exit_code':0,'metrics':evaluated_test['metrics']})
    report={'status':'PASSED_CUDA_SMOKE_ONLY','run':str(run),'formal_training_started':False,
        'budget':{'epochs':2,'batch':2,'nbs':2,'imgsz':64,'workers':0,'smoke_only_initial_loss_scale':1},'cuda_device':torch.cuda.get_device_name(0),
        'actual_optimizer_updates':actual_updates,
        'initialization':record,'preflight':probe,'interrupted_epoch1_resume':True,'saved_optimizer_steps':saved_steps,
        'resume':read_json(run/'resume_state.json'),'completed_training':result,'val_export':val,'test_export':test,
        'launcher_resume_finalize_guards':'passed against actual checkpoints',
        'public_val':evaluated_val['display_percent'],'public_test':evaluated_test['display_percent'],
        'summary':summary(run), 'run_code_sha':read_json(run/'config_identity.json')['model_code_sha'],
        'worktree_dirty_at_freeze':read_json(run/'config_identity.json')['worktree_dirty_at_freeze'],
        'scope_note':'Synthetic verification only; no production data training or accuracy claim'}
    write_json(args.out,report)

if __name__=='__main__': main()
