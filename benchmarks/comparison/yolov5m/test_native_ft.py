"""Actual bounded synthetic native lifecycle. Never a formal training/initialization entry."""
from __future__ import annotations
import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import HERE, PROJECT, read_json, write_json, sha256, identity, environment
from config import snapshot, read_yaml
from run import prepare, guard, summary, isolated_env, validate_cache
from worker import activate, options, validate_resume_options, build_initial_model

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root,assets=args.output.resolve(),args.assets.resolve()
    root.mkdir(parents=True,exist_ok=False)
    import yaml
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    data_root=root/'fixture'
    for split,count in (('train',4),('val',2),('test',2)):
        (data_root/'images'/split).mkdir(parents=True)
        (data_root/'labels'/split).mkdir(parents=True)
        for i in range(count):
            size=(112,80) if i%2==0 else (133,79)
            Image.new('RGB',size,(50+i*15,100,150)).save(data_root/'images'/split/f'{i:02d}.png')
            (data_root/'labels'/split/f'{i:02d}.txt').write_text('0 .5 .5 .4 .4\n',encoding='utf-8')
    data=root/'data.yaml'
    data.write_text(yaml.safe_dump({'path':str(data_root),'names':{0:'crack'},
        **{s:'images/'+s for s in ('train','val','test')}}),encoding='utf-8')
    base={'epochs':3,'batch':2,'imgsz':64,'workers':0,'device':'cpu','amp':False,
          'nbs':4,'plots':False,'close_mosaic':1,'degrees':0,'translate':0,'scale':0,
          'hsv_h':0,'hsv_s':0,'hsv_v':0,'fliplr':0,'mosaic':1.,'mixup':1.}
    candidate=root/'candidate.yaml'
    candidate.write_text(yaml.safe_dump({**base,'optimizer':'SGD','lr0':.003}),encoding='utf-8')
    def arguments(name,source=candidate):
        return SimpleNamespace(run=root/'runs'/name,run_id=name,config=source,assets=assets,
            asset_cache=assets,source_project=PROJECT,data=data,data_root=data_root,
            gt_cache=None,dataset_cache=None,resume_training=False)
    a=arguments('config_a')
    prepare(a,small_sample=True)
    b_source=root/'candidate_b.yaml'
    b_source.write_text(yaml.safe_dump({**base,'epochs':2,'close_mosaic':0,'optimizer':'AdamW',
        'lr0':.006,'nbs':6,'weight_decay':.007,'hsv_h':.1,'degrees':5,
        'mosaic':0.,'mixup':0.,'warmup_bias_lr':.02}),encoding='utf-8')
    b=arguments('config_b',b_source)
    prepare(b,small_sample=True)
    verified=activate(assets)
    from validation import preprocessing
    from utils.callbacks import Callbacks
    from bench_yolov5_runtime import validate_checkpoint, strict_amp_probe
    import train
    opt_a,opt_b=options(a.run,assets),options(b.run,assets)
    assert opt_a.batch_size==2 and opt_b.optimizer=='AdamW' and opt_b.nbs==6
    assert opt_a.hyp==str(a.run/'train_hyp.yaml') and opt_b.hyp==str(b.run/'train_hyp.yaml')
    report={'scope':'SMOKE_ONLY_SYNTHETIC','formal_training':False,
            'server_dual_batch16_imgsz640_capacity':'NOT_TESTED','identity':identity(),
            'preprocessing':preprocessing(root),'official_asset':{'bytes':Path(verified['weights']).stat().st_size,
                'sha256':sha256(verified['weights'])},'lifecycle':{}}
    if torch.cuda.is_available():
        model,_=build_initial_model(verified,snapshot(a.run))
        model=model.cuda().train()
        original=model.model[0].bn.running_mean.detach().clone()
        rng=torch.cuda.get_rng_state().clone()
        assert strict_amp_probe(model)
        assert model.training and torch.equal(original,model.model[0].bn.running_mean)
        assert torch.equal(rng,torch.cuda.get_rng_state())
        report['local_actual_model_amp_probe']='passed; private FP32/FP16 copy; BN and CUDA RNG preserved'
        del model;gc.collect();torch.cuda.empty_cache()
    else:report['local_actual_model_amp_probe']='NOT_RUN_NO_LOCAL_CUDA'
    def interrupt_after_epoch(run, opt):
        callbacks=Callbacks();calls=[]
        def stop():
            calls.append(True)
            if len(calls)==2:raise KeyboardInterrupt('Intentional synthetic interruption after epoch 1')
        callbacks.register_action('on_train_epoch_start',callback=stop)
        try:train.train(opt.hyp,opt,torch.device('cpu'),callbacks)
        except KeyboardInterrupt:pass
        else:raise AssertionError('Interruption was not exercised')
        state=read_json(run/'status.json')
        state['stages']['train']={'status':'interrupted','started':'2026-10-03T00:00:00Z','process_exit_code':130,
                                  'scope':'direct native synthetic callback, not formal launcher'}
        write_json(run/'status.json',state)
        summary(run)
        assert read_json(run/'summary.json')['training']['completed_epochs']==1
        assert not (run/'native/training_complete.json').exists()
    interrupt_after_epoch(a.run,opt_a)
    ckpt=torch.load(a.run/'native/weights/last.pt',map_location='cpu',weights_only=False)
    validate_checkpoint(ckpt,a.run/'native',resume=True)
    validate_resume_options(ckpt,options(a.run,assets,True))
    for changed in ({**ckpt,'comparison_identity':None},
                    {**ckpt,'comparison_state':{**ckpt['comparison_state'],'initialization_sha256':'changed'}}):
        try:validate_checkpoint(changed,a.run/'native',resume=True)
        except ValueError:pass
        else:raise AssertionError('Foreign/incomplete checkpoint accepted')
    try:validate_resume_options({**ckpt,'opt':{**ckpt['opt'],'nbs':123}},options(a.run,assets,True))
    except ValueError:pass
    else:raise AssertionError('Changed resume configuration accepted')
    saved_steps=ckpt['comparison_state']['optimizer_steps']
    assert saved_steps>0 and ckpt['optimizer'] and ckpt['ema'] is not None
    original_init=sha256(a.run/'native/initialization.json')
    del ckpt,changed;gc.collect()
    candidate.write_text(yaml.safe_dump({**base,'lr0':.02,'mosaic':.25,'mixup':0}),encoding='utf-8')
    c=arguments('config_c')
    prepare(c,small_sample=True)
    assert snapshot(a.run)['lr0']==.003 and snapshot(c.run)['lr0']==.02
    candidate.rename(root/'moved_candidate.yaml')
    guard(a)  # original external path is now gone
    with patch('run.identity',return_value={**identity(),'project_commit':'0'*40}):
        try:guard(a)
        except ValueError:pass
        else:raise AssertionError('Fixed code guard removed')
    def command(action,obj,*extra,expected=0):
        result=subprocess.run([sys.executable,str(HERE/'run.py'),action,'--run-id',obj.run_id,
            '--output-root',str(root/'runs'),*extra],cwd=PROJECT,env=isolated_env())
        assert result.returncode==expected,(action,result.returncode,expected)
    command('resume',a)  # true native optimizer/EMA/scaler/scheduler/RNG restore, then public val only
    assert sha256(a.run/'native/initialization.json')==original_init
    complete=read_json(a.run/'native/training_complete.json')
    assert complete['completed_epochs']==3 and complete['mosaic_closed']
    effective=read_json(a.run/'native/effective_training.json')
    resumed=read_json(a.run/'native/effective_resume_1.json')
    assert effective['hyp_after_nc_imgsz_decay_scaling']==resumed['hyp_after_nc_imgsz_decay_scaling']
    assert effective['hyp_raw']['cls']==.3 and math.isclose(effective['hyp_after_nc_imgsz_decay_scaling']['cls'],.00375,rel_tol=1e-12)
    assert read_json(a.run/'summary.json')['test']=='NOT_EXECUTED'
    ckpt=torch.load(a.run/'native/weights/last.pt',map_location='cpu',weights_only=False)
    assert ckpt['comparison_state']['optimizer_steps']>saved_steps
    try:validate_checkpoint(ckpt,a.run/'native',resume=True)
    except ValueError:pass
    else:raise AssertionError('Completed checkpoint accepted for more training')
    del ckpt;gc.collect()
    val_hash=sha256(a.run/'evaluation/val_unified_metrics.json')
    train_stages=[s for s in read_json(a.run/'status.json')['stages'] if s.startswith('train')]
    command('finalize',a)
    assert val_hash==sha256(a.run/'evaluation/val_unified_metrics.json')
    assert train_stages==[s for s in read_json(a.run/'status.json')['stages'] if s.startswith('train')]
    validate_cache(a.run,'test',metrics=True)
    command('finalize',a)  # identity-matched test cache reuses cleanly
    command('resume',a,'--config',str(b_source),expected=2)
    command('finalize',c,expected=1)
    report['lifecycle']['SGD']={'completed_epochs':complete['completed_epochs'],'interrupted_after':1,
        'initialization_preserved':True,'raw_hyp_not_rescaled_on_resume':True,
        'effective':effective,'resume_effective':resumed,'anchors':read_json(a.run/'native/autoanchor.json'),
        'first_update':read_json(a.run/'native/first_update.json'),'start_test':'NOT_EXECUTED',
        'finalize':'public test only; val hash and train stages unchanged; repeated finalize cache reused'}
    prediction=a.run/'evaluation/val_predictions.jsonl'
    raw=prediction.read_bytes();prediction.write_bytes(raw+b'\n')
    try:validate_cache(a.run,'val',metrics=True)
    except ValueError:pass
    else:raise AssertionError('Altered predictions cache accepted')
    prediction.write_bytes(raw)
    metrics_file=a.run/'evaluation/val_unified_metrics.json'
    raw_metrics=metrics_file.read_bytes()
    changed_metrics=read_json(metrics_file);changed_metrics['precision']=.123
    write_json(metrics_file,changed_metrics)
    try:validate_cache(a.run,'val',metrics=True)
    except ValueError:pass
    else:raise AssertionError('Edited metrics cache accepted')
    metrics_file.write_bytes(raw_metrics)
    interrupt_after_epoch(b.run,opt_b)
    ckpt=torch.load(b.run/'native/weights/last.pt',map_location='cpu',weights_only=False)
    validate_checkpoint(ckpt,b.run/'native',resume=True)
    validate_resume_options(ckpt,options(b.run,assets,True))
    del ckpt;gc.collect()
    b_source.unlink()
    command('resume',b)
    eff_b=read_json(b.run/'native/effective_training.json')
    assert eff_b['optimizer_type'].endswith('AdamW')
    assert eff_b['hyp_raw']['lr0']==.006 and eff_b['hyp_raw']['hsv_h']==.1
    assert eff_b['hyp_after_nc_imgsz_decay_scaling']['weight_decay']==.007
    assert eff_b['hyp_raw']['degrees']==5 and eff_b['nbs']==6
    assert eff_b['options']['optimizer']=='AdamW' and eff_b['optimizer_groups'][0]['betas']==[.937,.999]
    assert read_json(b.run/'native/training_complete.json')['completed_epochs']==2
    report['lifecycle']['AdamW']={'interrupted_after':1,'completed_epochs':2,'effective':eff_b,
        'first_update':read_json(b.run/'native/first_update.json'),'resume_after_source_deleted':'passed'}
    from utils.torch_utils import smart_optimizer
    tiny=torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.BatchNorm2d(4))
    adam=smart_optimizer(tiny,'Adam',.005,.9,.001)
    before=tiny[1].weight.detach().clone()
    tiny(torch.randn(2,3,4,4)).square().mean().backward()
    adam.step()
    assert type(adam).__name__=='Adam' and adam.state and not torch.equal(before,tiny[1].weight)
    report['native_Adam_optimizer']='real 3-group native smart_optimizer and update passed'
    for obj in (a,b):
        calls=[json.loads(line) for line in (obj.run/'native/native_validation_calls.jsonl').read_text().splitlines()]
        assert all(x['requested_batch']==2 and x['loader_batch']==2 and x['rect'] and x['nms_iou']==.6 for x in calls)
        audit=read_json(obj.run/'model_check.json')
        assert audit['transferred_tensors']==475 and audit['model_tensors']==481 and audit['parameters_unfused']==20871318
    invalid=root/'invalid.yaml';invalid.write_text('bogus: 1\n')
    bad=arguments('invalid_run',invalid)
    command('prepare',bad,'--config',str(invalid),expected=1)
    assert not bad.run.exists()
    sys.path.insert(0,str(HERE.parent/'evaluation'))
    from evaluate import evaluate
    from test_preparation import PreparationTests
    gt,ident,rows=PreparationTests().fixtures()
    rows[0]['predictions']=[{'category_id':1,'score':.9,'bbox':[10.,10.,30.,30.]}]
    metrics=evaluate(gt,iter(rows),ident)
    assert metrics['empty_prediction_images']==1 and abs(metrics['AP50']-.995)<1e-8
    report['public_empty_prediction_fixture']='all images retained; empty prediction AP fixture passed'
    report['source_change_isolation']={'old_lr0':snapshot(a.run)['lr0'],'new_lr0':snapshot(c.run)['lr0'],
        'old_mosaic':snapshot(a.run)['mosaic'],'new_mosaic':snapshot(c.run)['mosaic'],
        'old_config_sha256':read_json(a.run/'frozen.json')['config_sha256'],
        'new_config_sha256':read_json(c.run/'frozen.json')['config_sha256']}
    write_json(root/'native_ft_validation.json',report)
    print('PASSED bounded native lifecycle, source isolation, public val/test, and optimizer checks',flush=True)

if __name__=='__main__':main()
