"""Two synthetic epochs with native SGD, save/restore, and production public export.

This uses batch2/64 and an explicit SMOKE UUID. It is never formal training or
a proof of batch16/640 capacity for parallel server jobs.
"""
from __future__ import annotations
import argparse
from copy import copy
import gc
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch
import uuid
from support import (ROOT, ASSET, configure, read_json, native_recipe, recipe, run_identity,
                     validate_checkpoint, write_json, sha256, checked_weight, SOURCE, runtime_environment)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--config',type=Path,help='Freeze a strict candidate YAML, then apply explicitly recorded smoke batch2/64/2-epoch overrides')
    p.add_argument('--train-images',type=int,choices=(8,64),default=64)
    a=p.parse_args(); run=a.run.absolute()
    config=None
    smoke_id='SMOKE_'+uuid.uuid4().hex
    if a.config:
        from configuration import freeze_config,frozen_config
        config=freeze_config(run,a.config.read_bytes(),
            {'python':str(Path(sys.executable).resolve()),'source':str(SOURCE),'weights':str(ASSET)},
            smoke_id,require_clean=False)
    else:
        run.mkdir(parents=True,exist_ok=False)
    source=configure(run/'runtime')
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from ultralytics.data.build import build_dataloader
    from ultralytics.utils.torch_utils import ModelEMA
    from adapters import ComparisonTrainer, strict_amp_probe, reset_workers, capture_rng
    from run import load_initial_model
    from data import preflight
    torch.set_num_threads(2)
    device='0' if a.device=='cuda' else 'cpu'
    if a.device=='cuda':
        assert torch.cuda.is_available(), 'Requested CUDA smoke requires a GPU'
    report={'scope':'SMOKE_ONLY_SYNTHETIC','formal_training':False,
            'formal_batch16_imgsz640_parallel_capacity':'NOT_TESTED','device':a.device}
    asset_hash=sha256(checked_weight())
    if a.device=='cuda':
        model,record=load_initial_model(); model=model.cuda().train(); ema=ModelEMA(model)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.937)
        before={k:v.clone() for k,v in model.state_dict().items()}
        ema_before={k:v.clone() for k,v in ema.ema.state_dict().items()}
        rng=capture_rng(); assert strict_amp_probe(model)
        assert model.training and all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        assert all(torch.equal(v,ema.ema.state_dict()[k]) for k,v in ema_before.items())
        assert not optimizer.state and ema.updates==0
        after=capture_rng()
        assert rng['python']==after['python'] and np.array_equal(rng['numpy'][1],after['numpy'][1])
        assert torch.equal(rng['torch_cpu'],after['torch_cpu']) and all(torch.equal(x,y) for x,y in zip(rng['torch_cuda'],after['torch_cuda']))
        report['amp_isolation']='real COCO-adapted M; FP32/FP16 accuracy; model/BN/EMA/optimizer and all RNG streams unchanged'
        report['local_gpu']=torch.cuda.get_device_name(0)
        del model,ema,optimizer,before,ema_before; gc.collect(); torch.cuda.empty_cache()
    else:
        report['amp_isolation']='NOT_TESTED_ON_CPU'

    shared=run/'synthetic'
    for split in ('train','val','test'):
        (shared/'images'/split).mkdir(parents=True); (shared/'labels'/split).mkdir(parents=True)
        for i in range(a.train_images if split=='train' else 2):
            # Textured synthetic inputs avoid degenerate all-constant BatchNorm gradients.
            pixels=np.random.default_rng(42+i).integers(0,256,(79+i%5,101+i%7,3),dtype=np.uint8)
            Image.fromarray(pixels).save(shared/'images'/split/(str(i)+'.jpg'))
            (shared/'labels'/split/(str(i)+'.txt')).write_text('0 .5 .5 .4 .4\n' if i%2==0 else '',encoding='utf-8')
    data=run/'source.yaml'; data.write_text(yaml.safe_dump({'path':shared.as_posix(),'names':{0:'crack'},
        **{s:'images/'+s for s in ('train','val','test')}}),encoding='utf-8')
    manifest=preflight(data,run,shared,enforce_counts=False)
    environment=runtime_environment() if config is not None else None
    if environment is not None:
        write_json(run/'environment.json',environment)
    ident=run_identity(manifest,source,require_clean=False,run_id=smoke_id,config=config,environment=environment,
                       run_uuid=read_json(run/'run_id.json').get('run_uuid') if config is not None else None)
    if config is None:
        write_json(run/'run_id.json',{'run_id':ident['run_id']})
    write_json(run/'identity.json',ident)
    gradients=[]; parameters={}

    class SmokeTrainer(ComparisonTrainer):
        def get_dataloader(self,dataset_path,batch_size=2,rank=0,mode='train'):
            ds=self.build_dataset(dataset_path,mode,2)
            loader=build_dataloader(ds,2,0,shuffle=mode=='train',rank=-1)
            loader.reset=lambda:reset_workers(loader)
            return loader
        def optimizer_step(self):
            grads=[p.grad for p in self.model.parameters() if p.grad is not None]
            assert grads and torch.isfinite(self.loss).all()
            row={'tensors':len(grads),'nonzero':sum(bool(torch.count_nonzero(g)) for g in grads),
                 'finite_scaled_gradients':all(bool(torch.isfinite(g).all()) for g in grads),
                 'scaler_before':self.scaler.get_scale()}
            result=super().optimizer_step()
            row['scaler_after']=self.scaler.get_scale()
            row['native_amp_overflow_skip']=row['scaler_after']<row['scaler_before']
            gradients.append(row)
            return result

    overrides=native_recipe(config)
    overrides.update(model=str(ASSET),data=str(run/'data.yaml'),project=str(run),name='train',
        exist_ok=False,epochs=2,batch=2,imgsz=64,workers=0,plots=False,close_mosaic=0,amp=a.device=='cuda',device=device)
    t=SmokeTrainer(overrides=overrides,run=run,manifest=manifest,identity=ident,config=config)
    def before_updates(trainer):
        expected=config or recipe()
        assert type(trainer.optimizer).__name__==expected['optimizer']
        if expected['optimizer']=='SGD':
            assert all(g['nesterov'] for g in trainer.optimizer.param_groups)
        assert not trainer.model.model[-1].dfl.conv.weight.requires_grad
        assert torch.equal(trainer.model.model[-1].dfl.conv.weight.flatten().cpu(),torch.arange(16,dtype=torch.float32))
        assert [n for n,p in trainer.model.named_parameters() if not p.requires_grad]==['model.22.dfl.conv.weight']
        parameters['before']=next(trainer.model.parameters()).detach().cpu().clone()
        report['native_optimizer']={'type':type(trainer.optimizer).__name__,
            'groups':[{k:g.get(k) for k in ('lr','initial_lr','momentum','betas','weight_decay','nesterov')} for g in trainer.optimizer.param_groups],
            'trainable_elements':sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            'loss_weights':{k:getattr(trainer.args,k) for k in ('box','cls','dfl')},
            'nominal_accumulate_before_warmup':trainer.accumulate,'augmentation':trainer.train_loader.dataset.transform_report}
    t.add_callback('on_pretrain_routine_end',before_updates)
    def interrupt_after_complete_epoch(trainer):
        if trainer.epoch==1:
            raise KeyboardInterrupt('Intentional smoke interruption after complete epoch checkpoint')
    t.add_callback('on_train_epoch_start',interrupt_after_complete_epoch)
    try: t.train()
    except KeyboardInterrupt: pass
    else: raise AssertionError('Smoke interruption not exercised')
    write_json(run/'gradient_trace.json',{'gradients':gradients})
    assert not torch.equal(parameters['before'],next(t.model.parameters()).detach().cpu())
    assert any(g['finite_scaled_gradients'] and not g['native_amp_overflow_skip'] for g in gradients)
    report['nonzero_sgd_update']=True; report['gradients']=gradients
    ckpt=torch.load(t.last,map_location='cpu',weights_only=False)
    validate_checkpoint(ckpt,ident,resume=True,config=config)
    assert ckpt['train_args']['weight_decay']==(config or recipe())['weight_decay']
    report['initialization']=read_json(run/'initialization.json')
    assert report['initialization']['transferred_tensors']==469
    for change in ({'run_id':'other-pilot'},{'initialization_type':'random'}, {'model':'other-model'},
                   {'run_id':'SMOKE_foreign'},{'initialization_type':'coco'}):
        try: validate_checkpoint({**ckpt,'comparison_identity':{**ident,**change}},ident,resume=True,config=config)
        except ValueError: pass
        else: raise AssertionError('Foreign checkpoint accepted')
    original_record=(run/'initialization.json').read_bytes()
    expected_state=ckpt['comparison_training_state']
    del t,ckpt; gc.collect(); torch.cuda.empty_cache()
    overrides.update(model=str(run/'train/weights/last.pt'),resume=str(run/'train/weights/last.pt'))
    if config is not None:
        config=frozen_config(run)
    t=SmokeTrainer(overrides=overrides,run=run,manifest=manifest,identity=ident,config=config)
    def equal_state(x,y):
        if isinstance(x,torch.Tensor): return torch.equal(x,y.cpu())
        if isinstance(x,dict): return x.keys()==y.keys() and all(equal_state(v,y[k]) for k,v in x.items())
        if isinstance(x,(tuple,list)): return len(x)==len(y) and all(equal_state(v,w) for v,w in zip(x,y))
        if isinstance(x,np.ndarray): return np.array_equal(x,y)
        return x==y
    def verify_restore(trainer):
        assert equal_state(expected_state['model'],trainer.model.state_dict())
        assert equal_state(expected_state['optimizer'],trainer.optimizer.state_dict())
        assert equal_state(expected_state['ema'],trainer.ema.ema.state_dict())
        assert equal_state(expected_state['scaler'],trainer.scaler.state_dict())
        assert equal_state(expected_state['scheduler'],trainer.scheduler.state_dict())
        assert equal_state(expected_state['rng'],capture_rng())
        assert trainer.comparison_optimizer_steps>0 and trainer.ema.updates>0
        assert trainer.optimizer.state and trainer.start_epoch==1
        assert trainer.args.weight_decay==(config or recipe())['weight_decay']
        report['resume_raw_weight_decay']=trainer.args.weight_decay
        report['resume_effective_weight_decay']=[g['weight_decay'] for g in trainer.optimizer.param_groups]
        report['restore_before_next_update']='exact equality: FP32 training model, optimizer momentum/groups, EMA, scaler, scheduler, Python/NumPy/Torch CPU/CUDA RNG'
    t.add_callback('on_pretrain_routine_end',verify_restore)
    t.train()
    assert t.epoch+1==2 and original_record==(run/'initialization.json').read_bytes()
    assert sha256(ASSET)==asset_hash
    report['lifecycle']={'batch':2,'imgsz':64,'epochs':2,'amp':a.device=='cuda',
        'interrupted_after_epoch':1,'resumed_to_epoch':2,'initialization_record_preserved':True,
        'foreign_checkpoint_rejection':'passed','native_resume_not_bitwise_worker_replay':True}
    completed={'status':'completed','completed_epoch':2,'best_epoch':t.comparison_best_epoch,
        'best_sha256':sha256(t.best),'last_sha256':sha256(t.last),'scope':'SMOKE_ONLY',
        'exit_code':0,'stop_reason':'epoch_limit'}
    write_json(run/'train_status.json',completed)
    del t,expected_state; gc.collect(); torch.cuda.empty_cache()
    from export import export_split,evaluate_split,SETTINGS
    SETTINGS['device']=device
    with patch('export.run_identity',side_effect=lambda m,s,**kw:run_identity(m,s,require_clean=False,**kw)):
        for split in ('val','test'):
            result=export_split(run,split,manifest,source)
            assert result['images']==2
            metrics=evaluate_split(run,split,manifest,source)
            report['public_'+split]={'images':result['images'],'raw_0_1':metrics['raw_0_1'],
                'best_checkpoint_sha256':result['checkpoint_sha256'],'settings':result['settings']}
    assert report['public_val']['best_checkpoint_sha256']==report['public_test']['best_checkpoint_sha256']
    report['export_evaluate']='same selected smoke best; production FP32 640 exporter and shared evaluator; two synthetic images per split'
    report['candidate_config_sha256']=ident.get('config_sha256')
    report['smoke_overrides']={k:overrides[k] for k in ('epochs','batch','imgsz','workers','plots','close_mosaic','amp','device')}
    write_json(run/'pilot_smoke.json',report); print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
