"""Continuous native-loss/AMP/EMA training; exact resume and isolated capacity probe."""
from __future__ import annotations
import csv
import io
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid
import numpy as np
import torch
from augment_b19 import snapshot
from configuration import frozen_config,paths
from data import train_loader
from export import training_val
from model import build,load_coco,selected_checkpoint
from optimizer import build_optimizer,UpdateSchedule
from support import (HERE,LOCK,atomic_bytes,canonical,digest,read_json,sha256,status,write_json)


def seed_everything(cfg):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.deterministic=cfg['deterministic']; torch.backends.cudnn.benchmark=not cfg['deterministic']
    torch.use_deterministic_algorithms(cfg['deterministic'],warn_only=True)
    return {'seed':cfg['seed'],'worker_seed':'torch.initial_seed modulo 2^32 -> random/numpy',
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'warn_only':True,
        'limit':'CUDA grid_sample backward can be nondeterministic; no bitwise reproducibility claim',
        'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']}

def rng_state(generator):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'loader_generator':generator.get_state()}

def restore_rng(state,generator):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader_generator'])

def save_torch(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.parent/('.tmp_'+uuid.uuid4().hex)
    try:
        with temp.open('xb') as f: torch.save(obj,f); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists(): temp.unlink()

def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

def identity(run,cfg,manifest):
    run=Path(run); frozen=read_json(run/'config_identity.json'); rid=read_json(run/'run_id.json')
    return {'model':'official_dfine_m_configurable_to_crack','model_code_sha':frozen['model_code_sha'],
        'adapter_sha256':frozen['adapter_sha256'],'upstream_commit':read_json(LOCK)['commit'],
        'initialization':'coco_detection_pretrained','initialization_sha256':read_json(LOCK)['weights']['sha256'],
        'dataset_identity_sha256':manifest['dataset_identity_sha256'],'config_sha256':frozen['config_sha256'],
        'environment_sha256':frozen['environment_sha256'],'run_uuid':rid['run_uuid'],'run_id':rid['run_id'],
        'run_path':str(run.absolute()),'scope':rid['scope'],'ema':cfg['ema']}

def setup(run,cfg,manifest,capacity=False):
    run=Path(run); seed=seed_everything(cfg); model,criterion,post,EMA=build(cfg,paths(cfg)['source'],cfg['device'],
        run/'structure')
    init=load_coco(model,paths(cfg)['weights'],None if capacity else run/'initialization_load.json')
    opt,groups=build_optimizer(model,cfg)
    ema=EMA(model,decay=cfg['ema_decay'],warmups=cfg['ema_warmup_updates'],start=0).to(cfg['device']) if cfg['ema'] else None
    scaler=torch.cuda.amp.GradScaler(enabled=cfg['amp'] and cfg['device'].startswith('cuda'),
        init_scale=cfg['amp_init_scale'],growth_factor=cfg['amp_growth_factor'],
        backoff_factor=cfg['amp_backoff_factor'],growth_interval=cfg['amp_growth_interval'])
    if cfg['amp'] and not cfg['device'].startswith('cuda'): print('Explicit CPU recipe: AMP unavailable; FP32 used and recorded',flush=True)
    n=sum(r['split']=='train' for r in manifest['records'])
    batches=n//cfg['batch_size'] if cfg['drop_last'] else math.ceil(n/cfg['batch_size'])
    per_epoch=math.ceil(batches/cfg['gradient_accumulation_steps'])
    schedule=UpdateSchedule(opt,per_epoch*cfg['epochs'],round(per_epoch*cfg['warmup_epochs']),cfg['warmup_start_factor'],cfg['lrf'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    if not capacity:
        write_json(run/'optimizer_groups.json',groups)
        write_json(run/'schedule.json',{'actual_loader_batches':batches,'updates_per_epoch':per_epoch,
            'total_updates':schedule.total,'warmup_updates':schedule.warmup,'units':'actual optimizer updates',
            'first_factor':schedule.factor(0),'last_warmup_factor':schedule.factor(schedule.warmup-1) if schedule.warmup else None,
            'last_update_factor':schedule.factor(schedule.total-1),'group_target_lrs':schedule.targets,
            'effective_batch':cfg['batch_size']*cfg['gradient_accumulation_steps'],'implicit_batch_lr_scaling':False,
            'mixing_close_zero_based_epoch':cfg['epochs']-cfg['close_mosaic'] if cfg['close_mosaic'] else None})
        write_json(run/'seed_strategy.json',seed)
    return model,criterion,post,ema,opt,scaler,schedule,generator

def finite_gradients(model):
    for name,param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all(): raise FloatingPointError('Nonfinite gradient: '+name)

def optimizer_step(model,opt,scaler,schedule,ema,cfg):
    if scaler.is_enabled(): scaler.unscale_(opt)
    finite_gradients(model)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_max_norm'],error_if_nonfinite=True) if cfg['gradient_clip_max_norm']>0 else torch.tensor(0.)
    lr=schedule.apply()
    if scaler.is_enabled(): scaler.step(opt); scaler.update()
    else: opt.step()
    schedule.advance()
    if ema is not None: ema.update(model)
    opt.zero_grad(set_to_none=True)
    return float(norm),lr

def loss_batch(model,criterion,images,targets,cfg,epoch,index,batches):
    images=images.to(cfg['device'],dtype=torch.float32)
    targets=[{k:v.to(cfg['device']) if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
    enabled=cfg['amp'] and cfg['device'].startswith('cuda')
    with torch.autocast(device_type=torch.device(cfg['device']).type,enabled=enabled): output=model(images,targets=targets)
    if not torch.isfinite(output['pred_boxes']).all() or not torch.isfinite(output['pred_logits']).all(): raise FloatingPointError('Nonfinite final model outputs')
    with torch.autocast(device_type=torch.device(cfg['device']).type,enabled=False):
        losses=criterion(output,targets,epoch=epoch,step=index,global_step=epoch*batches+index,epoch_step=batches)
        total=sum(losses.values())
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite native total loss')
    return total,losses,output

def capacity_probe(run):
    run=Path(run); cfg=frozen_config(run); manifest=read_json(run/'data/manifest.json')
    if cfg['device'].startswith('cuda'): torch.cuda.reset_peak_memory_stats(torch.device(cfg['device']))
    model,criterion,post,ema,opt,scaler,schedule,generator=setup(run,cfg,manifest,capacity=True)
    loader=train_loader(manifest,cfg,0,generator); samples=[]; shapes={}; candidates={}; handles=[]
    for name,module in model.named_modules():
        if name in ('backbone','encoder'):
            def hook(m,inp,out,name=name): shapes[name]=[list(t.shape) for t in out]
            handles.append(module.register_forward_hook(hook))
        if isinstance(module,torch.nn.Conv2d) and name.startswith(('backbone.stages.','encoder.')) and len(handles)<20:
            def layer_hook(m,inp,out,name=name): candidates[name]=list(out.shape)
            handles.append(module.register_forward_hook(layer_hook))
    model.train(); criterion.train(); opt.zero_grad(set_to_none=True)
    steps=0
    # Fetch exactly the final configured batch. Probe updates do not touch any formal state.
    for index,(images,targets) in enumerate(loader):
        if len(images)!=cfg['batch_size']: raise ValueError('Capacity probe is not a complete configured batch')
        total,losses,out=loss_batch(model,criterion,images,targets,cfg,0,index,len(loader))
        scaler.scale(total).backward() if scaler.is_enabled() else total.backward()
        norm,lrs=optimizer_step(model,opt,scaler,schedule,ema,cfg)
        samples.append({'shape':list(images.shape),'range':[float(images.min()),float(images.max())],
            'target_counts':[len(t['labels']) for t in targets],'losses':{k:float(v.detach()) for k,v in losses.items()},
            'total_loss':float(total.detach()),'gradient_norm_before_clip':norm,'group_lrs':lrs,
            'output_keys':list(out),'ema_updates':ema.updates if ema else None,'scaler_scale':scaler.get_scale()})
        steps+=1
        if steps>=cfg['preflight_steps']: break
    for handle in handles: handle.remove()
    if steps<cfg['preflight_steps']: raise ValueError('Not enough actual probe batches')
    receipt={'status':'completed','batch_size':cfg['batch_size'],'imgsz':cfg['imgsz'],'optimizer_steps':steps,
        'scope':'ISOLATED_PROCESS_FINAL_RECIPE_CAPACITY','formal_state_mutated':False,'samples':samples,
        'layer_shapes':shapes,'candidate_layer_shapes':candidates,'actual_transforms':loader.dataset.transform_report,'augmentation_counts':snapshot(loader.dataset),
        'amp_enabled':scaler.is_enabled(),'amp_init_scale':cfg['amp_init_scale'],'ema':cfg['ema'],'clip_max_norm':cfg['gradient_clip_max_norm'],
        'max_allocated_bytes':torch.cuda.max_memory_allocated(torch.device(cfg['device'])) if cfg['device'].startswith('cuda') else None,
        'limit':'This checks configured batch capacity in this process; it does not guarantee concurrent full-training capacity'}
    write_json(run/'preflight_receipt.json',receipt)
    print('Isolated configured-batch forward/loss/backward/optimizer/EMA preflight completed',flush=True)

def preflight_process(run):
    run=Path(run); status(run,'preflight','running')
    command=[sys.executable,'-u',str(HERE/'run.py'),'_capacity','--run-dir',str(run)]
    completed=subprocess.run(command)
    if completed.returncode:
        status(run,'preflight','failed',exit_code=completed.returncode)
        raise RuntimeError('Configured-batch isolated preflight failed; no batch/input/initialization was changed')
    status(run,'preflight','completed',exit_code=0,receipt_sha256=sha256(run/'preflight_receipt.json'))

def validate_last(ckpt,expected,cfg,schedule,run):
    seals=read_json(Path(run)/'checkpoint_integrity.json')
    if sha256(Path(run)/'train/weights/last.pth')!=seals['last_sha256'] or ckpt.get('epoch')!=seals['epoch']:
        raise ValueError('Resume last checksum/epoch differs or pair save was interrupted')
    if ckpt.get('schema')!='dfine_m_last_v1' or ckpt.get('identity')!=expected: raise ValueError('Resume must use last from this same frozen D-FINE run')
    required={'model','ema','optimizer','scheduler','scaler','rng','best','epoch','criterion','rows'}
    if not required<=ckpt.keys(): raise ValueError('Incomplete last state')
    if not 0<=ckpt['epoch']<cfg['epochs']-1: raise ValueError('Resume epoch completed or outside plan')
    if cfg['ema'] and (ckpt['ema'] is None or ckpt['ema']['updates']!=ckpt['scheduler']['updates']): raise ValueError('Resume EMA/update state differs')
    if ckpt['scheduler']['updates']!=(ckpt['epoch']+1)*(schedule.total//cfg['epochs']): raise ValueError('Resume epoch/update state differs')
    best=selected_checkpoint(Path(run)/'train/weights/best.pth',expected)
    if sha256(Path(run)/'train/weights/best.pth')!=ckpt['best']['sha256'] or best['epoch']!=ckpt['best']['epoch']:
        raise ValueError('Resume best identity differs or interrupted checkpoint save')
    if len(ckpt['rows'])!=ckpt['epoch']+1 or best['val_mAP50_95']!=ckpt['best']['score']: raise ValueError('Resume best/rows differ')

def csv_rows(run,rows):
    keys=['epoch']+sorted({k for row in rows for k in row if k!='epoch'})
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n')
    writer.writeheader(); writer.writerows(rows); atomic_bytes(Path(run)/'train/results.csv',stream.getvalue().encode())

def train(run,resume=None,epoch_budget=None):
    run=Path(run); cfg=frozen_config(run); manifest=read_json(run/'data/manifest.json')
    expected=read_json(run/'identity.json')
    if identity(run,cfg,manifest)!=expected: raise ValueError('Run model/config/data/init identity changed')
    preflight=read_json(run/'preflight_status.json')
    if preflight['status']!='completed' or preflight['receipt_sha256']!=sha256(run/'preflight_receipt.json'):
        raise ValueError('Complete final-recipe preflight before training')
    if not resume and (run/'train/weights/last.pth').exists(): raise ValueError('Existing run requires explicit --resume, not COCO tuning')
    old=read_json(run/'train_status.json') if (run/'train_status.json').exists() else {}
    if old.get('status')=='completed': raise ValueError('Training is already completed')
    status(run,'train','running',resume=str(resume) if resume else None)
    model,criterion,post,ema,opt,scaler,schedule,generator=setup(run,cfg,manifest)
    start_epoch=0; rows=[]; best={'score':-float('inf'),'epoch':None,'sha256':None}
    if resume:
        if Path(resume).resolve()!=(run/'train/weights/last.pth').resolve(): raise ValueError('Resume explicitly requires this same run last.pth')
        ckpt=torch.load(resume,map_location='cpu',weights_only=False); validate_last(ckpt,expected,cfg,schedule,run)
        model.load_state_dict(ckpt['model'],strict=True); criterion.load_state_dict(ckpt['criterion'],strict=True)
        if ema is not None: ema.load_state_dict(ckpt['ema'],strict=True)
        opt.load_state_dict(ckpt['optimizer']); schedule.load_state_dict(ckpt['scheduler']); scaler.load_state_dict(ckpt['scaler'])
        best=ckpt['best']; rows=ckpt['rows']; start_epoch=ckpt['epoch']+1; restore_rng(ckpt['rng'],generator); del ckpt
    started=time.monotonic()
    for epoch in range(start_epoch,cfg['epochs']):
        loader=train_loader(manifest,cfg,epoch,generator); model.train(); criterion.train(); opt.zero_grad(set_to_none=True)
        write_json(run/f'augmentation/epoch_{epoch+1:03d}_objects.json',loader.dataset.transform_report)
        accum=cfg['gradient_accumulation_steps']; sums={}; total_sum=0.; grad_norm=0.; epoch_start=time.monotonic()
        print(f'\nEpoch {epoch+1}/{cfg["epochs"]}, actual batches={len(loader)}, mixing_closed={loader.dataset.closed}',flush=True)
        for index,(images,targets) in enumerate(loader):
            total,losses,_=loss_batch(model,criterion,images,targets,cfg,epoch,index,len(loader))
            divisor=min(accum,len(loader)-(index//accum)*accum)
            loss=total/divisor
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            total_sum+=float(total.detach())
            for k,v in losses.items(): sums[k]=sums.get(k,0.)+float(v.detach())
            if (index+1)%accum==0 or index+1==len(loader): grad_norm,lrs=optimizer_step(model,opt,scaler,schedule,ema,cfg)
            if index%cfg['print_freq']==0 or index+1==len(loader):
                print(f'TRAIN epoch={epoch+1} batch={index+1}/{len(loader)} loss={float(total.detach()):.6g} '
                    f'LR={[g["lr"] for g in opt.param_groups]} optimizer_updates={schedule.updates} EMA={ema.updates if ema else None}',flush=True)
        write_json(run/f'augmentation/epoch_{epoch+1:03d}_counts.json',snapshot(loader.dataset))
        selected=ema.module if cfg['ema'] else model
        if cfg['ema'] and ema.updates!=schedule.updates: raise ValueError('Continuous EMA updates differ')
        val=training_val(selected,post,manifest,cfg,expected,epoch)
        score=val['mAP50_95']
        if score is None or not math.isfinite(score): raise ValueError('Undefined/nonfinite public val mAP')
        write_json(run/f'train/val_metrics/epoch_{epoch+1:03d}.json',val)
        if score>=best['score']:  # Unrounded score; later epoch wins ties.
            best_path=run/'train/weights/best.pth'
            save_torch(best_path,{'schema':'dfine_m_selected_v1','identity':expected,'epoch':epoch,
                'selected_source':'ema' if cfg['ema'] else 'model','selected_key':'selected_state_dict',
                'selected_state_dict':cpu_state(selected),'val_mAP50_95':score,'ema_updates':ema.updates if ema else None})
            best={'score':score,'epoch':epoch,'sha256':sha256(best_path)}
        row={'epoch':epoch+1,'loss_total':total_sum/len(loader),**{k:v/len(loader) for k,v in sums.items()},
            **{'lr_'+g['role']:g['lr'] for g in opt.param_groups},
            **{'val_'+k:val[k] for k in ('precision','recall','AP50','AP75','mAP50_95')},
            'ema_updates':ema.updates if ema else 0,'optimizer_updates':schedule.updates,
            'amp_enabled':int(scaler.is_enabled()),'amp_scale':scaler.get_scale(),'gradient_norm_before_clip':grad_norm,
            'epoch_seconds':time.monotonic()-epoch_start}
        rows.append(row); csv_rows(run,rows)
        save_torch(run/'train/weights/last.pth',{'schema':'dfine_m_last_v1','identity':expected,'epoch':epoch,
            'model':cpu_state(model),'criterion':criterion.state_dict(),'ema':ema.state_dict() if ema else None,
            'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'scaler':scaler.state_dict(),
            'rng':rng_state(generator),'best':best,'rows':rows,'augmentation_closed':loader.dataset.closed})
        write_json(run/'checkpoint_integrity.json',{'best_sha256':best['sha256'],'last_sha256':sha256(run/'train/weights/last.pth'),
            'epoch':epoch,'selected_source':'ema' if cfg['ema'] else 'model','selected_key':'selected_state_dict'})
        print(f'VAL epoch={epoch+1} P={val["precision"]*100:.4f}% R={val["recall"]*100:.4f}% AP50={val["AP50"]*100:.4f}% '
            f'AP75={val["AP75"]*100:.4f}% mAP50-95={score*100:.4f}% BEST={best["epoch"]+1}',flush=True)
        from plotting import plot_training
        plot_training(run)
        status(run,'train','running',completed_epochs=epoch+1,best_epoch=best['epoch']+1,best_sha256=best['sha256'])
        if epoch_budget is not None and epoch+1-start_epoch>=epoch_budget and epoch+1<cfg['epochs']:
            if expected['scope']!='SMOKE_ONLY': raise ValueError('Epoch budget allowed only for isolated synthetic smoke')
            status(run,'train','smoke_interrupted',completed_epochs=epoch+1,best_sha256=best['sha256']); return
    if schedule.updates!=schedule.total: raise ValueError('Training did not complete actual planned optimizer updates')
    status(run,'train','completed',exit_code=0,completed_epochs=cfg['epochs'],best_epoch=best['epoch']+1,
        best_sha256=best['sha256'],best_mAP50_95=best['score'],selected_source='ema' if cfg['ema'] else 'model',
        selected_key='selected_state_dict',optimizer_updates=schedule.updates,elapsed_seconds=time.monotonic()-started,
        stop_reason='epoch_limit',early_stopping=False)
