"""Bounded SDB-P3 engineering preflight; never completes an epoch or evaluates val/test."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import warnings
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from init_sdb_p3 import (ROOT, MODEL_DIR, VARIANTS, build, controlled_models, build_training_model,
                         verify_model, require, sha256, write_json, runtime, is_added)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules.sdb_p3 import SDBP3
from ultralytics.utils import YAML, ASSETS
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.torch_utils import TORCH_2_4
from c19_lif_v1_probe import capture, targets


def state_exact(left, right, message='State mismatch'):
    require(set(left) == set(right), message + ': keys')
    for key, value in left.items():
        require(torch.equal(value.detach().cpu(), right[key].detach().cpu()), message + ': ' + key)
    return len(left)


def compare(a, b, atol=2e-6, rtol=2e-5):
    """Compare nested native outputs, preserving DN metadata and every tensor."""
    errors=[]
    def visit(left, right, name):
        if isinstance(left, torch.Tensor):
            require(isinstance(right, torch.Tensor) and left.shape == right.shape, name + ': shape')
            x,y=left.detach().float().cpu(),right.detach().float().cpu()
            require(torch.isfinite(x).all() and torch.isfinite(y).all(), name + ': nonfinite')
            error=float((x-y).abs().max()) if x.numel() else 0.
            require(torch.allclose(x,y,atol=atol,rtol=rtol), f'{name}: max error {error}, atol={atol}, rtol={rtol}')
            errors.append(error)
        elif isinstance(left, dict):
            require(set(left)==set(right),name+': dictionary keys')
            for key in left: visit(left[key],right[key],name+'.'+str(key))
        elif isinstance(left,(tuple,list)):
            require(len(left)==len(right),name+': sequence length')
            for i,(x,y) in enumerate(zip(left,right)):visit(x,y,name+'.'+str(i))
        else:require(left==right,name+': metadata')
    visit(a,b,'output')
    return dict(max_abs_error=max(errors,default=0.),tensor_count=len(errors),atol=atol,rtol=rtol)


def synthetic_batch(shape=(160,160),device='cpu'):
    generator=torch.Generator().manual_seed(392)
    return dict(img=torch.rand(2,3,*shape,generator=generator).to(device),
                bboxes=torch.tensor([[.35,.42,.2,.08],[.64,.63,.09,.23],[.4,.5,.16,.3]],device=device),
                cls=torch.zeros(3,1,device=device),batch_idx=torch.tensor([0,1,1],device=device))


def branch_grads(model):
    return {name:dict(norm=float(p.grad.detach().double().norm()) if p.grad is not None and torch.isfinite(p.grad).all() else None,
                     finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else False)
            for name,p in model.named_parameters() if '.sdb.' in name}


def native_scaler(enabled):
    return torch.amp.GradScaler('cuda',enabled=enabled) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=enabled)


def core_checks():
    rows=[]
    for h,w in [(4,6),(3,5),(3,6),(4,5)]:
        x=torch.arange(2*h*w,dtype=torch.float32).reshape(1,2,h,w)
        expected=torch.empty(1,8,(h+1)//2,(w+1)//2)
        for phase,(dy,dx) in enumerate([(0,0),(1,0),(0,1),(1,1)]):
            for c in range(2):
                for yy in range((h+1)//2):
                    for xx in range((w+1)//2):
                        expected[0,phase*2+c,yy,xx]=x[0,c,min(yy*2+dy,h-1),min(xx*2+dx,w-1)]
        got=SDBP3.phase_rearrange(x)
        require(torch.equal(got,expected),'Phase-major ordering/right-bottom replicate padding')
        rows.append(dict(shape=[h,w],elements=got.numel(),exact=True))
    core=SDBP3(64,256,32)
    p2=torch.randn(2,64,7,9,requires_grad=True);t3=torch.randn(2,256,4,5)
    require(torch.equal(core(p2,t3),t3),'Zero output projection is not exact identity')
    invalid=False
    try:core(p2,t3[...,:3,:])
    except (ValueError,RuntimeError):invalid=True
    require(invalid,'Spatial mismatch silently accepted')
    with torch.no_grad():core.W_o.weight.normal_(std=.01)
    delta=core(p2,t3)-t3
    delta.square().mean().backward()
    require(torch.isfinite(p2.grad).all() and p2.grad.abs().sum()>0,'Isolated delta has no P2 gradient')
    require(delta.abs().sum()>0,'Nonzero output projection has no residual')
    return dict(status='PASSED',phase_cases=rows,odd_core_identity=True,invalid_shape_rejected=True,
                isolated_delta_p2_gradient_norm=float(p2.grad.norm()),residual_norm=float(delta.norm()),
                gradient_scope='Independent P2, fixed T3; squared residual diagnostic only. Full network uses detection loss.')


def parent_from_target(target,variant):
    parent=build(variant,nc=1,baseline=True)
    parent.load_state_dict({k:target.state_dict()[k] for k in parent.state_dict()},strict=True)
    parent.nc=1
    return parent


def wiring_check(target):
    model=deepcopy(target).cpu().eval();seen={};handles=[]
    def output(name):
        def hook(module,args,value):seen[name]=value
        return hook
    def inputs(name):
        def hook(module,args):seen[name]=args
        return hook
    try:
        for index in (4,18,19):handles.append(model.model[index].register_forward_hook(output(str(index))))
        for index in (19,20,26):handles.append(model.model[index].register_forward_pre_hook(inputs(str(index)+'_inputs')))
        handles.append(model.model[19].sdb.register_forward_pre_hook(inputs('sdb_inputs')))
        with torch.no_grad():model(torch.rand(1,3,160,192))
        pairs=[('P2_to_wrapper',seen['4'],seen['19_inputs'][0][1]),
               ('concat_to_wrapper',seen['18'],seen['19_inputs'][0][0]),
               ('P2_to_core',seen['4'],seen['sdb_inputs'][0]),
               ('P3_to_downsample',seen['19'],seen['20_inputs'][0]),
               ('P3_to_decoder',seen['19'],seen['26_inputs'][0][0])]
        for name,left,right in pairs:require(left is right,name+' must pass the exact tensor object, without detach/copy')
        from ultralytics.nn.modules import RepC3
        with torch.no_grad():original=RepC3.forward(model.model[19],seen['18'])
        require(torch.equal(original,seen['sdb_inputs'][1]),'SDB semantic input must be the original RepC3 output')
        return dict(status='PASSED',exact_tensor_identity={name:True for name,_,_ in pairs},
                    core_T3_equals_original_RepC3=True,p2_shape=list(seen['4'].shape),p3_shape=list(seen['19'].shape))
    finally:
        for handle in handles:handle.remove()


def initial_equivalence(target,variant,device,amp=False):
    rows={};context=lambda:torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext()
    tol=(3e-3,3e-2) if amp else (2e-6,2e-5)
    for training,shapes in [(False,[(640,640),(320,640)]),(True,[(160,160),(160,192)])]:
        for shape in shapes:
            parent=parent_from_target(target,variant).to(device).train(training)
            child=deepcopy(target).to(device).train(training);child.nc=1
            batch=synthetic_batch(shape,device)
            if not training:batch['img']=batch['img'][:1]
            torch.manual_seed(193);torch.cuda.manual_seed_all(193) if device=='cuda' else None
            with torch.no_grad(),context():
                a=parent.predict(batch['img'],batch=targets(batch) if training else None)
                aloss=parent.loss(batch,preds=a)[0] if training else None
            torch.manual_seed(193);torch.cuda.manual_seed_all(193) if device=='cuda' else None
            with torch.no_grad(),context():
                b=child.predict(batch['img'],batch=targets(batch) if training else None)
                bloss=child.loss(batch,preds=b)[0] if training else None
            row=compare(a,b,*tol)
            if training:
                require(a[-1] and a[-1]['dn_num_split'][0]>0,'Valid GT did not exercise DN')
                row.update(loss=compare(aloss,bloss,*tol),dn_split=a[-1]['dn_num_split'])
                row['bn_buffers_exact']=state_exact({k:v for k,v in parent.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k},
                    {k:v for k,v in child.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k})
            rows[f'{"train" if training else "eval"}_{shape[0]}x{shape[1]}']=row
            del parent,child,a,b,batch;gc.collect()
    return dict(status='PASSED',device=device,amp=amp,matched_rng_and_bn=True,cases=rows)


def actual_train_api(initialized,target,variant):
    observed={}
    class StopProbe(Exception):pass
    class AuditTrainer(RTDETRTrainer):
        def __init__(self,overrides,_callbacks):
            self.args=SimpleNamespace(**{k:v for k,v in overrides.items() if k!='session'})
            self.data=dict(nc=1,channels=3);torch.manual_seed(42)
        def get_model(self,cfg=None,weights=None,verbose=True):
            model,audit=build_training_model(cfg,weights,self.data,variant)
            observed['rebuild_audit']=audit
            return model
        def train(self):
            observed.update(status='PASSED',states_exact=state_exact(target.state_dict(),self.model.state_dict()),
                            native_Model_train_path=True,native_get_model_used_by_controlled_helper=True,optimizer_steps=0)
            raise StopProbe()
    with patch('ultralytics.engine.model.checks.check_pip_update_available'):
        try:RTDETR(str(initialized)).train(trainer=AuditTrainer,data='unused_probe_data.yaml',seed=42,verbose=False)
        except StopProbe:pass
    require(observed.get('status')=='PASSED','Actual Model.train loading path not reached')
    return observed


def make_optimizer(model):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args=SimpleNamespace(warmup_bias_lr=.1,lr0=.0005,weight_decay=.0001)
    opt=trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    return opt,optimizer_coverage(model,opt)


def optimizer_coverage(model,opt):
    ids=[id(p) for group in opt.param_groups for p in group['params']]
    require(len(ids)==len(set(ids)),'Duplicate optimizer parameter')
    require(set(ids)=={id(p) for p in model.parameters() if p.requires_grad},'Optimizer misses trainable parameters')
    rows=[]
    for name,p in model.named_parameters():
        if '.sdb.' in name:
            group=next(g for g in opt.param_groups if any(p is item for item in g['params']))
            rows.append(dict(name=name,numel=p.numel(),count=ids.count(id(p)),group=group.get('param_group'),weight_decay=group['weight_decay']))
    return dict(status='PASSED',new=rows,all_parameters_once=True)


def loss_startup(target,device,folder,amp=False):
    """Real detection loss with DN; native default dynamic scaler, disposable optimizer."""
    model=deepcopy(target).to(device).train();model.nc=1
    batch=synthetic_batch(device=device);opt,coverage=make_optimizer(model)
    scaler=native_scaler(amp)
    ema=ModelEMA(model);steps=[];effective=0;upstream_seen=set();initial=None
    for attempt in range(24):
        opt.zero_grad(set_to_none=True);torch.manual_seed(42+attempt)
        with torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext():
            preds=model.predict(batch['img'],batch=targets(batch));loss=model.loss(batch,preds=preds)[0]
        require(torch.isfinite(loss),'Nonfinite real detection loss')
        require(preds[-1] and preds[-1]['dn_num_split'][0]>0,'DN path not used')
        scaler.scale(loss).backward();scaler.unscale_(opt);grads=branch_grads(model)
        finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        require(finite or amp,'Nonfinite unscaled FP32 gradient')
        before=model.model[19].sdb.W_o.weight.detach().clone();scale_before=scaler.get_scale()
        if finite:
            if effective==0:
                require(grads['model.19.sdb.W_o.weight']['norm']>0,'Initial W_o gradient is zero')
                require(all(v['norm']==0 for k,v in grads.items() if k!='model.19.sdb.W_o.weight'),'Upstream gradient must initially be zero')
                initial=grads
            else:
                for name in ('W_d','DW3','W_q','W_g'):
                    if grads[f'model.19.sdb.{name}.weight']['norm']>0:upstream_seen.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        scaler.step(opt);scaler.update()
        changed=not torch.equal(before,model.model[19].sdb.W_o.weight)
        if changed:effective+=1;ema.update(model)
        steps.append(dict(attempt=attempt,loss=float(loss.detach()),finite_unscaled_gradients=bool(finite),
                          scale_before=scale_before,scale_after=scaler.get_scale(),effective_update=changed,new_gradients=grads))
        del preds,loss
        if effective>=2 and len(upstream_seen)==4:break
    require(effective>=2 and len(upstream_seen)==4,'Branch did not start after dynamic-scaled optimizer updates')
    require(torch.count_nonzero(model.model[19].sdb.W_o.weight)>0,'Updated W_o still zero')
    checkpoint=folder/f'nonzero_{device}_{"amp" if amp else "fp32"}.pt'
    torch.save(dict(model=deepcopy(model).cpu(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                    ema=deepcopy(ema.ema).cpu(),updates=ema.updates,epoch=0,train_args=dict(task='detect')),checkpoint)
    loaded=torch_load(checkpoint,map_location=device)
    state_exact(model.state_dict(),loaded['model'].state_dict(),'Nonzero save/reload')
    restore=RTDETRTrainer.__new__(RTDETRTrainer);restore.model=loaded['model'];restore.optimizer,_=make_optimizer(restore.model)
    restore.scaler=native_scaler(amp);restore.ema=ModelEMA(restore.model)
    restore.resume=True;restore.epochs=200;restore.args=SimpleNamespace(model=str(checkpoint),close_mosaic=10)
    restore.resume_training(loaded)
    require(restore.start_epoch==1,'Native resume start epoch')
    compare(opt.state_dict(),restore.optimizer.state_dict(),0,0)
    state_exact(ema.ema.state_dict(),restore.ema.ema.state_dict(),'EMA restore')
    require(restore.scaler.state_dict()==scaler.state_dict(),'Scaler restore')
    # A resumed real loss step proves learned state survives loading and stays differentiable.
    restore.optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext(): resumed_loss=restore.model.loss(batch)[0]
    restore.scaler.scale(resumed_loss).backward();restore.scaler.unscale_(restore.optimizer)
    require(torch.isfinite(resumed_loss),'Resumed loss nonfinite')
    resumed_grads=branch_grads(restore.model)
    require(all(v['finite'] for v in resumed_grads.values()),'Resumed gradient nonfinite')
    torch.nn.utils.clip_grad_norm_(restore.model.parameters(),10.);restore.scaler.step(restore.optimizer);restore.scaler.update()
    report=dict(status='PASSED',scope='synthetic GT engineering only',device=device,amp=amp,batch=2,imgsz=160,
                optimizer=coverage,steps=steps,effective_updates=effective,initial_gradient=initial,upstream_seen=sorted(upstream_seen),
                nonzero_save_reload=True,native_resume=True,ema_exact=True,resume_gradients=resumed_grads,
                checkpoint_sha256=sha256(checkpoint),checkpoint=str(checkpoint))
    result=deepcopy(model).cpu().eval()
    del model,restore,loaded,opt,ema,batch;gc.collect()
    return report,result


def nonzero_fusion(model,variant,device,folder):
    model=deepcopy(model).to(device).eval()
    # Train-updated W_o is used, with deterministic nontrivial magnitude to avoid an identity-only check.
    with torch.no_grad():model.model[19].sdb.W_o.weight.add_(torch.linspace(-.02,.02,8192,device=device).reshape_as(model.model[19].sdb.W_o.weight))
    sdb_state={k:v.clone() for k,v in model.model[19].sdb.state_dict().items()}
    x=synthetic_batch((160,192),device)['img'][:1]
    disabled=deepcopy(model)
    with torch.no_grad():disabled.model[19].sdb.W_o.weight.zero_()
    with torch.no_grad():_,active=capture(model,x);_,inactive=capture(disabled,x)
    difference=float((active['scale_0']-inactive['scale_0']).abs().max())
    require(difference>1e-7,'Nonzero SDB has no network effect')
    fused=deepcopy(model).fuse(verbose=False)
    state_exact(sdb_state,fused.model[19].sdb.state_dict(),'Fusion changed SDB/GN')
    require(sum(isinstance(m,torch.nn.GroupNorm) for m in fused.model[19].sdb.modules())==2,'GN lost at fusion')
    require(sum(isinstance(m,torch.nn.BatchNorm2d) for m in fused.modules())<sum(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()),'Ordinary BN fusion disabled')
    if variant.startswith('cbr_'):require(hasattr(fused.model[20],'bn'),'Original LIF BN protection lost')
    expected=19973061 if variant.startswith('cbr_') else 19905812
    require(sum(p.numel() for p in fused.parameters())==expected,'Fused nc=1 parameter count drift')
    rows={}
    for precision in (['fp32','amp','half'] if device=='cuda' else ['fp32']):
        left,right=deepcopy(model),deepcopy(fused);image=x
        if precision=='half':left.half();right.half();image=x.half()
        param_ids={n:(id(p),p.dtype) for n,p in left.named_parameters()}
        context=lambda:torch.autocast('cuda',dtype=torch.float16) if precision=='amp' else nullcontext()
        with torch.no_grad(),context():_,a=capture(left,image);_,b=capture(right,image)
        tol=(2e-5,2e-4) if precision=='fp32' else (3e-3,3e-2)
        # Encoder topk row permutations are recorded and aligned only when complete candidate sets match.
        from c19_lif_v1_diagnostic import align_to_ids, selection_report
        selection=selection_report(a,b)
        write_json(folder/f'fusion_{device}_{precision}_selection.json',selection)
        keys=['scale_0','scale_1','scale_2','boxes','scores','enc_boxes','enc_scores']
        for record in (a,b):
            require(all(torch.isfinite(record[k]).all() for k in keys),'Nonfinite native fusion inference')
        continuous=compare({k:a[k] for k in keys[:3]},{k:b[k] for k in keys[:3]},*tol)
        drift=selection['kind']=='SET_DRIFT'
        require(not drift or precision!='fp32','FP32 fusion changed native top-k candidate set; review required')
        measured=dict(status='NATIVE_TOPK_DRIFT',allclose_claimed=False) if drift else compare(
            {k:a[k] for k in keys},{k:(align_to_ids(a,b) if selection['kind']!='IDENTICAL' else b)[k] for k in keys},*tol)
        with torch.no_grad(),context():_,replay=capture(right,image,fixed_ids=a['candidate_indices'])
        replay_report=compare({k:a[k] for k in keys},{k:replay[k] for k in keys},*tol)
        parent_control=None
        if drift:
            # Top-k is discrete. Diagnose both complete native candidate sets without accepting
            # partial intersections or relabelling differing natural outputs as equivalent.
            with torch.no_grad(),context():
                _,reverse=capture(left,image,fixed_ids=b['candidate_indices'])
            compare({k:reverse[k] for k in keys},{k:b[k] for k in keys},*tol)
            baseline=parent_from_target(model.cpu(),variant).eval().to(device)
            model.to(device)
            baseline_fused=deepcopy(baseline).fuse(verbose=False)
            if precision=='half':baseline.half();baseline_fused.half()
            with torch.no_grad(),context():
                _,pa=capture(baseline,image);_,pb=capture(baseline_fused,image)
                _,pr=capture(baseline_fused,image,fixed_ids=pa['candidate_indices'])
            parent_control=dict(selection=selection_report(pa,pb),
                continuous=compare({k:pa[k] for k in keys[:3]},{k:pb[k] for k in keys[:3]},*tol),
                fixed_query_replay=compare({k:pa[k] for k in keys},{k:pr[k] for k in keys},*tol),
                same_common_state=True,same_input_precision=True)
            del baseline,baseline_fused
        require(param_ids=={n:(id(p),p.dtype) for n,p in left.named_parameters()},'forward replaced Parameter/dtype')
        rows[precision]=dict(status='NATIVE_TOPK_DRIFT' if drift else 'PASSED',selection=selection,native_aligned=measured,
                             continuous_neck=continuous,fixed_query_replay=replay_report,parent_control=parent_control,
                             required_inference_checks='PASSED',native_output_equivalence=not drift,
                             explicit_model_half=precision=='half',parameter_identity_unchanged=True)
        write_json(folder/f'fusion_{device}_{precision}.json',rows[precision])
        del left,right;gc.collect()
    save=folder/f'fusion_nonzero_{device}.pt'
    torch.save(dict(model=model.cpu(),epoch=-1,train_args=dict(task='detect')),save)
    loaded=RTDETR(str(save)).model.eval().to(device)
    state_exact(model.state_dict(),loaded.state_dict(),'RTDETR nonzero checkpoint load')
    with torch.no_grad():restored=loaded(x);original=model.to(device)(x)
    restored_report=compare(original,restored,0,0)
    # Re-enter Model.train -> controlled native get_model with already-learned nc=1 weights.
    learned_api=actual_train_api(save,model.cpu(),variant)
    model.to(device)
    fused_again=deepcopy(fused).fuse(verbose=False)
    state_exact(fused.state_dict(),fused_again.state_dict(),'Repeated fusion')
    return dict(status='PASSED',branch_disabled_p3_max_difference=difference,precision=rows,
                acceptance='Strict FP32 complete-output fusion; AMP/half finite inference, continuous neck and full fixed-query replay. Native reduced-precision top-k drift remains explicitly disclosed.',
                sdb_states_unchanged=True,GN_preserved=True,reload=restored_report,fused_parameters=expected,
                learned_Model_train_api=learned_api,
                native_fuse_order='FP32 fuse then half',repeat_fuse_exact=True)


class BoundedStop(Exception):
    """Normal intentional preflight cutoff, never used by formal training."""


def server_capacity(initialized,variant,data,folder):
    from train_sdb_p3 import recipe, verify_data
    rows=[];new_seen=set();updates=0;batch_count=0;observed={};amp_messages=[]
    progress=dict(status='RUNNING',stage='resources',steps=rows,observed=observed,resources={})
    def persist_progress(stage=None,status=None):
        if stage is not None:progress['stage']=stage
        if status is not None:progress['status']=status
        progress.update(effective_updates=updates,batches=batch_count,upstream_seen=sorted(new_seen),native_amp_log=amp_messages)
        write_json(folder/'native_capacity_progress.json',progress)
    persist_progress()
    require(torch.cuda.is_available(),'Server B16/640 native AMP requires CUDA')
    inventory=verify_data(data);args,changes=recipe(initialized,variant,data)
    args.update(project=str(folder),name='native_b16_640',exist_ok=False)
    args.pop('save_dir',None)
    amp_model=ROOT/'yolo26n.pt';bus=Path(ASSETS)/'bus.jpg'
    progress['resources']={str(p):sha256(p) if p.is_file() else None for p in (amp_model,bus)}
    persist_progress()
    require(amp_model.is_file() and bus.is_file(),f'Native AMP resources missing: {amp_model}, {bus}. Prepare local compatible yolo26n.pt and assets/bus.jpg; no upgrade/download fallback.')
    # Load explicitly up front so native check_amp cannot silently accept an incompatible/offline resource.
    from ultralytics import YOLO
    probe=YOLO(str(amp_model));del probe
    class BoundedTrainer(RTDETRTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            model,audit=build_training_model(cfg,weights,self.data,variant);observed['rebuild']=audit;return model
        def _build_train_pipeline(self):
            require(self.batch_size==16 and self.args.batch==16,'B16 capacity cannot automatically reduce batch')
            return super()._build_train_pipeline()
        def _setup_train(self):
            persist_progress('native_setup')
            super()._setup_train()
            require(self.amp and self.args.imgsz==640 and self.batch_size==16,'Native AMP/B16/640 changed')
            observed['optimizer']=optimizer_coverage(self.model,self.optimizer)
            observed['train_augmentations']=repr(self.train_loader.dataset.transforms)
            observed['actual_args']=vars(self.args).copy()
            # Hook fires only when GradScaler really invokes optimizer.step (not on overflow skip).
            def stepped(opt,args,kwargs):
                nonlocal updates
                updates+=1
            self.optimizer.register_step_post_hook(stepped)
            persist_progress('setup_complete')
        def optimizer_step(self):
            before=updates;scale=self.scaler.get_scale();grads=branch_grads(self.model)
            row=dict(batch=batch_count,loss=float(self.loss.detach()),scale_before=scale,
                     new_scaled_gradients=grads,accumulate=self.accumulate,lrs=[float(g['lr']) for g in self.optimizer.param_groups])
            rows.append(row);persist_progress('native_optimizer_step')
            for name in ('W_d','DW3','W_q','W_g'):
                value=grads[f'model.19.sdb.{name}.weight']
                if value['finite'] and value['norm'] and value['norm']>0:new_seen.add(name)
            super().optimizer_step()
            row.update(scale_after=self.scaler.get_scale(),effective_update=updates>before)
            persist_progress('optimizer_step_complete')
        def validate(self):raise AssertionError('Preflight must stop before full validation')
        def final_eval(self):raise AssertionError('Preflight must stop before final evaluation')
    def start(trainer):
        observed['start_epoch']=getattr(trainer,'start_epoch',0)
        observed['epoch_attribute_present_on_start']=hasattr(trainer,'epoch')
        torch.cuda.reset_peak_memory_stats()
    def batch_end(trainer):
        nonlocal batch_count
        batch_count+=1
        require(trainer.batch_size==16 and trainer.args.imgsz==640 and trainer.amp,'Native capacity settings changed')
        require(torch.isfinite(trainer.loss),'Native real batch loss nonfinite')
        if updates>=3 and len(new_seen)==4:raise BoundedStop()
        require(batch_count<32,'No two effective AMP updates / upstream gradients within 32 real batches')
    def batch_start(trainer):
        # Native first-epoch OOM recovery otherwise changes batch before re-entering the pipeline.
        trainer._oom_retries=3
        persist_progress('native_forward_backward')
    from ultralytics.utils import LOGGER
    import logging
    class AmpCapture(logging.Handler):
        def emit(self,record):
            message=record.getMessage()
            if 'AMP:' in message:amp_messages.append(message)
    handler=AmpCapture();LOGGER.addHandler(handler)
    api=RTDETR(str(initialized));api.add_callback('on_train_start',start);api.add_callback('on_train_batch_end',batch_end)
    api.add_callback('on_train_batch_start',batch_start)
    begun=time.perf_counter();before_hash=sha256(initialized);old=os.getcwd()
    try:
        os.chdir(ROOT)
        with patch('ultralytics.engine.model.checks.check_pip_update_available'):
            try:api.train(trainer=BoundedTrainer,**args)
            except BoundedStop:pass
    except BaseException as error:
        progress['failure']=dict(error=repr(error),traceback=traceback.format_exc())
        persist_progress(status='FAILED')
        raise
    finally:os.chdir(old);LOGGER.removeHandler(handler)
    require(updates>=2 and len(new_seen)==4,'Native bounded optimizer updates incomplete')
    require(any('checks passed' in line for line in amp_messages) and not any('skipped' in line for line in amp_messages),'Native AMP check skipped/failed')
    require(sha256(initialized)==before_hash,'Preflight modified controlled initialization')
    torch.cuda.synchronize()
    persist_progress('bounded_preflight_complete','PASSED')
    return dict(status='PASSED',batch=16,imgsz=640,amp=True,effective_updates=updates,batches=batch_count,
                seconds=time.perf_counter()-begun,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),native_amp_log=amp_messages,
                data=inventory,recipe_differences=changes,steps=rows,upstream_seen=sorted(new_seen),**observed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','initialized','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--variant',choices=list(VARIANTS),default='cbr_lif_sdb_p3_v1')
    parser.add_argument('--data',type=Path)
    parser.add_argument('--server',action='store_true',help='Require real parent data and native B16/640 AMP steps; otherwise report remains PENDING.')
    args=parser.parse_args()
    args.output=args.output.resolve();args.source=args.source.resolve();args.initialized=args.initialized.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='FAILED',checks_status='FAILED',scope='server' if args.server else 'local',variant=args.variant,
                formal_training='NOT_STARTED',test='NOT_RUN',runtime=runtime(),checks={},
                warnings=[],
                capacity=dict(status='PENDING',reason='Requires --server, original dataset, B16/640 native AMP'))
    stage='setup';begun=time.perf_counter()
    def persist():write_json(args.output/'checks.json',report)
    original_warning=warnings.showwarning
    def warning(message,category,filename,lineno,file=None,line=None):
        record=dict(message=str(message),category=category.__name__,file=str(filename),line=lineno)
        if record not in report['warnings']:report['warnings'].append(record)
        original_warning(message,category,filename,lineno,file,line)
    warnings.showwarning=warning
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
        from train_sdb_p3 import code_identity, verify_data
        import ultralytics
        module_path=Path(inspect.getfile(SDBP3)).resolve()
        require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT) and module_path.is_relative_to(ROOT),'Imported ultralytics outside this worktree')
        report['imports']=dict(ultralytics=str(Path(ultralytics.__file__).resolve()),sdb=str(module_path))
        report['identity']=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            variant=args.variant,source_sha256=sha256(args.source),init_sha256=sha256(args.initialized),
            data_sha256=sha256(args.data) if args.data else None,dataset_inventory=verify_data(args.data) if args.data else None,
            files=code_identity(),dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True))
        stage='core';report['checks']['core']=core_checks()
        from check_sdb_p3_core import count_checks,warmup_checks
        report['checks']['thop_int_tensor_and_conv_accounting']=count_checks()
        report['checks']['finite_AutoBackend_warmup']=warmup_checks()
        persist()
        new_states=[]
        for variant in VARIANTS:
            stage=variant+'.controlled_init';print('STAGE '+stage,flush=True)
            parent,weights,mapping=controlled_models(args.source,variant)
            if variant==args.variant:
                loaded=RTDETR(str(args.initialized)).model
                state_exact(weights.state_dict(),loaded.state_dict(),'Saved controlled initializer differs')
                del loaded
            torch.manual_seed(42);target,audit=build_training_model(weights.yaml,weights,dict(nc=1,channels=3),variant)
            target.nc=1
            new_states.append({k:v.clone() for k,v in target.state_dict().items() if '.sdb.' in k})
            row=report['checks'][variant]=dict(mapping=mapping,trainer_rebuild=audit,topology=verify_model(target,variant,zero=True))
            row['actual_tensor_wiring']=wiring_check(target)
            expected=20177861 if variant.startswith('cbr_') else 20110868
            count=sum(p.numel() for p in target.parameters());require(count==expected,'Unfused nc=1 parameter count')
            parent_count=sum(p.numel() for p in parent_from_target(target,variant).parameters())
            row['complexity']=dict(nc=1,imgsz=640,unfused_parent_parameters=parent_count,unfused_target_parameters=count,
                delta_parameters=count-parent_count,analytical_conv_MACs=178790400,analytical_conv_GFLOPs=.3575808,
                scope='SDB convolutions only; GN/activation/elementwise excluded. Full-network FLOPs not claimed.')
            if variant==args.variant:row['actual_Model_train_api']=actual_train_api(args.initialized,target,variant)
            del parent,weights;gc.collect()
            if args.server and variant!=args.variant:
                row['runtime_checks']=dict(status='NOT_REQUESTED',reason='Default server round checks only the selected variant; both construction/common/new-state audits still run.')
                del target;gc.collect();persist()
                continue
            for device in ('cpu','cuda'):
                if device=='cuda' and not torch.cuda.is_available():row['cuda']=dict(status='PENDING',reason='CUDA unavailable');continue
                stage=variant+'.'+device+'.equivalence';print('STAGE '+stage,flush=True)
                device_started=time.perf_counter()
                if device=='cuda':torch.cuda.reset_peak_memory_stats()
                item=row[device]={};item['initial_equivalence']=initial_equivalence(target,variant,device)
                if device=='cuda':item['AMP_initial_equivalence']=initial_equivalence(target,variant,device,True)
                persist();stage=variant+'.'+device+'.detection_loss';print('STAGE '+stage,flush=True)
                folder=args.output/variant;folder.mkdir(exist_ok=True)
                item['detection_loss'],nonzero=loss_startup(target,device,folder)
                stage=variant+'.'+device+'.nonzero_fusion';print('STAGE '+stage,flush=True)
                item['nonzero_fusion']=nonzero_fusion(nonzero,variant,device,folder)
                if device=='cuda':
                    stage=variant+'.cuda.AMP_loss';print('STAGE '+stage,flush=True)
                    item['AMP_detection_loss'],unused=loss_startup(target,device,folder,True);del unused
                del nonzero;gc.collect()
                if device=='cuda':
                    torch.cuda.synchronize()
                    item['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                    item['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                    torch.cuda.empty_cache()
                item['seconds']=time.perf_counter()-device_started
                persist()
            del target;gc.collect()
        state_exact(new_states[0],new_states[1],'Variant SDB initialization differs')
        report['checks']['two_variant_new_states_exact']=True
        if args.server:
            require(args.data is not None,'--server requires --data')
            stage='server_capacity';persist();report['capacity']=server_capacity(args.initialized,args.variant,args.data,args.output)
        report['checks_status']='PASSED'
        require(sha256(args.initialized)==report['identity']['init_sha256'],'Preflight modified controlled initialization')
        report['status']='PASSED' if args.server and report['capacity']['status']=='PASSED' else 'PENDING'
    except BaseException as error:
        report['failure']=dict(stage=stage,error=repr(error),traceback=traceback.format_exc())
        if stage=='server_capacity' and (args.output/'native_capacity_progress.json').is_file():
            progress=json.loads((args.output/'native_capacity_progress.json').read_text(encoding='utf-8'))
            progress.update(status='FAILED',failure=report['failure'])
            write_json(args.output/'native_capacity_progress.json',progress)
        raise
    finally:
        warnings.showwarning=original_warning
        report['seconds']=time.perf_counter()-begun;report['last_stage']=stage;persist()


if __name__=='__main__':main()
