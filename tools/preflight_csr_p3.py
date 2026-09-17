"""Bounded CSR-P3 engineering checks; never starts an epoch or val/test evaluation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch
import warnings

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from init_csr_p3 import (ROOT, MODEL_DIR, VARIANTS, COUNTS, SOURCE_SHA256, build, controlled_models,
                        build_training_model, native_rebuild, verify_model, is_added, tensor_hash,
                        require, sha256, write_json, runtime)
from c19_lif_v1_data import dataset_inventory, real_batch
from c19_lif_v1_probe import capture, targets
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML, LOGGER
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA, TORCH_2_4


def native_scaler(enabled):
    return torch.amp.GradScaler('cuda',enabled=enabled) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=enabled)


def data_root(config):
    raw = YAML.load(config)
    require(raw.get('nc', len(raw['names'])) == 1 and len(raw['names']) == 1, 'Data must have one class')
    root = Path(raw.get('path', Path(config).parent))
    if not root.is_absolute():
        root = (Path(config).resolve().parent / root).resolve()
    for split in ('train', 'val', 'test'):
        require(raw[split].replace('\\', '/').rstrip('/') == 'images/'+split,
                'Data split must match the successful parent inventory layout')
    return root


def identity(variant, source, initialized, data, recipe):
    """Content identity for immutable preflight reports and strict formal-start gate."""
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    candidates = {ROOT/p for p in tracked if p and (
        p.startswith('ultralytics-main/ultralytics/') and p.endswith(('.py', '.yaml')) or
        p.startswith('tools/') and p.endswith('.py') or
        p.startswith('configs/csr_p3') and p.endswith('.yaml') or
        p.startswith('docs/csr_p3/') and p.endswith(('.yaml', '.json')))}
    # Include yet-uncommitted experiment files in development reports. Start also
    # checks the commit, so a report from this phase cannot authorize a later SHA.
    candidates |= set((ROOT/'tools').glob('*csr_p3*.py'))
    candidates |= set((ROOT/'ultralytics-main/ultralytics/nn/modules').glob('csr_p3.py'))
    candidates |= set(MODEL_DIR.glob('*csr-p3*.yaml'))
    dataset=data_root(data) if data and Path(data).is_file() else None
    available=dataset is not None and all((dataset/'images'/split).is_dir() for split in ('train','val','test'))
    inventory = dataset_inventory(dataset) if available else None
    if inventory is not None:
        parent = json.loads((ROOT/'docs/csr_p3/parent_dataset_inventory.json').read_text(encoding='utf-8'))
        require(inventory == parent, 'Dataset split/label identity differs from successful parent')
    return dict(variant=variant, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,text=True).strip(),
                code_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(candidates) if p.is_file()},
                source_sha256=sha256(source) if source and Path(source).is_file() else None,
                init_sha256=sha256(initialized) if initialized and Path(initialized).is_file() else None,
                data_config_sha256=sha256(data) if data and Path(data).is_file() else None,
                data_inventory=inventory, recipe_sha256=sha256(recipe) if recipe and Path(recipe).is_file() else None)


def compare(left, right, atol=2e-6, rtol=2e-5, label='tensor'):
    left, right = left.detach().float().cpu(), right.detach().float().cpu()
    require(left.shape == right.shape, label+' shape differs')
    require(torch.isfinite(left).all() and torch.isfinite(right).all(), label+' nonfinite')
    error = float((left-right).abs().max()) if left.numel() else 0.
    require(torch.allclose(left, right, atol=atol, rtol=rtol), f'{label} mismatch max_abs={error}, atol={atol}, rtol={rtol}')
    return dict(shape=list(left.shape), max_abs=error, atol=atol, rtol=rtol)


def synthetic_batch(shape=(160,160), device='cpu'):
    generator = torch.Generator().manual_seed(2026+shape[1])
    return dict(img=torch.rand(2,3,*shape,generator=generator).to(device),
                bboxes=torch.tensor([[.3,.4,.15,.3],[.65,.6,.2,.1],[.4,.45,.12,.4]],device=device),
                cls=torch.zeros(3,1,device=device), batch_idx=torch.tensor([0,1,1],device=device))


def gradients(model):
    result={}
    for name,p in model.named_parameters():
        if is_added(name):
            finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else True
            norm=float(p.grad.detach().float().norm()) if p.grad is not None else None
            result[name]=dict(norm=norm if norm is not None and np.isfinite(norm) else None,
                              finite=finite,nonfinite_norm=repr(norm) if norm is not None and not np.isfinite(norm) else None)
    return result


def assert_offset_gradient(rows):
    values = [r for n,r in rows.items() if '.offset_pw.' in n]
    require(values and all(r['finite'] and r['norm'] is not None and r['norm'] > 0 for r in values),
            'Initial offset projection has missing/zero/nonfinite task gradient')


def optimizer(model):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args = SimpleNamespace(warmup_bias_lr=.1, lr0=.0005, weight_decay=.0001)
    opt = trainer.build_optimizer(model, name='AdamW', lr=.0005, momentum=.937, decay=.0001)
    return opt, optimizer_coverage(model, opt)


def optimizer_coverage(model, opt):
    ids = [id(p) for group in opt.param_groups for p in group['params']]
    require(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad},
            'Optimizer parameter coverage is not exactly once')
    rows = []
    for name,p in model.named_parameters():
        if is_added(name):
            group = next(g for g in opt.param_groups if any(p is q for q in g['params']))
            rows.append(dict(name=name, occurrences=ids.count(id(p)), group=group.get('param_group'),
                             weight_decay=group['weight_decay'], numel=p.numel()))
    require(len(rows) == 7, 'Optimizer CSR inventory changed')
    return dict(parameter_tensors=len(ids), new=rows, exactly_once=True)


def actual_train_api(initialized, expected, variant):
    seen = {}
    class Stopped(Exception):
        pass
    class ProbeTrainer(RTDETRTrainer):
        def __init__(self, overrides, _callbacks):
            self.args = SimpleNamespace(**{k:v for k,v in overrides.items() if k != 'session'})
            self.data = dict(nc=1,channels=3)
            torch.manual_seed(42)
        def get_model(self, cfg=None, weights=None, verbose=False):
            model, audit = build_training_model(cfg,weights,self.data,variant,zero=True)
            seen['native_rebuild'] = audit
            return model
        def train(self):
            require(all(torch.equal(v,self.model.state_dict()[k]) for k,v in expected.state_dict().items()),
                    'Actual model.train path differs from audited native nc=1 rebuild')
            seen.update(status='PASSED', optimizer_steps=0, loading_path='RTDETR.train -> ProbeTrainer.get_model -> native RTDETRTrainer.get_model')
            raise Stopped()
    with patch('ultralytics.engine.model.checks.check_pip_update_available', return_value=None):
        try:
            RTDETR(str(initialized)).train(trainer=ProbeTrainer, data='unused_probe.yaml', seed=42, device='cpu')
        except Stopped:
            pass
    require(seen.get('status') == 'PASSED', 'Actual model.train path not reached')
    return seen


def equivalence(target, variant, shape):
    parent = build(variant,nc=1,baseline=True)
    parent.load_state_dict({k:target.state_dict()[k] for k in parent.state_dict()},strict=True)
    child = deepcopy(target)
    parent.nc = child.nc = 1
    batch = synthetic_batch(shape)
    report = dict(shape=list(shape), effective_gt=3)
    parent.eval(); child.eval()
    with torch.no_grad():
        report['eval'] = compare(parent(batch['img'])[0],child(batch['img'])[0],label='zero-offset eval')
    rows = []
    for model in (parent,child):
        model.train(); model.zero_grad(set_to_none=True)
        inputs = {**batch, 'img':batch['img'].clone().requires_grad_(True)}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(77)
            pred = model.predict(inputs['img'],batch=targets(inputs))
            require(pred[-1] is not None, 'Real DN path missing')
            loss = model.loss(inputs,preds=pred)[0]
            loss.backward()
        rows.append(dict(output=tuple(t.detach() for t in pred[:4]), loss=loss.detach(),
                         input_grad=inputs['img'].grad.detach(), dn=pred[-1]['dn_num_split']))
    report['train_outputs'] = [compare(a,b,label='zero-offset DN output') for a,b in zip(rows[0]['output'],rows[1]['output'])]
    report['loss'] = compare(rows[0]['loss'],rows[1]['loss'],label='zero-offset true loss')
    report['input_gradient'] = compare(rows[0]['input_grad'],rows[1]['input_grad'],label='zero-offset common input gradient')
    report['dn_num_split'] = rows[1]['dn']
    report['initial_gradients'] = gradients(child)
    assert_offset_gradient(report['initial_gradients'])
    report['status'] = 'PASSED'
    return report


def staged_updates(target, batch, device, amp, folder):
    model = deepcopy(target).to(device).train(); model.nc=1
    data = {k:v.to(device) for k,v in batch.items()}
    opt, coverage = optimizer(model)
    scaler = native_scaler(amp)
    effective = [0]
    hook = opt.register_step_post_hook(lambda *args:effective.__setitem__(0,effective[0]+1))
    rows=[]
    start = time.perf_counter()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    diagnostic=folder/('updates_'+device.replace(':','_')+('_amp' if amp else '_fp32')+'.json')
    try:
        for attempt in range(24 if amp else 3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext():
                pred = model.predict(data['img'],batch=targets(data))
                require(pred[-1] is not None, 'DN missing during updates')
                loss = model.loss(data,preds=pred)[0]
            require(torch.isfinite(loss), 'Nonfinite real detection loss')
            old_scale=float(scaler.get_scale()); before=effective[0]
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            grads=gradients(model)
            finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if finite:
                if before == 0: assert_offset_gradient(grads)
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            scaler.step(opt);scaler.update()
            rows.append(dict(attempt=attempt,loss=float(loss.detach()),scale_before=old_scale,
                             scale_after=float(scaler.get_scale()), effective=effective[0]>before,
                             gradients_finite=bool(finite),gradients=grads,dn_num_split=pred[-1]['dn_num_split']))
            write_json(diagnostic,dict(status='RUNNING',steps=rows,effective_updates=effective[0]))
            if effective[0]>=3:break
        require(effective[0]>=3,'Native dynamic scaler did not achieve three bounded effective updates')
        for name in rows[-1]['gradients']:
            require(any(r['effective'] and r['gradients'][name]['finite'] and (r['gradients'][name]['norm'] or 0)>0 for r in rows),
                    'CSR parameter did not receive staged task gradient: '+name)
        require(torch.count_nonzero(model.model[17].csr.offset_pw.weight)>0,'Offsets did not update')
        model.eval()
        with torch.no_grad():
            feature=[]
            handle=model.model[17].register_forward_pre_hook(lambda m,a:feature.append(a[0].detach()))
            model(data['img']);handle.remove()
            lateral=model.model[17].act(model.model[17].bn(model.model[17].conv(feature[0])))
            diagnostics=model.model[17].csr.diagnostics(lateral)
            require(torch.count_nonzero(model.model[17].csr.residual(lateral))>0,'Learned CSR residual remains zero')
        result=dict(status='PASSED',device=device,amp=amp,batch=len(data['img']),shape=list(data['img'].shape[-2:]),
                    effective_updates=effective[0],steps=rows,optimizer=coverage,diagnostics=diagnostics,
                    seconds=time.perf_counter()-start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)
        write_json(diagnostic,result)
        return model.cpu(),opt.state_dict(),scaler.state_dict(),result
    except BaseException as error:
        write_json(diagnostic,dict(status='FAILED',steps=rows,effective_updates=effective[0],
                                  last_gradients=gradients(model),error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        hook.remove()


def lifecycle_checks(learned,opt_state,scaler_state,variant,folder,device='cpu'):
    folder.mkdir(parents=True,exist_ok=True)
    phase_path=folder/('lifecycle_'+device.replace(':','_')+'_phase.json')
    def phase(name):
        print('Lifecycle '+device+': '+name,flush=True)
        write_json(phase_path,dict(status='RUNNING',phase=name))
    phase('checkpoint')
    model=deepcopy(learned).eval().to(device)
    require(torch.count_nonzero(model.model[17].csr.offset_pw.weight)>0,'Lifecycle needs learned nonzero state')
    path=folder/('learned_'+device.replace(':','_')+'.pt')
    torch.save(dict(epoch=1,model=deepcopy(model).cpu(),optimizer=opt_state,scaler=scaler_state),path)
    checkpoint=torch_load(path,map_location='cpu')
    restored=checkpoint['model']
    require(all(torch.equal(v.cpu(),restored.state_dict()[k]) for k,v in model.state_dict().items()),'Checkpoint lost learned state')
    phase('native resume reconstruction')
    resumed,audit=build_training_model(restored.yaml,restored,dict(nc=1,channels=3),variant,zero=False)
    opt,_=optimizer(resumed);opt.load_state_dict(checkpoint['optimizer'])
    for key,state in opt_state['state'].items():
        for name,value in state.items():
            if isinstance(value,torch.Tensor):
                require(torch.equal(value.cpu(),opt.state_dict()['state'][key][name].cpu()),'Optimizer resume changed state')
    phase('optimizer and scaler restore')
    loaded_scaler=native_scaler(bool(scaler_state))
    loaded_scaler.load_state_dict(checkpoint['scaler'])
    require(loaded_scaler.state_dict()==scaler_state,'Scaler resume changed state')
    # One genuine continuation step, compared to an uninterrupted copy with the
    # identical optimizer/RNG. This is an explicitly FP32 serialization diagnostic;
    # server native AMP updates remain a separate gate.
    phase('matched continuation update')
    uninterrupted=deepcopy(restored)
    opt_uninterrupted,_=optimizer(uninterrupted)
    opt_uninterrupted.load_state_dict(deepcopy(checkpoint['optimizer']))
    continued=[]
    for candidate,continuation_opt in ((uninterrupted,opt_uninterrupted),(resumed,opt)):
        candidate.nc=1;candidate.train();continuation_opt.zero_grad(set_to_none=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(991)
            continuation_loss=candidate.loss(synthetic_batch((160,192)))[0]
            require(torch.isfinite(continuation_loss),'Resumed true loss nonfinite')
            continuation_loss.backward()
        require(all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None),'Resumed gradient nonfinite')
        torch.nn.utils.clip_grad_norm_(candidate.parameters(),10.)
        continuation_opt.step()
        continued.append(float(continuation_loss.detach()))
    require(all(torch.equal(v,resumed.state_dict()[k]) for k,v in uninterrupted.state_dict().items()),
            'Native resumed continuation differs from uninterrupted model/optimizer')
    del uninterrupted,opt_uninterrupted,resumed,opt,checkpoint,restored
    gc.collect()
    phase('EMA')
    ema=ModelEMA(deepcopy(model))
    require(all(torch.equal(v,ema.ema.state_dict()[k]) for k,v in model.state_dict().items()),'EMA copy lost learned state')
    ema.update(model)
    require(all(torch.allclose(v,ema.ema.state_dict()[k],atol=1e-7,rtol=1e-6) for k,v in model.state_dict().items()),
            'EMA update changed identical learned state beyond rounding tolerance')
    ema_path=folder/('ema_'+device.replace(':','_')+'.pt')
    torch.save(dict(ema=deepcopy(ema.ema).cpu(),updates=ema.updates),ema_path)
    restored_ema=torch_load(ema_path,map_location='cpu')
    require(restored_ema['updates']==ema.updates and all(torch.equal(v.cpu(),restored_ema['ema'].state_dict()[k])
            for k,v in ema.ema.state_dict().items()),'EMA checkpoint changed learned state')
    del restored_ema
    del ema
    gc.collect()
    phase('full-model fusion')
    x=synthetic_batch((160,192),device)['img'][:1]
    fused=deepcopy(model).fuse(verbose=False)
    verify_model(fused,variant)
    with torch.no_grad():
        native_a,record_a=capture(model,x)
        native_b,record_b=capture(fused,x)
        shared_indices=torch.equal(record_a['candidate_indices'],record_b['candidate_indices'])
        # Numerical BN fusion can reorder close native top-k scores. Report this
        # explicitly and compare the same selected queries if needed.
        if shared_indices:
            output=compare(native_a[0],native_b[0],atol=2e-4,rtol=2e-3,label='learned full-model fusion')
        else:
            replay_a,_=capture(model,x,fixed_ids=record_a['candidate_indices'])
            replay_b,_=capture(fused,x,fixed_ids=record_a['candidate_indices'])
            output=compare(replay_a[0],replay_b[0],atol=2e-4,rtol=2e-3,label='learned fusion with recorded native query IDs')
        feature=torch.randn(1,128,20,24,device=device)
        lateral=compare(model.model[17](feature),fused.model[17](feature),atol=2e-5,rtol=2e-4,label='CSRConv.forward_fuse')
        plain=fused.model[17].act(fused.model[17].conv(feature))
        branch_effect=float((fused.model[17](feature)-plain).abs().max())
        require(branch_effect>0,'Fused CSR branch silently absent')
    result=dict(status='PASSED',checkpoint_sha256=sha256(path),checkpoint_exact=True,
                resume_native_get_model=audit,optimizer_resume_exact=True,scaler_resume_exact=True,
                continuation=dict(status='PASSED',precision='FP32 serialization diagnostic',
                                  native_rebuilt_and_uninterrupted_exact=True,losses=continued,effective_updates=1),
                ema_copy_learned_exact=True,ema_update_finite=True,
                ema_checkpoint_exact=True,ema_checkpoint_sha256=sha256(ema_path),
                full_model_fusion=output,native_query_indices_equal=shared_indices,
                forward_fuse=lateral,nonzero_fused_branch_max_abs=branch_effect,
                counts=dict(unfused=sum(p.numel() for p in model.parameters()),fused=sum(p.numel() for p in fused.parameters())))
    # Explicit half is tested on a learned copy even on CPU: local CSR grid_sample
    # remains FP32, while model convolution/attention support is runtime-dependent.
    phase('explicit half')
    half=deepcopy(model).half()
    identities={n:(id(p),p.dtype) for n,p in half.named_parameters()}
    half_error=None
    try:
        with torch.no_grad():
            half_output=half(x.half())[0]
    except RuntimeError as error:
        if device!='cpu':raise
        half_output=None;half_error=repr(error)
    require(identities=={n:(id(p),p.dtype) for n,p in half.named_parameters()},'Half forward replaced/cast Parameters')
    half_finite=half_output is not None and bool(torch.isfinite(half_output).all())
    if not half_finite and device=='cpu':
        parent_half=build(variant,nc=1,baseline=True)
        parent_half.load_state_dict({k:model.state_dict()[k].cpu() for k in parent_half.state_dict()},strict=True)
        parent_half.eval().half()
        parent_error=None
        try:
            with torch.no_grad():parent_finite=bool(torch.isfinite(parent_half(x.half())[0]).all())
        except RuntimeError as error:
            parent_finite=False;parent_error=repr(error)
        result['explicit_half']=dict(status='CPU_HALF_NUMERICAL_FAILURE',attempted=True,output_finite=False,
            corresponding_learned_parent_without_csr_finite=parent_finite,parameter_identity_preserved=True,
            target_error=half_error,parent_error=parent_error,
            reason='Observed nonfinite CPU whole-model half output; parent control recorded without assuming causation. '
                   'Instrumentation can change reproducibility in this runtime. CPU FP32 and explicit CUDA half are separate required gates; '
                   'no decoder, deterministic policy, or precision settings were changed to mask this result.')
        del parent_half
    else:
        require(half_finite,'Explicit CUDA half inference nonfinite')
        result['explicit_half']=dict(status='PASSED',output_dtype=str(half_output.dtype),parameter_identity_preserved=True)
    # Same learned model through real predict -> AutoBackend -> automatic fuse.
    del half,fused
    gc.collect()
    phase('predict AutoBackend automatic fusion')
    api_path=folder/('predict_'+device.replace(':','_')+'.pt')
    portable=deepcopy(model).cpu();portable.args={**DEFAULT_CFG_DICT,'task':'detect'}
    torch.save(dict(model=portable,epoch=-1,train_args=portable.args),api_path)
    api=RTDETR(str(api_path))
    api.predict(np.zeros((160,192,3),dtype=np.uint8),imgsz=192,device=device,verbose=False,save=False)
    verify_model(api.predictor.model.model,variant)
    result['predict_auto_fuse']=dict(status='PASSED',branch_preserved=True)
    write_json(phase_path,dict(status='PASSED',phase='complete'))
    return result


def enforced_amp_check(model, weights):
    """Execute unchanged native check_amp with local resources; reject silent skips."""
    import ultralytics
    from ultralytics.utils import ASSETS
    from ultralytics.utils.checks import check_amp
    require(Path(weights).is_file(), 'Local AMP-check weights absent: '+str(weights))
    require((ASSETS/'bus.jpg').is_file(), 'Local native AMP asset absent: '+str(ASSETS/'bus.jpg'))
    real_yolo=ultralytics.YOLO
    log=io.StringIO();handler=logging.StreamHandler(log);LOGGER.addHandler(handler)
    try:
        with patch('ultralytics.YOLO',side_effect=lambda name,*a,**kw:real_yolo(str(weights) if name=='yolo26n.pt' else name,*a,**kw)):
            passed=check_amp(model)
    finally:
        LOGGER.removeHandler(handler)
    require(passed and 'checks passed' in log.getvalue(),'Native AMP check failed/skipped: '+log.getvalue())
    return True


def server_capacity(args,folder):
    require(torch.cuda.is_available(),'CUDA unavailable for server capacity')
    require(args.recipe and Path(args.recipe).is_file(),'Full successful-parent recipe required')
    recipe=YAML.load(args.recipe)
    parent_recipe=YAML.load(ROOT/'configs/csr_p3_parent_args.yaml')
    require(set(recipe)==set(parent_recipe),'Recipe field inventory differs from archived successful parent')
    changes={key:dict(parent=parent_recipe[key],target=recipe[key]) for key in parent_recipe if parent_recipe[key]!=recipe[key]}
    require(set(changes)<={'model','name','project','data','save_dir'},'Non-identity parent recipe changes: '+str(changes))
    for key,value in dict(batch=16,imgsz=640,amp=True,nbs=64,epochs=200,optimizer='AdamW',lr0=.0005,
                          warmup_epochs=5,warmup_bias_lr=.1,deterministic=True).items():
        require(recipe.get(key)==value,'Recipe changed: '+key)
    # Only disposable output/model/data identities differ. The real trainer builds
    # the original online augmentation pipeline and optimizer without an epoch.
    recipe.update(model=str(args.init),data=str(args.data),device=args.device,
                  project=str(folder),name='native_capacity',exist_ok=False)
    recipe.pop('save_dir',None)
    class FiniteTrainer(RTDETRTrainer):
        def get_model(self,cfg=None,weights=None,verbose=False):
            model,audit=build_training_model(cfg,weights,self.data,args.variant,zero=True)
            self.csr_init_audit=audit
            return model
    with patch('ultralytics.engine.trainer.check_amp',side_effect=lambda model:enforced_amp_check(model,args.amp_check_weights)):
        trainer=FiniteTrainer(overrides=recipe)
        trainer._setup_train()
    require(trainer.amp and trainer.batch_size==16 and trainer.args.imgsz==640,'Native setup changed B16/640/AMP')
    trainer.epoch=trainer.start_epoch
    trainer._model_train()
    coverage=optimizer_coverage(trainer.model,trainer.optimizer)
    effective=[0];hook=trainer.optimizer.register_step_post_hook(lambda *a:effective.__setitem__(0,effective[0]+1))
    nb=len(trainer.train_loader);nw=max(round(trainer.args.warmup_epochs*nb),100)
    # Execute the current native warmup block, not a separately approximated schedule.
    source=inspect.getsource(RTDETRTrainer._do_train)
    begin=source.index('                if ni <= nw:');end=source.index('                # Forward',begin)
    warmup=textwrap.dedent(source[begin:end])
    rows=[];trainer.optimizer.zero_grad();last_opt_step=-1
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    diagnostic=folder/'server_capacity_steps.json'
    try:
        for i,batch in enumerate(trainer.train_loader):
            require(i<64,'Native scaler failed to produce two effective updates within bounded preflight')
            ni=i+nb*trainer.epoch
            exec(warmup,dict(self=trainer,ni=ni,nw=nw,epoch=trainer.epoch,np=np))
            batch=trainer.preprocess_batch(batch)
            require(batch['img'].shape==(16,3,640,640) and batch['bboxes'].numel()>0,'Real B16/640 effective GT required')
            with torch.autocast('cuda',dtype=torch.float16):
                pred=trainer.model.predict(batch['img'],batch=targets(batch))
                require(pred[-1] is not None,'Native DN missing')
                loss=trainer.model.loss(batch,preds=pred)[0]
            require(torch.isfinite(loss),'Nonfinite native AMP loss')
            old_scale=float(trainer.scaler.get_scale());before=effective[0]
            trainer.scaler.scale(loss).backward()
            grads=gradients(trainer.model)
            for row in grads.values():
                if row['norm'] is not None:row['norm']/=old_scale
            step=ni-last_opt_step>=trainer.accumulate
            if step:
                # Unscale/clip/step/update/EMA are the original native method.
                trainer.optimizer_step();last_opt_step=ni
            rows.append(dict(batch=i,loss=float(loss.detach()),effective=effective[0]>before,
                             scale_before=old_scale,scale_after=float(trainer.scaler.get_scale()),
                             accumulate=trainer.accumulate,gradients=grads,gt=len(batch['bboxes']),
                             dn_num_split=pred[-1]['dn_num_split'],files=[str(p) for p in batch['im_file']]))
            write_json(diagnostic,dict(status='RUNNING',steps=rows,effective_updates=effective[0]))
            if effective[0]>=2 and all(any(r['effective'] and r['gradients'][n]['finite'] and
                (r['gradients'][n]['norm'] or 0)>0 for r in rows) for n in grads):break
        require(effective[0]>=2,'Fewer than two native effective optimizer updates')
        require(torch.count_nonzero(trainer.model.model[17].csr.offset_pw.weight)>0,'Server offsets did not update')
        require(all(any(r['effective'] and r['gradients'][n]['finite'] and (r['gradients'][n]['norm'] or 0)>0
                for r in rows) for n in grads),'Server staged CSR gradients incomplete')
        torch.cuda.synchronize()
        result=dict(status='PASSED',batch=16,imgsz=640,amp=True,effective_updates=effective[0],
                    native_amp_check='PASSED_WITH_LOCAL_RESOURCES',original_online_augmentation=True,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    seconds=time.perf_counter()-started,steps=rows,optimizer=coverage,
                    native_warmup_sha256=hashlib.sha256(warmup.encode()).hexdigest(),formal_training='NOT_STARTED')
        from ultralytics.utils import ASSETS
        result['amp_resources']=dict(weights=dict(path=str(args.amp_check_weights.resolve()),sha256=sha256(args.amp_check_weights)),
                                     bus=dict(path=str((ASSETS/'bus.jpg').resolve()),sha256=sha256(ASSETS/'bus.jpg')))
        require(torch.count_nonzero(trainer.ema.ema.model[17].csr.offset_pw.weight)>0,'Actual server EMA lost learned CSR')
        result['native_ema_updates']=trainer.ema.updates
        del pred,loss,batch
        result['learned_lifecycle']=lifecycle_checks(trainer.model.cpu(),trainer.optimizer.state_dict(),
                              trainer.scaler.state_dict(),args.variant,folder/'server_learned',str(trainer.device))
        write_json(diagnostic,result)
        return result
    except BaseException as error:
        write_json(diagnostic,dict(status='FAILED',steps=rows,effective_updates=effective[0],
                                  last_gradients=gradients(trainer.model),error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        hook.remove()
        del trainer
        gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant',choices=VARIANTS)
    parser.add_argument('--mode',choices=('local','server'),default='local')
    for name in ('source','init','data','recipe','amp-check-weights'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--device',default='0')
    parser.add_argument('--cuda-small',action='store_true',help='Also run real-data small CUDA FP32/native dynamic AMP diagnostics')
    args=parser.parse_args()
    require(not args.report.exists(),'Preserve existing preflight report: '+str(args.report))
    folder=args.report.parent/(args.report.stem+'_artifacts')
    folder.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    report=dict(status='FAILED',mode=args.mode,variant=args.variant,created_at=datetime.now(timezone.utc).isoformat(),
                formal_training='NOT_STARTED',test='NOT_RUN',runtime=runtime(),checks={},pending=[],
                determinism=dict(enabled=True,warn_only=True,cudnn_benchmark=False),warnings=[],
                precision='CPU FP32 plus explicitly labelled optional CUDA diagnostics; native server AMP remains separate')
    original_showwarning=warnings.showwarning
    def record_warning(message,category,filename,lineno,file=None,line=None):
        row=dict(category=category.__name__,message=str(message),file=filename,line=lineno)
        if row not in report['warnings'] and len(report['warnings'])<50:report['warnings'].append(row)
        original_showwarning(message,category,filename,lineno,file,line)
    warnings.showwarning=record_warning
    try:
        report['identity']=identity(args.variant,args.source,args.init,args.data,args.recipe)
        missing=[name for name in ('source','init') if not getattr(args,name) or not getattr(args,name).is_file()]
        if missing:
            report['pending'].append('Controlled initialization checks need unavailable local files: '+', '.join(missing))
            report['server_capacity']=dict(status='PENDING',reason='Required inputs unavailable')
            report['status']='PENDING'
            return
        require(report['identity']['source_sha256']==SOURCE_SHA256,'Unified source wrong SHA')
        if not args.data or not args.data.is_file():
            report['pending'].append('Real dataset identity and real-image checks: data config unavailable')
        elif report['identity']['data_inventory'] is None:
            report['pending'].append('Real dataset identity and real-image checks: dataset split directories unavailable')
        parent,fresh,mapping=controlled_models(args.source,args.variant)
        loaded=RTDETR(str(args.init)).model
        require(all(torch.equal(v,loaded.state_dict()[k]) for k,v in fresh.state_dict().items()),'Init differs from controlled source')
        torch.manual_seed(42)
        target,adapt=build_training_model(loaded.yaml,loaded,dict(nc=1,channels=3),args.variant,zero=True)
        target.nc=1
        report['checks'].update(initialization=mapping,trainer_nc1=adapt,topology=verify_model(target,args.variant,zero=True),
                                actual_train_api=actual_train_api(args.init,target,args.variant))
        del parent,fresh,loaded;gc.collect()
        report['checks']['cpu_equivalence']=[equivalence(target,args.variant,shape) for shape in ((160,160),(160,192))]
        write_json(args.report,report)
        learned,opt,scaler,updates=staged_updates(target,synthetic_batch(),'cpu',False,folder)
        report['checks']['cpu_updates']=updates
        report['checks']['cpu_lifecycle']=lifecycle_checks(learned,opt,scaler,args.variant,folder)
        del learned,opt;gc.collect()
        report['checks']['parameter_counts']={v:dict(parent_unfused=COUNTS[v][0]-17548,
                target_unfused=COUNTS[v][0],parent_fused=COUNTS[v][1]-17548,target_fused=COUNTS[v][1],delta=17548) for v in VARIANTS}
        report['checks']['complexity_scope']='Parameter counts verified against actual model topology; sampling is not counted as ordinary convolution GFLOPs.'
        if args.cuda_small:
            if torch.cuda.is_available() and report['identity']['data_inventory'] is not None:
                batch,records=real_batch(data_root(args.data),160,2)
                report['checks']['real_samples']=records
                for amp in (False,True):
                    learned,opt,scaler,updates=staged_updates(target,batch,'cuda:0',amp,folder)
                    report['checks']['cuda_small_'+('amp' if amp else 'fp32')]=updates
                    if amp:
                        report['checks']['cuda_lifecycle']=lifecycle_checks(learned,opt,scaler,args.variant,folder,'cuda:0')
                    del learned,opt;gc.collect();torch.cuda.empty_cache()
            else:
                report['pending'].append('CUDA small real-data checks: local CUDA/data unavailable')
        if args.mode=='server':
            missing=[]
            if not torch.cuda.is_available():missing.append('CUDA')
            for name in ('data','recipe','amp_check_weights'):
                value=getattr(args,name)
                if not value or not value.is_file():missing.append(name)
            from ultralytics.utils import ASSETS
            if not (ASSETS/'bus.jpg').is_file():missing.append(str(ASSETS/'bus.jpg'))
            if report['identity']['data_inventory'] is None:missing.append('dataset split directories')
            if missing:
                report['server_capacity']=dict(status='PENDING',missing=missing)
                report['pending'].append('Server native B16/640: '+', '.join(missing))
            else:
                report['server_capacity']=server_capacity(args,folder)
        else:
            report['server_capacity']=dict(status='PENDING',reason='Not executed on training server',batch=16,imgsz=640,amp=True)
            report['pending'].append('Actual server B16/640 online augmentation/native AMP capacity and two effective optimizer updates')
        half_checks=[report['checks'].get('cuda_lifecycle',{}).get('explicit_half',{}),
                     report.get('server_capacity',{}).get('learned_lifecycle',{}).get('explicit_half',{})]
        if not any(row.get('status')=='PASSED' for row in half_checks):
            report['pending'].append('Explicit whole-model CUDA half inference on learned nonzero state')
        require(sha256(args.init)==report['identity']['init_sha256'],'Preflight modified controlled initialization')
        report['status']='PENDING' if report['pending'] else 'PASSED'
    except BaseException as error:
        report['failure']=dict(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        warnings.showwarning=original_showwarning
        write_json(args.report,report)
        print(json.dumps(dict(status=report['status'],report=str(args.report.resolve()),pending=report['pending']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
