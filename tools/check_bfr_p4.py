"""Bounded local engineering diagnostics; never formal epochs or split evaluation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from init_bfr_p4 import (ROOT, VARIANTS, controlled_models, build_training_model, native_rebuild,
                        verify_model, require, write_json, runtime, is_added, sha256)
from c19_lif_v1_probe import capture, targets
from c19_lif_v1_data import real_batch, dataset_inventory
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import BFRP4, BFRRepC3
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA


@contextmanager
def strict_tf32():
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield old
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def compare(a, b, atol=2e-5, rtol=2e-4, enforce=True):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    require(a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(), 'Shape/nonfinite mismatch')
    result = dict(max_abs=float((a-b).abs().max()), relative_l2=float((a-b).norm()/(a.norm()+1e-12)),
                  close=bool(torch.allclose(a,b,atol=atol,rtol=rtol)), atol=atol, rtol=rtol)
    if enforce: require(result['close'], f'Numerical mismatch: {result}')
    return result


def optimizer(model):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args = SimpleNamespace(warmup_bias_lr=.1, lr0=.0005, weight_decay=.0001)
    opt = trainer.build_optimizer(model, name='AdamW', lr=.0005, momentum=.937, decay=.0001)
    ids = [id(p) for group in opt.param_groups for p in group['params']]
    require(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad},
            'Optimizer coverage must be exact')
    return opt


def state_equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict): return set(a)==set(b) and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a, (tuple,list)): return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b


def wiring(model, image):
    seen = {}; handles = []
    def complete_parent(m,args,out): seen['complete_parent_output'] = out
    def before(m,args): seen['original_p4'] = args[0]; seen['expected'] = args[0].detach().clone()
    def after(m,args,out): seen['result'] = out
    def consumer(m,args): seen['p5_input'] = args[0]
    def decoder(m,args): seen['decoder_p4'] = args[0][1]
    handles += [model.model[22].cv3.register_forward_hook(complete_parent),
                model.model[22].bfr.register_forward_pre_hook(before), model.model[22].bfr.register_forward_hook(after),
                model.model[23].register_forward_pre_hook(consumer), model.model[26].register_forward_pre_hook(decoder)]
    try:
        with torch.no_grad(): model(image)
        require(seen['result'] is seen['p5_input'] and seen['result'] is seen['decoder_p4'], 'P4 consumers lost exact BFR output')
        require(seen['original_p4'] is seen['complete_parent_output'], 'BFR input is not the full original RepC3 output')
        require(torch.equal(seen['original_p4'],seen['expected']), 'BFR mutated original RepC3 result')
        return dict(status='PASSED',input_shape=list(seen['original_p4'].shape),both_consumers_same_object=True,
                    complete_parent_output_same_object=True,nodes=len(model.model),wrapper_repeat=len(model.model[22].m))
    finally:
        for h in handles:h.remove()


def initial_alignment(parent,target,batch):
    parent.eval(); target.eval()
    report={}
    with torch.no_grad():
        for h,w in ((160,192),(640,640)):
            x=torch.rand(1,3,h,w)
            a,ra=capture(parent,x); b,rb=capture(target,x)
            report[f'eval_{h}x{w}']=compare(a[0],b[0],0,0)
            for key in ('scale_0','scale_1','scale_2','candidate_indices'):
                require(torch.equal(ra[key],rb[key]), 'Initial parent mismatch '+key)
            report[f'wiring_{h}x{w}']=wiring(target,x)
    p,t=deepcopy(parent).train(),deepcopy(target).train();p.nc=t.nc=1
    rng=torch.get_rng_state()
    pa,pr=capture(p,batch['img'],targets(batch));pl=p.loss(batch,preds=pa)[0]
    torch.set_rng_state(rng)
    ta,tr=capture(t,batch['img'],targets(batch));tl=t.loss(batch,preds=ta)[0]
    report['train_loss']=compare(pl,tl,0,0)
    require(pr['dn_split']==tr['dn_split'] and pr['dn_split'] is not None,'DN not aligned')
    report['train_dn_split']=pr['dn_split']
    report['bn_buffers_exact']=all(torch.equal(v,t.state_dict()[k]) for k,v in p.state_dict().items())
    require(report['bn_buffers_exact'],'Training BN/shared parameters drift')
    return report


def train_copy(target,batch,device,amp,budget=16):
    model=deepcopy(target).to(device).train();model.nc=1
    sample={k:v.to(device) for k,v in batch.items()}
    opt=optimizer(model);scaler=torch.amp.GradScaler('cuda',enabled=amp)
    ema=ModelEMA(model);rows=[];effective=0;first=None
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for attempt in range(budget if amp else 2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,enabled=amp):
            pred,taps=capture(model,sample['img'],targets(sample))
            loss=model.loss(sample,preds=pred)[0]
        require(bool(torch.isfinite(loss)), 'Nonfinite detection loss')
        scaler.scale(loss).backward();scaler.unscale_(opt)
        grads={n:float(p.grad.float().norm()) if p.grad is not None else None
               for n,p in model.named_parameters() if is_added(n)}
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        before=model.model[22].bfr.wo.weight.detach().clone();scale=float(scaler.get_scale())
        if finite:
            require(grads['model.22.bfr.wo.weight']>0,'W_o did not receive an effective gradient')
            if effective:
                for key in ('wd.weight','gn.weight','fc1.weight','fc1.bias','fc2.weight','fc2.bias'):
                    require(grads['model.22.bfr.'+key]>0,'No identifiable upstream gradient: '+key)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        elif not amp:raise RuntimeError('FP32 nonfinite gradients')
        scaler.step(opt);scaler.update()
        updated=not torch.equal(before,model.model[22].bfr.wo.weight)
        if updated:
            effective+=1;ema.update(model)
            if first is None:first=grads
        rows.append(dict(attempt=attempt,loss=float(loss.detach()),gt=int(sample['cls'].numel()),dn_split=taps['dn_split'],
                         finite_gradients=finite,scale=scale,next_scale=float(scaler.get_scale()),updated=updated,
                         new_grad_norms={k:v if v is None or v<float('inf') else 'NONFINITE' for k,v in grads.items()}))
        del pred,loss,taps
        if effective>=2:break
    require(effective>=2,'Finite budget exhausted before two effective updates')
    for key,value in first.items():
        if key!='model.22.bfr.wo.weight':require(value==0,'Initial upstream gradient should be zero at W_o=0')
    require(torch.count_nonzero(model.model[22].bfr.wo.weight)>0,'Learned W_o remains zero')
    if device=='cuda':torch.cuda.synchronize()
    result=dict(status='PASSED',device=device,amp=amp,budget=budget if amp else 2,actual_batches=len(rows),effective_updates=effective,
                batch=len(sample['img']),image_shape=list(sample['img'].shape),gt_source='real train labels',steps=rows,
                elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                gn_bias_exception='Spatially constant; DC is excluded. Exact-arithmetic zero direction, retained in optimizer.',
                optimizer_parameter_coverage='exactly once',initial_scaler='native default 65536' if amp else 'disabled FP32 diagnostic',
                scope='B2/160 fixed real images for engineering only; not B16/640 capacity or formal training')
    return model,opt,scaler,ema,result


def fusion_trace(model,image,device,precision):
    left=deepcopy(model).float().eval();right=deepcopy(left).fuse(verbose=False)
    require(type(right.model[22]) is BFRRepC3 and type(right.model[22].bfr) is BFRP4,'Fusion dropped BFR')
    require(state_equal(left.model[22].bfr.state_dict(),right.model[22].bfr.state_dict()),'Fusion changed BFR state')
    if precision=='half':left.half();right.half();image=image.half()
    with torch.no_grad(),torch.autocast(device_type=device,enabled=precision=='amp'):
        pa,a=capture(left,image);pb,b=capture(right,image)
        replay_a,ra=capture(left,image,fixed_ids=a['candidate_indices'])
        replay_b,rb=capture(right,image,fixed_ids=a['candidate_indices'])
    atol,rtol=(2e-5,2e-4) if precision=='fp32' else (3e-3,3e-2)
    pre={k:compare(a[k],b[k],atol,rtol) for k in ('scale_0','scale_1','scale_2','encoder_features','candidate_scores')}
    natural=compare(pa[0],pb[0],atol,rtol,False)
    replay=compare(replay_a[0],replay_b[0],atol,rtol)
    ids_equal=torch.equal(a['candidate_indices'],b['candidate_indices'])
    return dict(status='PASSED_OPERATOR_DIAGNOSTIC',precision=precision,parameters=sum(p.numel() for p in right.parameters()),
                bfr_preserved=True,continuous_features=pre,native_candidates_equal=ids_equal,
                changed_candidate_positions=int((a['candidate_indices']!=b['candidate_indices']).sum()),
                native_output=natural,fixed_candidate_replay=replay,
                equivalence='NATIVE_OUTPUT_CLOSE' if natural['close'] else 'NATIVE_SELECTION_DRIFT; fixed replay is diagnostic only')


def lifecycle(model,opt,scaler,ema,folder,device,variant=None):
    variant=variant or ('cbr_lif_bfr_p4_v1' if hasattr(model.model[-1],'cbr') else 'bfr_p4_v1')
    model.eval();x=torch.rand(1,3,160,192,device=device)
    require(torch.count_nonzero(model.model[22].bfr.wo.weight)>0,'Lifecycle needs learned nonzero BFR')
    ema.update(model)
    require(torch.count_nonzero(ema.ema.model[22].bfr.wo.weight)>0,'EMA dropped branch')
    quantized=deepcopy(model).cpu().half()
    payload=dict(model=quantized,ema=deepcopy(ema.ema).cpu().half(),updates=ema.updates,optimizer=opt.state_dict(),
                 scaler=scaler.state_dict(),epoch=0,best_fitness=0.,train_args=dict(task='detect'))
    path=folder/(device+('_amp' if scaler.is_enabled() else '_fp32')+'_learned.pt')
    torch.save(payload,path);loaded=torch_load(path,map_location='cpu')
    require(state_equal(loaded['model'].state_dict(),quantized.state_dict()),'Same-quantization reload mismatch')
    # The actual native Trainer get_model reconstruction must retain learned
    # shared/BFR tensors, not just a deserialized module passed to resume_training.
    restored,loading=build_training_model(loaded['model'].yaml,loaded['model'].float(),
                                         dict(nc=1,channels=3),variant)
    restored=restored.to(device)
    trainer=RTDETRTrainer.__new__(RTDETRTrainer);trainer.model=restored
    trainer.optimizer=optimizer(restored);trainer.scaler=torch.amp.GradScaler('cuda',enabled=scaler.is_enabled())
    trainer.ema=ModelEMA(restored);trainer.resume=True;trainer.epochs=200
    trainer.args=SimpleNamespace(model=str(path),close_mosaic=10)
    trainer.resume_training(loaded)
    require(trainer.start_epoch==1 and state_equal(trainer.optimizer.state_dict(),opt.state_dict()),'Native resume optimizer/epoch mismatch')
    require(trainer.scaler.state_dict()==scaler.state_dict(),'Native resume scaler mismatch')
    require(state_equal(restored.model[22].bfr.state_dict(),quantized.float().model[22].bfr.state_dict()),'Native resume lost learned BFR')
    require(state_equal(trainer.ema.ema.state_dict(),loaded['ema'].float().state_dict()),'Native resume EMA tensors mismatch')
    require(trainer.ema.updates==loaded['updates'],'Native resume EMA updates mismatch')
    report=dict(learned_nonzero=True,save_reload_same_half_quantization=True,ema_nonzero=True,native_resume_training=True,
                native_learned_get_model=loading,ema_tensors_exact=True,ema_updates_exact=True,
                epoch=trainer.start_epoch,optimizer_exact=True,scaler_exact=True,checkpoint=str(path),checkpoint_sha256=sha256(path),
                checkpoint_scope='Disposable half checkpoint; never controlled init or formal resume input')
    restored.eval()
    with torch.no_grad():report['reload_inference']=compare(restored(x)[0],quantized.to(device)(x)[0],0,0)
    report['fusion_default_fp32']=fusion_trace(model,x,device,'fp32')
    with strict_tf32() as old:
        report['tf32_original']=list(old)
        report['fusion_strict_fp32']=fusion_trace(model,x,device,'fp32')
    if device=='cuda':
        report['half_fusion']=fusion_trace(model,x,device,'half')
        report['amp_fusion']=fusion_trace(model,x,device,'amp')
        from ultralytics.nn.autobackend import AutoBackend
        backend=AutoBackend(model=deepcopy(model).float(),device=torch.device('cuda'),fp16=True,verbose=False)
        with torch.no_grad():
            backend.warmup((1,3,160,192))
            output=backend(x.half())[0]
        require(torch.isfinite(output).all(),'Native AutoBackend half failed')
        report['autobackend_half_finite_warmup']=True
    require((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)==tuple(report['tf32_original']),'TF32 flags not restored')
    return report


def benchmark(model,device):
    model=deepcopy(model).eval().to(device);x=torch.rand(1,3,640,640,device=device)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    times=[]
    with torch.no_grad():
        for i in range(4):
            if device=='cuda':torch.cuda.synchronize()
            t=time.perf_counter();y=model(x)[0]
            if device=='cuda':torch.cuda.synchronize()
            require(torch.isfinite(y).all(),'Benchmark nonfinite')
            times.append(1000*(time.perf_counter()-t))
    return dict(device=device,batch=1,imgsz=640,first_ms=times[0],subsequent_ms=times[1:],
                cache='No persistent DCT cache: constants regenerated in FP32 on every call',
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=list(VARIANTS),required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--devices',nargs='+',choices=['cpu','cuda'],default=['cpu'])
    args=p.parse_args();torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',variant=args.variant,runtime=runtime(),formal_training='NOT_STARTED',final_test='NOT_RUN')
    dest=args.output/'checks.json'
    try:
        inventory=dataset_inventory(args.dataset)
        expected=json.loads((ROOT/'docs/bfr_p4/parent_dataset_inventory.json').read_text())
        require(inventory==expected,'Real dataset identity differs from successful parent')
        report['dataset_identity']=inventory
        parent80,target80,report['controlled']=controlled_models(args.source,args.variant)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            parent=native_rebuild(parent80.yaml,parent80,nc=1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            target,report['trainer']=build_training_model(target80.yaml,target80,dict(nc=1,channels=3),args.variant)
        del parent80,target80;gc.collect()
        batch,report['train_samples']=real_batch(args.dataset,160,2)
        report['initial_alignment']=initial_alignment(parent,target,batch)
        del parent;gc.collect();write_json(dest,report)
        report['modes']={}
        for device in args.devices:
            if device=='cuda' and not torch.cuda.is_available():
                report['modes']['cuda']=dict(status='PENDING',reason='CUDA unavailable');continue
            report['benchmark_'+device]=benchmark(target,device)
            for amp in ([False,True] if device=='cuda' else [False]):
                key=device+('_amp' if amp else '_fp32')
                learned,opt,scaler,ema,entry=train_copy(target,batch,device,amp)
                report['modes'][key]=entry;write_json(dest,report)
                entry['lifecycle']=lifecycle(learned,opt,scaler,ema,args.output,device)
                del learned,opt,scaler,ema;gc.collect()
                if device=='cuda':torch.cuda.empty_cache()
                write_json(dest,report)
        report['server_capacity']=dict(status='PENDING',reason='Separate real augmented B16/640 native AMP server gate required')
        report['status']='PASSED_LOCAL'
    except BaseException as e:
        report.update(status='FAILED',error=repr(e));raise
    finally:write_json(dest,report)
    print(json.dumps(dict(status=report['status'],output=str(dest)),indent=2))


if __name__=='__main__':main()
