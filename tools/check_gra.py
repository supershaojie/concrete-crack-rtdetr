"""Bounded GRA engineering checks; never launches formal training or final test."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import torch
from init_gra import (ROOT, VARIANTS, controlled_models, require, runtime, sha256,
                      verify_model, write_json)
from c19_lif_v1_probe import capture, targets
from c19_lif_v1_diagnostic import PRE_KEYS, selection_report
from c19_lif_v1_data import dataset_inventory, real_batch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.patches import torch_load


def comparison(a, b, atol=2e-5, rtol=2e-4):
    device_a,device_b=str(a.device),str(b.device)
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.shape != b.shape:
        return dict(status='FAILED', shape_a=list(a.shape), shape_b=list(b.shape))
    aa, bb = a.double(), b.double()
    finite = bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
    d = (aa-bb).abs()
    bad = d > atol + rtol*bb.abs()
    return dict(status='PASSED' if finite and not bad.any() else 'FAILED',
                shape=list(a.shape), dtype_a=str(a.dtype), dtype_b=str(b.dtype), device_a=device_a,device_b=device_b,finite=finite,
                max_abs=float(d.max()) if d.numel() else 0.,
                relative_L2=float(torch.linalg.vector_norm(aa-bb)/torch.linalg.vector_norm(bb).clamp_min(1e-30)),
                exceeded_fraction=float(bad.double().mean()) if bad.numel() else 0., atol=atol, rtol=rtol)


def equal_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return set(sa) == set(sb) and all(torch.equal(v.cpu(), sb[k].cpu()) for k,v in sa.items())


@contextmanager
def strict_fp32():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def wiring(model, image, batch=None):
    saved, rows, handles = {}, {}, []
    for index in (15,16,17,19,20,22,25):
        def hook(module, args, value, index=index): saved[index] = value
        handles.append(model.model[index].register_forward_hook(hook))
    def before(module,args):
        require(len(args[0]) == 3, 'GRA requires H,U,L')
        rows['inputs'] = [dict(source=i, shape=list(v.shape),dtype=str(v.dtype),device=str(v.device),
                               same_object=v is saved[i], equal=torch.equal(v,saved[i]))
                          for i,v in zip((15,16,17),args[0])]
        require(all(r['same_object'] and r['equal'] for r in rows['inputs']), 'GRA receives wrong source tensor')
    def after(module,args,value):
        rows['lateral'] = comparison(value[:,256:],saved[17],0,0)
        rows['nearest_residual'] = comparison(value[:,:256],saved[16],0,0)
        saved[18] = value
    def downstream(module,args):
        rows['layer20_uses_19'] = args[0] is saved[19]
    def decoder(module,args):
        rows['decoder_uses_19_22_25'] = [v is saved[i] for i,v in zip((19,22,25),args[0])]
    handles += [model.model[18].register_forward_pre_hook(before),model.model[18].register_forward_hook(after),
                model.model[20].register_forward_pre_hook(downstream),model.model[26].register_forward_pre_hook(decoder)]
    try:
        with torch.no_grad(): _, records = capture(model,image,batch)
    finally:
        for h in handles:h.remove()
    rows.update(nodes=len(model.model),savelist=model.save,source15_actually_read=True,
                output_shapes={str(k):list(v.shape) for k,v in saved.items()},
                layer20_class=type(model.model[20]).__name__,decoder_class=type(model.model[26]).__name__)
    require(rows['lateral']['status']=='PASSED' and rows['layer20_uses_19'] and all(rows['decoder_uses_19_22_25']), 'Downstream wiring mismatch')
    return rows,records


def make_optimizer(model):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args=SimpleNamespace(warmup_bias_lr=.1,lr0=.0005,weight_decay=.0001)
    opt=trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    expected={id(p) for p in model.parameters() if p.requires_grad}
    require(len(ids)==len(set(ids)) and set(ids)==expected,'Optimizer coverage mismatch')
    return opt,dict(trainable_tensors=len(ids),each_exactly_once=True,
                    gra=[dict(name=n,numel=p.numel()) for n,p in model.named_parameters() if n.startswith('model.18.')])


def gra_stats(model,image):
    m=model.model[18];record={};handles=[]
    def tap(module,args,value):
        delta=.25*module.offset_logits(args[0]).float().tanh()
        correction=value[:,:256].float()-args[0][1].float()
        record.update(offset_abs_max=float(delta.abs().max()),offset_abs_mean=float(delta.abs().mean()),
                      saturation_fraction=float((delta.abs()>.249).float().mean()),finite=bool(torch.isfinite(delta).all()),
                      residual_abs_max=float(correction.abs().max()),offset_weight_nonzero=int(torch.count_nonzero(module.offset.weight)))
    handles.append(m.register_forward_hook(tap))
    try:
        with torch.no_grad():model(image)
    finally:
        for h in handles:h.remove()
    return record


def scaler_for(amp):
    # CUDA API exists on Torch 2.1; default scale/backoff are intentionally native.
    return torch.cuda.amp.GradScaler(enabled=amp)


def state_tree_equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict): return a.keys()==b.keys() and all(state_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(state_tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def lifecycle(model,opt,scaler,ema,image,folder):
    model.eval(); learned=deepcopy(model).cpu();path=folder/'learned_disposable.pt'
    checkpoint=dict(model=learned,ema=deepcopy(ema.ema).cpu(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                    epoch=0,updates=ema.updates,best_fitness=.1,train_args=dict(task='detect'))
    torch.save(checkpoint,path);loaded=torch_load(path,map_location='cpu')
    require(equal_state(learned,loaded['model']),'Learned reload changed model')
    require(torch.count_nonzero(loaded['model'].model[18].offset.weight)>0,'Reload lost learned GRA')
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.model=deepcopy(loaded['model']).to(image.device);trainer.optimizer,_=make_optimizer(trainer.model)
    trainer.scaler=scaler_for(scaler.is_enabled());trainer.ema=ModelEMA(trainer.model)
    trainer.resume=True;trainer.epochs=200;trainer.args=SimpleNamespace(model=str(path),close_mosaic=10)
    RTDETRTrainer.resume_training(trainer,loaded)
    require(trainer.start_epoch==1 and trainer.ema.updates==ema.updates,'Native resume epoch/EMA update mismatch')
    require(state_tree_equal(opt.state_dict(),trainer.optimizer.state_dict()),'Native resume optimizer mismatch')
    require(state_tree_equal(scaler.state_dict(),trainer.scaler.state_dict()),'Native resume scaler mismatch')
    require(equal_state(ema.ema,trainer.ema.ema),'Native resume EMA mismatch')
    cache=lambda m:dict(shapes=m.model[26].shapes,anchors_dtype=str(m.model[26].anchors.dtype),
                        anchors_device=str(m.model[26].anchors.device),anchors_shape=list(m.model[26].anchors.shape))
    caches=dict(ema_original=cache(ema.ema),native_resume_ema=cache(trainer.ema.ema))
    ema_direct=deepcopy(loaded['ema']).to(image.device).eval()
    with torch.no_grad():
        orig=model(image)[0];restored=trainer.model.eval()(image)[0]
        ea=ema.ema.eval()(image)[0];eb=ema_direct(image)[0];native_ema=trainer.ema.ema.eval()(image)[0]
    report=dict(status='PASSED',reload_state_exact=True,native_resume=True,start_epoch=trainer.start_epoch,
                optimizer_exact=True,scaler_exact=True,scaler_state=trainer.scaler.state_dict(),ema_own_reload=comparison(ea,eb,0,0),
                model_reload=comparison(orig,restored,0,0),ema_updates=ema.updates,checkpoint_sha256=sha256(path),
                native_resume_ema_output=comparison(ea,native_ema,0,0),native_resume_ema_caches=caches)
    write_json(folder/'lifecycle.json',report)
    require(report['model_reload']['status']==report['ema_own_reload']['status']=='PASSED','Learned reload forward mismatch')
    require(report['native_resume_ema_output']['finite'],'Nonfinite native resume EMA output')
    if report['native_resume_ema_output']['status']!='PASSED':
        # Original native resume creates EMA from the immediate model, then loads
        # EMA parameters. Shape-only non-state_dict anchor cache can consequently
        # retain AMP precision. Diagnose without editing native code or weights.
        left,right=deepcopy(ema.ema),deepcopy(trainer.ema.ema)
        left.model[26].shapes=[];right.model[26].shapes=[]
        with strict_fp32(),torch.no_grad():a=left(image)[0];b=right(image)[0]
        report['native_resume_ema_recache_diagnostic']=comparison(a,b,0,0)
        require(report['native_resume_ema_recache_diagnostic']['status']=='PASSED','Unexplained native EMA resume difference')
        report['native_resume_ema_note']='PRECISION_NOTE: unchanged native resume inherits immediate-model shape-only anchor cache; regenerating both diagnostic caches gives exact EMA output. Learned parameters/buffers are exact.'
        del left,right
    # Actual native setup_model -> get_model selects EMA from the checkpoint and
    # reconstructs fresh caches; verify the learned, nonzero state on this path.
    from train_gra import RecordingTrainer
    variant='cbr_lif_gra_v1' if hasattr(model.model[26],'cbr') else 'gra_v1'
    native=RecordingTrainer.__new__(RecordingTrainer);native.gra_variant=variant
    native.model=str(path.resolve());native.args=SimpleNamespace(pretrained=False)
    native.data=dict(nc=1,channels=3);native.setup_model()
    require(equal_state(native.model,loaded['ema']),'Native learned get_model differs from checkpoint EMA')
    require(torch.count_nonzero(native.model.model[18].offset.weight)>0,'Native get_model reset learned GRA')
    report['native_learned_setup_model']=dict(status='PASSED',checkpoint_model_selection='EMA',all_state_exact=True)
    if image.device.type=='cuda':
        half_a=deepcopy(model).half();half_path=folder/'learned_half.pt';torch.save(half_a,half_path)
        half_b=torch_load(half_path,map_location=image.device)
        with torch.no_grad(): a=half_a(image.half())[0];b=half_b(image.half())[0]
        report['half_reload']=comparison(a,b,0,0)
        half_a.float();half_b.float()
        with torch.no_grad():a=half_a(image)[0];b=half_b(image)[0]
        report['same_quantization_half_float']=comparison(a,b,0,0)
        require(report['half_reload']['status']==report['same_quantization_half_float']['status']=='PASSED','Half learned lifecycle failed')
    else:report['half']='PENDING_CUDA; CPU whole-network half is not the CUDA criterion'
    write_json(folder/'lifecycle.json',report)
    del trainer,learned,loaded,native,ema_direct;gc.collect()
    return report


def learned_fusion(model,image):
    model.eval();fused=deepcopy(model).fuse(verbose=False)
    require(equal_state(model.model[18],fused.model[18]),'Fusion lost GRA state')
    require(torch.count_nonzero(fused.model[18].offset.weight)>0,'Fusion reset offset')
    report=dict(gra_state_exact=True,gra_class=type(fused.model[18]).__name__,
                unfused_parameters=sum(p.numel() for p in model.parameters()),
                fused_parameters=sum(p.numel() for p in fused.parameters()),
                default_tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32,cudnn=torch.backends.cudnn.allow_tf32))
    for mode in ('default','strict'):
        with (strict_fp32() if mode=='strict' else nullcontext()),torch.no_grad():
            _,a=capture(model,image);_,b=capture(fused,image)
            _,replay=capture(fused,image,fixed_ids=a['candidate_indices'])
        pre={k:comparison(a[k],b[k]) for k in PRE_KEYS if k not in ('anchors','valid_mask')}
        fixed={k:comparison(a[k],replay[k]) for k in ('boxes','scores','enc_boxes','enc_scores')}
        native={k:comparison(a[k],b[k]) for k in ('boxes','scores')}
        ids_equal=torch.equal(a['candidate_indices'],b['candidate_indices'])
        passed=all(x['status']=='PASSED' for x in list(pre.values())+list(fixed.values()))
        native_pass=all(x['status']=='PASSED' for x in native.values())
        native_finite=all(x['finite'] for x in native.values())
        report[mode]=dict(status='PASSED' if passed and native_pass else ('PRECISION_NOTE' if passed and native_finite and not ids_equal else 'FAILED'),
                          continuous=pre,fixed_candidate_replay=fixed,native_output=native,candidate_indices_equal=ids_equal,
                          selection=selection_report(a,b),native_output_allclose=native_pass)
    # Same-input GRA operator itself must survive fusion exactly even with nonzero learned offsets.
    taps={}
    handle=model.model[18].register_forward_pre_hook(lambda m,a:taps.update(inputs=[v.detach() for v in a[0]]))
    try:
        with torch.no_grad():model(image)
    finally:handle.remove()
    with torch.no_grad():report['same_input_gra']=comparison(model.model[18](taps['inputs']),fused.model[18](taps['inputs']),0,0)
    require(report['strict']['status']!='FAILED' and report['same_input_gra']['status']=='PASSED','Strict continuous fusion failed')
    require(all(v['finite'] for v in report['default']['native_output'].values()),'Default native fused output nonfinite')
    if image.device.type=='cuda':
        h=deepcopy(fused).half()
        with torch.no_grad():y=h(image.half())[0]
        report['fused_half']=dict(status='PASSED' if torch.isfinite(y).all() else 'FAILED',shape=list(y.shape),dtype=str(y.dtype))
        require(report['fused_half']['status']=='PASSED','Fused CUDA half nonfinite')
    return report


def detection_updates(initial,batch,device,amp,folder):
    folder.mkdir(parents=True,exist_ok=False)
    model=deepcopy(initial).to(device).train();model.nc=1
    batch={k:v.to(device) for k,v in batch.items()};image=batch['img'][:1]
    opt,groups=make_optimizer(model);scaler=scaler_for(amp);ema=ModelEMA(model)
    rows=[];updates=0;start=time.perf_counter()
    for i in range(16 if amp else 2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,enabled=amp):
            preds=model.predict(batch['img'],batch=targets(batch));loss=model.loss(batch,preds=preds)[0]
        require(bool(torch.isfinite(loss)),'Nonfinite forward detection loss')
        scale=float(scaler.get_scale());scaler.scale(loss).backward();scaler.unscale_(opt)
        finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        grads={n:float(p.grad.float().norm()) if p.grad is not None and torch.isfinite(p.grad).all() else None
               for n,p in model.named_parameters() if n.startswith('model.18.')}
        require(amp or finite,'FP32 gradient nonfinite')
        if finite:torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        scaler.step(opt);scaler.update();effective=finite
        if effective:updates+=1;ema.update(model)
        rows.append(dict(batch=i,loss=float(loss.detach()),GT=len(batch['cls']),dn_split=preds[-1]['dn_num_split'],
                         scaler_before=scale,scaler_after=float(scaler.get_scale()),gradients_finite=finite,effective_update=effective,gra_gradients=grads))
        del preds,loss
        if updates>=2:break
    require(updates>=2,'Bounded smoke exhausted without two effective updates')
    model.eval();stats=gra_stats(model,image)
    require(stats['offset_weight_nonzero']>0 and 0<stats['offset_abs_max']<=.25 and stats['finite'],'GRA did not learn finite nonzero offsets')
    report=dict(status='PASSED',device=device,amp=amp,batch=len(batch['img']),shape=list(batch['img'].shape),
                scope='disposable small real-image detection-loss AdamW updates; not formal B16/640 capacity',
                optimizer=groups,steps=rows,effective_updates=updates,seconds=time.perf_counter()-start,learned=stats)
    report['lifecycle']=lifecycle(model,opt,scaler,ema,image,folder)
    if not amp:report['fusion']=learned_fusion(model,image)
    else:
        with torch.no_grad(),torch.autocast('cuda'):out=model(image)[0]
        report['native_amp_forward']=dict(finite=bool(torch.isfinite(out).all()),dtype=str(out.dtype),shape=list(out.shape))
        report['fusion']=learned_fusion(model.float(),image.float())
    del model,opt,ema;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda','all'],default='all')
    args=parser.parse_args();require(not args.output.exists(),'Preserve existing check output')
    args.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(42)
    report=dict(status='RUNNING',formal_training='NOT_STARTED',final_test='NOT_RUN',runtime=runtime(),source_sha256=sha256(args.source),variants={},
                server_capacity=dict(status='PENDING',reason='Requires actual server B16/640 online augmentation native AMP'),
                torch21=dict(status='PENDING',reason='Run this same CLI in existing server Torch 2.1 environment'))
    def persist():write_json(args.output/'checks.json',report)
    persist()
    try:
        from check_gra_module import run_checks
        report['module_cpu']=run_checks('cpu')
        cuda=torch.cuda.is_available() and args.device in ('cuda','all')
        report['module_cuda']=run_checks('cuda') if cuda else dict(status='PENDING',reason='CUDA not available/requested')
        batch=None
        if args.dataset and args.dataset.is_dir():
            report['data']=dataset_inventory(args.dataset);batch,report['samples']=real_batch(args.dataset,size=160,count=2)
        else:report['data']=dict(status='PENDING',reason='Real data unavailable; no invented detection checks')
        for variant in VARIANTS:
            print('CHECK '+variant,flush=True)
            parent,target,init=controlled_models(args.source,variant);parent.eval();target.eval();verify_model(target,variant,zero=True)
            row=report['variants'][variant]=dict(initialization=init)
            x=torch.rand(1,3,640,640)
            row['wiring_eval'],got=wiring(target,x)
            with torch.no_grad():_,reference=capture(parent,x)
            row['zero_common']={k:comparison(reference[k],got[k],0,0) for k in PRE_KEYS if k not in ('anchors','valid_mask')}
            require(all(v['status']=='PASSED' for v in row['zero_common'].values()),'Initial common feature mismatch')
            require(row['wiring_eval']['nearest_residual']['status']=='PASSED','Initial layer18 differs from cat(U,L)')
            target.train()
            train_batch=batch if batch is not None else dict(img=torch.rand(2,3,160,160),cls=torch.zeros(2,1),
                bboxes=torch.tensor([[.4,.5,.2,.3],[.6,.4,.2,.1]]),batch_idx=torch.tensor([0,1]))
            row['wiring_train'],_=wiring(target,train_batch['img'],targets(train_batch));target.eval()
            # Restore untouched controlled state after train-mode BN wiring probe.
            _,target,_=controlled_models(args.source,variant);target.eval()
            counts=dict(parent_unfused=sum(p.numel() for p in parent.parameters()),candidate_unfused=sum(p.numel() for p in target.parameters()),
                        parent_fused=sum(p.numel() for p in deepcopy(parent).fuse(verbose=False).parameters()),
                        candidate_fused=sum(p.numel() for p in deepcopy(target).fuse(verbose=False).parameters()))
            require(counts['candidate_unfused']-counts['parent_unfused']==8744 and counts['candidate_fused']-counts['parent_fused']==8744,'Parameter delta drift')
            row['cost']=dict(parameters=counts,nc=1,imgsz=640,new_conv_MACs=36249600,new_conv_FLOPs_2_per_MAC=72499200,
                status='PARTIAL',excluded='Two FP32 grid_sample calls (3,276,800 scalar sampled outputs), tanh, SiLU, interpolation, grid arithmetic, residual arithmetic; not zero cost')
            del parent,got,reference,x;gc.collect();persist()
            if batch is not None:
                row['cpu_fp32']=detection_updates(target,batch,'cpu',False,args.output/variant/'cpu_fp32');persist()
                if cuda:
                    row['cuda_fp32']=detection_updates(target,batch,'cuda',False,args.output/variant/'cuda_fp32');persist()
                    row['cuda_amp']=detection_updates(target,batch,'cuda',True,args.output/variant/'cuda_amp');persist()
            del target;gc.collect()
        report['status']='PASSED_LOCAL_CHECKS_SERVER_PENDING'
        if str(torch.__version__).startswith('2.1.'):
            report['torch21']=dict(status='PASSED',reason='All requested checks actually executed in this Torch 2.1 process')
    except BaseException as error:
        report['status']='FAILED';report['error']=repr(error);report['traceback']=traceback.format_exc();raise
    finally:
        from train_gra import source_manifest
        report['source_manifest']=source_manifest()
        report['tested_hashes']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted(set(list((ROOT/'tools').glob('*gra*.py'))+list((ROOT/'ultralytics-main/ultralytics').rglob('*.py'))+
                                list((ROOT/'ultralytics-main/ultralytics/cfg/models/rt-detr').glob('*gra*.yaml'))))}
        persist()


if __name__=='__main__':main()
