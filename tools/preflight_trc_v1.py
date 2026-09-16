"""Bounded TRC engineering checks on copies; --server adds mandatory real B16/640 AMP gate."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from init_trc_v1 import (ROOT, MODEL_DIR, VARIANTS, DEFAULT_VARIANT, NEW_KEYS, SOURCE_SHA256, build,
                        controlled_models, build_training_model, verify_model, require, sha256, write_json, runtime)
from trc_v1_common import fingerprint
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA, init_seeds, TORCH_2_4
from c19_lif_v1_probe import capture, targets
from c19_lif_v1_data import real_batch, dataset_inventory


@contextmanager
def strict_fp32():
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try: yield dict(matmul_tf32=False,cudnn_tf32=False,restore=list(previous))
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def optimizer(model):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args=SimpleNamespace(warmup_bias_lr=.1,lr0=.0005,weight_decay=.0001)
    opt=trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    require(len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad},'Optimizer omission/duplication')
    groups={n:dict(occurrences=ids.count(id(p)),group=next(g['param_group'] for g in opt.param_groups if any(q is p for q in g['params'])))
            for n,p in model.named_parameters() if n in NEW_KEYS}
    require(len(groups)==3 and all(r['occurrences']==1 for r in groups.values()),'TRC optimizer registration failed')
    return opt,groups


def tensor_compare(a,b,atol=2e-5,rtol=2e-4):
    require(a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(),'Nonfinite or shape mismatch')
    error=float((a-b).abs().max()) if a.numel() else 0.
    require(torch.allclose(a,b,atol=atol,rtol=rtol),f'Numerical mismatch: max_abs={error}, atol={atol}, rtol={rtol}')
    return dict(max_abs=error,atol=atol,rtol=rtol,status='PASSED')


def api_rebuild(initialized,expected):
    observed={}
    class Done(Exception): pass
    class AuditTrainer(RTDETRTrainer):
        def __init__(self,overrides,_callbacks):
            self.args=SimpleNamespace(**{k:v for k,v in overrides.items() if k!='session'})
            self.data=dict(nc=1,channels=3); torch.manual_seed(42)
        def train(self):
            require(all(torch.equal(v,self.model.state_dict()[k]) for k,v in expected.state_dict().items()),'Actual train API rebuild differs')
            observed.update(status='PASSED',native_get_model=True,states_exact=len(expected.state_dict()),optimizer_steps=0)
            raise Done()
    with patch('ultralytics.engine.model.checks.check_pip_update_available'):
        try: RTDETR(str(initialized)).train(trainer=AuditTrainer,model=str(initialized),data=str(ROOT/'configs/crack.yaml'),seed=42)
        except Done: pass
    require(observed,'Native train API was not reached')
    return observed


def serialization_resume(model,variant):
    report={}
    for active in (False,True):
        copy=deepcopy(model).eval()
        if active:
            with torch.no_grad():
                copy.model[9].trc.coefficient.weight.fill_(.001)
                copy.model[9].trc.coefficient.bias.copy_(torch.linspace(-.1,.1,8))
        opt,groups=optimizer(copy)
        # Populate optimizer moments only on a disposable copy to exercise native restoration.
        for name,p in copy.named_parameters():
            if name in NEW_KEYS: p.grad=torch.full_like(p,.001)
        if active: opt.step()
        ema=ModelEMA(copy); ema.update(copy)
        payload=dict(model=copy,ema=ema.ema,optimizer=opt.state_dict(),updates=ema.updates,
                     scaler=None,epoch=0,best_fitness=.1)
        buffer=io.BytesIO();torch.save(payload,buffer);buffer.seek(0)
        saved=torch_load(buffer,map_location='cpu')
        restored,rebuild=build_training_model(saved['model'].yaml,saved['model'],dict(nc=1,channels=3),variant)
        require(all(torch.equal(v,restored.state_dict()[k]) for k,v in copy.state_dict().items()),'Checkpoint reconstruction lost state')
        trainer=RTDETRTrainer.__new__(RTDETRTrainer)
        trainer.model=restored;trainer.optimizer,_=optimizer(restored);trainer.ema=ModelEMA(restored)
        trainer.resume=True;trainer.args=SimpleNamespace(model='disposable',close_mosaic=10)
        trainer.epochs=200;trainer.start_epoch=0;trainer.batch_size=16
        trainer.resume_training(saved)
        require(trainer.start_epoch==1 and trainer.ema.updates==1,'Native resume epoch/EMA update count failed')
        require(all(torch.equal(v,trainer.ema.ema.state_dict()[k]) for k,v in ema.ema.state_dict().items()),'EMA restore lost tensors')
        for k,row in opt.state_dict()['state'].items():
            for name,v in row.items():
                if isinstance(v,torch.Tensor):require(torch.equal(v,trainer.optimizer.state_dict()['state'][k][name]),'Optimizer moments changed')
        require(all(torch.equal(v,restored.state_dict()[k]) for k,v in copy.state_dict().items()),'Resume changed learned parameters')
        report['nonzero' if active else 'zero']=dict(status='PASSED',checkpoint_exact=True,ema_exact=True,native_resume=True,optimizer_exact=True)
    return report


def single_fusion(unfused,fused,image,device,out):
    """Strict FP32 original-C2 fusion check with explicit candidate identity alignment."""
    from c19_lif_v1_diagnostic import PRE_KEYS,compare_records,selection_report,align_to_ids
    report=dict(status='FAILED',atol=2e-5,rtol=2e-4)
    try:
        with strict_fp32() as flags,torch.no_grad():
            _,a=capture(unfused,image);_,b=capture(fused,image)
            report['tf32']=flags
            report['pre_selection']=compare_records(a,b,2e-5,2e-4,'single.fusion.pre',device,'fp32',keys=PRE_KEYS)
            report['selection']=selection_report(a,b)
            report['natural_unaligned']=compare_records(a,b,2e-5,2e-4,'single.fusion.natural',device,'fp32',collect=True)
            require(report['selection']['kind'] in {'IDENTICAL','PERMUTATION'},'Single-module fusion changed candidate set; diagnostic retained')
            report['all_candidates_aligned']=compare_records(a,align_to_ids(a,b),2e-5,2e-4,'single.fusion.aligned',device,'fp32')
            _,replay=capture(fused,image,fixed_ids=a['candidate_indices'])
            report['fixed_query_replay']=compare_records(a,replay,2e-5,2e-4,'single.fusion.replay',device,'fp32')
            report['status']='PASSED' if report['selection']['kind']=='IDENTICAL' else 'PASS_WITH_CANDIDATE_PERMUTATION'
    finally:write_json(out/(device+'_fp32_fusion/fuse_diagnostic.json'),report)
    return dict(status='PASSED',acceptance=report['status'],report=str(out/(device+'_fp32_fusion/fuse_diagnostic.json')))


def model_numerics(parent,model,variant,device,out):
    a=deepcopy(parent).to(device).eval(); b=deepcopy(model).to(device).eval()
    x=torch.rand(1,3,640,640,device=device)
    shapes=[]
    handle=b.model[9].register_forward_pre_hook(lambda m,args:shapes.append(list(args[0].shape)))
    with strict_fp32() as tf32,torch.no_grad():
        ya=a(x)[0]; yb=b(x)[0]
    handle.remove();require(shapes==[[1,256,20,20]],'Wrong S5 AIFI input')
    report=dict(status='PASSED',shape=shapes[0],zero_full_network=tensor_compare(ya,yb),fp32=tf32)
    del a,ya,yb,x
    # A deliberately learned state on an isolated copy, including original branches.
    with torch.no_grad():
        b.model[9].trc.coefficient.weight.normal_(std=.003)
        b.model[9].trc.coefficient.bias.copy_(torch.linspace(-.1,.1,8,device=device))
    if variant==DEFAULT_VARIANT:
        from check_c19_lif_v1 import activate
        activate(b,bn=True)
    x=torch.rand(1,3,160,192,device=device)
    summaries=[]
    def summary_hook(module,inputs):
        with torch.no_grad():
            bias,stats=module.trc(inputs[0].flatten(2).permute(0,2,1),return_diagnostics=True)
            r,lam=stats['r'],stats['lambda']
            summaries.append(dict(r_min=float(r.min()),r_max=float(r.max()),r_mean=float(r.mean()),r_std=float(r.std()),
                                  lambda_mean=float(lam.mean()),lambda_positive_fraction=float((lam>0).float().mean()),
                                  lambda_negative_fraction=float((lam<0).float().mean()),bias_abs_max=float(bias.abs().max())))
    summary_handle=b.model[9].register_forward_pre_hook(summary_hook)
    with torch.no_grad():b(x)
    summary_handle.remove();report['optional_nonzero_summary']=summaries[0]
    fused=deepcopy(b).fuse(verbose=False)
    require(all(torch.equal(v,fused.state_dict()[k]) for k,v in b.state_dict().items() if k in NEW_KEYS),'Fusion changed TRC')
    if variant==DEFAULT_VARIANT:
        from c19_lif_v1_diagnostic import fusion_protocol
        from c19_lif_v1_cutoff import fusion_accepted
        with strict_fp32() as tf32:
            fusion=fusion_protocol(b,fused,x,out/(device+'_fp32_fusion'),device,'fp32')
        require(fusion_accepted(fusion,device,'fp32'),'Original strict LIF fusion gate rejected')
        report['nonzero_fusion']=dict(status='PASSED',report=str(out/(device+'_fp32_fusion/fuse_diagnostic.json')),tf32=tf32)
    else:
        report['nonzero_fusion']=single_fusion(b,fused,x,device,out)
    if device=='cuda':
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):value=b(x)[0]
        require(torch.isfinite(value).all(),'Whole-model AMP nonfinite')
        half=deepcopy(fused).half()
        with torch.no_grad(): value=half(x.half())[0]
        require(torch.isfinite(value).all(),'Whole-model fused half nonfinite')
        report['native_amp']=dict(status='PASSED',dtype='float16 autocast')
        report['fused_half']=dict(status='PASSED',dtype=str(next(half.parameters()).dtype),order='FP32 fuse then half')
    return report


def complexity(parent,model):
    import thop
    from ultralytics.nn.modules.trc_aifi import TokenRedundancyCalibration
    def count_trc(module,inputs,output):
        batch,tokens,channels=inputs[0].shape
        macs=batch*(tokens*channels*32+tokens*channels*8+tokens*tokens*32)
        module.total_ops += module.total_ops.new_tensor([macs])
    rows=[]
    for fused in (False,True):
        for name,original in (('parent',parent),('candidate',model)):
            copy=deepcopy(original).cpu().eval()
            if fused:copy.fuse(verbose=False)
            params=sum(p.numel() for p in copy.parameters())
            with strict_fp32(),torch.no_grad():
                macs,_=thop.profile(copy,inputs=(torch.zeros(1,3,640,640),),custom_ops={TokenRedundancyCalibration:count_trc},verbose=False)
            rows.append(dict(model=name,fused=fused,nc=1,imgsz=640,batch=1,parameters=params,thop_MACs=macs,thop_GFLOPs=macs*2/1e9))
    for fused in (False,True):
        p,c=[r for r in rows if r['fused']==fused]
        require(c['parameters']-p['parameters']==10248,'Profile parameter delta changed')
        require(abs(c['thop_MACs']-p['thop_MACs']-9216000)<1,'TRC function MACs missing/double counted')
    return dict(status='PASSED',rows=rows,trc_matrix_MACs=9216000,trc_matrix_GFLOPs=.018432,
        B16_FP32_bias_bytes=81920000,
        uncovered=['TRC functional LN/L2/exp/log/sum/clamp/tanh/bias/mask elementwise',
                   'native MHA QKV/attention functional operations not fully covered by THOP',
                   'native deformable grid sampling, functional activations and CBR/LIF elementwise operations'],
        scope='Partial THOP operator count, not complete measured FLOPs or latency; only added matrix delta is explicit')


def loss_probe(model,parent,batch,device,amp,nb=1):
    from check_c19_lif_v1 import native_warmup
    copy=deepcopy(model).to(device).train();copy.nc=1
    if hasattr(copy,'criterion'):del copy.criterion
    sample={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    opt,groups=optimizer(copy)
    # Native Trainer creates its scheduler before warmup; it registers initial_lr.
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda epoch:1.)
    scaler=torch.amp.GradScaler('cuda',enabled=amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=amp)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    attempts=[];updates=0
    for attempt in range(16):
        opt.zero_grad(set_to_none=True)
        warmup=native_warmup(opt,updates,nb)
        torch.manual_seed(123+updates)
        with torch.autocast(device_type=device,enabled=amp):
            prediction=copy.predict(sample['img'],batch=targets(sample))
            loss,items=copy.loss(sample,preds=prediction)
        require(torch.isfinite(loss),'Nonfinite original loss')
        before=float(scaler.get_scale())
        scaler.scale(loss).backward();scaler.unscale_(opt)
        finite=all(torch.isfinite(p.grad).all() for p in copy.parameters() if p.grad is not None)
        grad={n:float(p.grad.float().norm()) if p.grad is not None and torch.isfinite(p.grad).all() else None
              for n,p in copy.named_parameters() if n in NEW_KEYS}
        if finite:torch.nn.utils.clip_grad_norm_(copy.parameters(),10.)
        # Native GradScaler skips Inf gradients and backs off; never force scale=1.
        scaler.step(opt);scaler.update()
        attempts.append(dict(attempt=attempt,optimizer_update=updates,loss=float(loss),scale_before=before,
                             scale_after=float(scaler.get_scale()),unscaled_gradients_finite=finite,grad_norms=grad,
                             dn_split=prediction[-1]['dn_num_split'] if prediction[-1] else None,warmup=warmup))
        if not finite and not amp:raise RuntimeError('FP32 backward nonfinite')
        if finite:
            require(grad['model.9.trc.coefficient.weight'] is not None and grad['model.9.trc.coefficient.weight']>0,'Coefficient gradient missing')
            updates+=1
        del prediction,loss
        if updates>=2:break
    if updates<2:
        # Same batch, common state, RNG and scaler value; no interpretation of scaled Inf alone.
        contrast=deepcopy(parent).to(device).train();contrast.nc=1
        contrast.load_state_dict({k:copy.state_dict()[k] for k in contrast.state_dict()},strict=True)
        if hasattr(contrast,'criterion'):del contrast.criterion
        popt,_=optimizer_for_parent(contrast)
        pscaler=torch.amp.GradScaler('cuda',enabled=amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=amp)
        if amp:pscaler.load_state_dict(scaler.state_dict())
        torch.manual_seed(123+updates)
        with torch.autocast(device_type=device,enabled=amp):ploss=contrast.loss(sample)[0]
        pscale=float(pscaler.get_scale());pscaler.scale(ploss).backward();pscaler.unscale_(popt)
        pfinite=all(torch.isfinite(p.grad).all() for p in contrast.parameters() if p.grad is not None)
        if pfinite:torch.nn.utils.clip_grad_norm_(contrast.parameters(),10.)
        pscaler.step(popt);pscaler.update()
        error=RuntimeError('Persistent native AMP overflow; parent same-batch comparison attached')
        error.diagnostics=dict(attempts=attempts,parent_loss=float(ploss),parent_scale_before=pscale,
                               parent_scale_after=float(pscaler.get_scale()),seed=123+updates,
                               parent_unscaled_gradients_finite=pfinite)
        raise error
    return dict(status='PASSED',batch=len(sample['img']),imgsz=list(sample['img'].shape[-2:]),amp=amp,device=device,
                native_initial_scale=attempts[0]['scale_before'],attempts=attempts,disposable_optimizer_steps=updates,
                gt_groups=targets(sample)['gt_groups'],optimizer=groups,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None)


def optimizer_for_parent(model):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args=SimpleNamespace(warmup_bias_lr=.1)
    return trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001),None


def server_capacity(model,parent,data,out,variant):
    from train_trc_v1 import recipe,ARCHIVE,ensure_amp_resources,verify_data_config
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.checks import check_amp
    require(torch.cuda.is_available(),'Server CUDA required')
    require(os.environ.get('CONDA_DEFAULT_ENV')=='rtdetr','Activate existing rtdetr environment')
    verify_data_config(data)
    ensure_amp_resources(Path(os.environ.get('TRC_V1_MAIN','/root/autodl-tmp/projects/Crack_RTDETR')),out/'amp_resources.json')
    probe=deepcopy(model).cuda()
    require(check_amp(probe),'Native check_amp rejected: AMP recipe may not silently change')
    del probe;torch.cuda.empty_cache()
    args,_=recipe(ARCHIVE,variant,Path('diagnostic_only.pt'),data=data)
    trainer=RTDETRTrainer.__new__(RTDETRTrainer);trainer.args=get_cfg(overrides=args)
    cfg=YAML.load(data);trainer.data={**cfg,'nc':1,'channels':3};trainer.device=torch.device('cuda')
    init_seeds(42,deterministic=True)
    loader=trainer.get_dataloader(str(Path(cfg['path'])/'images/train'),batch_size=16,rank=-1,mode='train')
    require(loader.dataset.augment,'Online augmentation is disabled')
    batch=trainer.preprocess_batch(next(iter(loader)))
    require(tuple(batch['img'].shape)==(16,3,640,640),'Server batch must remain B16/640')
    result=loss_probe(model,parent,batch,'cuda',True,len(loader))
    result.update(online_augmentation=True,split='train',workers=trainer.args.workers,
                  recipe={k:getattr(trainer.args,k) for k in ('batch','imgsz','amp','mosaic','mixup','hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','perspective','flipud','fliplr')})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','initialized','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS,default=DEFAULT_VARIANT)
    p.add_argument('--data',type=Path);p.add_argument('--server',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    require(not args.output.exists(),'Preflight output already exists; preserve evidence')
    args.output.mkdir(parents=True)
    initial_hash=sha256(args.initialized)
    tested_fingerprint=fingerprint(args.variant,args.source,args.initialized,args.data)
    report=dict(status='RUNNING',variant=args.variant,runtime=runtime(),formal_optimizer_steps=0,
                formal_training='NOT_STARTED',test='NOT_RUN',server_capacity=dict(status='PENDING',reason='Requires explicit --server real B16/640/native AMP'))
    def save():write_json(args.output/'preflight.json',report)
    save()
    try:
        import importlib.util
        import unittest
        spec=importlib.util.spec_from_file_location('trc_unit_checks',ROOT/'ultralytics-main/tests/test_trc_aifi.py')
        tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
        with (args.output/'module_tests.log').open('x',encoding='utf-8') as log:
            unit=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        require(unit.wasSuccessful(),'TRC module checks failed; see module_tests.log')
        report['module_checks']=dict(status='PASSED' if not unit.skipped else 'PENDING',run=unit.testsRun,
                                    skipped=[dict(test=str(test),reason=reason) for test,reason in unit.skipped])
        require(sha256(args.source)==SOURCE_SHA256,'Wrong fixed source')
        parent,expected,init_report=controlled_models(args.source,args.variant)
        actual=torch_load(args.initialized,map_location='cpu')
        require(actual['epoch']==-1 and all(actual.get(k) is None for k in ('optimizer','ema','scaler','updates')),'Formal init has trained state')
        loaded=actual['model'].float()
        require(set(loaded.state_dict())==set(expected.state_dict()) and all(torch.equal(v,loaded.state_dict()[k]) for k,v in expected.state_dict().items()),'Initialized file differs from controlled state')
        report['initialization']=init_report
        other=build('trc_v1' if args.variant==DEFAULT_VARIANT else DEFAULT_VARIANT)
        require(all(torch.equal(loaded.state_dict()[k],other.state_dict()[k]) for k in NEW_KEYS),'Both TRC initial states differ')
        report['both_variant_trc_initial_exact']=True
        torch.manual_seed(42)
        model,rebuild=build_training_model(loaded.yaml,loaded,dict(nc=1,channels=3),args.variant)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            from init_c19_lif_v1 import native_rebuild
            parent=native_rebuild(str(MODEL_DIR/VARIANTS[args.variant][1]),parent,1,3);parent.nc=1
        report['native_nc1']=rebuild;report['native_train_api']=api_rebuild(args.initialized,model)
        report['topology']=verify_model(model,args.variant,zero=True)
        opt,report['optimizer']=optimizer(model);del opt,other,expected,actual,loaded
        report['serialization_resume']=serialization_resume(model,args.variant);save()
        report['cpu']=model_numerics(parent,model,args.variant,'cpu',args.output);save()
        report['complexity']=complexity(parent,model);save()
        if torch.cuda.is_available():
            report['cuda']=model_numerics(parent,model,args.variant,'cuda',args.output)
        else:report['cuda']=dict(status='PENDING',reason='CUDA unavailable')
        if args.data:
            cfg=YAML.load(args.data);dataset=Path(cfg['path'])
            report['dataset']=dataset_inventory(dataset)
            batch,records=real_batch(dataset,160,2)
            report['bounded_train_samples']=records
            report['cpu_loss']=loss_probe(model,parent,batch,'cpu',False)
            if torch.cuda.is_available():
                report['cuda_fp32_loss']=loss_probe(model,parent,batch,'cuda',False)
                report['cuda_amp_loss']=loss_probe(model,parent,batch,'cuda',True)
        else:report['real_data']=dict(status='PENDING',reason='--data not supplied')
        save();gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        if args.server:
            require(args.data is not None,'--server requires --data')
            report['server_capacity']=server_capacity(model,parent,args.data,args.output,args.variant)
        require(initial_hash==sha256(args.initialized),'Formal initialization changed during diagnostic')
        report['fingerprint']=fingerprint(args.variant,args.source,args.initialized,args.data)
        require(report['fingerprint']==tested_fingerprint,'Source/config/weights changed while preflight was running')
        report['status']='PASSED' if args.server and report['cuda']['status']=='PASSED' else 'PENDING'
        report['local_checks']='PASSED'
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),diagnostics=getattr(error,'diagnostics',None))
        raise
    finally:save()
    print(json.dumps(dict(status=report['status'],report=str(args.output/'preflight.json')),ensure_ascii=False))


if __name__=='__main__':main()
