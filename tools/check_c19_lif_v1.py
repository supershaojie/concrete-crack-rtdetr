"""Finite engineering checks. Never runs epochs or a complete val/test split."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from init_c19_lif_v1 import (ROOT,MODEL_DIR,CONFIGS,BASE_COMMIT,C19_COMMIT,C2_COMMIT,build,controlled_models,
                            build_training_model,verify_model,is_added,require,write_json,sha256,runtime,topology)
from check_lif_down import module_checks,compare
from c19_lif_v1_probe import capture,targets
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load


def tensor_hash(v):
    return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def geometry_gradient():
    from ultralytics.nn.modules import CrackBoundaryRefinement
    module=CrackBoundaryRefinement(16,32)
    with torch.no_grad():module.offset_out.bias.fill_(.3)
    boxes=torch.rand(2,7,4,requires_grad=True)
    grid=module.sampling_grid(boxes);require(not grid.requires_grad,'Historical sampling detach lost')
    image=torch.rand(2,16,9,13,requires_grad=True);query=torch.rand(2,7,32,requires_grad=True)
    result=module(image,query,boxes);result.sum().backward()
    expected=torch.ones_like(boxes);expected[...,2:]+=module.rho*torch.tensor(.3).tanh()
    return dict(sampling_grid_detached=True,box_width_height_path=compare(boxes.grad,expected),
                rule='constant original offset gives d(sum(output))/d(w,h)=1+rho*tanh(0.3)')


from c19_lif_v1_data import dataset_inventory,real_batch


from c19_lif_v1_diagnostic import compare_records, fusion_protocol, atomic_json, PASS, DiagnosticError


def activate(model,lif=True,cbr=True,bn=False):
    t=topology()
    with torch.no_grad(),torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        module=model.model[t['p3_to_p4']['downsample']]
        if hasattr(module,'O_proj'):
            if lif:module.O_proj.weight.normal_(std=.01)
            else:module.O_proj.weight.zero_()
            if bn:
                module.bn.running_mean.copy_(torch.linspace(-.3,.4,256,device=module.bn.weight.device))
                module.bn.running_var.copy_(torch.linspace(.4,1.8,256,device=module.bn.weight.device))
                module.bn.weight.copy_(torch.linspace(.6,1.4,256,device=module.bn.weight.device))
                module.bn.bias.copy_(torch.linspace(-.2,.3,256,device=module.bn.weight.device))
        if hasattr(model.model[-1],'cbr'):
            out=model.model[-1].cbr.offset_out
            if cbr:out.weight.normal_(std=.02);out.bias.fill_(.15)
            else:out.weight.zero_();out.bias.zero_()


def degenerations(target,device):
    report={}
    for kind,lif,cbr in [('C2',False,False),('C19',False,True),('LIF',True,False)]:
        pair=deepcopy(target).eval();activate(pair,lif,cbr)
        parent=build(kind,nc=1).eval();parent.load_state_dict({k:pair.state_dict()[k] for k in parent.state_dict()},strict=True)
        pair.to(device);parent.to(device)
        for shape in ((640,640),(160,192)):
            x=torch.rand(1,3,*shape,device=device)
            with torch.no_grad():_,a=capture(parent,x);_,b=capture(pair,x)
            report[f'{kind}_{shape[0]}x{shape[1]}']=compare_records(a,b)
        del pair,parent,x;gc.collect()
    return report


def historical_regressions(target,batch,folder):
    report={}
    for kind,commit in [('C2',C2_COMMIT),('LIF',BASE_COMMIT),('C19',C19_COMMIT)]:
        hist=folder/'history'/kind;hist.mkdir(parents=True)
        archive=hist/'source.zip'
        subprocess.run(['git','archive','--format=zip','--output='+str(archive),commit,'ultralytics-main/ultralytics'],cwd=ROOT,check=True)
        with zipfile.ZipFile(archive) as z:z.extractall(hist)
        model=build(kind,nc=1).eval();model.load_state_dict({k:target.state_dict()[k] for k in model.state_dict()},strict=True)
        activate(model)
        x=torch.rand(1,3,160,192,generator=torch.Generator().manual_seed(987))
        payload=hist/'payload.pt'; result=hist/'result.pt'
        torch.save(dict(yaml=model.yaml,state=model.state_dict(),image=x,batch=batch),payload)
        command=[sys.executable,str(ROOT/'tools/c19_lif_v1_probe.py'),'--root',str(hist),'--payload',str(payload),'--output',str(result)]
        with (hist/'regression.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=hist,stdout=log,stderr=subprocess.STDOUT,check=True)
        old=torch_load(result,map_location='cpu')
        fresh=build(kind);fresh_hash={k:tensor_hash(v) for k,v in fresh.state_dict().items()}
        require(old['fresh']==fresh_hash,kind+' historical constructor changed')
        with torch.no_grad():_,evaluation=capture(model,x)
        model.nc=1;model.train();torch.manual_seed(123)
        pred,training=capture(model,batch['img'],targets(batch));loss=model.loss(batch,preds=pred)[0];loss.backward()
        gradrows={k:compare(v,dict(model.named_parameters())[k].grad) for k,v in old['gradients'].items()}
        report[kind]=dict(commit=commit,source=old['source'],constructor_states_exact=len(fresh_hash),
                          eval=compare_records(old['eval'],evaluation),train=compare_records(old['train'],training),
                          gradient_tensors=len(gradrows),gradient_max_abs=max(v['max_abs_error'] for v in gradrows.values()),
                          loss_old=old['loss'],loss_current=float(loss),isolated_process=True)
        del model,fresh,old,pred,loss;gc.collect()
        print(kind+' isolated historical regression passed',flush=True)
    return report


def actual_train_api(initialized,target):
    expected={k:tensor_hash(v) for k,v in target.state_dict().items()};observed={}
    class Stopped(Exception):pass
    class AuditTrainer(RTDETRTrainer):
        # Stop before loading data or training; inherit the actual get_model.
        def __init__(self,overrides,_callbacks):
            self.args=SimpleNamespace(**{k:v for k,v in overrides.items() if k!='session'})
            self.data=dict(nc=1,channels=3);torch.manual_seed(42)
        def train(self):
            require({k:tensor_hash(v) for k,v in self.model.state_dict().items()}==expected,'Actual train API rebuild differs')
            observed.update(status='PASSED',states_exact=len(expected),inherited_native_get_model=True,optimizer_steps=0)
            raise Stopped()
    args=YAML.load(ROOT/'docs/c19_lif_v1/c2_args.yaml');args.update(model=str(initialized),name='bounded_api_probe',save_dir=str(ROOT/'outputs/bounded_api_probe'))
    with patch('ultralytics.engine.model.checks.check_pip_update_available'):
        try:RTDETR(str(initialized)).train(trainer=AuditTrainer,**args)
        except Stopped:pass
    require(observed,'Actual train API not reached')
    return observed


def optimizer(model):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args=SimpleNamespace(warmup_bias_lr=.1,lr0=.0005,weight_decay=.0001)
    opt=trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    require(len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad},'Optimizer missing/duplicate parameters')
    rows=[]
    for n,p in model.named_parameters():
        group=next(g for g in opt.param_groups if any(p is v for v in g['params']))
        if is_added(n):
            expected='bias' if n.endswith('.bias') else 'weight'
            require(group['param_group']==expected and group['weight_decay']==(0. if expected=='bias' else .0001),'New parameter grouping drift')
            rows.append(dict(name=n,numel=p.numel(),group=expected,lr=group['lr'],weight_decay=group['weight_decay'],norm=False))
    for group in opt.param_groups:group['initial_lr']=group['lr']
    return opt,dict(total_parameter_tensors=len(ids),new=rows,no_missing_or_duplicate=True)


def native_warmup(opt,ni,nb):
    # Execute the warmup block taken from the unchanged native Trainer method.
    source=inspect.getsource(RTDETRTrainer._do_train)
    start=source.index('                if ni <= nw:');end=source.index('                # Forward',start)
    import textwrap
    trainer=SimpleNamespace(args=SimpleNamespace(nbs=64,warmup_bias_lr=.1,warmup_momentum=.8,momentum=.937),
                            batch_size=16,optimizer=opt,lf=lambda epoch:1.,accumulate=4)
    scope=dict(self=trainer,ni=ni,nw=max(round(5*nb),100),epoch=0,np=np)
    exec(textwrap.dedent(source[start:end]),scope)
    return dict(batch=ni,nw=scope['nw'],accumulate=trainer.accumulate,
                groups=[dict(kind=g['param_group'],lr=float(g['lr']),weight_decay=g['weight_decay']) for g in opt.param_groups],
                implementation_sha256=hashlib.sha256(source[start:end].encode()).hexdigest())


def loss_checks(target,batch,device,folder,nb,amp=False):
    model=deepcopy(target).to(device).train();model.nc=1;activate(model)
    batch={k:v.to(device) for k,v in batch.items()};opt,groups=optimizer(model);rows=[]
    scaler=torch.cuda.amp.GradScaler(enabled=amp,init_scale=128.)
    for i in range(3):
        opt.zero_grad(set_to_none=True);warmup=native_warmup(opt,i,nb);torch.manual_seed(123+i)
        with torch.autocast(device_type=device,enabled=amp):
            prediction,taps=capture(model,batch['img'],targets(batch));loss=model.loss(batch,preds=prediction)[0]
        scaler.scale(loss).backward();scaler.unscale_(opt)
        require(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Nonfinite native loss/backward')
        grads={n:float(p.grad.float().norm()) if p.grad is not None else None for n,p in model.named_parameters() if is_added(n)}
        require(all(v is not None and v>0 for v in grads.values()),'Activated branch lost gradients')
        require(model.model[0].conv.weight.grad is not None,'Shared backbone gradient missing')
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        scaler.step(opt);scaler.update()
        rows.append(dict(loss=float(loss),dn_split=taps['dn_split'],total_queries=taps['boxes'].shape[2],new_grad_norms=grads,warmup=warmup,
                         parameter_norms={n:float(p.detach().float().norm()) for n,p in model.named_parameters() if is_added(n)}))
        del prediction,loss
    dest=folder/f'smoke_{device}_{amp}.pt';torch.save(dict(model=model.cpu(),optimizer=opt.state_dict()),dest)
    reloaded=torch_load(dest,map_location='cpu');opt2,_=optimizer(reloaded['model']);opt2.load_state_dict(reloaded['optimizer'])
    require(all(torch.equal(v,reloaded['model'].state_dict()[k]) for k,v in model.state_dict().items()),'Smoke model reload mismatch')
    for k,group in opt.state_dict()['state'].items():
        for n,v in group.items():
            if isinstance(v,torch.Tensor):require(torch.equal(v.cpu(),opt2.state_dict()['state'][k][n].cpu()),'Optimizer reload mismatch')
    # Different GT group sizes use the real denoising generator; no assumed total Q.
    model.eval();head=model.model[-1];head.train();dynamic=[]
    features=[torch.rand(2,256,h,w) for h,w in ((20,24),(10,12),(5,6))]
    for group in ([1,3],[2,7],[0,0]):
        count=sum(group);batch2=dict(cls=torch.zeros(count,dtype=torch.long),bboxes=torch.rand(count,4),
                    batch_idx=torch.tensor([0]*group[0]+[1]*group[1],dtype=torch.long),gt_groups=group)
        with torch.no_grad():out=head([v.clone() for v in features],batch2)
        dynamic.append(dict(gt_groups=group,total_queries=out[0].shape[2],dn_split=out[-1]['dn_num_split'] if out[-1] else None))
    require(len({r['total_queries'] for r in dynamic})==3,'Dynamic DN not exercised')
    return dict(device=device,AMP=amp,smoke_scaler=128,formal_scaler='native unchanged',batch=len(batch['img']),imgsz=batch['img'].shape[-1],
                steps=rows,optimizer=groups,dynamic_DN=dynamic,model_optimizer_reload_exact=True,updated_checkpoint='DISCARDED; never formal init')


def benchmark(target,device):
    model=deepcopy(target).eval().to(device);x=torch.rand(1,3,640,640,device=device)
    with torch.no_grad():
        model(x)
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        for _ in range(3):model(x)
        if device=='cuda':torch.cuda.synchronize()
    return dict(scope='unfused synthetic inference only',device=device,batch=1,imgsz=640,warmup=1,iterations=3,
                ms_per_image=(time.perf_counter()-started)*1000/3,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)



def fuse_checks(target,device,folder,report=None,persist=lambda:None):
    from c19_lif_v1_diagnostic import rng_state
    if report is None:report={}
    model=deepcopy(target).eval();activate(model,bn=True);model.to(device)
    x=torch.rand(1,3,160,192,device=device)
    try:
        fused=deepcopy(model).fuse(verbose=False)
        lif_index=topology()['p3_to_p4']['downsample'];lif=fused.model[lif_index]
        require(hasattr(lif,'bn') and torch.equal(lif.O_proj.weight,model.model[lif_index].O_proj.weight),'LIF fuse lost residual/BN')
        require(sum(isinstance(m,torch.nn.BatchNorm2d) for m in fused.modules()) < sum(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()),'Global fuse disabled')
    except BaseException as error:
        evidence=folder/(device+'_fuse_setup_failure.pt')
        torch.save(dict(image=x.cpu(),state=model.cpu().state_dict(),rng=rng_state()),evidence)
        report['setup_failure']=dict(status='FAILED_INCOMPLETE_DIAGNOSTIC',stage=device+'.fp32.fuse.setup',
            error=repr(error),evidence=str(evidence),sha256=sha256(evidence));persist()
        raise
    report.update(nontrivial_bn=True,both_branches_active=True,lif_bn_retained=True)
    def parent_factory():
        parent=build('C2',nc=1).eval()
        parent.load_state_dict({k:model.state_dict()[k].float().cpu() for k in parent.state_dict()},strict=True)
        return parent
    def protocol(label,left,right,image,precision):
        destination=folder/(device+'_'+label+'_fusion')
        report[label]=dict(status='RUNNING',diagnostic=str(destination/'fuse_diagnostic.json'));persist()
        try:
            result=fusion_protocol(left,right,image,destination,device,precision,parent_factory)
        finally:
            path=destination/'fuse_diagnostic.json'
            if path.is_file():report[label]=json.loads(path.read_text(encoding='utf-8'))
            persist()
        return result
    protocol('FP32',model,fused,x,'fp32')
    with torch.no_grad():_,before=capture(model,x);_,after=capture(fused,x)
    fused.fuse(verbose=False)
    with torch.no_grad():_,again=capture(fused,x)
    report['repeat_fuse']=compare_records(after,again)
    checkpoint=folder/f'nonzero_{device}.pt'
    torch.save(dict(model=model.cpu(),epoch=-1,train_args=dict(task='detect')),checkpoint)
    loaded=RTDETR(str(checkpoint)).model.eval().to(device)
    require(all(torch.equal(v.cpu(),loaded.state_dict()[k].cpu()) for k,v in model.state_dict().items()),'Nonzero save/reload changed state')
    with torch.no_grad():_,restored=capture(loaded,x)
    report['save_reload']=compare_records(before,restored)
    if device=='cuda':
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):amp=loaded(x)[0]
        require(torch.isfinite(amp).all(),'AMP inference failed')
        half=deepcopy(loaded).half()
        with torch.no_grad():half_out=half(x.half())[0]
        require(torch.isfinite(half_out).all() and next(half.parameters()).dtype==torch.float16,'True-half inference failed')
        # Compare same precision before/after fuse; cross-precision is not bitwise equality.
        # Native AutoBackend fuses FP32 first, then converts to half. Calling
        # half().fuse() hits the inherited generic Conv helper's FP32 bias dtype.
        half_fused_model=deepcopy(fused).half()
        protocol('half_fuse',half,half_fused_model,x.half(),'half')
        protocol('amp_fuse',loaded,fused,x,'amp')
        report['true_half']=dict(status='PASSED',parameter_dtype=str(next(half.parameters()).dtype),output_dtype=str(half_out.dtype))
        report['half_fuse_order']='Native order: FP32 fuse then half; inherited half().fuse() is unsupported and unchanged'
    else:report['AMP']=report['true_half']='NOT_RUN on CPU; CUDA checked separately'
    # Real predict -> AutoBackend -> automatic fuse using the serialized nonzero model.
    api=RTDETR(str(checkpoint));image=np.zeros((160,192,3),dtype=np.uint8)
    predictions=api.predict(image,imgsz=192,device=device,verbose=False,save=False)
    require(hasattr(api.predictor.model.model.model[lif_index],'bn'),'AutoBackend predict lost LIF BN')
    report['predict_auto_fuse']=dict(status='PASSED',images=len(predictions),checkpoint_sha256=sha256(checkpoint))
    persist()
    require(all(report[k]['status'] in PASS for k in ('FP32','half_fuse','amp_fuse') if k in report),'Fusion requires review; natural candidate membership drift blocks formal startup')
    return report,checkpoint


def bounded_eval(checkpoint,dataset,folder):
    from c19_lif_v1_results import evaluate
    sample=folder/'bounded_dataset'
    for split in ('val','test'):
        (sample/'images'/split).mkdir(parents=True);(sample/'labels'/split).mkdir(parents=True)
        for image in sorted((dataset/'images/train').glob('*.jpg'))[:2]:
            shutil.copyfile(image,sample/'images'/split/image.name)
            shutil.copyfile(dataset/'labels/train'/image.with_suffix('.txt').name,sample/'labels'/split/image.with_suffix('.txt').name)
    config=sample/'data.yaml';YAML.save(config,dict(path=str(sample),train='images/val',val='images/val',test='images/test',names={0:'crack'}))
    result={}
    for split in ('val','test'):
        result[split]=evaluate(checkpoint,config,split,folder/('bounded_'+split),device='cpu',
            val_report=folder/'bounded_val/metrics.json' if split=='test' else None,evidence_scope='two_train_images_engineering_only')
    return {k:dict(status=v['status'],images=v['images'],ground_truth=v['ground_truth'],predictions=v['predictions'],policy=v['policy'],
                   checkpoint_sha256=v['checkpoint_sha256'],evidence_scope=v['evidence_scope']) for k,v in result.items()}


def capacity(initialized,dataset,batch_size):
    require(batch_size==16 and torch.cuda.is_available(),'Formal capacity check requires CUDA B16')
    weights=RTDETR(str(initialized)).model;torch.manual_seed(42)
    model,_=build_training_model(weights.yaml,weights,dict(nc=1,channels=3));model.cuda().train();model.nc=1
    data,records=real_batch(dataset,640,16);data={k:v.cuda() for k,v in data.items()};opt,_=optimizer(model)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.float16):loss=model.loss(data)[0]
    loss.backward()
    require(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'B16/640 AMP failed')
    torch.cuda.synchronize()
    return dict(status='PASSED',batch=16,imgsz=640,AMP=True,loss=float(loss),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.perf_counter()-started,samples=records,optimizer_steps=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('source','initialized','real-dataset','output'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--capacity-batch',type=int,choices=[16]);args=parser.parse_args()
    args.output=args.output.resolve();args.initialized=args.initialized.resolve();args.source=args.source.resolve();args.real_dataset=args.real_dataset.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
    report=dict(status='FAILED',runtime=runtime(),formal_training='NOT_RUN',full_val_test='NOT_RUN',capacity=dict(status='NOT_RUN'),
                precision_tolerances=dict(parent_FP32=[2e-6,2e-5],fuse_FP32=[2e-5,2e-4],fuse_half_AMP=[3e-3,3e-2]),
                determinism='Local subprocess only: TF32 off, four CPU threads, deterministic warn_only. Native grid_sample/replicate backward may warn.')
    try:
        report['LIF_module']=module_checks()
        subprocess.run([sys.executable,str(ROOT/'ultralytics-main/tests/test_cbr.py')],cwd=ROOT,check=True)
        report['original_CBR_tests']=dict(status='PASSED',tests=3)
        report['CBR_geometry_gradient']=geometry_gradient()
        _,pair,mapping=controlled_models(args.source);weights=RTDETR(str(args.initialized)).model
        require(all(torch.equal(v,weights.state_dict()[k]) for k,v in pair.state_dict().items()),'Existing init differs from fresh controlled source')
        torch.manual_seed(42);target,adapt=build_training_model(weights.yaml,weights,dict(nc=1,channels=3))
        report['mapping']=mapping;report['trainer_rebuild']=adapt;report['actual_train_api']=actual_train_api(args.initialized,target)
        report['topology']=verify_model(target,zero=True);report['graph']=[dict(index=m.i,type=type(m).__name__,source=m.f) for m in target.model]
        report['parameters']={kind:sum(p.numel() for p in build(kind,nc=1).parameters()) for kind in CONFIGS}
        report['dataset']=dataset_inventory(args.real_dataset);batch,samples=real_batch(args.real_dataset)
        report['samples']=samples
        report['historical_regressions']=historical_regressions(target,batch,args.output)
        nb=(report['dataset']['train']['images']+15)//16
        for device in ('cpu','cuda'):
            if device=='cuda' and not torch.cuda.is_available():report[device]='NOT_RUN';continue
            torch.manual_seed(321)
            report[device]=dict(degenerations=degenerations(target,device))
            print(device+' parent degenerations passed',flush=True)
            report[device]['fuse']={}
            _,checkpoint=fuse_checks(target,device,args.output,report[device]['fuse'],lambda:atomic_json(args.output/'checks.json',report))
            if device=='cpu':report['bounded_val_test']=bounded_eval(checkpoint,args.real_dataset,args.output)
            report[device]['loss']=loss_checks(target,batch,device,args.output,nb)
            if device=='cuda':report[device]['AMP_loss']=loss_checks(target,batch,device,args.output,nb,amp=True)
            report[device]['benchmark']=benchmark(target,device)
            print(device+' fuse/loss/DN/optimizer passed',flush=True);gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        if args.capacity_batch:report['capacity']=capacity(args.initialized,args.real_dataset,args.capacity_batch)
        report['status']='PASSED'
    except BaseException as error:
        report['failure']=getattr(error,'detail',dict(error=repr(error)))
        raise
    finally:
        report['source_sha256']={p.relative_to(ROOT).as_posix():sha256(p) for p in list((ROOT/'tools').glob('*c19_lif_v1*.py'))+
                                 [ROOT/'ultralytics-main/ultralytics/nn/modules/cbr.py',ROOT/'ultralytics-main/ultralytics/nn/modules/lif_down.py',
                                  ROOT/'ultralytics-main/ultralytics/nn/modules/transformer.py',ROOT/'ultralytics-main/ultralytics/nn/tasks.py']}
        atomic_json(args.output/'checks.json',report)


if __name__=='__main__':main()
