"""Finite engineering checks; no training epoch, full validation or test split inference."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import warnings
from contextlib import nullcontext
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('YOLO_AUTOINSTALL','false')
import numpy as np
import torch
from init_c17_lif_v1 import (ROOT, CONFIGS, VARIANT, MODULE_HASHES, BASE_COMMIT, C17_COMMIT, C2_COMMIT, controlled_models,
    initialize, native_rebuild, verify_model, runtime, is_added, require, write_json, sha256, build_training_model)
from c17_lif_v1_topology import state_mapping
from lif_down_topology import locate
from check_lif_down import module_checks, real_batch, targets
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.patches import torch_load


def compare(a,b,atol=2e-6,rtol=2e-5):
    a,b=a.detach().cpu().float(),b.detach().cpu().float()
    require(a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(),'Nonfinite/shape mismatch')
    delta=(a-b).abs()
    row=dict(shape=list(a.shape),max_abs=float(delta.max()) if delta.numel() else 0.,max_rel=float((delta/a.abs().clamp_min(1e-8)).max()) if delta.numel() else 0.,atol=atol,rtol=rtol)
    require(torch.allclose(a,b,atol=atol,rtol=rtol),'Numerical mismatch: '+str(row))
    return row


def tree_compare(a,b,**tolerance):
    if isinstance(a,torch.Tensor): return compare(a,b,**tolerance)
    if isinstance(a,(tuple,list)):
        require(type(a)==type(b) and len(a)==len(b),'Output tree mismatch')
        return [tree_compare(x,y,**tolerance) for x,y in zip(a,b)]
    if isinstance(a,dict):
        require(set(a)==set(b),'Output keys changed')
        return {k:tree_compare(a[k],b[k],**tolerance) for k in a}
    require(a==b,'Output metadata mismatch');return a


def capture(model,x,fixed_queries=None):
    p=locate(model.yaml);taps={};handles=[]
    def encoder_hook(m,args,out):
        taps['encoder_scores_before_topk']=out.detach().cpu()
        taps['query_indices']=torch.topk(out.max(-1).values,300,dim=1).indices.detach().cpu()
    handles.append(model.model[-1].enc_score_head.register_forward_hook(encoder_hook))
    original_topk=torch.topk
    def fixed_topk(input,k,dim=None,**kwargs):
        if input.ndim==2 and k==300 and dim==1:
            indices=fixed_queries.to(input.device)
            return torch.return_types.topk((input.gather(1,indices),indices))
        return original_topk(input,k,dim=dim,**kwargs)
    for name,i in [('down',p['p3_to_p4']['downsample']),*[(f'p{s}',p[k]['input']) for s,k in [(3,'p3_to_p4'),(4,'p4_to_p5')]],('p5',p['p4_to_p5']['output'])]:
        def hook(m,args,out,key=name):taps[key]=out.detach().cpu()
        handles.append(model.model[i].register_forward_hook(hook))
    try:
        with torch.no_grad(), patch.object(torch,'topk',side_effect=fixed_topk) if fixed_queries is not None else nullcontext(): out=model(x)
        taps['raw']=out
        taps['query_bbox']=out[0][...,:4];taps['query_score']=out[0][...,4:]
        return taps
    finally:
        for h in handles:h.remove()


def compare_fused(a,b,x,atol=2e-5,rtol=2e-4):
    left,right=capture(a,x),capture(b,x)
    report={'features':{k:compare(left[k],right[k],atol,rtol) for k in ['down','p3','p4','p5','encoder_scores_before_topk']}}
    indices_equal=torch.equal(left['query_indices'],right['query_indices'])
    report['natural_topk_indices_equal']=indices_equal
    report['natural_output_row_max_abs']=float((left['raw'][0].float()-right['raw'][0].float()).abs().max())
    report['topk_set_intersection']=len(set(left['query_indices'].flatten().tolist()) & set(right['query_indices'].flatten().tolist()))
    if not indices_equal:
        # Selection is discontinuous near tied scores. Re-run with the unfused anchor IDs for both
        # copies, preserving the actual score/features. This is an isolated diagnostic, never production.
        right=capture(b,x,fixed_queries=left['query_indices'])
        report['alignment']='same unfused encoder anchor IDs supplied to local torch.topk diagnostic; original score tensors unchanged'
    else:report['alignment']='native query ordering already identical'
    report['query_aligned']=tree_compare(left,right,atol=atol,rtol=rtol)
    return report


def activate(model,cs=True,lif=True,bn=False):
    torch.manual_seed(770)
    with torch.no_grad():
        for m in model.modules():
            if type(m).__name__=='CSCEFv51':
                m.output_projection.weight.normal_(0,.01) if cs else m.output_projection.weight.zero_()
            if type(m).__name__=='LIFDown':
                m.O_proj.weight.normal_(0,.01) if lif else m.O_proj.weight.zero_()
                if bn:
                    m.bn.running_mean.copy_(torch.linspace(-.3,.3,256,device=m.bn.weight.device))
                    m.bn.running_var.copy_(torch.linspace(.5,1.5,256,device=m.bn.weight.device))
                    m.bn.weight.copy_(torch.linspace(.7,1.3,256,device=m.bn.weight.device))
                    m.bn.bias.copy_(torch.linspace(-.1,.1,256,device=m.bn.weight.device))


def degeneration(models,device):
    report={}
    for parent,cs,lif in [('c2',False,False),('c17',True,False),('lif',False,True)]:
        a,b=deepcopy(models[parent]).to(device).eval(),deepcopy(models[VARIANT]).to(device).eval()
        activate(b,cs,lif)
        m=state_mapping(a.yaml,b.yaml,a.state_dict())
        a.load_state_dict({k:b.state_dict()[v] for k,v in m.items()},strict=True)
        rows={}
        for h,w in [(160,192),(640,640)]:
            torch.manual_seed(301);x=torch.rand(1,3,h,w,device=device)
            rows[f'{h}x{w}']=tree_compare(capture(a,x),capture(b,x))
        report[parent]=rows
        del a,b,x;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    return report


def cscef_startup(model):
    cs=deepcopy(next(m for m in model.modules() if type(m).__name__=='CSCEFv51'))
    torch.manual_seed(123);x=torch.randn(2,256,9,13,requires_grad=True);s=torch.randn_like(x,requires_grad=True)
    out=cs([x,s]);require(torch.equal(out,x),'CSCEF zero output not identity')
    out.square().mean().backward()
    grads={n:float(p.grad.norm()) for n,p in cs.named_parameters()}
    require(grads['output_projection.weight']>0 and all(v==0 for n,v in grads.items() if n!='output_projection.weight'),'Original zero-init gradient staging changed')
    return grads


def flow_effect(model):
    a=deepcopy(model).eval();activate(a,True,False)
    b=deepcopy(a);activate(b,True,True)
    x=torch.rand(1,3,160,192)
    left,right=capture(a,x),capture(b,x)
    result={'p3':compare(left['p3'],right['p3'],0,0)}
    for k in ['down','p4','p5']:
        result[k+'_effect_max_abs']=float((left[k]-right[k]).abs().max())
        require(result[k+'_effect_max_abs']>0,'Active LIF failed to affect '+k)
    return result


def historical(models,folder):
    report={}
    for kind,commit in [('c2',C2_COMMIT),('c17',C17_COMMIT),('lif',BASE_COMMIT)]:
        model=deepcopy(models[kind]).eval();activate(model)
        torch.manual_seed(301);x=torch.rand(1,3,160,192)
        payload=folder/(kind+'_input.pt');out=folder/(kind+'_history.pt')
        content=dict(yaml=model.yaml,state=model.state_dict(),image=x,kind=kind)
        if kind!='c2':
            module=next(m for m in model.modules() if type(m).__name__==('CSCEFv51' if kind=='c17' else 'LIFDown'))
            content['module_state']=module.state_dict()
        torch.save(content,payload)
        worker=str(ROOT/'tools/c17_lif_v1_history_worker.py')
        env=dict(os.environ,PYTHONPATH=str(ROOT/'ultralytics-main'),YOLO_AUTOINSTALL='false')
        fixture_report=None
        if kind!='c2':
            fixture=ROOT/'ultralytics-main/tests/fixtures/c17_lif_v1'
            for n in MODULE_HASHES:
                require(sha256(fixture/n)==MODULE_HASHES[n],'Historical fixture hash mismatch')
            fixture_out=folder/(kind+'_fixture.pt');current_out=folder/(kind+'_current.pt')
            subprocess.run([sys.executable,worker,str(ROOT),str(payload),str(fixture_out),str(fixture)],cwd=folder,env=env,check=True)
            subprocess.run([sys.executable,worker,str(ROOT),str(payload),str(current_out)],cwd=folder,env=env,check=True)
            fixture_report=tree_compare(torch_load(fixture_out,map_location='cpu')['module'],torch_load(current_out,map_location='cpu')['module'],atol=0,rtol=0)
        present=subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=ROOT,capture_output=True).returncode==0
        if not present:
            require(fixture_report is not None,'Missing required C2 historical ancestor')
            report[kind]=dict(commit=commit,full_parent_model='NOT_RUN: historical Git object absent',original_module_isolated_fixture=fixture_report)
            continue
        root=folder/('history_'+kind);root.mkdir()
        archive=subprocess.check_output(['git','archive',commit,'ultralytics-main/ultralytics'],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            for member in tf.getmembers():
                dest=root/member.name
                require(dest.resolve().is_relative_to(root.resolve()) and not member.issym() and not member.islnk(),'Unsafe historical archive')
                if member.isfile():
                    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(tf.extractfile(member).read())
        env=dict(os.environ,PYTHONPATH=str(root/'ultralytics-main'),YOLO_AUTOINSTALL='false')
        subprocess.run([sys.executable,str(ROOT/'tools/c17_lif_v1_history_worker.py'),str(root),str(payload),str(out)],cwd=folder,env=env,check=True)
        result=torch_load(out,map_location='cpu')
        require(Path(result['import_path']).resolve().is_relative_to(root.resolve()),'Historical import contamination')
        with torch.no_grad(): current=model(x)
        row=dict(commit=commit,import_path=result['import_path'],prediction=tree_compare(current,result['prediction'],atol=0,rtol=0),original_module_isolated_fixture=fixture_report)
        if kind!='c2':
            # Reproduce the same probe in a separate CURRENT subprocess, avoiding any import cache ambiguity.
            current_out=folder/(kind+'_current.pt')
            subprocess.run([sys.executable,str(ROOT/'tools/c17_lif_v1_history_worker.py'),str(ROOT),str(payload),str(current_out)],cwd=folder,env=dict(env,PYTHONPATH=str(ROOT/'ultralytics-main')),check=True)
            current_result=torch_load(current_out,map_location='cpu')
            row['nonzero_module_and_gradients']=tree_compare(current_result['module'],result['module'],atol=0,rtol=0)
            require(float(result['module']['lateral_grad'].norm())>0,'No lateral gradient')
            if kind=='c17':require(float(result['module']['semantic_grad'].norm())>0,'No semantic content gradient')
        report[kind]=row
    return report


def fusion(model,device,folder):
    a=deepcopy(model).to(device).eval();activate(a,bn=True)
    b=deepcopy(a).fuse(verbose=False)
    t=verify_model(a);i=t['p3_to_p4']['downsample']
    require(hasattr(b.model[i],'bn') and torch.equal(a.model[i].O_proj.weight,b.model[i].O_proj.weight),'LIF fuse lost BN/residual')
    before=sum(isinstance(m,torch.nn.BatchNorm2d) for m in a.modules());after=sum(isinstance(m,torch.nn.BatchNorm2d) for m in b.modules())
    require(after<before,'Normal fusion globally disabled')
    report=dict(bn_before=before,bn_after=after,lif_bn_preserved=True,fp32={})
    for h,w in [(160,192),(640,640)]:
        x=torch.rand(1,3,h,w,device=device)
        report['fp32'][f'{h}x{w}']=compare_fused(a,b,x)
    b.fuse(verbose=False)
    report['repeat_fuse']=compare_fused(a,b,x)
    weights=folder/f'nonzero_{device}.pt'
    a.args={'task':'detect','model':CONFIGS[VARIANT]};a.task='detect'
    torch.save(dict(model=a.cpu(),train_args=a.args,epoch=-1,ema=None),weights)
    loaded=RTDETR(str(weights)).model.eval().to(device)
    a.to(device)
    report['reload']=tree_compare(capture(a,x),capture(loaded,x),atol=0,rtol=0)
    api=RTDETR(str(weights))
    prediction=api.predict(source=x,device=device,verbose=False,save=False,imgsz=640,conf=.001,half=False)
    require(hasattr(api.predictor.model.model.model[i],'bn'),'API automatic fusion lost LIF BN')
    report['predict_auto_fuse']=compare_fused(a,api.predictor.model.model,x)
    report['predict_results']=len(prediction)
    if device=='cuda':
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            report['AMP_fuse']=compare_fused(a,b,x,atol=.008,rtol=.04)
        a.half();b.half()
        report['half_fuse']=compare_fused(a,b,x.half(),atol=.008,rtol=.04)
        require(capture(b,x.half())['raw'][0].dtype==torch.float16,'Not true half')
    return report


def optimizer_check(model):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    opt=trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    names={id(p):n for n,p in model.named_parameters() if p.requires_grad}
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    require(len(ids)==len(set(ids))==len(names) and set(ids)==set(names),'Optimizer coverage mismatch')
    rows=[dict(group=g['param_group'],lr=g['lr'],weight_decay=g['weight_decay'],names=[names[id(p)] for p in g['params']]) for g in opt.param_groups]
    added=[dict(name=names[id(p)],group=g['param_group'],lr=g['lr'],weight_decay=g['weight_decay'],classification='weight') for g in opt.param_groups for p in g['params'] if is_added(names[id(p)])]
    require(len(added)==10 and all(r['group']=='weight' for r in added),'Innovation optimizer groups changed')
    return opt,dict(groups=rows,innovation_parameters=added,coverage=len(ids),effective_decay=.0001,initial_accumulate=4)


def warmup(opt,ni,nb):
    nw=max(round(5*nb),100)
    accumulate=max(1,int(np.interp(ni,[0,nw],[1,64/16]).round()))
    for g in opt.param_groups:
        g['lr']=float(np.interp(ni,[0,nw],[.1 if g['param_group']=='bias' else 0.,.0005]))
    return dict(batch_index=ni,nw=nw,accumulate=accumulate,lrs={g['param_group']:g['lr'] for g in opt.param_groups})


def loss_checks(model,batch,device,folder,nb=756,capacity=False):
    b=deepcopy(model).to(device).train();b.nc=1;activate(b)
    batch={k:v.to(device) for k,v in batch.items()}
    opt,report=optimizer_check(b)
    report.update(scope='B16/640 AMP capacity' if capacity else 'two real train images, finite three updates',input=list(batch['img'].shape),steps=[],warnings=[])
    modes=[True] if capacity else ([False,True] if device=='cuda' else [False])
    for amp in modes:
        scaler=torch.cuda.amp.GradScaler(enabled=amp,init_scale=128.)
        for ni in range(1 if capacity else 3):
            opt.zero_grad(set_to_none=True);w=warmup(opt,ni,nb)
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter('always')
                with torch.autocast(device_type=device,enabled=amp,dtype=torch.float16 if device=='cuda' else torch.bfloat16):
                    pred=b.predict(batch['img'],batch=targets(batch));loss=b.loss(batch,preds=pred)[0]
                require(pred[-1] is not None and pred[-1]['dn_num_split'][-1]==300,'DN disabled')
                scaler.scale(loss).backward();scaler.unscale_(opt)
            report['warnings'] += sorted(set(str(v.message) for v in seen))
            grads={n:float(p.grad.float().norm()) for n,p in b.named_parameters() if is_added(n) and p.grad is not None}
            require(torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters()),'Native loss/backward nonfinite')
            require(len(grads)==10 and all(v>0 for v in grads.values()),'Activated innovation gradient absent')
            main={n:float(p.grad.float().norm()) for n,p in b.named_parameters() if n in ['model.17.conv.weight','model.15.conv.weight']}
            torch.nn.utils.clip_grad_norm_(b.parameters(),10.0)
            scaler.step(opt);scaler.update()
            report['steps'].append(dict(w,amp=amp,loss=float(loss),dn=pred[-1]['dn_num_split'],gradients=grads,main_gradients=main,
                parameter_norms={n:float(p.float().norm()) for n,p in b.named_parameters() if is_added(n)},smoke_scaler=128))
            del pred,loss
    dest=folder/f'optimizer_{device}.pt'
    torch.save(dict(model=b.state_dict(),optimizer=opt.state_dict()),dest)
    checkpoint=torch_load(dest,map_location=device)
    c=deepcopy(model).to(device);c.load_state_dict(checkpoint['model'],strict=True)
    opt2,_=optimizer_check(c);opt2.load_state_dict(checkpoint['optimizer'])
    report['model_state_reload']=tree_compare(b.state_dict(),c.state_dict(),atol=0,rtol=0)
    report['optimizer_reload']=tree_compare(opt.state_dict(),opt2.state_dict(),atol=0,rtol=0)
    report['criterion']=type(b.criterion).__name__
    if device=='cuda':report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--real-dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--capacity',action='store_true')
    p.add_argument('--data-config',type=Path)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    report=dict(status='FAILED',runtime=runtime(),formal_training='NOT_RUN',full_val_test='NOT_RUN',server_B16_640='NOT_RUN',
                diagnostic_settings='Isolated CLI process; TF32 off, 4 threads, deterministic warn_only. Formal recipe unchanged.')
    start=time.perf_counter()
    try:
        models,mapping=controlled_models(args.source)
        report['initialization']=initialize(args.source,args.output/'untrained_init.pt')
        rebuilt={}
        for kind,m in models.items():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42);rebuilt[kind]=native_rebuild(m).eval()
        models=rebuilt
        report['parameters']={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()}
        report['topology']=verify_model(models[VARIANT],zero=True)
        report['original_cscef_zero_step']=cscef_startup(models[VARIANT])
        report['decoder_dependency_effect']=flow_effect(models[VARIANT])
        report['historical_regression']=historical(models,args.output)
        print('Historical isolated regressions PASS',flush=True)
        report['lif_module']=module_checks()
        batch,records=real_batch(args.real_dataset,size=160);report['samples']=records
        train_images=list((args.real_dataset/'images/train').glob('*.jpg'));nb=(len(train_images)+15)//16
        for device in ['cpu','cuda']:
            if device=='cuda' and not torch.cuda.is_available():report[device]='NOT_RUN';continue
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            report[device]={}
            report[device]['degeneration']=degeneration(models,device)
            print(device+' three nonzero parent-degeneration checks PASS',flush=True)
            report[device]['fusion']=fusion(models[VARIANT],device,args.output)
            print(device+' nonzero BN/fuse/reload/predict checks PASS',flush=True)
            report[device]['loss']=loss_checks(models[VARIANT],batch,device,args.output,nb)
            print(device+' native DN/loss/optimizer/reload PASS',flush=True)
            gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
        if args.capacity:
            require(torch.cuda.is_available(),'CUDA capacity requested')
            from ultralytics.cfg import get_cfg
            from ultralytics.utils import YAML
            from ultralytics.data.utils import check_det_dataset
            require(args.data_config is not None,'Capacity requires actual data configuration')
            trainer=RTDETRTrainer.__new__(RTDETRTrainer)
            trainer.args=get_cfg(YAML.load(ROOT/'docs/c17_lif_v1/c2_args.yaml'))
            trainer.data=check_det_dataset(str(args.data_config),autodownload=False)
            dataset=trainer.build_dataset(trainer.data['train'],mode='train',batch=16)
            batch16=dataset.collate_fn([dataset[i] for i in range(16)])
            batch16={k:batch16[k] for k in ['img','bboxes','cls','batch_idx']}
            batch16['img']=batch16['img'].float()/255
            torch.cuda.reset_peak_memory_stats()
            report['server_B16_640']=loss_checks(models[VARIANT],batch16,'cuda',args.output,nb,capacity=True)
        report['status']='PASSED'
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        write_json(args.output/'checks.json',report)
        print('Report: '+str(args.output/'checks.json'),flush=True)


if __name__=='__main__':main()
