"""Bounded LSRT checks. No epochs, final test, or formal optimizer updates."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback
import warnings

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import numpy as np
from init_lsrt_v1 import (ROOT, VARIANTS, LSRT_KEYS, build, controlled_models, build_training_model,
                          require, write_json, runtime, fingerprint, verify_model, sha256)
from ultralytics import RTDETR
from ultralytics.nn.modules.lsrt import LocalSemanticResidualTransport
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.patches import torch_load
from check_c19_lif_v1 import actual_train_api, optimizer as parent_optimizer, activate
from c19_lif_v1_probe import capture, targets
from c19_lif_v1_diagnostic import compare_records, fusion_protocol
from c19_lif_v1_cutoff import fusion_accepted


def optimizer(model):
    opt,report=parent_optimizer(model)
    ids=[id(p) for group in opt.param_groups for p in group['params']]
    report['LSRT_occurrences']={name:ids.count(id(p)) for name,p in model.named_parameters() if name in LSRT_KEYS}
    require(set(report['LSRT_occurrences'])==LSRT_KEYS and all(n==1 for n in report['LSRT_occurrences'].values()),'LSRT optimizer coverage mismatch')
    report['LSRT_groups']=[dict(name=name,group=index,weight_decay=group['weight_decay'])
                          for name,p in model.named_parameters() if name in LSRT_KEYS
                          for index,group in enumerate(opt.param_groups) if any(p is item for item in group['params'])]
    return opt,report


def single_fusion(unfused,fused,image,folder,device,precision):
    """Same ordered tolerances/replay as parent, with no LIF-specific assertions."""
    from contextlib import nullcontext
    from c19_lif_v1_diagnostic import selection_report, align_to_ids, PRE_KEYS, schema
    folder.mkdir(parents=True,exist_ok=False)
    atol,rtol=(2e-5,2e-4) if precision=='fp32' else (3e-3,3e-2)
    def trace(model,ids=None):
        with torch.no_grad(),(torch.autocast('cuda',dtype=torch.float16) if precision=='amp' else nullcontext()):
            return capture(model,image,fixed_ids=ids)[1]
    report=dict(status='FAILED',device=device,precision=precision,tolerances=dict(atol=atol,rtol=rtol))
    try:
        a,b=trace(unfused),trace(fused)
        report['pre_selection']=compare_records(a,b,atol,rtol,device=device,precision=precision,keys=PRE_KEYS)
        report['selection']=selection_report(a,b)
        report['fixed_replay']=compare_records(trace(unfused,a['candidate_indices']),trace(fused,a['candidate_indices']),atol,rtol,device=device,precision=precision)
        require(report['selection']['kind']!='SET_DRIFT','LSRT-only fusion changed candidate set; explicit review required')
        report['aligned']=compare_records(a,align_to_ids(a,b),atol,rtol,device=device,precision=precision,
                                          keys=[k for k in schema(a) if k not in PRE_KEYS+['candidate_indices']])
        report['status']='PASSED'
    finally:write_json(folder/'fuse_diagnostic.json',report)
    return report


def sample(device='cpu', size=(160,192)):
    return dict(img=torch.rand(2,3,*size,device=device),
                cls=torch.zeros(4,1,device=device),
                bboxes=torch.tensor([[.3,.4,.2,.3],[.6,.7,.1,.2],[.5,.5,.4,.2],[.7,.3,.1,.1]],device=device),
                batch_idx=torch.tensor([0,1,1,1],device=device))


def initial_equivalence(parent, target, device):
    result={}
    for shape in ((160,192),(640,640)):
        a,b=deepcopy(parent).to(device).eval(),deepcopy(target).to(device).eval()
        x=torch.rand(1,3,*shape,device=device)
        with torch.no_grad():
            _,left=capture(a,x);_,right=capture(b,x)
        comparison=compare_records(left,right,stage='LSRT_zero_initial',device=device)
        require(torch.equal(left['boxes'],right['boxes']) and torch.equal(left['scores'],right['scores']), 'Zero LSRT output must be exact')
        result[str(shape)]=dict(status='PASSED', output_exact=True, comparison=comparison)
        del a,b,x
    a,b=deepcopy(parent).to(device).train(),deepcopy(target).to(device).train()
    batch=sample(device)
    torch.manual_seed(717)
    _,left=capture(a,batch['img'],targets(batch))
    torch.manual_seed(717)
    _,right=capture(b,batch['img'],targets(batch))
    result['training_same_rng']=compare_records(left,right,stage='training_same_rng',device=device)
    require(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()), 'Separate BN updates differ')
    return dict(status='PASSED', checks=result)


def loss_backward(target, device, amp=False):
    rows=[]
    for nonzero in (False, True):
        m=deepcopy(target).to(device).train(); m.nc=1
        if nonzero:
            activate(m)
            with torch.no_grad(): m.model[18].lsrt.out_proj.weight.normal_(std=.01)
        batch=sample(device)
        with torch.autocast(device_type=device,enabled=amp):
            pred=m.predict(batch['img'],batch=targets(batch))
            loss=m.loss(batch,preds=pred)[0]
        require(torch.isfinite(loss),'Nonfinite native loss')
        loss.backward()
        require(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),'Nonfinite native gradient')
        norms={n:float(p.grad.float().norm()) if p.grad is not None else None for n,p in m.named_parameters() if n in LSRT_KEYS}
        require(norms['model.18.lsrt.out_proj.weight'] > 0, 'First LSRT output gradient missing')
        if nonzero:
            require(all(v is not None and v>0 for v in norms.values()), 'Activated LSRT gradient missing')
        else:
            require(all(v==0 for k,v in norms.items() if k!='model.18.lsrt.out_proj.weight'), 'Zero Wo upstream gradient mismatch')
        require(m.model[0].conv.weight.grad is not None,'Backbone gradient absent')
        rows.append(dict(nonzero_out_proj=nonzero,loss=float(loss.detach()),gradients=norms,
                         dn_num_split=pred[4]['dn_num_split'],GT_per_image=targets(batch)['gt_groups']))
        del m,pred,loss;gc.collect()
    return dict(status='PASSED',device=device,AMP=amp,optimizer_steps=0,steps=rows)


def reload_fusion(target, variant, device, folder):
    report={}
    for active in (False,True):
        m=deepcopy(target).eval().to(device)
        if active:
            activate(m,bn=True)
            with torch.no_grad():m.model[18].lsrt.out_proj.weight.normal_(std=.01)
        expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        ckpt=folder/f'{device}_reload_{active}.pt'
        with ckpt.open('xb') as f:
            torch.save(dict(model=deepcopy(m).cpu(),epoch=-1,train_args=dict(task='detect')),f)
        restored=RTDETR(str(ckpt)).model.eval().to(device)
        require(all(torch.equal(v,restored.state_dict()[k].cpu()) for k,v in expected.items()),'Nonzero checkpoint lost state')
        state_copy=build(variant,nc=1);state_copy.load_state_dict(expected,strict=True)
        ema=ModelEMA(restored)
        require(all(torch.equal(v,ema.ema.state_dict()[k].cpu()) for k,v in expected.items()),'EMA clone changed state')
        ema.update(restored)
        require(torch.count_nonzero(ema.ema.model[18].lsrt.out_proj.weight)==torch.count_nonzero(restored.model[18].lsrt.out_proj.weight),'EMA cleared Wo')
        fused=deepcopy(restored).fuse(verbose=False)
        require(all(torch.equal(restored.state_dict()[k],fused.state_dict()[k]) for k in LSRT_KEYS),'Fuse changed LSRT')
        if variant=='cbr_lif_lsrt_v1':require(hasattr(fused.model[20],'bn'),'LIF shared BN removed')
        row=dict(checkpoint_exact=True,state_dict_exact=True,EMA_preserved=True,LSRT_fuse_exact=True)
        x=torch.rand(1,3,160,192,device=device)
        # Parent project's candidate-aware protocol and original tolerances are reused unchanged.
        precisions=('fp32','amp','half') if device=='cuda' else ('fp32',)
        for precision in precisions:
            a,b=deepcopy(restored),deepcopy(fused)
            xx=x
            if precision=='half':a.half();b.half();xx=x.half()
            if variant=='cbr_lif_lsrt_v1':
                diag=fusion_protocol(a,b,xx,folder/f'{device}_{active}_{precision}',device,precision)
                require(fusion_accepted(diag,device,precision),'Fusion gate requires review: '+diag['status'])
            else:
                diag=single_fusion(a,b,xx,folder/f'{device}_{active}_{precision}',device,precision)
            row[precision]=dict(status=diag['status'],report=str(folder/f'{device}_{active}_{precision}/fuse_diagnostic.json'))
            del a,b
        report['nonzero' if active else 'zero']=row
        del m,restored,fused,state_copy,ema;gc.collect()
    return dict(status='PASSED',checks=report)


def complexity(parent,target):
    import thop
    def count_lsrt(module,inputs,output):
        low,high=inputs;b,c,hh,ww=low.shape;h,w=high.shape[-2:];d=32
        module.total_ops += torch.tensor([b*(c*d*hh*ww+2*c*d*h*w+d*c*hh*ww+2*9*d*hh*ww)],dtype=torch.float64,device=module.total_ops.device)
    rows={}
    x=torch.zeros(1,3,640,640)
    for label,model in (('parent',parent),('LSRT',target)):
        rows[label]={}
        for fused in (False,True):
            m=deepcopy(model).eval()
            if fused:m.fuse(verbose=False)
            params=sum(p.numel() for p in m.parameters())
            with torch.no_grad():macs,_=thop.profile(m,inputs=(x,),custom_ops={LocalSemanticResidualTransport:count_lsrt},verbose=False)
            rows[label]['fused' if fused else 'unfused']=dict(parameters=params,thop_MACs=macs,thop_2MAC_GFLOPs=2*macs/1e9)
    for fused in ('unfused','fused'):
        require(rows['LSRT'][fused]['parameters']-rows['parent'][fused]['parameters']==32777,'Parameter delta wrong')
        require(rows['LSRT'][fused]['thop_MACs']-rows['parent'][fused]['thop_MACs']==134758400,'LSRT THOP functional hook missing/double counted')
    return dict(status='PASSED',nc=1,imgsz=640,device='cpu',precision='FP32',mode='eval',rows=rows,
                LSRT_core_MACs=134758400,LSRT_core_2MAC_GFLOPs=.2695168,
                limitations='THOP operator coverage estimate, not measured latency or all operations. LSRT excludes normalization, softmax, mask, difference and memory traffic.')


def capacity(target,data_yaml,folder):
    from train_lsrt_v1 import dataset_inventory
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.data.utils import check_det_dataset
    from types import SimpleNamespace
    require(torch.cuda.is_available(),'Real capacity requires CUDA')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    inv=dataset_inventory(data_yaml)
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    args=YAML.load(ROOT/'docs/c19_lif_v1/c2_args.yaml')
    trainer.args=SimpleNamespace(**args);trainer.data=check_det_dataset(str(data_yaml),autodownload=False)
    dataset=trainer.build_dataset(trainer.data['train'],mode='train',batch=16)
    require(len(dataset)>=16,'Fewer than 16 training images')
    # Two fixed real train batches: native order and more GT. Test never selects samples.
    many=sorted(range(len(dataset)),key=lambda i:len(dataset.labels[i]['cls']),reverse=True)[:16]
    selections=[('native_order',list(range(16))),('high_GT_train',many)]
    rows=[]
    for label,indices in selections:
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        batch=dataset.collate_fn([dataset[i] for i in indices])
        keep={k:v for k,v in batch.items() if k in ('img','cls','bboxes','batch_idx')}
        keep={k:v.cuda() for k,v in keep.items()};keep['img']=keep['img'].float()/255
        require(keep['img'].shape==(16,3,640,640),'Capacity batch geometry changed')
        model=deepcopy(target).cuda().train();model.nc=1
        opt,opt_report=optimizer(model)
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.float16):
            pred=model.predict(keep['img'],batch=targets(keep));loss=model.loss(keep,preds=pred)[0]
        loss.backward();torch.cuda.synchronize()
        require(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Capacity loss/gradient nonfinite')
        rows.append(dict(batch_kind=label,loss=float(loss.detach()),GT_per_image=targets(keep)['gt_groups'],
                         dn_num_split=pred[4]['dn_num_split'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.perf_counter()-started,
                         images=[str(dataset.im_files[i]) for i in indices],optimizer_coverage=opt_report))
        del model,opt,loss,pred,batch,keep;gc.collect();torch.cuda.empty_cache()
    return dict(status='PASSED',batch=16,imgsz=640,AMP=True,optimizer_steps=0,native_training_augmentation=True,random_numpy_torch_seed=42,
                loss=rows[0]['loss'],batches=rows),inv


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='FAILED',variant=args.variant,runtime=runtime(),fingerprints=fingerprint(args.variant,args.source,args.initialized),
                formal_training='NOT_STARTED',final_test='NOT_RUN',formal_optimizer_steps=0,checks={},
                capacity=dict(status='PENDING',reason='Requires explicit --capacity, CUDA, real training data, B16/640 native AMP'),
                precision_tolerances=dict(parent_FP32=[2e-6,2e-5],fuse_FP32=[2e-5,2e-4],fuse_half_AMP=[3e-3,3e-2]))
    path=args.output/'report.json'
    def step(name,fn):
        print('CHECK',name,flush=True)
        try: report['checks'][name]=fn()
        except Exception as error:
            report['checks'][name]=dict(status='FAILED',error=repr(error),traceback=traceback.format_exc())
        write_json(path,report)
    from check_lsrt_module import run_checks
    old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,
         torch.backends.cudnn.deterministic,torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.manual_seed(42)
        parent80,expected,mapping=controlled_models(args.source,args.variant)
        weights=RTDETR(str(args.initialized)).model
        require(all(torch.equal(v,weights.state_dict()[k]) for k,v in expected.state_dict().items()),'Initialization file not controlled initial state')
        torch.manual_seed(42);target,adapt=build_training_model(weights.yaml,weights,dict(nc=1,channels=3),args.variant)
        torch.manual_seed(42);parent=build(args.variant,nc=1,baseline=True)
        parent.load_state_dict({k:target.state_dict()[k] for k in parent.state_dict()},strict=True)
        report['checks']['initialization']=mapping;report['checks']['native_trainer']=adapt
        report['checks']['topology']=dict(status='PASSED',**verify_model(target,args.variant,zero=True))
        step('actual_train_API',lambda:actual_train_api(args.initialized,target))
        step('AdamW_coverage',lambda:dict(status='PASSED',**optimizer(target)[1]))
        step('complexity',lambda:complexity(parent,target))
        devices=['cpu']+(['cuda'] if args.device=='cuda' and torch.cuda.is_available() else [])
        for device in devices:
            step(device+'_module',lambda:run_checks(device))
            step(device+'_initial_equivalence',lambda:initial_equivalence(parent,target,device))
            step(device+'_FP32_loss',lambda:loss_backward(target,device))
            step(device+'_reload_fusion',lambda:reload_fusion(target,args.variant,device,args.output))
            if device=='cuda':step('cuda_AMP_loss',lambda:loss_backward(target,'cuda',True))
        if args.data:
            from train_lsrt_v1 import dataset_inventory
            report['data_inventory']=dataset_inventory(args.data)
        if args.capacity:
            require(args.device=='cuda' and args.data,'--capacity requires --device cuda --data')
            try:report['capacity'],report['data_inventory']=capacity(target,args.data,args.output)
            except Exception as error:
                report['capacity']=dict(status='FAILED',error=repr(error),OOM=isinstance(error,torch.cuda.OutOfMemoryError),optimizer_steps=0)
        pending=[]
        if 'cuda' not in devices:pending.append('CUDA FP32/native AMP/half')
        if report['capacity']['status']=='PENDING':pending.append('real train B16/640 AMP capacity, GT/DN/peak memory')
        server=report['runtime']['python'].startswith('3.10.') and report['runtime']['torch']=='2.1.2+cu121'
        if not server:pending.append('Repeat on existing server Python3.10 / torch2.1.2+cu121 / 4090')
        report['pending']=pending
        failed=any(v.get('status')=='FAILED' for v in report['checks'].values()) or report['capacity']['status']=='FAILED'
        report['status']='FAILED' if failed else ('PENDING' if pending else 'PASSED')
        report['determinism']='TF32 disabled; deterministic=True warn_only; CUDA sampling backward is not claimed bitwise deterministic'
    except Exception as error:
        report['error']=repr(error);report['traceback']=traceback.format_exc()
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=old[:4]
        torch.use_deterministic_algorithms(old[4],warn_only=old[5]);report['precision_flags_restored']=True
        write_json(path,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS,default='cbr_lif_lsrt_v1')
    for name in ('source','initialized','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--data',type=Path);parser.add_argument('--capacity',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        result=run(args)
    result['actual_warnings']=sorted({str(w.message) for w in recorded})
    write_json(args.output/'report.json',result)
    print(json.dumps(dict(status=result['status'],report=str(args.output/'report.json')),indent=2))
    raise SystemExit(1 if result['status']=='FAILED' else 0)
