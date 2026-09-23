"""ROR 有界数学、真实模型接线、初始化、优化器一步和生命周期检查。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import math
from pathlib import Path
import subprocess
import types
from datetime import datetime, timezone

import torch
from init_c19_lif_v1 import (ROOT, controlled_models, build_training_model, verify_model,
                             runtime, write_json, require, sha256)
from ultralytics.models.utils.ror import RORLoss, image_loss, matched_loss, aligned_iou, ramp, CONFIG
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.patches import torch_load
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ror_v1_training import RORTrainer, install, epoch_start

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"


def close(a, b, atol=1e-7, rtol=1e-6):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def reference(q0, q1, z, cls):
    terms = []
    for i in range(len(z)):
        for j in range(len(z)):
            if i == j or cls[i] != cls[j] or q0[j]-q0[i] < .02 or q1[i]-q1[j] < .02:
                continue
            qi, qj = min(1-1e-4,max(1e-4,q1[i])), min(1-1e-4,max(1e-4,q1[j]))
            m = min(2., math.log(qi/(1-qi))-math.log(qj/(1-qj)))
            d = max(0., m-(z[i]-z[j]))
            phi = .5*d*d if d <= 1 else d-.5
            terms.append((q1[i]-q1[j])*phi)
    return sum(terms)/max(1,len(terms)), len(terms)


def math_checks(device="cpu"):
    def tensor(values, grad=False):
        return torch.tensor(values, device=device, dtype=torch.float32, requires_grad=grad)
    q0, q1 = tensor([.55,.70],True), tensor([.82,.73],True)
    z = torch.logit(tensor([.60,.75])).detach().requires_grad_()
    cls = torch.zeros(2, device=device, dtype=torch.long)
    loss, row = image_loss(q0,q1,z,cls)
    expected, count = reference(q0.tolist(),q1.tolist(),z.tolist(),cls.tolist())
    close(loss,tensor(expected)); require(abs(float(loss)-.06434)<1e-5,"Specified example")
    loss.backward(); close(z.grad,tensor([-.09,.09])); require(q0.grad is None and q1.grad is None,"Quality detached")
    for before, after, labels in (([.7,.55],[.82,.73],[0,0]),([.55,.56],[.82,.73],[0,0]),
                                  ([.55,.70],[.74,.73],[0,0]),([.55,.70],[.82,.73],[0,1]),([.55],[.82],[0])):
        zz=tensor([0.]*len(before),True)
        value, info=image_loss(tensor(before),tensor(after),zz,torch.tensor(labels,device=device))
        value.backward();require(value==0 and info['eligible_pairs']==0 and torch.count_nonzero(zz.grad)==0,"Ineligible case")
    value,info=image_loss(tensor([0,.02]),tensor([.02,0]),tensor([0.,0.],True),cls)
    require(info['eligible_pairs']==1,"Inclusive FP32 delta")
    for quality in ([.82,.73],[1.,0.]):
        q=tensor(quality); zz=torch.logit(q.clamp(1e-4,1-1e-4)).requires_grad_()
        value,_=image_loss(q0,q,zz,cls)
        close(value,tensor(0.),atol=1e-12,rtol=0)  # torch.logit vs log/log1p FP32 last-bit difference
    q0a,q1a,za=tensor([.1,.5,.9]),tensor([.9,.5,.1]),tensor([0.,0.,-5.],True)
    value,info=image_loss(q0a,q1a,za,torch.zeros(3,device=device,dtype=torch.long))
    ref,n=reference(q0a.tolist(),q1a.tolist(),za.tolist(),[0,0,0]); close(value,tensor(ref))
    require(info['eligible_pairs']==n==3 and info['violating_pairs']==1,"Eligible denominator includes satisfied pairs")
    for vals in ([1e20,-1e20],[-1e20,1e20]):
        zz=tensor(vals,True); value,_=image_loss(q0,q1,zz,cls); value.backward()
        require(torch.isfinite(value) and torch.isfinite(zz.grad).all(),"Extreme finite logits")
    # Construct exact aligned qualities via contained boxes: IoU is area ratio.
    gt=tensor([[.5,.5,1.,1.],[.5,.5,1.,1.],[.5,.5,1.,1.]])
    before=tensor([[[.5,.5,.55,1.],[.5,.5,.70,1.]],[[.5,.5,.4,1.],[.5,.5,.5,1.]]],True)
    after=tensor([[[.5,.5,.82,1.],[.5,.5,.73,1.]],[[.5,.5,.6,1.],[.5,.5,.5,1.]]],True)
    logits=torch.logit(tensor([[[.6],[.75]],[[.5],[.4]]])).detach().requires_grad_()
    targets=dict(bboxes=gt,cls=torch.zeros(3,device=device,dtype=torch.long),gt_groups=[2,1])
    indices=[(torch.tensor([0,1]),torch.tensor([0,1])),(torch.tensor([0]),torch.tensor([2]))]
    value,rows=matched_loss(before,after,logits,targets,indices)
    close(value,tensor(expected/2));value.backward()
    require(before.grad is None and after.grad is None and logits.grad[0,0]<0 and logits.grad[0,1]>0,"Direct box/logit gradients")
    empty=dict(bboxes=gt[:0],cls=targets['cls'][:0],gt_groups=[0,0])
    empty_indices=[(torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long))]*2
    zero,_=matched_loss(before,after,logits,empty,empty_indices);require(zero.dtype==torch.float32 and zero.requires_grad and zero==0,"Graph-connected empty zero")
    # Test endpoints ordinary IoU and reject invalid data, never sanitize.
    close(aligned_iou(gt[:1],gt[:1]),tensor([1.]))
    bad=gt[:1].clone();bad[0,2]=-1
    try: aligned_iou(bad,gt[:1])
    except ValueError: pass
    else: raise AssertionError("Invalid box accepted")
    try: image_loss(q0,q1,tensor([float('nan'),0]),cls)
    except FloatingPointError: pass
    else: raise AssertionError("Nonfinite logits accepted")
    require([ramp(e) for e in (0,5,6,20,199)]==[0.,0.,1/15,1.,1.],"Schedule boundaries")
    with torch.autocast(device_type=torch.device(device).type,enabled=device!='cpu'):
        value,_=image_loss(q0,q1,z,cls)
    require(value.dtype==torch.float32 and torch.isfinite(value),"AMP ROR FP32")
    return dict(status="PASS",device=device,example_loss=float(loss),example_grad=z.grad.tolist(),
                independent_reference=True,box_direct_gradient=None,fp32_atol=1e-7,fp32_rtol=1e-6)


def parent_module(path, name):
    source=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT).decode('utf-8')
    module=types.ModuleType(name); module.__file__=BASE+':'+path
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module


def targets_of(batch):
    return dict(bboxes=batch['bboxes'],cls=batch['cls'].long().view(-1),batch_idx=batch['batch_idx'].long(),
                gt_groups=[int((batch['batch_idx']==i).sum()) for i in range(len(batch['img']))])


def split_predictions(result, details=None):
    boxes,scores,enc_b,enc_s,meta=result
    db,ds=None,None
    if meta:
        db,boxes=boxes.split(meta['dn_num_split'],dim=2); ds,scores=scores.split(meta['dn_num_split'],dim=2)
        if details is not None: details={k:v.split(meta['dn_num_split'],dim=1)[1] for k,v in details.items()}
    return (torch.cat([enc_b[None],boxes]),torch.cat([enc_s[None],scores])),dict(dn_bboxes=db,dn_scores=ds,dn_meta=meta),details


def integration(source, device="cpu"):
    torch.manual_seed(42)
    _,init,init_report=controlled_models(source)
    model,adapt=build_training_model(init.yaml,init,dict(nc=1,channels=3))
    verify_model(model,zero=True)
    parent_tasks=parent_module('ultralytics-main/ultralytics/nn/tasks.py','ultralytics.nn._ror_parent_tasks')
    parent_losses=parent_module('ultralytics-main/ultralytics/models/utils/loss.py','ultralytics.models.utils._ror_parent_loss')
    # Baseline really runs the exact parent source, including traversal and loss.
    baseline=parent_tasks.RTDETRDetectionModel(deepcopy(model.yaml),nc=1,verbose=False)
    baseline.load_state_dict(model.state_dict(),strict=True)
    baseline.criterion=parent_losses.RTDETRDetectionLoss(nc=1,use_vfl=True)
    model=install(model,dict(config=CONFIG,check='bounded'))
    require(set(model.state_dict())==set(baseline.state_dict()),"State keys differ")
    batch=dict(img=torch.rand(2,3,160,160,device=device),
               bboxes=torch.tensor([[.25,.25,.2,.1],[.6,.6,.3,.2],[.7,.2,.1,.25],[.5,.5,.3,.3]],device=device),
               cls=torch.zeros(4,1,device=device),batch_idx=torch.tensor([0,0,0,1],device=device))
    targets=targets_of(batch)
    model.to(device).train(); baseline.to(device).train()
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    def restore():
        torch.set_rng_state(rng)
        if cuda_rng: torch.cuda.set_rng_state_all(cuda_rng)
    with torch.no_grad():
        restore(); a=baseline.predict(batch['img'],batch=targets)
        restore(); b,details=model.predict(batch['img'],batch=targets,return_cbr_details=True)
        for left,right in zip(a[:4],b[:4]): close(left,right,atol=0,rtol=0)
        aa,kw,_=split_predictions(a);bb,_,side=split_predictions(b,details)
        native=baseline.criterion(aa,targets,**kw)
        model.criterion.set_epoch(20)
        active=model.criterion(bb,targets,**kw,ror_enabled=True,ror_details=side)
        for key in native: close(native[key],active[key],atol=0,rtol=0)
        require(set(active)==set(native)|{'loss_ror'},"ROR repeated in auxiliary/DN")
        require(torch.equal(side['after'],bb[0][-1]),"DN query dimension")
        # Native batch-flat offset with unequal GT counts.
        idx=model.criterion.matcher(bb[0][-1],bb[1][-1],targets['bboxes'],targets['cls'],[3,1])
        require(set(idx[0][1].tolist())=={0,1,2} and idx[1][1].tolist()==[3],"GT offset")
    model.criterion.set_epoch(0)
    # BN counters also had identical diagnostic forwards; compare actual independent optimizer updates.
    helper=RTDETRTrainer.__new__(RTDETRTrainer)
    optim_a=helper.build_optimizer(baseline,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    optim_b=helper.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    restore();loss_a,show_a=baseline.loss(batch); loss_a.backward()
    restore();loss_b,show_b=model.loss(batch); loss_b.backward()
    close(loss_a,loss_b,atol=0,rtol=0);close(show_a,show_b,atol=0,rtol=0)
    gradient_error=0.0;gradient_rows={}
    for (name,a),(other,b) in zip(baseline.named_parameters(),model.named_parameters()):
        require(name==other and (a.grad is None)==(b.grad is None),'Gradient parameter identity')
        if a.grad is not None:
            if device=='cpu':close(a.grad,b.grad,atol=0,rtol=0)
            delta=float((a.grad-b.grad).abs().max());gradient_error=max(gradient_error,delta)
            gradient_rows[name]=dict(ror_vs_parent_max_abs=delta)
    native_repeat_error=0.0;repeat_model=None;repeat_optimizer=None
    if device!='cpu':
        # Isolate native CUDA backward variability on exactly the same inputs/RNG/weights.
        original_grads={n:p.grad.clone() for n,p in baseline.named_parameters() if p.grad is not None}
        original_buffers={n:b.clone() for n,b in baseline.named_buffers()}
        baseline.zero_grad(set_to_none=True);restore();repeat_loss,_=baseline.loss(batch)
        close(repeat_loss,loss_a,atol=0,rtol=0);repeat_loss.backward()
        repeat_model=deepcopy(baseline)
        for (name,param),(other,replica) in zip(baseline.named_parameters(),repeat_model.named_parameters()):
            if param.grad is not None:replica.grad=param.grad.detach().clone()
        for name,buffer in repeat_model.named_buffers():buffer.copy_(original_buffers[name])
        repeat_optimizer=helper.build_optimizer(repeat_model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
        for name,param in baseline.named_parameters():
            if name in original_grads:
                repeat_delta=float((param.grad-original_grads[name]).abs().max())
                native_repeat_error=max(native_repeat_error,repeat_delta)
                gradient_rows[name]['native_repeat_max_abs']=repeat_delta
                gradient_rows[name]['bound']=max(2e-7,4*repeat_delta)
                param.grad=original_grads[name]
        for name,buffer in baseline.named_buffers():buffer.copy_(original_buffers[name])
        write_json(ROOT/'outputs/ror_v1/checks/cuda_backward_variation.json',gradient_rows)
        require(all(row['ror_vs_parent_max_abs']<=row['bound'] for row in gradient_rows.values()),
                'ROR gradient difference exceeds per-parameter native CUDA backward variability; see variation report')
    torch.nn.utils.clip_grad_norm_(baseline.parameters(),10.);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
    optim_a.step();optim_b.step()
    if repeat_model is not None:
        torch.nn.utils.clip_grad_norm_(repeat_model.parameters(),10.);repeat_optimizer.step()
    changed=0;max_update_error=0.0
    step_rows={}
    for key,value in model.state_dict().items():
        delta=float((value-baseline.state_dict()[key]).abs().max())
        if repeat_model is None:close(value,baseline.state_dict()[key],atol=0,rtol=0)
        else:
            native_delta=float((repeat_model.state_dict()[key]-baseline.state_dict()[key]).abs().max())
            step_rows[key]=dict(ror_vs_parent_max_abs=delta,native_repeat_max_abs=native_delta,bound=max(2e-7,4*native_delta))
        max_update_error=max(max_update_error,delta)
        changed+=int(not torch.equal(value,before[key]))
    if step_rows:
        # The first Adam step is sensitive to gradients near eps=1e-8. Verify both
        # independent updates against the analytic native AdamW first-step formula,
        # instead of enlarging all weight tolerances to hide a sign change near zero.
        for candidate,optimizer in ((baseline,optim_a),(model,optim_b)):
            names={id(param):name for name,param in candidate.named_parameters()}
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is None:continue
                    name=names[id(param)];g=param.grad
                    expected=before[name]*(1-group['lr']*group['weight_decay'])-group['lr']*g/(g.abs()+group['eps'])
                    close(param,expected,atol=2e-7,rtol=2e-6)
        write_json(ROOT/'outputs/ror_v1/checks/cuda_step_variation.json',step_rows)
    require(changed>0,"No effective optimizer update")
    # No-DN and no-GT path, then ordinary inference and supplied-prediction validation loss.
    model.zero_grad(set_to_none=True);model.criterion.set_epoch(20);model.model[-1].num_denoising=0
    for count in (0,1):
        mini={**batch,'bboxes':batch['bboxes'][:count],'cls':batch['cls'][:count],'batch_idx':batch['batch_idx'][:count]}
        val,_=model.loss(mini);require(torch.isfinite(val),"Empty/single GT active path");val.backward();model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        out=model.predict(batch['img']);require(out[0].shape==(2,300,5),"Normal inference return")
        val,shown=model.loss(batch,preds=out);require(torch.isfinite(val) and shown.numel()==3,"Validation L0")
    # Actual Trainer.get_model and native resume state restoration (no full training).
    dummy=RORTrainer.__new__(RORTrainer);dummy.resume=False;dummy.data=dict(nc=1,channels=3)
    dummy.ror_plan=dict(identity=model.ror_v1);dummy.ror_output=ROOT/'outputs/ror_v1/checks'
    torch.manual_seed(42);rebuilt=dummy.get_model(init.yaml,init,False)
    require(type(rebuilt.criterion) is RORLoss,"Actual Trainer rebuild missed ROR")
    rebuilt.criterion.set_epoch(6);ema=ModelEMA(rebuilt);ema.ema.criterion.set_epoch(6)
    optimizer=torch.optim.AdamW(rebuilt.parameters(),lr=.0005)
    checkpoint=dict(epoch=6,model=None,ema=ema.ema,updates=7,optimizer=optimizer.state_dict(),scaler={},best_fitness=.1)
    stream=io.BytesIO();torch.save(checkpoint,stream);stream.seek(0);restored=torch_load(stream,map_location='cpu')
    dummy.resume=True;dummy.model=rebuilt;dummy.epochs=200;dummy.ema=ema;dummy.optimizer=optimizer
    dummy.scaler=torch.cuda.amp.GradScaler(enabled=False);dummy.args=types.SimpleNamespace(model='bounded.pt',close_mosaic=10)
    dummy.resume_training(restored)
    require(dummy.start_epoch==7 and rebuilt.criterion.ror_weight==.1*2/15,"Resume restarted warmup")
    dummy.epoch=20;epoch_start(dummy);require(dummy.model.criterion.ror_weight==dummy.ema.ema.criterion.ror_weight==.1,"Epoch callback/EMA")
    from c19_lif_v1_diagnostic import fusion_protocol
    from c19_lif_v1_cutoff import fusion_accepted
    model.float().eval();fused=deepcopy(model).fuse(verbose=False)
    fusion_folder=ROOT/'outputs/ror_v1/checks'/('fusion_'+device.replace(':','_')+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    fusion=fusion_protocol(model,fused,batch['img'],fusion_folder,
                           torch.device(device).type,'fp32')
    require(fusion_accepted(fusion),"Fusion candidate-identity audit failed")
    return dict(status='PASS',device=device,source_sha256=sha256(source),parameters=20149765,
                added_parameters=0,state_keys=len(before),class_adaptation=adapt['ALLOWED_CLASS_ADAPTATION'],
                changed_states_after_step=changed,zero_weight_update_max_abs=max_update_error,
                gradient_max_abs=gradient_error,native_repeat_gradient_max_abs=native_repeat_error,
                gradient_comparison='CPU exact; CUDA per-parameter max(2e-7, 4*native repeated-backward max_abs), forward/loss exact',
                optimizer_update_tolerances='CPU exact; CUDA per-state max(2e-7,4*native repeated-step max_abs)',
                cuda_direct_step_variation_check=all(row['ror_vs_parent_max_abs']<=row['bound'] for row in step_rows.values()) if step_rows else None,
                cuda_step_reference='each actual clipped-gradient first AdamW update checked independently: atol=2e-7,rtol=2e-6' if step_rows else None,
                diagnostics_predictions='bitwise exact',active_L0='bitwise exact',DN=True,no_DN=True,
                empty_GT=True,single_GT=True,validation='L0 only',resume_e=7,schedule_at_20=.1,
                fused_parameters=sum(p.numel() for p in fused.parameters()),fusion_status=fusion['status'],
                fusion_tolerances=fusion['tolerances'],fusion_report=str(fusion_folder/'fuse_diagnostic.json'))


def criterion_fixture():
    # Controlled matched identities with a genuine nonzero ROR and DN tail protection.
    b0=torch.tensor([[[.5,.5,.55,1.],[.5,.5,.7,1.]]]); b1=b0.clone();b1[0,:,2]=torch.tensor([.82,.73])
    scores=torch.logit(torch.tensor([[[.6],[.75]]])).requires_grad_()
    targets=dict(bboxes=torch.tensor([[.5,.5,1.,1.],[.5,.5,1.,1.]]),cls=torch.zeros(2,dtype=torch.long),gt_groups=[2])
    criterion=RORLoss();criterion.set_epoch(20)
    calls=[]
    class CountingMatcher(torch.nn.Module):
        def forward(self,*args,**kwargs):
            calls.append(args[0].clone())
            return [(torch.tensor([0,1]),torch.tensor([0,1]))]
    criterion.matcher=CountingMatcher()
    preds=(torch.stack([b1,b1,b1,b1]),torch.stack([scores,scores,scores,scores]))
    native=RTDETRDetectionLoss(nc=1,use_vfl=True);native.matcher=CountingMatcher()
    expected=native(preds,targets);calls.clear()
    loss=criterion(preds,targets,ror_enabled=True,ror_details=dict(before=b0,after=b1))
    require(len(calls)==4,"Final rematched or aux stopped matching")
    require(set(loss)==set(expected)|{'loss_ror'} and float(loss['loss_ror'])>0,"Active final-only loss missing")
    for k in expected: close(loss[k],expected[k],atol=0,rtol=0)
    close(loss['loss_ror'],torch.tensor(.006434),atol=1e-6)
    loss['loss_ror'].backward();require(scores.grad[0,0,0]<0 and scores.grad[0,1,0]>0,"Wiring gradient")
    try: criterion(preds,targets,ror_enabled=True)
    except RuntimeError: pass
    else: raise AssertionError("Missing state silently accepted")
    return dict(status='PASS',final_match_calls=1,aux_match_calls=3,weighted=float(loss['loss_ror']))


def amp_smoke(source):
    """One disposable B2/160 native AMP update and native EMA-half validation call."""
    from ultralytics.models.rtdetr.val import RTDETRValidator
    _,initial,_=controlled_models(source)
    model,_=build_training_model(initial.yaml,initial,dict(nc=1,channels=3))
    install(model,dict(config=CONFIG,scope='AMP smoke'));model.nc=1;model.cuda().train();model.criterion.set_epoch(20)
    batch=dict(img=torch.rand(2,3,160,160,device='cuda'),
               bboxes=torch.tensor([[.3,.3,.2,.2],[.6,.6,.2,.3],[.5,.5,.3,.3]],device='cuda'),
               cls=torch.zeros(3,1,device='cuda'),batch_idx=torch.tensor([0,0,1],device='cuda'))
    helper=RTDETRTrainer.__new__(RTDETRTrainer)
    optimizer=helper.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    scaler=torch.cuda.amp.GradScaler(enabled=True)
    scales=[]
    for attempt in range(12):  # bounded native GradScaler calibration; never change AMP/batch/recipe
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True): loss,shown=model(batch)
        require(torch.isfinite(loss),'AMP nonfinite loss');scaler.scale(loss).backward();scaler.unscale_(optimizer)
        finite_grads=bool(torch.stack([torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None]).all())
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        prior_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        scales.append(dict(before=prior_scale,after=scaler.get_scale(),finite_gradients=finite_grads))
        if finite_grads:break
    require(finite_grads,'AMP gradients still nonfinite after 12 bounded native scaler attempts')
    ema=ModelEMA(model)
    sample={k:v.detach().cpu() for k,v in batch.items()};sample['img']=(sample['img']*255).round().byte()
    class OneBatch:
        dataset=[0,1]
        def __len__(self):return 1
        def __iter__(self):yield deepcopy(sample)
    observed={}
    class Validator(RTDETRValidator):
        def init_metrics(self,model):
            observed.update(half=self.args.half,model_dtype=str(next(model.parameters()).dtype))
        def update_metrics(self,preds,batch):
            require(all(torch.isfinite(v).all() for p in preds for v in p.values()),'Native EMA validation nonfinite')
        def gather_stats(self):pass
        def get_stats(self):return {}
        def finalize_metrics(self):pass
        def print_results(self):pass
    validator=Validator(dataloader=OneBatch(),save_dir=ROOT/'outputs/ror_v1/checks/amp_validator',
                        args=dict(imgsz=160,plots=False,save_json=False,save_txt=False,workers=0,task='detect'))
    trainer=types.SimpleNamespace(device=torch.device('cuda'),data=dict(nc=1,names={0:'crack'}),amp=True,ema=ema,model=model,
             args=types.SimpleNamespace(compile=False),loss_items=torch.zeros(3,device='cuda'),
             stopper=types.SimpleNamespace(possible_stop=False),epoch=20,epochs=200,world_size=1,
             label_loss_items=lambda loss,prefix:{prefix+'/'+str(i):float(v) for i,v in enumerate(loss)})
    output=validator(trainer)
    require(observed['half'] and observed['model_dtype']=='torch.float16' and torch.isfinite(validator.loss).all(),
            'Native half validator not exercised')
    return dict(status='PASS',batch=2,imgsz=160,optimizer_update=True,loss=float(loss),native_scaler_steps=scales,
                EMA_native_half_validation=observed,validation_loss=output,formal_training='NOT_RUN')


def run(source=None, device='cpu'):
    torch.set_num_threads(4)
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(42,deterministic=True)
    report=dict(runtime=runtime(),math=math_checks(device),criterion=criterion_fixture())
    report['integration']=integration(source,device) if source and Path(source).is_file() else dict(status='SKIPPED',reason='PENDING_ASSET: unified source')
    write_json(ROOT/'outputs/ror_v1'/('checks_progress_'+device.replace(':','_')+'.json'),report)
    if device!='cpu' and source and Path(source).is_file():
        import gc
        gc.collect();torch.cuda.empty_cache();report['native_AMP']=amp_smoke(source)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--device',default='cpu')
    parser.add_argument('--report',type=Path,default=ROOT/'outputs/ror_v1/checks.json')
    args=parser.parse_args()
    try: result=run(args.source,args.device)
    except BaseException as error:
        write_json(args.report,dict(status='FAIL',error=repr(error)));raise
    result['status']='REVIEW_REQUIRED' if result.get('integration',{}).get('cuda_direct_step_variation_check') is False else 'PASS'
    write_json(args.report,result);print(result)
