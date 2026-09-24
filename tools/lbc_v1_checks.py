"""Independent formula/reference checks and bounded engineering integration tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import gc
import math
import random
import time
import numpy as np
import torch

from lbc_v1_training import (LBCTrainer, LBCDetectionModel, initialize, native_copy, audit_wrapper,
                              validate_checkpoint, deploy, epoch_start, batch_end, epoch_end)
from init_c19_lif_v1 import ROOT, build, require, write_json
from ultralytics.models.rtdetr.lbc import LBCHead, LBC_CONFIG, HEAD_KEYS, regions, sample_indices, score_loss, ramp
from ultralytics.utils.torch_utils import ModelEMA, EarlyStopping
from ultralytics.utils.patches import torch_load
from ultralytics.models.rtdetr.train import RTDETRTrainer


def fixture():
    return dict(img=torch.rand(2, 3, 160, 160),
                bboxes=torch.tensor([[.30,.35,.18,.35],[.72,.62,.15,.12],[.60,.40,.12,.25]]),
                cls=torch.zeros(3, 1), batch_idx=torch.tensor([0.,0.,1.]))


def reference(batch, hf, wf):
    """Independent scalar Python geometry and sampling, with no production helpers."""
    h, w = batch['img'].shape[-2:]
    sx, sy = w/wf, h/hf
    by_image = [[] for _ in batch['img']]
    for j, (b, index) in enumerate(zip(batch['bboxes'].tolist(), batch['batch_idx'].tolist())):
        cx, cy, bw, bh = b
        if not all(math.isfinite(x) for x in b) or bw <= 0 or bh <= 0:
            continue
        left, top, right, bottom = max(0,(cx-bw/2)*w), max(0,(cy-bh/2)*h), min(w,(cx+bw/2)*w), min(h,(cy+bh/2)*h)
        if right > left and bottom > top:
            by_image[int(index)].append((j, [left,top,right,bottom]))
    result = []
    for image, boxes in enumerate(by_image):
        def expand(b, cells, factor):
            dx, dy = max(cells*sx, factor*(b[2]-b[0])), max(cells*sy, factor*(b[3]-b[1]))
            return [max(0,b[0]-dx),max(0,b[1]-dy),min(w,b[2]+dx),min(h,b[3]+dy)]
        excluded = [expand(b,1,.1) for _,b in boxes]
        for j,b in boxes:
            outer = expand(b,4,.5)
            p,n = [],[]
            for y in range(hf):
                for x in range(wf):
                    px,py = (x+.5)*sx,(y+.5)*sy
                    def contains(r):
                        return r[0] <= px < r[2] and r[1] <= py < r[3]
                    if contains(b): p.append(y*wf+x)
                    if contains(outer) and not any(contains(e) for e in excluded): n.append(y*wf+x)
            if not p or len(n) < 16: continue
            def sample(v):
                return v if len(v)<=128 else [v[(t*(len(v)-1))//127] for t in range(128)]
            ps,ns = sample(p),sample(n)
            result.append(dict(image=image, gt_index=j, positive=ps, negative=ns, k=min(4,len(ps),len(ns))))
    return result


def unit_checks():
    b = dict(img=torch.zeros(2,3,256,320),
             bboxes=torch.tensor([[.30,.35,.38,.35],[.38,.40,.2,.2],[.01,.01,.1,.12],
                                   [.50,.5,.0001,.4],[.5,.5,1.,1.],[.8,.8,.1,.15],
                                   [.6,.6,0.,.1],[float('nan'),.5,.1,.1]]),
             cls=torch.zeros(8,1), batch_idx=torch.tensor([0,0,0,0,1,0,1,1]))
    before = torch.get_rng_state().clone()
    pairs, stats, _ = regions(b,(32,40))
    ref = reference(b,32,40)
    require(len(pairs)==len(ref)>0, 'Reference pair count')
    for a,r in zip(pairs,ref):
        require(all(a[k]==r[k] for k in ('image','gt_index','k')), 'Reference image/key')
        require(all(a[k].tolist()==r[k] for k in ('positive','negative')), 'Reference geometric samples')
    require(stats['empty_positive']>0 and stats['insufficient_background']>0 and stats['invalid_box']==2, 'Skip branches')
    for n in (1,127,128,129,255,10000):
        actual=sample_indices(torch.arange(n)).tolist()
        expect=list(range(n)) if n<=128 else [(t*(n-1))//127 for t in range(128)]
        require(actual==expect and len(set(actual))==len(actual), '128 sampling rule')
    require(torch.equal(before,torch.get_rng_state()), 'Geometry consumed RNG')
    z=torch.linspace(-.9,.9,2*32*40).reshape(2,32,40).requires_grad_()
    raw, details=score_loss(z,pairs)
    values=[]
    for r in ref:
        v=z.detach()[r['image']].flatten().tolist()
        p=sum(sorted((v[i] for i in r['positive']),reverse=True)[:r['k']])/r['k']
        n=sum(sorted((v[i] for i in r['negative']),reverse=True)[:r['k']])/r['k']
        values.append(math.log1p(math.exp((.2+n-p)/.2))+.25*math.log1p(math.exp(n/.2)))
    require(abs(float(raw)-sum(values)/len(values))<2e-6, 'Independent target-weighted mean/formula')
    raw.backward();require(torch.isfinite(z.grad).all(), 'Score gradient finite')
    r=[dict(image=0,positive=torch.tensor([0]),negative=torch.tensor([1]),k=1,
            positive_count=1,negative_count=16)]
    loss=lambda p,n: float(score_loss(torch.tensor([[[p,n]]],dtype=torch.float32),r)[0])
    require(loss(.2,.1)<loss(.1,.1)<loss(.1,.2), 'Loss direction')
    require(math.isfinite(loss(-1000.,1000.)), 'Stable softplus')
    h=LBCHead(); f=torch.zeros(2,128,4,5,requires_grad=True)
    h(f).sum().backward();require(torch.isfinite(f.grad).all(), 'Zero-norm eps branch')
    try: h(torch.full_like(f,float('nan')))
    except ValueError: pass
    else: raise AssertionError('Nonfinite feature must fail')
    empty=deepcopy(b);empty.update(bboxes=torch.empty(0,4),cls=torch.empty(0,1),batch_idx=torch.empty(0))
    require(regions(empty,(32,40))[1]['pairs']==0, 'Empty GT')
    malformed=deepcopy(empty);malformed['batch_idx']=torch.tensor([0])
    try: regions(malformed,(32,40))
    except ValueError: pass
    else: raise AssertionError('Malformed labels were hidden')
    require([ramp(e) for e in (0,5,6,20)]==[0.,0.,1/15,1.], 'Schedule boundaries')
    return dict(status='PASSED', geometry=stats, independent_reference=True, rng_unchanged=True,
                raw=float(raw), shared_k=True, schedule={'5':0,'6':1/15,'20':1})


def bare_trainer(model):
    t=LBCTrainer.__new__(LBCTrainer)
    t.model=model;t.args=SimpleNamespace(lr0=.0005,momentum=.937,weight_decay=.0001,
        nbs=64,batch=16,epochs=200,cos_lr=True,lrf=.01,close_mosaic=10,model='controlled-lifecycle-fixture')
    t.data=dict(nc=1,channels=3);t.resume=False;t.start_epoch=0;t.epochs=200
    t.lbc_effective_updates=t.lbc_overflow_skips=t.lbc_consecutive_skips=0
    t.lbc_resume_state=None;t._lbc_ni=-1;t._lbc_last_opt_step=-1
    builder = t.build_optimizer if isinstance(model,LBCDetectionModel) else lambda *a,**kw: RTDETRTrainer.build_optimizer(t,*a,**kw)
    t.optimizer=builder(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    t.scaler=torch.cuda.amp.GradScaler(enabled=False)
    t._setup_scheduler();t.ema=ModelEMA(model);t.accumulate=4;t.train_loader=[None,None]
    return t


def model_checks(folder, init):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    source=torch_load(init,map_location='cpu')['model']
    t=LBCTrainer.__new__(LBCTrainer);t.data=dict(nc=1,channels=3);t.resume=False
    state=torch.get_rng_state().clone()
    model=t.get_model(source.yaml,source,verbose=False).train()
    model.nc=1  # Native Trainer.set_model_attributes normally sets this before loss.
    native=native_copy(model).train()
    report=dict(status='RUNNING',adaptation=t.lbc_adaptation,audit=audit_wrapper(native,model))
    del source
    b=fixture();rng=torch.get_rng_state().clone()
    op0=bare_trainer(native).optimizer
    op1=bare_trainer(model).optimizer
    head0={k:v.clone() for k,v in model.lbc_head.state_dict().items()}
    torch.set_rng_state(rng);l0,items0=native(b);l0.backward()
    torch.set_rng_state(rng);l1,items1=model(b);l1.backward()
    require(torch.equal(l0,l1) and torch.equal(items0,items1),'Inactive original loss mismatch')
    p0,p1=dict(native.named_parameters()),dict(model.named_parameters())
    for n,p in p0.items():
        q=p1[n]
        require((p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad), 'Inactive public gradient '+n)
    require(all(p.grad is None for p in model.lbc_head.parameters()),'Inactive auxiliary grad')
    torch.nn.utils.clip_grad_norm_(native.parameters(),10.)
    torch.nn.utils.clip_grad_norm_([p for n,p in model.named_parameters() if n not in HEAD_KEYS],10.)
    op0.step();op1.step()
    require(all(torch.equal(v,model.state_dict()[n]) for n,v in native.state_dict().items()),'Inactive AdamW public update')
    require(all(torch.equal(v,model.lbc_head.state_dict()[k]) for k,v in head0.items()),'Inactive head moved')
    report['inactive']=dict(loss=float(l0),public_gradients_exact=True,public_update_exact=True,head_grad_none=True)
    del op0,op1,l0,l1
    native.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)
    targets=dict(cls=b['cls'].long().flatten(),bboxes=b['bboxes'],batch_idx=b['batch_idx'].long(),gt_groups=[2,1])
    torch.set_rng_state(rng);normal=native.predict(b['img'],batch=targets)
    torch.set_rng_state(rng);original,f3=model.predict_with_s3(b['img'],targets)
    require(all(torch.equal(a,c) for a,c in zip(normal[:4],original[:4])),'S3 read changed native predictions')
    # Native L0 consumes the same predictions; this does not run a second decoder/DN forward.
    a,_=native.loss(b,normal);c,_=model.loss(b,original);require(torch.equal(a,c),'Active native L0 mismatch')
    pairs,stats,_=regions(b,f3.shape[-2:]);raw,_=score_loss(model.lbc_head(f3),pairs)
    named=list(model.named_parameters());grads=torch.autograd.grad(raw,[p for _,p in named],allow_unused=True,retain_graph=True)
    upstream=[]
    for (name,_),grad in zip(named,grads):
        if name.startswith('lbc_head.') or (name.startswith('model.') and int(name.split('.')[1])<=5):
            if grad is not None and torch.count_nonzero(grad): upstream.append(name)
        else: require(grad is None,'Direct LBC path leaked to '+name)
    require(any(n.startswith('lbc_head') for n in upstream) and any(n.startswith('model.5.') for n in upstream), 'Missing S3/head gradients')
    (c+.05*raw).backward()
    require(model.model[20].O_proj.weight.grad is not None and model.model[26].cbr.offset_out.weight.grad is not None,
            'Total loss lost LIF/CBR paths')
    report['routing']=dict(s3_shape=list(f3.shape),predictions_exact=True,L0_exact=True,
        direct_gradient_parameters=upstream,no_direct_gradient_after_S3=True,LIF_CBR_total_path=True,pairs=stats['pairs'])
    del native,normal,original,raw,f3,grads,a,c
    model.zero_grad(set_to_none=True);gc.collect()
    t=bare_trainer(model);model.lbc_epoch=20
    head0={k:v.clone() for k,v in model.lbc_head.state_dict().items()}
    backbone0=model.model[5].blocks[0].branch2a.conv.weight.detach().clone()
    t._lbc_ni=0
    loss,_=model(b);loss.backward()
    saved_grads=[p.grad.clone() for p in model.lbc_head.parameters()]
    no_pairs=deepcopy(b);no_pairs['bboxes']=torch.tensor([[.5,.5,1.,1.],[.5,.5,1.,1.]])
    no_pairs['cls']=torch.zeros(2,1);no_pairs['batch_idx']=torch.tensor([0.,1.])
    empty_loss,_=model(no_pairs);empty_loss.backward()
    require(all(torch.equal(p.grad,g) for p,g in zip(model.lbc_head.parameters(),saved_grads)), 'Empty microbatch erased accumulated head gradient')
    t.optimizer_step()
    require(any(not torch.equal(v,model.lbc_head.state_dict()[k]) for k,v in head0.items()), 'Active head did not update')
    require(not torch.equal(backbone0,model.model[5].blocks[0].branch2a.conv.weight),'Backbone did not update')
    head1={k:v.clone() for k,v in model.lbc_head.state_dict().items()}
    steps1={n:int(t.optimizer.state[p]['step']) for n,p in model.named_parameters() if n in HEAD_KEYS}
    for _ in range(2):
        loss,_=model(no_pairs);loss.backward()
    require(all(p.grad is None for p in model.lbc_head.parameters()),'No-pair window materialized grad')
    t._lbc_ni=1;t.optimizer_step()
    require(all(torch.equal(v,model.lbc_head.state_dict()[k]) for k,v in head1.items()),'No-pair AdamW moved head')
    require(all(int(t.optimizer.state[p]['step'])==steps1[n] for n,p in model.named_parameters() if n in HEAD_KEYS),'No-pair optimizer state advanced')
    report['accumulation']=dict(valid_then_empty_preserved=True,empty_window_no_update=True,active_head_backbone_update=True,
                                optimizer_groups=t.lbc_optimizer_groups,clip=t.lbc_clip)
    # A controlled two-microbatch epoch boundary with a pending, scaled gradient.
    model.lbc_epoch=19;t.epoch=19;t._lbc_ni=39;t._lbc_last_opt_step=38
    loss,_=model(b);loss.backward()
    t.scheduler.last_epoch=19;t.scheduler.step()
    t.best_fitness=t.fitness=.1;t.metrics={};t.save_period=-1;t.stopper=EarlyStopping(patience=50)
    t.wdir=folder/'lifecycle';t.last=t.wdir/'last.pt';t.best=t.wdir/'best.pt'
    t.csv=folder/'absent.csv'
    t.save_model();ckpt=torch_load(t.last,map_location='cpu');validate_checkpoint(ckpt)
    factory=LBCTrainer.__new__(LBCTrainer);factory.data=dict(nc=1,channels=3);factory.resume=True
    rebuilt=factory.get_model(ckpt['ema'].yaml,ckpt['ema'].float(),verbose=False)
    rebuilt.nc=1
    restored=bare_trainer(rebuilt)
    restored.resume=True;restored.stopper=EarlyStopping(patience=50)
    restored.resume_training(ckpt);restored.scheduler.load_state_dict(ckpt['scheduler'])
    require(restored.initial_optimizer_step_index()==38 and restored.start_epoch==20,'Resume epoch/window')
    restored.initialize_train_gradients()
    require(restored.model.lbc_epoch==20 and ramp(restored.model.lbc_epoch)==1,'Resume reset schedule')
    require(restored.scheduler.state_dict()==t.scheduler.state_dict() and restored.scaler.state_dict()==t.scaler.state_dict(), 'Scheduler/scaler restoration')
    require(all(torch.equal(v,restored.model.state_dict()[k]) for k,v in model.state_dict().items()),'Raw model restoration')
    require(torch.equal(model.model[-1].anchors,restored.model.model[-1].anchors), 'Raw decoder cache restoration')
    require(torch.equal(ckpt['ema'].model[-1].anchors,restored.ema.ema.model[-1].anchors), 'EMA decoder cache restoration')
    require(all((p.grad is None and dict(restored.model.named_parameters())[n].grad is None) or
                torch.equal(p.grad,dict(restored.model.named_parameters())[n].grad) for n,p in model.named_parameters()),'Pending gradient restoration')
    t._lbc_ni=40;restored._lbc_ni=40;t.optimizer_step();restored.optimizer_step()
    require(all(torch.equal(v,restored.model.state_dict()[k]) for k,v in model.state_dict().items()),'Resumed optimizer continuation mismatch')
    report['lifecycle']=dict(status='PASSED',epoch_resumed=20,scheduler_scaler_restored=True,
        raw_model_optimizer_pending_gradient_continuation_exact=True,raw_and_EMA_decoder_caches_restored=True,
        real_get_model_rebuild=True,ema_updates=restored.ema.updates,
        boundary='controlled 2-batch epoch fixture; no arbitrary interruption reproducibility claim')
    report['deployment']=deploy(t.last,folder/'lifecycle_deploy.pt')
    report['status']='PASSED'
    write_json(folder/'model_checks.json',report)
    return report


def run_checks(folder, source):
    torch.set_num_threads(1)  # CPU embedding accumulation uses a fixed reduction order.
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    result=dict(status='RUNNING',units=unit_checks())
    init=folder/'controlled_init.pt'
    if not init.exists(): result['initialization']=initialize(source,init)
    result['model']=model_checks(folder,init)
    result['status']='PASSED';write_json(folder/'checks.json',result)
    return result
