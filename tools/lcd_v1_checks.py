"""Bounded LCD formula, native learning, routing and strict FP32 fusion checks."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import inspect
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from init_lcd_v1 import require, verify_model, PREFIX, NEW_KEYS
from ultralytics.nn.modules import LCD
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import ModelEMA
from c19_lif_v1_diagnostic import (rng_state, restore_rng, compare_records, align_to_ids,
                                  selection_report, schema, LIF_KEYS, PRE_KEYS, atomic_json)


def optimizer(model):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    opt = trainer.build_optimizer(model, name='AdamW', lr=.0005, momentum=.937, decay=.0001)
    ids = [id(p) for g in opt.param_groups for p in g['params']]
    require(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}, 'Optimizer coverage differs')
    rows = []
    for n, p in model.named_parameters():
        if n in NEW_KEYS:
            g = next(g for g in opt.param_groups if any(p is v for v in g['params']))
            kind = 'bias' if n.endswith('bias') else ('bn' if '.norm.' in n else 'weight')
            require(g['param_group'] == kind and g['weight_decay'] == (.0001 if kind == 'weight' else 0), 'LCD grouping differs')
            rows.append(dict(name=n, numel=p.numel(), group=kind, lr=g['lr'], decay=g['weight_decay'], occurrences=ids.count(id(p))))
    for g in opt.param_groups: g['initial_lr'] = g['lr']
    return opt, rows


def reference(m, x, ka=None, kb=None):
    u = F.conv2d(x, m.P.weight).permute(0, 2, 3, 1)
    u = F.layer_norm(u, (32,), m.norm.weight, m.norm.bias, 1e-6).permute(0, 3, 1, 2)
    a = F.conv2d(u, m.shared_dw.weight if ka is None else ka, padding=1, dilation=1, groups=32)
    b = F.conv2d(u, m.shared_dw.weight if kb is None else kb, padding=2, dilation=2, groups=32)
    return x + F.conv2d(F.silu(a) * torch.tanh(F.conv2d(b-a, m.G.weight)), m.O.weight)


def formula():
    state = torch.get_rng_state().clone(); m = LCD().double()
    require(torch.equal(state, torch.get_rng_state()), 'LCD constructor consumes public RNG')
    require(sum(p.numel() for p in m.parameters()) == 9568 and len(m.state_dict()) == 6, 'Budget/keys changed')
    x = torch.randn(2, 128, 5, 7, dtype=torch.double)
    require(torch.equal(m(x), x), 'Zero O is not identity')
    target = torch.randn_like(x)
    (m(x)*target).mean().backward()
    first = {n: float(p.grad.norm()) for n, p in m.named_parameters()}
    require(first['O.weight'] > 0 and all(v == 0 for n,v in first.items() if n != 'O.weight'), 'Initial gradients differ')
    with torch.no_grad(): m.O.weight.add_(m.O.weight.grad, alpha=-.1)
    m.zero_grad(); (m(x)*target).mean().backward()
    later = {n: float(p.grad.norm()) for n,p in m.named_parameters()}
    require(all(v > 0 for v in later.values()), 'LCD branch failed to learn after O update')
    with torch.no_grad(): m.O.weight.normal_(0, .03)
    errors = []
    for value in (x, torch.ones_like(x), torch.zeros_like(x), F.pad(torch.ones(2,128,1,1,dtype=x.dtype),(3,3,2,2))):
        torch.testing.assert_close(m(value), reference(m,value), atol=1e-12, rtol=1e-12)
        errors.append(float((m(value)-reference(m,value)).abs().max()))
    # Independent leaves prove the registered K receives the sum from both paths.
    ka=m.shared_dw.weight.detach().clone().requires_grad_(); kb=ka.detach().clone().requires_grad_()
    ga,gb=torch.autograd.grad((reference(m,x,ka,kb)*target).sum(),(ka,kb))
    shared=torch.autograd.grad((m(x)*target).sum(),m.shared_dw.weight)[0]
    torch.testing.assert_close(shared,ga+gb,atol=1e-10,rtol=1e-10)
    require(ga.norm()>0 and gb.norm()>0, 'Shared second route is detached')
    # Directional finite difference on a nonzero-output copy, including input and K.
    direction=torch.randn_like(x); eps=1e-5
    xx=x.clone().requires_grad_(); analytic=(torch.autograd.grad((m(xx)*target).sum(),xx)[0]*direction).sum()
    numerical=((m(x+eps*direction)-m(x-eps*direction))*target).sum()/(2*eps)
    torch.testing.assert_close(analytic,numerical,atol=1e-6,rtol=1e-6)
    with torch.no_grad():
        u=m.norm(m.P(torch.ones_like(x)).permute(0,2,3,1)).permute(0,3,1,2)
        difference=F.conv2d(u,m.shared_dw.weight,padding=2,dilation=2,groups=32)-m.shared_dw(u)
    require(difference.abs().max()>0, 'Constant-input padding edge check degenerated')
    return dict(status='PASSED',parameters=9568,state_keys=list(m.state_dict()),first_gradients=first,
                after_update_gradients=later,reference_max_errors=errors,shared_K_gradient_sum=True,
                directional_gradient=True,constant_boundary_difference=float(difference.abs().max()),
                conv_MACs_80x80=62668800,functional_second_DW_MACs=1843200)


def routes(pair, lcd):
    a,b=deepcopy(pair).eval(),deepcopy(lcd).eval()
    x=torch.rand(1,3,160,160)
    with torch.no_grad():
        pa,pb=a(x)[0],b(x)[0]
    require(torch.equal(pa,pb), 'Initial CPU forward differs')
    with torch.no_grad(): b.model[5].lcd.O.weight.normal_(0,.03)
    seen={}; handles=[]
    def record(name):
        def hook(module,args,value): seen[name]=value.detach().clone()
        return hook
    def incoming(name):
        def hook(module,args): seen[name]=args[0].detach().clone()
        return hook
    handles += [b.model[5].blocks.register_forward_hook(record('X')), b.model[5].register_forward_hook(record('Y'))]
    handles += [b.model[i].register_forward_pre_hook(incoming(str(i))) for i in (6,17)]
    try:
        with torch.no_grad(): changed=b(x)[0]
    finally:
        for h in handles:h.remove()
    require(not torch.equal(seen['X'],seen['Y']), 'Nonzero LCD was not exercised')
    require(torch.equal(seen['Y'],seen['6']) and torch.equal(seen['Y'],seen['17']), 'Both paths must read Y')
    require(not torch.equal(pa,changed), 'LCD has no prediction effect')
    require(a.save==b.save and [m.f for m in a.model]==[m.f for m in b.model], 'Global route/save changed')
    with torch.no_grad():
        z=torch.zeros(1,3,640,640)
        for layer in b.model[:6]:z=layer(z)
    require(z.shape==(1,128,80,80),'640 input S3 shape differs')
    return dict(status='PASSED',initial_forward_exact=True,nonzero_changes_prediction=True,
                layer5_shape=list(seen['Y'].shape),layer6_reads_Y=True,layer17_reads_saved_Y=True,
                layer5_shape_at_640=list(z.shape),original_save_list=b.save,global_from_unchanged=True)


def precision_settings():
    return dict(cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_autocast=torch.is_autocast_enabled(),cpu_autocast=torch.is_autocast_cpu_enabled())


@contextmanager
def strict_precision(evidence):
    # Minimal precision scope from RDL 91abf119 (read-only reference), no RDL model/loss.
    before=evidence['before']=precision_settings()
    try:
        torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
        with torch.autocast('cuda',enabled=False),torch.autocast('cpu',enabled=False):
            evidence['inside']=precision_settings(); yield
    finally:
        torch.backends.cudnn.allow_tf32=before['cudnn_tf32']
        torch.backends.cuda.matmul.allow_tf32=before['matmul_tf32']
        torch.set_float32_matmul_precision(before['float32_matmul_precision'])
        evidence['after']=precision_settings();evidence['restored']=evidence['after']==before
        require(evidence['restored'],'Precision not restored')


def fusion(model, image, output):
    from c19_lif_v1_probe import capture
    report=dict(status='FAILED',tolerance=dict(atol=3e-5,rtol=3e-5),precision={})
    try:
        with strict_precision(report['precision']),torch.no_grad():
            a=deepcopy(model).float().eval(); b=deepcopy(a).fuse(verbose=False).eval()
            require(hasattr(b.model[20],'bn'),'LIF fusion protection lost')
            require(all(torch.equal(v,b.model[5].lcd.state_dict()[k]) for k,v in a.model[5].lcd.state_dict().items()),'LCD fusion changed')
            # Two forwards only; preserve all original rows before optional ID alignment.
            _,ra=capture(a,image.float()); _,rb=capture(b,image.float())
            report['original_rows']=compare_records(ra,rb,atol=3e-5,rtol=3e-5,collect=True)
            require(all(torch.isfinite(v).all() for r in (ra,rb) for k,v in r.items() if isinstance(v,torch.Tensor)),'Nonfinite fusion evidence')
            if all(v.get('allclose_failed_count')==0 and not v.get('status') for v in report['original_rows'].values()):
                report['status']='PASSED'
            else:
                report['selection']=selection_report(ra,rb)
                require(report['selection']['kind']=='PERMUTATION','Fusion differs beyond candidate order')
                report['aligned']=compare_records(ra,align_to_ids(ra,rb),atol=3e-5,rtol=3e-5)
                report['status']='PASS_CANDIDATE_PERMUTATION'
            report['lif_bn_retained']=True;report['lcd_state_exact']=True
    finally: atomic_json(output,report)
    return report


def learning(initial,batch,device='cpu',amp=False,max_batches=16,seconds=900):
    saved=rng_state()
    try:return _learning(initial,batch,device,amp,max_batches,seconds)
    finally:restore_rng(saved)


def _learning(initial,batch,device='cpu',amp=False,max_batches=16,seconds=900):
    """Native unscale/clip/step/EMA, native warmup/accumulate, at most two executed steps."""
    from check_c19_lif_v1 import native_warmup
    start=time.monotonic(); rows=[]
    # Fixed disposable diagnostic RNG; restore the caller even on failed checks.
    torch.manual_seed(42)
    model=deepcopy(initial).to(device).train();model.nc=1
    model.model[5].lcd.collect_diagnostics=True
    opt,groups=optimizer(model)
    trainer=SimpleNamespace(model=model,optimizer=opt,scaler=torch.cuda.amp.GradScaler(enabled=amp),ema=ModelEMA(model))
    batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
    observed={};model.criterion=model.init_criterion()
    def loss_record(module,args,value):observed['loss_keys']=sorted(value)
    def dn_record(module,args,value):observed['dn_num_split']=value[-1]['dn_num_split'] if value[-1] else None
    handles=[model.criterion.register_forward_hook(loss_record),model.model[-1].register_forward_hook(dn_record)]
    effective=attempts=skipped=0; last=-1; internal=False
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    def step_count():
        return max([int(s['step'].item()) for s in opt.state.values() if 'step' in s] or [0])
    try:
        for i in range(max_batches):
            if time.monotonic()-start>=seconds:break
            warmup=native_warmup(opt,i,378)
            with torch.autocast(device_type=device.split(':')[0],enabled=amp):
                loss,items=model(batch); loss=loss.sum()
            require(torch.isfinite(loss),'Nonfinite native loss')
            trainer.scaler.scale(loss).backward()
            grads={n:float(p.grad.detach().float().norm()/trainer.scaler.get_scale()) if p.grad is not None else None
                   for n,p in model.named_parameters() if n in NEW_KEYS}
            finite=all(v is not None and __import__('math').isfinite(v) for v in grads.values())
            internal=finite and all(grads[PREFIX+k]>0 for k in ('shared_dw.weight','G.weight','P.weight'))
            row=dict(batch=i,loss=float(loss.detach()),loss_items=items.detach().cpu().tolist(),warmup=warmup,
                     gradients=grads if finite else 'AMP scaled overflow',scale=trainer.scaler.get_scale(),
                     diagnostics=dict(model.model[5].lcd.diagnostics),optimizer_update=False,**observed)
            # After the second effective update, one backward-only probe verifies inner learning.
            if effective==2:
                rows.append(row);break
            if i-last>=warmup['accumulate']:
                old=step_count(); RTDETRTrainer.optimizer_step(trainer); now=step_count()
                last=i;attempts+=1;effective+=int(now>old);skipped+=int(now==old)
                row['optimizer_update']=now>old
            rows.append(row)
        require(1<=effective<=2,'No effective optimizer update within bounds')
        require(torch.count_nonzero(model.model[5].lcd.O.weight)>0,'O did not change')
        require(internal,'Post-update internal LCD gradients missing/nonfinite within bounds')
        require(all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite updated parameters')
        require(observed['dn_num_split'] and any('dn' in k for k in observed['loss_keys']),'Native DN loss not exercised')
        require(type(model.criterion).__name__=='RTDETRDetectionLoss','Original criterion lost')
        report=dict(status='PASSED',device=device,AMP=amp,batch=int(batch['img'].shape[0]),shape=list(batch['img'].shape),
                    batches=len(rows),effective_updates=effective,attempts=attempts,skipped_overflows=skipped,
                    elapsed_seconds=time.monotonic()-start,diagnostic_seed=42,steps=rows,optimizer=groups,
                    criterion=inspect.getfile(type(model.criterion)),loss_gain=model.criterion.loss_gain,
                    matcher_cost=model.criterion.matcher.cost_gain,loss='original complete L0 incl encoder/aux/DN',
                    reused_batch=True,backward_only_final_probe=True,formal_weights_untouched=True)
        if device.startswith('cuda'):
            report.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        model.model[5].lcd.collect_diagnostics=False;model.zero_grad()
        return model.cpu(),trainer,report
    finally:
        for h in handles:h.remove()


def latency(pair,lcd,device='cpu'):
    times={}
    for label,m in [('pair',pair),('lcd',lcd)]:
        m=deepcopy(m).to(device).eval(); x=torch.rand(1,3,160,160,device=device)
        with torch.no_grad():
            for _ in range(2):m(x)
            if device.startswith('cuda'):torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(5):m(x)
            if device.startswith('cuda'):torch.cuda.synchronize()
            times[label]=(time.perf_counter()-start)*1000/5
    return dict(ms=times,device=device,batch=1,imgsz=160,warmup=2,repeats=5,paper_comparison=False)
