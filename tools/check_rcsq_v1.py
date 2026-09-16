"""Finite RCS-Q semantic probes. Diagnostic copies only; no epoch or final test."""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import math
import time

import torch
from init_rcsq_v1 import ROOT, build, require
from ultralytics.nn.modules.rcs_q import RegionCenteredSupportQuery, build_query_valid
from ultralytics.models.utils.ops import get_cdn_group
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.patches import torch_load
from c19_lif_v1_probe import capture, targets
from c19_lif_v1_diagnostic import compare_records, fusion_protocol
from c19_lif_v1_cutoff import fusion_accepted


def compare(a, b, atol=2e-6, rtol=2e-5):
    a, b = a.detach().float(), b.detach().float()
    require(a.shape == b.shape, 'Tensor shape changed')
    finite = torch.isfinite(a) & torch.isfinite(b)
    # Original anchor logits may legally contain matching infinities.
    require(torch.equal(torch.isposinf(a), torch.isposinf(b)) and
            torch.equal(torch.isneginf(a), torch.isneginf(b)), 'Infinity pattern changed')
    require(not torch.isnan(a).any() and not torch.isnan(b).any(), 'NaN in comparison')
    diff = (a[finite] - b[finite]).abs()
    bad = ~torch.isclose(a, b, atol=atol, rtol=rtol)
    row = dict(max_abs_error=float(diff.max()) if diff.numel() else 0., atol=atol, rtol=rtol,
               first_failure=bad.nonzero()[0].tolist() if bad.any() else None)
    require(not bad.any(), 'Numerical mismatch: ' + str(row))
    return row


def compare_tree(a, b):
    if torch.is_tensor(a):
        return compare(a, b)
    if isinstance(a, (list, tuple)):
        require(type(a) is type(b) and len(a) == len(b), 'Return layout changed')
        return [compare_tree(x, y) for x, y in zip(a, b)]
    if isinstance(a, dict):
        require(a.keys() == b.keys(), 'Return metadata keys changed')
        return {k: compare_tree(a[k], b[k]) for k in a}
    require(a == b, 'Return metadata changed')
    return a


def grad_norms(module):
    rows = {n: float(p.grad.float().norm()) if p.grad is not None else None
            for n, p in module.named_parameters()}
    require(all(v is None or math.isfinite(v) for v in rows.values()), 'Nonfinite gradient')
    return rows


def synthetic_batch(device='cpu', groups=(1, 3), shape=(160, 192)):
    boxes = torch.tensor([[.4, .45, .23, .17], [.6, .5, .3, .2], [.2, .7, .1, .12],
                          [.7, .2, .15, .3]], device=device).repeat((sum(groups)+3)//4, 1)[:sum(groups)]
    return dict(img=torch.rand(len(groups), 3, *shape, device=device), bboxes=boxes,
                cls=torch.zeros(sum(groups), 1, device=device),
                batch_idx=torch.tensor([i for i, n in enumerate(groups) for _ in range(n)], device=device))


def geometry_checks(device='cpu'):
    """Analytic coordinate ramps independently identify x/y order and border semantics."""
    module = RegionCenteredSupportQuery().to(device)
    height, width = 5, 9
    feature = torch.zeros(1, 256, height, width, device=device)
    feature[0, 0] = (torch.arange(width, device=device) + .5)[None] / width
    feature[0, 1] = (torch.arange(height, device=device) + .5)[:, None] / height
    feature[0, 2] = 2.
    boxes = torch.tensor([[[.5, .5, .6, .4], [.02, .03, .8, .7], [0., 0., .6, .8]]], device=device)
    query = torch.randn(1, 3, 256, device=device)
    with torch.no_grad():
        module.p3_proj.weight.zero_()
        for channel in range(3):
            module.p3_proj.weight[channel, channel, 0, 0] = 1.
        _, diag = module(feature, query, torch.logit(boxes), return_diagnostics=True)
    # Deliberately independent of grid_offsets/meshgrid and the implementation's
    # sampling operation: affine pixel-center ramps interpolate to clamped u/v.
    centers = (-.4, -.2, 0., .2, .4)
    expected_uv = torch.tensor([[[[cx + tx * w, cy + ty * h] for ty in centers for tx in centers]
                                 for cx, cy, w, h in boxes[0].tolist()]], device=device)
    grid = compare(diag['grid'], expected_uv * 2 - 1)
    expected_valid = ((expected_uv >= 0) & (expected_uv <= 1)).all(-1)
    require(torch.equal(diag['valid_points'], expected_valid), 'Geometry validity differs from analytic coordinates')
    expected_sample = torch.zeros(1, 3, 25, 64, device=device)
    expected_sample[..., 0] = expected_uv[..., 0].clamp(.5 / width, 1 - .5 / width)
    expected_sample[..., 1] = expected_uv[..., 1].clamp(.5 / height, 1 - .5 / height)
    expected_sample[..., 2] = 2.
    samples = compare(diag['sampled'], expected_sample)
    expected_mean = torch.stack([expected_sample[0, i, mask].mean(0)
                                 for i, mask in enumerate(expected_valid[0])])[None]
    region_mean = compare(diag['region_mean'], expected_mean)
    expected_centered = torch.where(expected_valid[..., None], expected_sample - expected_mean[..., None, :], 0.)
    centered = compare(diag['centered'], expected_centered)
    require(torch.count_nonzero(diag['centered'][~expected_valid]) == 0, 'Invalid centered sample not exactly zero')
    require(torch.count_nonzero(diag['attention_weights'][..., 1:].permute(0, 1, 3, 2)[~expected_valid]) == 0,
            'Invalid points retained real attention mass')
    return dict(status='PASSED',feature_shape=[height, width],grid=grid,samples=samples,
                region_mean=region_mean,centered=centered,valid_counts=expected_valid.sum(-1).tolist(),
                order='row-major y then x; last dimension (x,y)',
                reference='analytic pixel-center x/y ramps with border extension, no grid_sample reference call')


def module_checks(device='cpu'):
    torch.manual_seed(42)
    module = RegionCenteredSupportQuery().to(device)
    require(sum(p.numel() for p in module.parameters()) == 58370, 'RCS-Q parameter count')
    require(module.query_norm.eps == 1e-5, 'LN eps')
    q = torch.randn(2, 7, 256, device=device)
    p3 = torch.randn(2, 256, 9, 13, device=device, requires_grad=True)
    boxes = torch.tensor([.5, .5, .65, .6], device=device).expand(2, 7, 4).clone()
    logits = torch.logit(boxes).requires_grad_()
    valid = torch.ones(2, 7, dtype=torch.bool, device=device)
    before = [x.detach().clone() for x in (p3, q, logits)]
    out, diag = module(p3, q, logits, valid, return_diagnostics=True)
    require(torch.equal(out, q), 'Zero output projection must exactly preserve query')
    compare(diag['attention_weights'].sum(-1), torch.ones(2, 7, 2, device=device))
    out.square().mean().backward()
    first = grad_norms(module)
    require(first['out_proj.weight'] > 0 and all(v == 0 for n, v in first.items() if n != 'out_proj.weight'),
            'First gradient reachability wrong')
    require(logits.grad is None, 'Sampling boxes not detached')
    require(all(torch.equal(a, b) for a, b in zip(before, (p3, q, logits))), 'Input mutation')
    # Isolated diagnostic update. Never saved as formal initialization.
    with torch.no_grad():
        module.out_proj.weight.add_(module.out_proj.weight.grad, alpha=-.1)
    module.zero_grad(set_to_none=True); p3.grad = None
    module(p3, q, logits, valid).square().mean().backward()
    second = grad_norms(module)
    require(all(v is not None and v > 0 for v in second.values()), 'Activated RCS-Q lost internal gradients')
    require(p3.grad is not None and p3.grad.norm() > 0, 'P3 gradient missing')
    with torch.no_grad():
        module.out_proj.weight.normal_(std=.02)
    with torch.no_grad():
        _, d = module(p3, q, logits, valid, return_diagnostics=True)
        require(d['delta'].norm() > 0, 'Nonuniform evidence inactive')
        constant = torch.ones_like(p3) * torch.randn(2, 256, 1, 1, device=device)
        constant_out = module(constant, q, logits, valid)
        constant_error = compare(constant_out, q)
        invalid = valid.clone(); invalid[:, :3] = False
        inv_out, inv = module(p3, q, logits, invalid, return_diagnostics=True)
        require(torch.equal(inv_out[:, :3], q[:, :3]), 'Padding/all-invalid query was corrected')
        require(torch.equal(inv['attention_weights'][:, :3, :, 0], torch.ones(2, 3, 2, device=device)), 'Null did not take all mass')
        # Saturated width=0 is a legal sigmoid result from an infinite input logit.
        degenerate = logits.detach().clone(); degenerate[..., 2] = -torch.inf
        require(torch.equal(module(p3, q, degenerate, valid), q), 'Zero-area region corrected')
        edge = boxes.clone(); edge[..., :2] = .02
        edge_out, edge_diag = module(p3, q, torch.logit(edge), valid, return_diagnostics=True)
        counts = edge_diag['valid_points'].sum(-1)
        require(((counts > 0) & (counts < 25)).all(), 'Partially outside points not masked')
        require(torch.isfinite(edge_out).all(), 'Edge correction nonfinite')
        weight = module.null_logit.weight.clone(); bias = module.null_logit.bias.clone()
        module.null_logit.weight.zero_(); module.null_logit.bias.fill_(-20.)
        _, low = module(p3, q, logits, valid, return_diagnostics=True)
        module.null_logit.bias.fill_(20.)
        _, high = module(p3, q, logits, valid, return_diagnostics=True)
        require(high['delta'].norm() < low['delta'].norm() * .01, 'Null competition has no amplitude effect')
        module.null_logit.weight.copy_(weight); module.null_logit.bias.copy_(bias)
        # Zero real logits, 25 valid slots -> null exactly one half, NOT unit real mass.
        null_fixture = deepcopy(module); null_fixture.q_proj.weight.zero_()
        _, half = null_fixture(p3, q, logits, valid, return_diagnostics=True)
        compare(half['attention_weights'][..., 0], torch.full((2, 7, 2), .5, device=device))
        # A concatenated DN prefix cannot change normal-query output in this local operator.
        prefix_q = torch.randn(2, 11, 256, device=device)
        prefix_b = torch.zeros(2, 11, 4, device=device)
        base_out = module(p3, q, logits, valid)
        joined = module(p3, torch.cat((prefix_q, q), 1), torch.cat((prefix_b, logits), 1),
                        torch.cat((torch.ones(2, 11, dtype=torch.bool, device=device), valid), 1))
        independence = compare(base_out, joined[:, 11:])
        separate = torch.cat([module(p3, q[:, i:i+1], logits[:, i:i+1], valid[:, i:i+1]) for i in range(7)], 1)
        compare(base_out, separate)
        for length in (1, 19, 300):
            module(p3, torch.randn(2, length, 256, device=device), torch.zeros(2, length, 4, device=device),
                   torch.ones(2, length, dtype=torch.bool, device=device))
    rejected = []
    for which in range(3):
        args = [p3.detach().clone(), q.clone(), logits.detach().clone(), valid]
        args[which].reshape(-1)[0] = torch.nan
        try:
            module(*args)
        except (ValueError, RuntimeError, FloatingPointError):
            rejected.append(which)
    require(rejected == [0, 1, 2], 'True NaN inputs must fail diagnosis')
    return dict(status='PASSED',parameters=58370,first_gradient_norms=first,
                after_diagnostic_update_gradient_norms=second,diagnostic_updates=1,formal_updates=0,
                constant_region=constant_error,partial_valid_counts=counts.tolist(),
                null_suppression_ratio=float(high['delta'].norm()/low['delta'].norm()),
                query_independence=independence,NaN_rejections=rejected,non_square_feature=[9,13],
                geometry=geometry_checks(device))


def dn_checks(device='cpu'):
    # PyTorch 2.1.2 exposes this dispatch mode. Capture the actual two CDN scatter
    # writes, independently of the production modulo rule or an embedding witness.
    from torch.utils._python_dispatch import TorchDispatchMode
    class ScatterCapture(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.writes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.index_put_.default and args[0].ndim == 3:
                self.writes.append(dict(shape=tuple(args[0].shape), indices=tuple(t.detach().clone() for t in args[1])))
            return func(*args, **(kwargs or {}))

    rows = []
    for groups in ((0,0), (0,3), (1,3), (2,7)):
        batch = targets(synthetic_batch(device, groups))
        # Nonzero unique class embedding is a test witness for actual scatter occupancy.
        embedding = torch.arange(1,257,device=device).float()[None]
        with ScatterCapture() as intercepted:
            de, db, mask, meta = get_cdn_group(batch, 1, 300, embedding, training=True)
        count = 0 if meta is None else meta['dn_num_split'][0]
        normal = torch.randn(2,300,256,device=device)
        query = normal if de is None else torch.cat((de, normal), dim=1)
        valid = build_query_valid(query,batch,meta,300)
        require(valid[:,count:].all(), 'Normal validity changed')
        if meta:
            occupied = (de != 0).any(-1)  # Test witness only; production uses metadata.
            require(torch.equal(valid[:,:count], occupied), 'DN validity differs from actual generator scatter')
            require(len(intercepted.writes) == 2 and
                    [w['shape'] for w in intercepted.writes] == [tuple(de.shape), tuple(db.shape)],
                    'Did not intercept the actual CDN class and noisy-box scatter writes')
            for write in intercepted.writes:
                scatter = torch.zeros_like(occupied)
                scatter[write['indices']] = True
                require(torch.equal(valid[:,:count], scatter), 'DN validity differs from intercepted scatter indices')
            box_indices = intercepted.writes[1]['indices']
            noisy_boxes = db[box_indices].sigmoid()
            clean_boxes = batch['bboxes'].repeat(2 * meta['dn_num_group'], 1)
            require((noisy_boxes - clean_boxes).abs().max() > 1e-4, 'CDN noise unexpectedly absent')
            module=RegionCenteredSupportQuery().to(device)
            with torch.no_grad(): module.out_proj.weight.normal_(std=.02)
            p3=torch.rand(2,256,13,19,device=device)
            refs=torch.cat((db,torch.zeros(2,300,4,device=device)),1)
            out=module(p3,query,refs,valid)
            require(torch.equal(out[:,:count][~occupied],query[:,:count][~occupied]),'DN padding corrected')
            require((out[:,:count] - query[:,:count])[occupied].norm()>1e-5,'Valid noisy DN never uses RCS-Q')
        else:
            require(not intercepted.writes, 'Empty-GT generator unexpectedly scattered DN')
        rows.append(dict(gt_groups=list(groups),DN=count,valid_slots=valid.sum(1).tolist(),scatter_exact=True,
                         actual_scatter_writes=len(intercepted.writes)))
    require(build_query_valid(torch.zeros(2,300,256,device=device),None,None,300).all(),'No DN case')
    return dict(status='PASSED',cases=rows,scatter_witness='intercepted actual aten.index_put_ indices + independent nonzero embedding witness')


def optimizer_check(model):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args=SimpleNamespace(warmup_bias_lr=.1,lr0=.0005,weight_decay=.0001)
    optimizer=trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    require(len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad},
            'Optimizer missing or duplicate registered parameters')
    rows=[]
    for name,p in model.named_parameters():
        if '.rcsq.' in name:
            group=next(g for g in optimizer.param_groups if any(p is v for v in g['params']))
            rows.append(dict(name=name,numel=p.numel(),weight_decay=group['weight_decay'],lr=group['lr']))
    require(sum(r['numel'] for r in rows)==58370,'Optimizer RCS-Q coverage')
    return optimizer,dict(status='PASSED',all_registered_parameters_exactly_once=True,rcsq=rows)


def reload_checks(model, device):
    rows={}
    for nonzero in (False,True):
        m=deepcopy(model).to(device).eval()
        if nonzero:
            with torch.no_grad(): m.model[-1].rcsq.out_proj.weight.normal_(std=.02)
        state=deepcopy(m.state_dict()); other=deepcopy(model).to(device)
        other.load_state_dict(state,strict=True)
        require(all(torch.equal(v,other.state_dict()[k]) for k,v in state.items()),'state_dict reload changes state')
        buffer=BytesIO();torch.save(dict(model=m,ema=ModelEMA(m).ema,epoch=1 if nonzero else -1),buffer);buffer.seek(0)
        restored=torch_load(buffer,map_location=device)
        for key in ('model','ema'):
            require(all(torch.equal(v,restored[key].state_dict()[k]) for k,v in state.items()),key+' checkpoint changed state')
            restored[key].eval()
        ema=ModelEMA(m)
        expected_ema=deepcopy(ema.ema.model[-1].rcsq.state_dict())
        decay=ema.decay(ema.updates+1)
        # Native EMA performs two rounded FP32 operations even when source and
        # EMA start identical. Test its actual update law, not identity to source.
        for key,value in expected_ema.items():
            value *= decay
            value += (1-decay)*m.model[-1].rcsq.state_dict()[key].detach()
        ema.update(m)
        actual_ema=ema.ema.model[-1].rcsq.state_dict()
        require(all(torch.equal(value,actual_ema[key]) for key,value in expected_ema.items()),'EMA changed the native RCS-Q update law')
        ema_comparison=compare(ema.ema.model[-1].rcsq.out_proj.weight,m.model[-1].rcsq.out_proj.weight)
        if nonzero:
            require(ema.ema.model[-1].rcsq.out_proj.weight.norm()>0,'EMA reset learned RCS-Q')
        x=torch.rand(1,3,160,192,device=device)
        with torch.no_grad(): comparison=compare_tree(m(x),restored['model'](x))
        rows[str(nonzero)]=dict(exact_state=True,checkpoint=True,EMA=True,EMA_native_update_exact=True,
                                EMA_source_rounding=ema_comparison,eval=comparison,
                                out_norm=float(m.model[-1].rcsq.out_proj.weight.norm()))
    return rows


def initial_equivalence(parent, model, device):
    parent=deepcopy(parent).to(device).eval(); model=deepcopy(model).to(device).eval()
    x=torch.rand(1,3,160,192,device=device)
    with torch.no_grad():
        a=parent(x);b=model(x)
        require(b[0].shape==(1,300,5),'Normal query output changed')
        evaluation=compare_tree(a,b)
        parent.model[-1].export=True;model.model[-1].export=True
        export=compare_tree(parent(x),model(x))
    parent.model[-1].export=False;model.model[-1].export=False
    parent.train();model.train()
    batch=synthetic_batch(device); batch_targets=targets(batch)
    torch.manual_seed(123)
    with torch.no_grad(): a=parent.predict(batch['img'],batch=batch_targets)
    torch.manual_seed(123)
    with torch.no_grad(): b=model.predict(batch['img'],batch=batch_targets)
    training=compare_tree(a,b)
    return dict(status='PASSED',eval=evaluation,export=export,train_DN=training,
                total_queries=b[0].shape[2],normal_queries=300,independent_copies=True,RNG_reset=True)


def native_loss_checks(model,device,amp=False):
    m=deepcopy(model).to(device).train();m.nc=1
    batch=synthetic_batch(device)
    opt,groups=optimizer_check(m)
    with (torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext()):
        pred=m.predict(batch['img'],batch=targets(batch));loss=m.loss(batch,preds=pred)[0]
    loss.backward()
    require(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),
            'Native loss/backward nonfinite')
    norms=grad_norms(m.model[-1].rcsq)
    require(norms['out_proj.weight']>0,'Post-detach integration lost first output gradient')
    require(all(v==0 for n,v in norms.items() if n!='out_proj.weight'),'Zero-output upstream gradient unexpected')
    first_loss=float(loss); del pred,loss
    # Controlled nonzero projection activates the branch; no optimizer step performed here.
    with torch.no_grad():m.model[-1].rcsq.out_proj.weight.normal_(std=.01)
    opt.zero_grad(set_to_none=True)
    with (torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext()): loss=m.loss(batch)[0]
    loss.backward();active=grad_norms(m.model[-1].rcsq)
    require(torch.isfinite(loss) and all(v is not None and v>0 for v in active.values()),'Native active branch gradient missing')
    require(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),'Active native backward nonfinite')
    return dict(status='PASSED',device=device,AMP=amp,batch=2,imgsz=[160,192],loss=first_loss,
                active_loss=float(loss),first_gradients=norms,active_gradients=active,optimizer=groups,
                optimizer_steps=0,formal_updates=0)


def activate(model):
    with torch.no_grad(),torch.random.fork_rng(devices=[]):
        torch.manual_seed(512)
        model.model[-1].rcsq.out_proj.weight.normal_(std=.01)
        for m in model.modules():
            if type(m).__name__=='LIFDown':
                m.O_proj.weight.normal_(std=.01)
                m.bn.running_mean.copy_(torch.linspace(-.3,.4,256,device=m.bn.weight.device))
                m.bn.running_var.copy_(torch.linspace(.4,1.8,256,device=m.bn.weight.device))
            if type(m).__name__=='CrackBoundaryRefinement':
                m.offset_out.weight.normal_(std=.02);m.offset_out.bias.fill_(.15)


def fusion_checks(model,device,folder):
    source=deepcopy(model).float().eval().to(device);activate(source)
    fused=deepcopy(source).fuse(verbose=False)
    require(torch.equal(source.model[-1].rcsq.out_proj.weight,fused.model[-1].rcsq.out_proj.weight),'Fuse changed RCS-Q')
    # A fixed local fixture is independent of earlier DN noise and diagnostic probes.
    x=torch.rand(1,3,160,192,device=device,generator=torch.Generator(device=device).manual_seed(42));rows={}
    def parent_factory():
        base=build('baseline',nc=1).eval()
        base.load_state_dict({k:source.state_dict()[k].cpu() for k in base.state_dict()},strict=True)
        return base
    for precision in (('fp32','amp','half') if device=='cuda' else ('fp32',)):
        a=deepcopy(source);b=deepcopy(fused);image=x
        if precision=='half':a.half();b.half();image=x.half()
        if hasattr(source.model[-1],'cbr'):
            row=fusion_protocol(a,b,image,folder/precision,device,precision,parent_factory,parent_source=source)
            require(fusion_accepted(row,device,precision),'Fusion evidence blocked')
        else:
            from rcsq_v1_single_fusion import single_fusion
            row=single_fusion(a,b,image,device,precision,folder/precision,parent_factory,source)
        rows[precision]=row
        del a,b
    return dict(status='PASSED',nonzero_rcsq=True,nonzero_original_branches=True,modes=rows,fixture_seed=42,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                unfused_parameters=sum(p.numel() for p in source.parameters()),
                fused_parameters=sum(p.numel() for p in fused.parameters()))


def complexity(models,device='cpu'):
    import thop
    rows={}; x=torch.zeros(1,3,640,640,device=device)
    for kind,source in models.items():
        rows[kind]={}
        for fused in (False,True):
            m=deepcopy(source).eval().to(device)
            if fused:m.fuse(verbose=False)
            parameters=sum(p.numel() for p in m.parameters())
            with torch.no_grad(): macs,_=thop.profile(m,inputs=(x,),verbose=False)
            # Functional FP32 RCS-Q projections are invisible to module-level THOP hooks.
            custom_macs=(256*64*80*80 + 300*256*64 + 2*300*25*64*64 +
                         300*256*2 + 2*300*25*64 + 300*64*256) if hasattr(m.model[-1],'rcsq') else 0
            with torch.no_grad():
                m(x)
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
                for _ in range(3):m(x)
                if device=='cuda':torch.cuda.synchronize()
            rows[kind]['fused' if fused else 'unfused']=dict(parameters=parameters,thop_gflops=2*macs/1e9,
                rcsq_functional_projection_attention_gflops=2*custom_macs/1e9,
                thop_plus_rcsq_major_gflops=2*(macs+custom_macs)/1e9,
                milliseconds_per_image=(time.perf_counter()-start)*1000/3,iterations=3,warmup=1,
                batch=1,imgsz=640,device=device)
            del m
    return dict(models=rows,tool='THOP '+getattr(thop,'__version__','unknown'),
                uncovered=['grid_sample','centering/reductions','functional normalization','softmax','elementwise and indexing'],
                note='THOP does not count functional RCSQ projections; analytic major MACs are separate. Timing is local synthetic inference, not training capacity or a speed claim.')
