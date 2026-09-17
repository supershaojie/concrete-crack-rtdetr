"""Bounded SRE mathematics, wiring and disposable synthetic detection checks.

No epochs, training-data loader, val split or test split are run here. The
separate preflight owns the required real-data B16/640/native AMP capacity gate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F
import ultralytics
from ultralytics.nn.modules import SRE, SRERepC3, RepC3, LIFDown, RTDETRDecoderCBR
from ultralytics.nn.modules.sre import count_sre_projection_macs_partial
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import ModelEMA, TORCH_2_4
from ultralytics.utils.patches import torch_load
from c19_lif_v1_probe import capture, targets

CONFIGS = {
    "cbr_lif_sre_v1": ("rtdetr-resnet18-lite-cbr-lif-down.yaml", "rtdetr-resnet18-lite-cbr-lif-sre-v1.yaml", 20149765, 19944965),
    "sre_v1": ("rtdetr-resnet18-lite.yaml", "rtdetr-resnet18-lite-sre-v1.yaml", 20082772, 19877716),
}
MODEL_DIR = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, name, atol=2e-6, rtol=2e-5):
    require(torch.isfinite(a).all() and torch.isfinite(b).all(), name + ": nonfinite")
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)
    return float((a - b).abs().max()) if a.numel() else 0.0


def reference(q, v):
    """Independent per-pixel reference: enumerate absolute neighbor coordinates.

    No production offsets, slicing, shifts, pad, roll or aggregate are reused.
    Stack/add builds a differentiable graph for the gradient oracle.
    """
    b, c, h, w = v.shape
    rows = []
    for y in range(h):
        columns = []
        for x in range(w):
            num = v[:, :, y, x] * 0
            support = v[:, :1, y, x] * 0
            for j in range(max(0, y - 2), min(h, y + 3)):
                for k in range(max(0, x - 2), min(w, x + 3)):
                    if j == y and k == x:
                        continue
                    cosine = (q[:, :, y, x] * q[:, :, j, k]).sum(1, keepdim=True).clamp(-1, 1)
                    weight = torch.maximum(cosine, torch.zeros_like(cosine)) ** 2
                    num = num + weight * torch.maximum(v[:, :, j, k] - v[:, :, y, x], torch.zeros_like(num))
                    support = support + weight
            columns.append(num / (1 + support))
        rows.append(torch.stack(columns, -1))
    return torch.stack(rows, -2)


def reference_forward(m, x):
    z = F.conv2d(x.float(), m.W_d.weight.float())
    z = F.group_norm(z, 4, m.GN.weight.float(), m.GN.bias.float(), 1e-5)
    q = F.normalize(F.conv2d(z, m.W_q.weight.float()), dim=1, eps=1e-6)
    v = F.softplus(z, beta=1, threshold=20)
    return x + F.conv2d(reference(q, v), m.W_o.weight.float()).to(x.dtype)


def math_checks(device):
    report = {"status": "PASSED", "shapes": {}, "hand_cases": {}}
    require(len(SRE.OFFSETS) == len(set(SRE.OFFSETS)) == 24 and (0, 0) not in SRE.OFFSETS, "24 offsets")
    require(set(SRE.OFFSETS) == {(y, x) for y in range(-2, 3) for x in range(-2, 3)} - {(0, 0)}, "Full neighborhood")
    torch.manual_seed(310)
    for h, w in [(1, 1), (1, 7), (7, 1), (3, 5), (5, 3), (7, 7)]:
        q = F.normalize(torch.randn(2, 8, h, w, device=device), dim=1, eps=1e-6)
        v = torch.rand(2, 32, h, w, device=device) * 3
        d = SRE.aggregate(q, v)
        error = close(d, reference(q, v), f"reference {h}x{w}")
        maximum, bound = torch.zeros_like(v), torch.zeros_like(v)
        counts = []
        for y in range(h):
            for x in range(w):
                neighbors = [(j, k) for j in range(max(0, y-2), min(h, y+3)) for k in range(max(0, x-2), min(w, x+3)) if (j,k)!=(y,x)]
                counts.append(len(neighbors)); s = torch.zeros(2, 1, device=device)
                m = torch.zeros(2, 32, device=device)
                for j,k in neighbors:
                    weight = (q[:,:,y,x]*q[:,:,j,k]).sum(1,keepdim=True).clamp(-1,1).relu().square()
                    require(((weight>=0)&(weight<=1)).all(), "weight bounds")
                    s = s + weight; m = torch.maximum(m,(v[:,:,j,k]-v[:,:,y,x]).relu())
                maximum[:,:,y,x] = m; bound[:,:,y,x] = s/(1+s)*m
        require((d >= 0).all() and (d <= bound+1e-6).all() and (bound <= maximum+1e-6).all(), "D bounds")
        report["shapes"][f"{h}x{w}"] = dict(max_abs_error=error,valid_neighbors_min=min(counts),valid_neighbors_max=max(counts))
    q = torch.tensor([[[[1.,1.]],[[0.,0.]]]],device=device)
    v = torch.tensor([[[[1.,3.]],[[5.,2.]]]],device=device)
    close(SRE.aggregate(q,v),torch.tensor([[[[1.,0.]],[[0.,1.5]]]],device=device),"one-sided, channel directional")
    report["hand_cases"]["single_sided_channel_directional"] = True
    negative=q.clone();negative[:,:,:,1]*=-1
    for name,qq in [("negative_similarity",negative),("zero_descriptor",q*0)]:
        require(torch.count_nonzero(SRE.aggregate(qq,v))==0,name)
        report["hand_cases"][name]=True
    require(torch.count_nonzero(SRE.aggregate(q,torch.ones_like(v)))==0,"equal values")
    report["hand_cases"]["equal_values"] = True
    orthogonal=q.clone();orthogonal[0,:,0,1]=torch.tensor([0.,1.],device=device)
    require(torch.count_nonzero(SRE.aggregate(orthogonal,v))==0,"orthogonal similarity")
    corner_q=torch.ones(1,1,3,3,device=device);corner_v=torch.ones(1,1,3,3,device=device);corner_v[...,0,0]=0
    close(SRE.aggregate(corner_q,corner_v)[0,0,0,0],torch.tensor(8/9,device=device),"corner has eight votes, no repeated padding")
    report["hand_cases"]["orthogonal_similarity_and_exact_corner_count"] = True
    weak=q.clone();weak[0,:,0,1]=torch.tensor([.5,3**.5/2],device=device)
    close(SRE.aggregate(weak,v)[0,0,0,0],torch.tensor(.4,device=device),"weak support 0.25*2/1.25")
    require(SRE.aggregate(weak,v)[0,0,0,0]<SRE.aggregate(q,v)[0,0,0,0],"fixed V descriptor sensitivity")
    changed=v.clone();changed[0,0,0,1]=5
    close(SRE.aggregate(q,changed)[0,0,0,0],torch.tensor(2.,device=device),"fixed Q stronger neighbor")
    report["hand_cases"]["fixed_Q_and_fixed_V_sensitivity"] = True
    q3=torch.ones(1,1,1,3,device=device);v3=torch.tensor([[[[0.,1.,2.]]]],device=device)
    close(SRE.aggregate(q3,v3)[0,0,0,1],torch.tensor(1/3,device=device),"positive differences before sum")
    require(SRE.aggregate(q3,v3)[0,0,0,2]==0,"local max")
    report["hand_cases"]["positive_difference_before_sum_and_fixed_one_denominator"] = True
    # A wraparound implementation would communicate these endpoints; valid
    # boundaries must not. Descriptor zero separates all intermediate pixels.
    edgeq=torch.zeros(1,1,1,7,device=device);edgeq[...,0]=1;edgeq[...,-1]=1
    edgev=torch.zeros(1,1,1,7,device=device);edgev[...,-1]=9
    require(torch.count_nonzero(SRE.aggregate(edgeq,edgev))==0,"no wraparound")
    report["hand_cases"]["no_wraparound_or_duplicate_edge_votes"] = True
    m=SRE().to(device);n=deepcopy(m)
    with torch.no_grad():m.W_o.weight.normal_(std=.015);n.load_state_dict(m.state_dict())
    x=torch.randn(2,256,3,5,device=device,requires_grad=True);y=x.detach().clone().requires_grad_(True)
    a=m(x);b=reference_forward(n,y);report["nonzero_output_error"]=close(a,b,"nonzero output")
    objective=torch.randn_like(a);(a*objective).sum().backward();(b*objective).sum().backward()
    report["input_gradient_error"]=close(x.grad,y.grad,"input gradient",2e-5,2e-4)
    report["parameter_gradient_error"]={k:close(p.grad,dict(n.named_parameters())[k].grad,k,2e-5,2e-4) for k,p in m.named_parameters()}
    require(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters()),"nonzero all core gradients")
    # A tensor-free helper counts each functional projection once and supports
    # both THOP counter representations. The reported count stays PARTIAL.
    hook=SRE();arg=torch.empty(1,256,80,80)
    hook.total_ops=0;count_sre_projection_macs_partial(hook,(arg,),None);require(hook.total_ops==106496000,"THOP int")
    hook.total_ops=torch.tensor(0.);count_sre_projection_macs_partial(hook,(arg,),None);require(int(hook.total_ops)==106496000,"THOP tensor")
    valid_pairs=sum(max(0,80-abs(dy))*max(0,80-abs(dx)) for dy,dx in SRE.OFFSETS)
    report["cost"]={"status":"PARTIAL","parameters":sum(p.numel() for p in hook.parameters()),"projection_MAC_B1_640":106496000,"projection_GFLOPs_2_per_MAC":.212992,
        "valid_directed_neighbor_pairs_B1_80x80":valid_pairs,
        "neighborhood_dot_multiplications":valid_pairs*8,"neighborhood_dot_additions":valid_pairs*7,
        "weighted_positive_difference_multiplies":valid_pairs*32,
        "positive_difference_subtractions_and_relu_each":valid_pairs*32,
        "omitted_from_projection_MACs":"dot reductions, clamp/ReLU/square, support and numerator accumulations, final division, GN, descriptor normalization, Softplus, residual add; no whole-network GFLOPs claim",
        "memory_note":"per-offset autograd intermediates remain; B16 memory must be measured independently"}
    return report


def build(filename):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model=RTDETRDetectionModel(str(MODEL_DIR/filename),nc=1,verbose=False)
    model.nc=1
    return model


def rng_snapshot():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:torch.cuda.set_rng_state_all(state[1])


@contextmanager
def strict_fp32():
    matmul=torch.backends.cuda.matmul.allow_tf32;cudnn=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=matmul;torch.backends.cudnn.allow_tf32=cudnn


def synthetic_batch(device):
    generator=torch.Generator().manual_seed(123)
    return dict(img=torch.rand(2,3,160,160,generator=generator).to(device),
        cls=torch.zeros(3,1,device=device),batch_idx=torch.tensor([0,1,1],device=device),
        bboxes=torch.tensor([[.40,.45,.24,.30],[.65,.55,.18,.25],[.25,.3,.14,.19]],device=device))


def structure(variant,device):
    parent_file,filename,parent_count,parent_fused_count=CONFIGS[variant]
    parent=build(parent_file);model=build(filename)
    require(len(model.model)==27,"node count")
    wrapper=model.model[19]
    require(isinstance(wrapper,SRERepC3) and len(wrapper.m)==3 and wrapper.f==-1,"wrapper/repeats/from")
    require(sum(isinstance(m,SRE) for m in model.modules())==1,"one SRE")
    require(not isinstance(model.model[17],SRERepC3),"original model.17")
    head=model.model[26]
    require(head.f==[19,22,25] and head.num_queries==300 and len(head.decoder.layers)==3 and head.decoder.eval_idx==2,"head topology")
    if variant=="cbr_lif_sre_v1":
        require(isinstance(model.model[20],LIFDown) and isinstance(head,RTDETRDecoderCBR),"original CBR/LIF")
        require(head.cbr.rho==head.cbr.normal_fraction==.10,"CBR constants")
    else:
        require(not isinstance(model.model[20],LIFDown) and not isinstance(head,RTDETRDecoderCBR),"C2 ablation")
    ps=parent.state_dict();ms=model.state_dict()
    require(all(k in ms and torch.equal(v,ms[k]) for k,v in ps.items()),"parent constructor common RNG/state")
    new=sorted(set(ms)-set(ps));require(new and all(k.startswith('model.19.sre.') for k in new),"state path")
    require(sum(p.numel() for p in model.parameters())==parent_count+16704,"unfused count")
    fused_count=sum(p.numel() for p in deepcopy(model).eval().fuse(verbose=False).parameters())
    require(fused_count==parent_fused_count+16704,"fused count")
    before=rng_snapshot();SRE();after=rng_snapshot();require(torch.equal(before[0],after[0]),"CPU RNG isolation")
    require(before[1] is None or all(torch.equal(a,b) for a,b in zip(before[1],after[1])),"CUDA RNG untouched")
    parent.to(device);model.to(device);batch=synthetic_batch(device)
    taps={};handles=[]
    def hook(name):
        def save(m,a,v):taps[name]=v.detach().clone()
        return save
    def pre(name):
        def save(m,a):taps[name]=a[0].detach().clone()
        return save
    handles.extend([model.model[19].cv3.register_forward_hook(hook('complete_RepC3')),
        model.model[19].sre.register_forward_pre_hook(pre('SRE_input')),
        model.model[19].register_forward_hook(hook('enriched')),
        model.model[20].register_forward_pre_hook(pre('downsample_input')),
        head.register_forward_pre_hook(lambda m,a:taps.update(decoder_input=a[0][0].detach().clone()))])
    try:
        parent.train();model.train();rng=rng_snapshot()
        with torch.no_grad():
            pp=parent.predict(batch['img'],batch=targets(batch));pl=parent.loss(batch,preds=pp)[0]
            restore_rng(rng);mp=model.predict(batch['img'],batch=targets(batch));ml=model.loss(batch,preds=mp)[0]
        require(all(torch.equal(pp[i],mp[i]) for i in range(4)),"initial train outputs including DN")
        require(torch.equal(pl,ml),"initial true detection loss")
        require(pp[-1] is not None and pp[-1]['dn_num_split'][0]>0,"DN covered")
        require(torch.equal(taps['complete_RepC3'],taps['SRE_input']),"SRE after full RepC3")
        require(torch.equal(taps['enriched'],taps['downsample_input']) and torch.equal(taps['enriched'],taps['decoder_input']),"output consumed both paths")
        require(all(torch.equal(v,model.state_dict()[k]) for k,v in parent.state_dict().items()),"BN sync after train forward")
    finally:
        for h in handles:h.remove()
    result=dict(status="PASSED",node_count=27,internal_RepConv=3,wrapper_count=1,from_decoder=head.f,
        common_state_tensors=len(ps),new_state_keys=new,unfused_parameters=parent_count+16704,fused_parameters=fused_count,
        additional_parameters=16704,training_initial_outputs_exact=True,training_initial_loss=float(ml),
        synchronized_BN_CPU_CUDA_RNG_GT_DN=True,DN_split=mp[-1]['dn_num_split'],
        hook_complete_RepC3_to_SRE_to_downsample_and_decoder=True)
    del parent,pp,mp,pl,ml,taps;gc.collect()
    return model,result


def optimizer_for(model):
    trainer=RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args=SimpleNamespace(warmup_bias_lr=.1,lr0=.0005,weight_decay=.0001)
    optimizer=trainer.build_optimizer(model,name='AdamW',lr=.0005,momentum=.937,decay=.0001)
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    require(len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad},"optimizer exactly once")
    return optimizer


def detection_updates(model,device,amp=False):
    model.train();batch=synthetic_batch(device);opt=optimizer_for(model)
    scaler=(torch.amp.GradScaler('cuda',enabled=amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=amp))
    rows=[];updates=0;budget=12 if amp else 2
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for attempt in range(budget):
        opt.zero_grad(set_to_none=True);torch.manual_seed(815+attempt)
        with torch.autocast(device_type=device,enabled=amp):
            preds=model.predict(batch['img'],batch=targets(batch));loss=model.loss(batch,preds=preds)[0]
        require(torch.isfinite(loss),"finite detection loss")
        require(preds[-1] is not None and preds[-1]['dn_num_split'][0]>0,"DN in loss")
        scaler.scale(loss).backward();scaler.unscale_(opt)
        finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        raw_grads={name:float(p.grad.float().norm()) if p.grad is not None else None for name,p in model.model[19].sre.named_parameters()}
        nonfinite_sre=[name for name,p in model.model[19].sre.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        grads={name:None if name in nonfinite_sre else value for name,value in raw_grads.items()}
        old=model.model[19].sre.W_o.weight.detach().clone();scale=float(scaler.get_scale())
        if finite:
            require(grads['W_o.weight']>0,"W_o gradient startup")
            if updates==0:
                require(all(grads[name]==0 for name in grads if name!='W_o.weight'),"expected zero upstream first gradient")
            else:require(all(value is not None and value>0 for value in grads.values()),"upstream active after real update")
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        elif not amp:raise AssertionError("FP32 nonfinite gradients")
        scaler.step(opt);scaler.update();effective=not torch.equal(old,model.model[19].sre.W_o.weight)
        require(effective==bool(finite),"scaler update accounting")
        if effective:updates+=1
        rows.append(dict(attempt=attempt,loss=float(loss),GT=3,DN_split=preds[-1]['dn_num_split'],gradients=grads,
            all_model_gradients_finite=bool(finite),nonfinite_SRE_gradient_names=nonfinite_sre,
            scale_before=scale,scale_after=float(scaler.get_scale()),effective_update=effective,
            W_o_norm=float(model.model[19].sre.W_o.weight.norm())))
        del preds,loss
        if updates>=2:break
    require(updates>=2,"finite update budget exhausted")
    if device=='cuda':torch.cuda.synchronize()
    return opt,scaler,dict(status="PASSED",scope="disposable synthetic B2/160 detection loss; not B16 capacity",amp=amp,
        native_initial_scale=65536. if amp else None,budget_batches=budget,effective_updates=updates,rows=rows,
        optimizer_every_trainable_once=True,seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        native_resume="owned by real-data preflight; not implied by this smoke")


def continuous_comparison(a,b):
    keys=['scale_0','scale_1','scale_2','flattened_features','candidate_scores','encoder_features']
    return {k:close(a[k],b[k],k,2e-5,2e-4) for k in keys}


def lifecycle(model,opt,scaler,device,folder):
    model.eval();learned={k:v.detach().clone() for k,v in model.model[19].sre.state_dict().items()}
    require(learned['W_o.weight'].abs().sum()>0,"learned state required")
    checkpoint=folder/'disposable_smoke.pt'
    torch.save(dict(model=deepcopy(model).cpu(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=0),checkpoint)
    reloaded=torch_load(checkpoint,map_location=device)['model'].eval()
    require(all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items()),"model reload exact")
    x=synthetic_batch(device)['img'][:1]
    with torch.no_grad():
        original,orig_trace=capture(model,x)
        restored,_=capture(reloaded,x)
    close(original[0],restored[0],"reload output",0,0)
    ema=ModelEMA(model)
    require(all(torch.equal(v,ema.ema.model[19].sre.state_dict()[k]) for k,v in learned.items()),"EMA initial copy")
    ema.update(model)
    decay=ema.decay(ema.updates)
    expected_ema={k:v*decay+v*(1-decay) for k,v in learned.items()}
    require(all(torch.equal(v,ema.ema.model[19].sre.state_dict()[k]) for k,v in expected_ema.items()),"EMA native update formula")
    with torch.no_grad():
        ema_result=ema.ema(x)
    require(torch.isfinite(ema_result[0]).all(),"EMA finite")
    fused=deepcopy(model).fuse(verbose=False)
    require(isinstance(fused.model[19].sre.GN,torch.nn.GroupNorm),"GN retained")
    require(all(torch.equal(v,fused.model[19].sre.state_dict()[k]) for k,v in learned.items()),"fusion preserves SRE")
    with torch.no_grad():
        fused_output,fused_trace=capture(fused,x)
    # Default device flags are documented independently; strict diagnostics
    # scope restores both flags even if comparison throws.
    default=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        output_max_abs_error=float((original[0]-fused_output[0]).abs().max()),
        candidate_ids_equal=torch.equal(orig_trace['candidate_indices'],fused_trace['candidate_indices']))
    with strict_fp32(),torch.no_grad():
        strict,st=capture(model,x);other,ft=capture(fused,x)
        continuous=continuous_comparison(st,ft)
        ids_equal=torch.equal(st['candidate_indices'],ft['candidate_indices'])
        ids_same_set=torch.equal(st['candidate_indices'].sort(1).values,ft['candidate_indices'].sort(1).values)
        permutation_error=None
        if ids_equal:
            final=close(strict[0],other[0],"strict native fused output",2e-5,2e-4)
            replay=None
        else:
            replay_output,replay_trace=capture(fused,x,fixed_ids=st['candidate_indices'])
            replay=close(strict[0],replay_output[0],"fixed candidate diagnostic only",2e-5,2e-4)
            final=float((strict[0]-other[0]).abs().max())
            if ids_same_set:
                # Both outputs are untouched native runs. Sorting their query
                # outputs by the captured original spatial candidate ID checks
                # set equivalence when top-k changes only the output order. No
                # candidate replay is used in this native detection-set check.
                left=st['candidate_indices'].argsort(1).to(strict[0].device)
                right=ft['candidate_indices'].argsort(1).to(other[0].device)
                aligned_left=strict[0].gather(1,left.unsqueeze(-1).expand_as(strict[0]))
                aligned_right=other[0].gather(1,right.unsqueeze(-1).expand_as(other[0]))
                permutation_error=close(aligned_left,aligned_right,"native same-set candidate permutation",2e-5,2e-4)
        require(torch.isfinite(other[0]).all(),"fused native finite")
    result=dict(status="PASSED" if ids_equal or ids_same_set else "PENDING",learned_W_o_nonzero=True,save_reload_states_exact=True,save_reload_outputs_exact=True,
        EMA_initial_copy_exact=True,EMA_update_matches_native_formula=True,fusion_learned_states_exact=True,GroupNorm_retained=True,default_precision=default,
        strict_FP32_continuous_errors=continuous,strict_FP32_native_candidates_equal=ids_equal,
        strict_FP32_native_candidate_sets_equal=ids_same_set,native_candidate_permutation_aligned_output_error=permutation_error,
        strict_FP32_native_output_error=final,fixed_candidate_replay_diagnostic_error=replay,
        native_output_equivalence="PASSED" if ids_equal else ("PASSED_NATIVE_DETECTION_SET_AFTER_ORDER_ALIGNMENT" if ids_same_set else "PENDING_DISCRETE_TOPK_DIAGNOSTIC"),
        TF32_flags_restored=True,checkpoint="temporary and discarded")
    if device=='cuda':
        half_model=deepcopy(model).half();quantized=deepcopy(half_model).float()
        with torch.no_grad():
            half_result,half_trace=capture(half_model,x.half())
            quant_result,quant_trace=capture(quantized,x.half().float())
        require(torch.isfinite(half_result[0]).all(),"explicit CUDA half finite")
        require(all(torch.equal(v.half(),half_model.model[19].sre.state_dict()[k]) for k,v in learned.items()),"half preserved quantized SRE")
        half_path=folder/'half_smoke.pt';torch.save(half_model.cpu(),half_path)
        half_loaded=torch_load(half_path,map_location='cuda').eval()
        with torch.no_grad():half_reloaded=half_loaded(x.half())
        close(half_result[0],half_reloaded[0],"same quantized half reload",0,0)
        # Native checkpoint/backend loading promotes to float before fusion,
        # then chooses inference dtype. Preserve the same quantized source.
        half_fused=deepcopy(half_loaded).float().fuse(verbose=False).half()
        require(all(torch.equal(v,half_fused.model[19].sre.state_dict()[k]) for k,v in half_loaded.model[19].sre.state_dict().items()),"half fusion preserves learned quantized SRE")
        with torch.no_grad():half_fused_result,half_fused_trace=capture(half_fused,x.half())
        require(torch.isfinite(half_fused_result[0]).all(),"explicit fused CUDA half finite")
        result['CUDA_half']=dict(status="PASSED",finite=True,quantized_reload_exact=True,
            fused_finite=True,fused_learned_SRE_preserved=True,
            fused_path="same half-quantized checkpoint -> float -> native fuse -> half",
            fused_native_candidate_ids_equal=torch.equal(half_trace['candidate_indices'],half_fused_trace['candidate_indices']),
            fused_native_final_max_abs_error=float((half_result[0].float()-half_fused_result[0]).abs().max()),
            candidate_ids_equal=torch.equal(half_trace['candidate_indices'],quant_trace['candidate_indices']),
            final_vs_same_quantized_FP32_max_abs_error=float((half_result[0].float()-quant_result[0]).abs().max()),
            continuous_features_max_abs_error={k:float((half_trace[k].float()-quant_trace[k]).abs().max()) for k in ['scale_0','scale_1','scale_2','candidate_scores']},
            note="FP16/FP32 continuous and discrete differences are diagnostic; no native equivalence claim")
        del half_model,quantized,half_loaded,half_fused
    else:result['CUDA_half']={"status":"PENDING","reason":"separate CUDA execution required; CPU half whole network is not a gate"}
    del reloaded,fused,ema;gc.collect()
    return result


def run_checks(device='cpu',variants=None,math_only=False):
    from sre_common import code_identity, source_contract
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT/'ultralytics-main'),"wrong import")
    if device=='cuda':require(torch.cuda.is_available(),"CUDA unavailable")
    report=dict(status="RUNNING",scope="local engineering; no accuracy evidence",device=device,identity=code_identity(),original_module_hashes=source_contract(),runtime=dict(python=sys.version,
        torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        ultralytics_file=ultralytics.__file__,device=device),mathematics=math_checks(device),variants={})
    if not math_only:
        selected=list(variants or CONFIGS)
        initial_states=[]
        with tempfile.TemporaryDirectory(prefix='sre-check-') as tmp:
            for variant in selected:
                print('SRE checking '+variant+' on '+device,flush=True)
                model,topology=structure(variant,device)
                initial_states.append({k:v.detach().cpu().clone() for k,v in model.model[19].sre.state_dict().items()})
                opt,scaler,fp32=detection_updates(model,device,False)
                life=lifecycle(model,opt,scaler,device,Path(tmp))
                report['variants'][variant]=dict(structure=topology,FP32_detection=fp32,lifecycle=life)
                del model,opt,scaler;gc.collect()
                if device=='cuda':
                    torch.cuda.empty_cache();amp_model=build(CONFIGS[variant][1]).to(device)
                    amp_opt,amp_scaler,amp=detection_updates(amp_model,device,True)
                    report['variants'][variant]['native_AMP_detection']=amp
                    del amp_model,amp_opt,amp_scaler;gc.collect();torch.cuda.empty_cache()
        if len(initial_states)>1:
            require(all(torch.equal(v,initial_states[1][k]) for k,v in initial_states[0].items()),"variant initial states")
            report['new_initial_states_between_variants_exact']=True
    report['status']='PASSED' if all(row['lifecycle']['status']=='PASSED' for row in report['variants'].values()) else 'PENDING'
    report['PENDING']=['Real train B16/640/native AMP memory/time and two effective updates: separate preflight gate',
        'Formal training NOT_STARTED; final test NOT_RUN; engineering results do not measure precision/recall']
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--variant',choices=['both']+list(CONFIGS),default='both')
    parser.add_argument('--math-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(min(4,os.cpu_count() or 1))
    try:report=run_checks(args.device,None if args.variant=='both' else [args.variant],args.math_only)
    except Exception as error:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(status='FAILED',error=repr(error)),indent=2),encoding='utf-8')
        raise
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],output=str(args.output)),indent=2))


if __name__=='__main__':main()
