"""PBI-v1 explicit nonzero formula/gradient, RNG, structure and dtype audits; never trains a run."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import traceback

import torch
from torch.nn import functional as F

from pbi_common import (PBI_KEYS, VARIANTS, build, build_training_model, native_rebuild, require,
                        runtime, tensor_sha256, verify_model, write_json)
from ultralytics.nn.modules import Conv, PBI, PBIConv
from ultralytics.utils.torch_utils import fuse_conv_and_bn

ATOL, RTOL = 2e-5, 2e-4


def metric(a, b):
    a,b=a.detach().float(),b.detach().float()
    return dict(raw_allclose=bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL)),
                max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


def explicit(x, weights, amp=False):
    u,v=F.conv2d(x,weights[0]),F.conv2d(x,weights[1])
    with torch.autocast(device_type=x.device.type,enabled=False):
        return (x.float()+F.conv2d(u.float()*v.float(),weights[2].float())).to(x.dtype)


def rng_check():
    before=torch.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    module=PBI()
    require(torch.equal(before,torch.get_rng_state()),'PBI consumes public CPU RNG')
    require(all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all() if cuda else [])), 'PBI consumes CUDA RNG')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        weights=[torch.nn.Conv2d(256,32,1,bias=False),torch.nn.Conv2d(256,32,1,bias=False),torch.nn.Conv2d(32,256,1,bias=False)]
        torch.nn.init.xavier_uniform_(weights[0].weight,gain=1)
        torch.nn.init.xavier_uniform_(weights[1].weight,gain=1)
        torch.nn.init.zeros_(weights[2].weight)
    require(all(torch.equal(a.weight,b.weight) for a,b in zip((module.W1,module.W2,module.Wo),weights)), 'Xavier initialization differs')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(51)
        base=Conv(128,256,1,1,None,1,1,False)
        after_base=torch.get_rng_state()
        torch.random.default_generator.manual_seed(51)
        wrapped=PBIConv(128,256,1,1,None,1,1,False,32)
        require(torch.equal(after_base,torch.get_rng_state()),'Wrapper consumes extra public RNG')
    require(all(torch.equal(v,wrapped.state_dict()[k]) for k,v in base.state_dict().items()),'Original Conv state changed')
    require(not torch.equal(module.W1.weight,module.W2.weight), 'W1/W2 identical')
    require(len({p.untyped_storage().data_ptr() for p in module.parameters()}) == 3, 'Weights share storage')
    return dict(status='PASSED',constructor_preserves_cpu_cuda_rng=True,
                original_conv_state_and_later_cpu_rng_preserved=True,xavier_gain1_exact=True,
                independent_input_projections=True,no_storage_sharing=True,new_buffers=[])


def formula_check(device):
    module=PBI().to(device)
    require(sum(p.numel() for p in module.parameters())==24576,'Parameter count')
    rows=[]
    for shape in ((1,256,80,80),(2,256,9,11),(1,256,1,1)):
        x=torch.randn(shape,device=device)
        with torch.no_grad():
            require(torch.equal(x,module(x)),'Initial PBI is not identity')
        rows.append(dict(shape=list(shape),initial_identity_exact=True))
    with torch.no_grad():
        module.Wo.weight.normal_(std=.005)
    x=torch.randn(2,256,9,11,device=device,requires_grad=True)
    x2=x.detach().clone().requires_grad_(True)
    reference=[p.detach().clone().requires_grad_(True) for p in module.parameters()]
    y=module(x); ref=explicit(x2,reference)
    output=metric(y,ref)
    require(output['raw_allclose'] and output['finite'],'Explicit nonzero formula differs')
    require(float((y-x).abs().max())>0,'Zero residual cannot validate formula')
    probe=torch.randn_like(y)
    ((y*probe).square().mean()).backward();((ref*probe).square().mean()).backward()
    grads={name:metric(p.grad,r.grad) for (name,p),r in zip(module.named_parameters(),reference)}
    grads['input']=metric(x.grad,x2.grad)
    require(all(r['raw_allclose'] and r['finite'] for r in grads.values()),'Explicit gradient differs')
    # Mathematical startup is distinct from real detection-loss lifecycle checks.
    fresh=PBI().to(device); opt=torch.optim.SGD(fresh.parameters(),lr=.01)
    steps=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss=(fresh(x.detach())-.1).square().mean();loss.backward()
        norms={n:float(p.grad.norm()) for n,p in fresh.named_parameters()}
        require(all(torch.isfinite(p.grad).all() for p in fresh.parameters()),'Nonfinite gradient')
        require(norms['Wo.weight']>0 and (all(norms[n]==0 for n in ('W1.weight','W2.weight')) if step==0
                else all(v>0 for v in norms.values())), 'Unexpected first/second step gradient')
        opt.step(); steps.append(dict(loss=float(loss.detach()),gradient_norms=norms))
    wrapped=PBIConv(128,256,1,1,None,1,1,False,32).eval().to(device)
    wrapped.pbi.load_state_dict(module.state_dict(),strict=True)
    fused=deepcopy(wrapped)
    fused.conv=fuse_conv_and_bn(fused.conv,fused.bn);delattr(fused,'bn');fused.forward=fused.forward_fuse
    calls=[]
    hook=fused.pbi.register_forward_hook(lambda *_:calls.append(1))
    try:
        image=torch.randn(2,128,9,11,device=device)
        with torch.no_grad():
            fusion=metric(wrapped(image),fused(image))
        require(fusion['raw_allclose'] and fusion['finite'] and len(calls)==1,'Fused wrapper lost PBI')
    finally:
        hook.remove()
    report=dict(device=device,status='PASSED',new_trainable_parameters=24576,shapes=rows,
                nonzero_formula=output,nonzero_gradients=grads,nonzero_residual_max=float((y-x).abs().max().detach()),
                gradient_steps=steps,nonzero_wrapper_fusion={**fusion,'pbi_calls':len(calls)})
    if device=='cuda':
        half=deepcopy(module).half(); hx=x.detach().half()
        with torch.no_grad():
            hy=half(hx);hr=explicit(hx,list(half.parameters()))
            with torch.autocast('cuda',dtype=torch.float16):
                ay=module(x.detach());ar=explicit(x.detach(),list(module.parameters()))
        report['cuda_half']=metric(hy,hr);report['native_amp']=metric(ay,ar)
        require(all(report[k]['finite'] and report[k]['raw_allclose'] for k in ('cuda_half','native_amp')), 'CUDA precision formula failed')
        amp_module=PBI().cuda();opt=torch.optim.AdamW(amp_module.parameters(),lr=.0005)
        scaler=torch.cuda.amp.GradScaler(enabled=True)
        updates=[]
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss=(amp_module(x.detach())-.1).square().mean()
            scaler.scale(loss).backward();scaler.unscale_(opt)
            require(all(torch.isfinite(p.grad).all() for p in amp_module.parameters()), 'AMP gradient nonfinite')
            before=amp_module.Wo.weight.detach().clone(); scale=scaler.get_scale()
            scaler.step(opt);scaler.update()
            changed=not torch.equal(before,amp_module.Wo.weight)
            require(changed,'Native GradScaler skipped mathematical update')
            updates.append(dict(loss=float(loss.detach()),scale_before=scale,scale_after=scaler.get_scale(),effective_update=changed))
        report['native_amp_optimizer_steps']=updates
    return report


def flatten(value):
    if isinstance(value,torch.Tensor): return [value]
    if isinstance(value,(tuple,list)): return [t for v in value for t in flatten(v)]
    if isinstance(value,dict): return [t for v in value.values() for t in flatten(v)]
    return []


def full_structure():
    rows={};hashes={};modules=[]
    for variant in VARIANTS:
        parent,target=build(variant,baseline=True),build(variant)
        verify_model(target,variant,zero=True)
        public=parent.state_dict()
        require(set(target.state_dict())-set(public)==set(PBI_KEYS),'Common state keys changed')
        require(all(torch.equal(v,target.state_dict()[k]) for k,v in public.items()),'nc80 constructor common values differ')
        target,audit=build_training_model(target.yaml,target,dict(nc=1,channels=3),variant)
        with torch.random.fork_rng(devices=[]):
            parent=native_rebuild(parent.yaml,target,1,3)
        require(all(torch.equal(v,target.state_dict()[k]) for k,v in parent.state_dict().items()),'nc1 public values differ')
        parent.eval();target.eval(); inp=torch.rand(1,3,640,640)
        traces=[[],[]];handles=[];pbi_calls=[]
        try:
            for model,trace in zip((parent,target),traces):
                for layer in model.model:
                    def capture(m,inputs,output,trace=trace):
                        tensors=flatten(output)
                        trace.append(dict(node=m.i,source=m.f,type=type(m).__name__,
                                          shapes=[list(t.shape) for t in tensors],hashes=[tensor_sha256(t) for t in tensors]))
                    handles.append(layer.register_forward_hook(capture))
            handles.append(target.model[17].pbi.register_forward_hook(lambda *_:pbi_calls.append(1)))
            with torch.no_grad():
                pa,ta=parent(inp),target(inp)
            require(len(pbi_calls)==1,'PBI execution count wrong')
            require(all(a['shapes']==b['shapes'] and a['hashes']==b['hashes'] for a,b in zip(*traces)), 'Initial full graph differs')
            require(all(torch.equal(a,b) for a,b in zip(flatten(pa),flatten(ta))), 'Initial complete outputs differ')
        finally:
            for h in handles:h.remove()
        require(traces[1][17]['shapes']==[[1,256,80,80]],'640 insertion shape changed')
        unfused=sum(p.numel() for p in target.parameters());parent_count=sum(p.numel() for p in parent.parameters())
        fused=deepcopy(target).fuse(verbose=False);verify_model(fused,variant,zero=True)
        require(unfused-parent_count==24576,'Increment changed')
        modules.append(target.model[17].pbi)
        hashes[variant]={k:tensor_sha256(target.state_dict()[k]) for k in PBI_KEYS}
        rows[variant]=dict(status='PASSED',nc80_public_exact=True,nc1_native_rebuild=audit,
                           full_model_initial_equality=True,node_trace=traces[1],pbi_calls=1,
                           parameters=dict(parent_unfused=parent_count,unfused=unfused,
                                           fused=sum(p.numel() for p in fused.parameters()),added=24576))
    require(hashes['pbi_v1']==hashes['cbr_lif_pbi_v1'],'Two variants PBI initial hashes differ')
    require(all(a.untyped_storage().data_ptr()!=b.untyped_storage().data_ptr()
                for a,b in zip(modules[0].parameters(),modules[1].parameters())), 'Cross-variant storage sharing')
    return dict(status='PASSED',variants=rows,variant_initial_hashes=hashes,variant_initial_values_exact=True,
                cross_variant_no_storage_sharing=True,
                cost=dict(input=[1,256,80,80],macs=157286400,projection_gflops=.3145728,
                          elementwise_operations=1843200,elementwise_gops=.0018432,
                          scope='PBI only: all three projections including functional Wo, two FLOPs/MAC; no dtype/memory cost'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('all','cpu','cuda'),default='all')
    args=parser.parse_args();require(not args.output.exists(),'Preserve existing report')
    torch.set_num_threads(4)
    report=dict(status='FAILED',report_kind='pbi_math_audit',contract_version='pbi_acceptance_v1',
                tolerance=dict(atol=ATOL,rtol=RTOL),runtime=runtime(),formal_training='NOT_STARTED',final_test='NOT_RUN')
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(42)
            report['rng_and_initialization']=rng_check()
            devices=['cpu','cuda'] if args.device=='all' else [args.device]
            report['devices']=[]
            for device in devices:
                if device=='cuda' and not torch.cuda.is_available():
                    report['devices'].append(dict(device='cuda',status='PENDING',reason='CUDA unavailable'))
                else:report['devices'].append(formula_check(device))
            report['structure_and_counts']=full_structure()
        report['status']='PASSED' if all(d['status']=='PASSED' for d in report['devices']) else 'PENDING'
    except Exception as exc:
        report['error']=str(exc);report['traceback']=traceback.format_exc();raise
    finally:
        from train_pbi import code_identity
        report['code_identity']=code_identity()
        write_json(args.output,report)
    print(json.dumps(dict(status=report['status'],output=str(args.output.resolve()))))


if __name__=='__main__':main()
