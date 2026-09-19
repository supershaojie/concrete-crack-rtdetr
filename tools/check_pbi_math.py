"""PBI-v1 explicit nonzero formula/gradient, RNG, structure and dtype audits; never trains a run."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
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
MATH_EVIDENCE_VERSION = 'pbi_math_precision_v2'
FUSION_EVIDENCE_VERSION = 'pbi_fusion_precision_v1'
FUSION_PRECISION_POLICY = 'native_diagnostic_strict_fp32_reference_v1'


def math_source_identity():
    """Bind the evidence to the actual audit, validator and unchanged operators."""
    root = Path(__file__).resolve().parents[1]
    names = ('tools/check_pbi_math.py', 'tools/train_pbi.py',
             'ultralytics-main/ultralytics/nn/modules/pbi.py',
             'ultralytics-main/ultralytics/nn/modules/conv.py')
    return {name: hashlib.sha256((root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in names}


def precision_settings():
    return dict(matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision())


@contextmanager
def strict_fp32_reference(record):
    """Only the wrapper reference forward pair; restore coupled settings even on failure."""
    before = precision_settings()
    record.update(before=before, restored=False, exited_via_exception=False)
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        record['active'] = precision_settings()
        yield
    except BaseException:
        record['exited_via_exception'] = True
        raise
    finally:
        # Setting allow_tf32=True can turn 'medium' into 'high'. Restore the
        # coupled precision selector last, then verify every captured setting.
        torch.backends.cuda.matmul.allow_tf32 = before['matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = before['cudnn_allow_tf32']
        torch.set_float32_matmul_precision(before['float32_matmul_precision'])
        record['after'] = precision_settings()
        record['restored'] = record['after'] == before
        require(record['restored'], 'PRECISION_RESTORE: wrapper reference changed caller settings')


class FusionAuditError(RuntimeError):
    """Keep the complete measured evidence when a hard fusion condition fails."""
    def __init__(self, reason, evidence):
        super().__init__(reason + ': PBI wrapper fusion audit failed; see nonzero_wrapper_fusion evidence')
        self.reason, self.evidence = reason, evidence


def metric(a, b):
    a,b=a.detach().float(),b.detach().float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    difference = (a-b).abs()
    valid = difference[torch.isfinite(difference)]
    return dict(raw_allclose=bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL)),
                max_abs=float(difference.max()) if finite else None, finite=finite,
                finite_max_abs=float(valid.max()) if valid.numel() else None,
                nonfinite_a=int((~torch.isfinite(a)).sum()),nonfinite_b=int((~torch.isfinite(b)).sum()))


def _fusion_capture(wrapped, fused, image):
    """Observe each real forward, with all hooks removed before returning/raising."""
    states = {label: {k: tensor_sha256(v) for k,v in model.state_dict().items()}
              for label,model in (('unfused',wrapped),('fused',fused))}
    pbi_states = {label: {k: tensor_sha256(v) for k,v in model.pbi.state_dict().items()}
                  for label,model in (('unfused',wrapped),('fused',fused))}
    row = dict(precision=precision_settings(),input_sha256=tensor_sha256(image),
               wrapper_state_sha256=states, pbi_calls={'unfused':0,'fused':0},
               pbi_state_sha256=pbi_states,pbi_state_equal=pbi_states['unfused']==pbi_states['fused'],
               before_pbi=None,after_pbi=None,errors=[])
    taps, handles = {}, []
    def attach(label, model):
        def projection(_module, _args, value):
            taps[label+'_before'] = value.detach().clone()
        def branch(_module, _args, _value):
            row['pbi_calls'][label] += 1
        handles.append(model.act.register_forward_hook(projection))
        handles.append(model.pbi.register_forward_hook(branch))
    try:
        attach('unfused',wrapped); attach('fused',fused)
        with torch.no_grad():
            outputs = {'unfused':wrapped(image), 'fused':fused(image)}
        row['before_pbi'] = metric(taps['unfused_before'], taps['fused_before'])
        row['after_pbi'] = metric(outputs['unfused'],outputs['fused'])
        row['finite'] = row['before_pbi']['finite'] and row['after_pbi']['finite']
        row['residual_abs_max'] = {
            label: float((out-taps[label+'_before']).abs().max())
            if bool(torch.isfinite(out).all() and torch.isfinite(taps[label+'_before']).all()) else None
            for label,out in outputs.items()}
    except Exception as exc:
        row['errors'].append(dict(reason='EXECUTION_ERROR',detail=repr(exc)))
        row['finite'] = None
        row['capture_missing'] = [key for key in ('unfused_before','fused_before') if key not in taps]
    finally:
        for handle in handles:
            handle.remove()
    if row['pbi_calls'] != {'unfused':1,'fused':1}:
        row['errors'].append(dict(reason='PBI_CALL_COUNT',detail=row['pbi_calls']))
    if row['finite'] is False:
        row['errors'].append(dict(reason='NONFINITE',detail='Input projection or wrapper output is nonfinite'))
    if not row['pbi_state_equal']:
        row['errors'].append(dict(reason='PBI_STATE',detail='PBI parameters differ after fusion'))
    if row['finite'] and any(v <= 0 for v in row['residual_abs_max'].values()):
        row['errors'].append(dict(reason='ZERO_RESIDUAL',detail=row['residual_abs_max']))
    row['raw_allclose'] = all((row.get(key) or {}).get('raw_allclose') is True for key in ('before_pbi','after_pbi'))
    row['status'] = 'FAILED' if row['errors'] else 'PASSED' if row['raw_allclose'] else 'PRECISION_NOTE'
    return row


def wrapper_fusion_audit(wrapped, image, fused=None):
    """Native diagnostic and strict reference share input and source/PBI weights.

    Each uses the identical Conv/BN fusion flow, including its matrix multiply,
    under its recorded precision. The optional fused object is for fault probes.
    """
    before = precision_settings()
    report = dict(evidence_version=FUSION_EVIDENCE_VERSION, precision_policy=FUSION_PRECISION_POLICY,
                  scope='nonzero_wrapper_fusion_only', tolerance=dict(atol=ATOL,rtol=RTOL),
                  settings_before=before, status='FAILED', precision_context={},
                  fusion_flow='deepcopy_fuse_conv_and_bn_forward_fuse' if fused is None else 'injected_test_fixture',
                  device=image.device.type, device_index=image.device.index,
                  input_shape=list(image.shape), input_dtype=str(image.dtype))
    def prepare():
        if fused is not None:
            return fused
        result=deepcopy(wrapped)
        result.conv=fuse_conv_and_bn(result.conv,result.bn)
        delattr(result,'bn')
        result.forward=result.forward_fuse
        return result
    try:
        report['native'] = _fusion_capture(wrapped,prepare(),image)
        with strict_fp32_reference(report['precision_context']):
            report['strict_fp32'] = _fusion_capture(wrapped,prepare(),image)
            native, strict = report['native'],report['strict_fp32']
            report['same_input_and_weights'] = (native['input_sha256'] == strict['input_sha256'] and
                native['wrapper_state_sha256']['unfused'] == strict['wrapper_state_sha256']['unfused'] and
                native['pbi_state_sha256'] == strict['pbi_state_sha256'])
            if not report['same_input_and_weights']:
                raise FusionAuditError('PBI_STATE',report)
            for row in (native,strict):
                if row['errors']:
                    raise FusionAuditError(row['errors'][0]['reason'],report)
            if not strict['raw_allclose']:
                strict['status']='FAILED'
                strict['errors'].append(dict(reason='TOLERANCE',detail='Strict FP32 pre/post-PBI comparison exceeds unchanged tolerance'))
                raise FusionAuditError('TOLERANCE',report)
        report['status']='PASSED'  # Explicit policy pass, not a native allclose claim.
        report['native_precision_status']=native['status']
        return report
    except FusionAuditError:
        raise
    except Exception as exc:
        report['execution_error']=repr(exc)
        raise FusionAuditError('EXECUTION_ERROR',report) from exc
    finally:
        report['settings_after']=precision_settings()
        report['settings_restored']=report['settings_after']==before
        if not report['settings_restored']:
            report['status']='FAILED'
            raise FusionAuditError('PRECISION_RESTORE',report)


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


def formula_check(device, report=None):
    report = {} if report is None else report
    report.update(device=device,status='FAILED')
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
    image=torch.randn(2,128,9,11,device=device)
    report.update(new_trainable_parameters=24576,shapes=rows,
                  nonzero_formula=output,nonzero_gradients=grads,
                  nonzero_residual_max=float((y-x).abs().max().detach()),gradient_steps=steps)
    try:
        report['nonzero_wrapper_fusion']=wrapper_fusion_audit(wrapped,image)
    except FusionAuditError as exc:
        report['nonzero_wrapper_fusion']=exc.evidence
        report['failure_reason']=exc.reason
        raise
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
    report['status']='PASSED'
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
    from train_pbi import code_identity, git_head
    initial_code, initial_head = code_identity(), git_head()
    report=dict(status='FAILED',report_kind='pbi_math_audit',contract_version='pbi_acceptance_v1',
                math_evidence_version=MATH_EVIDENCE_VERSION,precision_policy=FUSION_PRECISION_POLICY,
                audit_source=math_source_identity(),code_identity=initial_code,git_head=initial_head,
                precision_before=precision_settings(),
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
                else:
                    device_report={}
                    report['devices'].append(device_report)
                    formula_check(device,device_report)
            report['structure_and_counts']=full_structure()
        report['status']='PASSED' if all(d['status']=='PASSED' for d in report['devices']) else 'PENDING'
        report['native_fusion_status']='PRECISION_NOTE' if any(
            d.get('nonzero_wrapper_fusion',{}).get('native_precision_status')=='PRECISION_NOTE'
            for d in report['devices']) else report['status']
    except Exception as exc:
        report['error']=str(exc);report['traceback']=traceback.format_exc();raise
    finally:
        report['precision_after']=precision_settings()
        report['precision_restored']=report['precision_before']==report['precision_after']
        report['code_identity_unchanged_during_run']=(code_identity()==initial_code and git_head()==initial_head)
        if not report['precision_restored'] or not report['code_identity_unchanged_during_run']:
            report['status']='FAILED'
            report['identity_error']='Audit changed precision settings, or source/HEAD changed during execution'
        write_json(args.output,report)
    require(report['status']=='PASSED','Mathematical audit incomplete or identity/precision restoration failed')
    print(json.dumps(dict(status=report['status'],output=str(args.output.resolve()))))


if __name__=='__main__':main()
