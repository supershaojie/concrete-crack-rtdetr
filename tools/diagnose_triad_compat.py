"""Opt-in bounded feature/box/precision diagnostics; never dataset evaluation or architecture search."""
from __future__ import annotations
import argparse
import time
import torch
from triad_compat import *
from init_triad_compat import controlled_models
from check_triad_compat import feature_forward, errors, tensors
from check_triad_compat_gradients import activate
from ultralytics.nn.modules import GISCCAAIFI, StableReferenceCBR


def rms(x):return float(x.detach().float().square().mean().sqrt())


def precision_selection(model,image):
    """Separate encoder top-k permutations from value errors without changing selection."""
    selected=[]
    handle=model.model[-1].enc_score_head.register_forward_hook(
        lambda m,a,o:selected.append(torch.topk(o.max(-1).values,model.model[-1].num_queries,dim=1).indices.detach()))
    try:
        fp=model(image)[0]
        with torch.autocast('cuda',dtype=torch.float16):amp=model(image)[0]
    finally:handle.remove()
    matches=selected[0][0,:,None]==selected[1][0,None,:]
    i,j=matches.nonzero(as_tuple=True)
    return dict(raw_output_error=errors(fp,amp),same_position_query_indices=int((selected[0]==selected[1]).sum()),
        shared_encoder_indices=int(i.numel()),matched_query_value_error=errors(fp[:,i],amp[:,j]),
        interpretation='Native encoder top-k may permute/change queries under AMP; matched values are a diagnostic, not a forced selection.')


def diagnose(source,variant,nonzero=False):
    reference,model,_=controlled_models(source,variant)
    torch.manual_seed(73)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval();model.model[-1].shapes=None
    if nonzero:activate(model)
    roles=topology(model,variant)['roles'];image=torch.rand(1,3,160,192,device=device)
    captures={};hooks=[]
    for module in model.modules():
        if isinstance(module,GISCCAAIFI):
            hooks.append(module.register_forward_pre_hook(lambda m,a: captures.update(scca_src=a[0].flatten(2).permute(0,2,1))))
            hooks.append(module.ma.register_forward_hook(lambda m,a,o: captures.update(scca_s=o[0])))
        if isinstance(module,StableReferenceCBR):
            hooks.append(module.register_forward_pre_hook(lambda m,a: captures.update(cbr_args=a)))
    with torch.no_grad():
        output,features,_=feature_forward(model,image,roles)
        for h in hooks:h.remove()
        report=dict(runtime=runtime(),variant=variant,diagnostic_only=True,synthetic_nonzero_intervention=nonzero,
            input=[1,3,160,192],features={k:dict(shape=list(v.shape),dtype=str(v.dtype),rms=rms(v),finite=bool(torch.isfinite(v).all())) for k,v in features.items() if isinstance(v,torch.Tensor)})
        report['cscef_delta_to_base_rms']=rms(features['P3_enh']-features['P3_base'])/(rms(features['P3_base'])+1e-8)
        for module in model.modules():
            if isinstance(module,GISCCAAIFI):
                delta,attention=module.scca_channel(captures['scca_src'],captures['scca_s'])
                report['scca']=dict(delta_rms=rms(delta),delta_to_src_rms=rms(delta)/(rms(captures['scca_src'])+1e-8),
                    delta_to_s_rms=rms(delta)/(rms(captures['scca_s'])+1e-8),attention_shape=list(attention.shape),finite=bool(torch.isfinite(delta).all()))
            if isinstance(module,StableReferenceCBR):
                _,d=module(*captures['cbr_args'],return_diagnostics=True)
                report['cbr']=dict(mean_abs_box_correction=float((d['after']-d['before']).abs().mean()),
                    saturated_fraction=float((d['tanh_offsets'].abs()>=.95).float().mean()),evidence_rms=rms(d['evidence']),
                    finite=bool(torch.isfinite(d['after']).all()),rho=module.rho)
        batch_output=model(torch.cat((image,torch.rand_like(image))))[0]
        report['same_image_batch_error']=errors(output[0],batch_output[:1])
        torch.testing.assert_close(output[0],batch_output[:1],atol=2e-5,rtol=2e-5)
        if device=='cuda':
            with torch.autocast('cuda',dtype=torch.float16):amp=model(image)
            report['AMP_vs_FP32']=errors(output[0],amp[0]);require(torch.isfinite(amp[0]).all(),'AMP nonfinite')
            report['precision_selection']=precision_selection(model,image)
            reference.to(device).eval();reference.model[-1].shapes=None
            report['baseline_precision_selection']=precision_selection(reference,image)
            reference.cpu()
        else:report['AMP_vs_FP32']={'status':'NOT_RUN'}
        for _ in range(3):model(image)
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(10):model(image)
        if device=='cuda':torch.cuda.synchronize()
        report['benchmark']=dict(mean_forward_ms=(time.perf_counter()-start)*100,iterations=10,warmups=3,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
            scope='local synthetic B1 160x192 eval; not AutoDL4090/formal training throughput')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--report',required=True);p.add_argument('--nonzero-diagnostic',action='store_true')
    a=p.parse_args();torch.set_num_threads(4);write_json(a.report,diagnose(a.source,a.variant,a.nonzero_diagnostic))
