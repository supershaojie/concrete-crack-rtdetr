"""Nonzero side-path derivatives, staged learning and preserved main gradients."""
from __future__ import annotations
import argparse
from copy import deepcopy
import torch
from triad_compat import require, write_json
from ultralytics.nn.modules import CSCEFv51, DRCSCEFv6, SCCAAIFI, GISCCAAIFI, CrackBoundaryRefinement, StableReferenceCBR


def activate(module):
    """Diagnostic only; formal initialization remains exactly zero."""
    with torch.no_grad():
        for name,p in module.named_parameters():
            if any(part in name for part in ("output_projection", "scca_o", "offset_out")):
                p.normal_(0,.02)


def grads(module, prefix=None):
    result={n: None if p.grad is None else float(p.grad.abs().sum()) for n,p in module.named_parameters()
            if prefix is None or n.startswith(prefix)}
    require(all(v is not None and v>0 and __import__('math').isfinite(v) for v in result.values()), f"Inactive parameters: {result}")
    return result


def cscef_check():
    torch.manual_seed(42)
    m=DRCSCEFv6(256,256,256)
    base=torch.randn(2,256,8,12,requires_grad=True)
    lateral=torch.randn_like(base,requires_grad=True)
    semantic=torch.randn(2,256,4,6,requires_grad=True)
    m([base,lateral,semantic]).square().mean().backward()
    first={n:float(p.grad.abs().sum()) for n,p in m.named_parameters()}
    require(first['output_projection.weight']>0 and all(v==0 for n,v in first.items() if n!='output_projection.weight'), "CSCEF zero-init staging")
    m.zero_grad(); base.grad=None; activate(m)
    y=m([base,lateral,semantic]); y.sum().backward()
    require(torch.equal(base.grad,torch.ones_like(base)), "CSCEF main-base derivative changed")
    require(lateral.grad is None and semantic.grad is None, "CSCEF refs leak side gradient")
    parameter_grads=grads(m)
    # References can still train through the ORIGINAL main-path consumer.
    (lateral.square().sum()+semantic.square().sum()).backward()
    require(lateral.grad.abs().sum()>0 and semantic.grad.abs().sum()>0, "Refs cannot receive main gradients")
    old=CSCEFv51(256,256); old.load_state_dict(m.state_dict())
    old_l=lateral.detach().requires_grad_(); old_s=semantic.detach().requires_grad_()
    (old([old_l,old_s])-old_l).square().sum().backward()
    require(old_s.grad.abs().sum()>0, "Old semantic path unexpectedly absent")
    with torch.no_grad():
        a=m.side_residual(lateral,semantic)
        b=m.side_residual(lateral[:1],semantic[:1])
        batch_error=float((a[:1]-b).abs().max())
        require(torch.allclose(a[:1],b,atol=1e-6,rtol=1e-5), "CSCEF batch dependence")
    return dict(status="passed", refs_side_gradient="None (no edge)", base_derivative_max_abs_error=0,
                main_ref_gradients_present=True, first_backward=first, active_parameter_grads=parameter_grads,
                old_semantic_grad_l1=float(old_s.grad.abs().sum()), batch_max_abs=batch_error)


def scca_check():
    torch.manual_seed(43)
    result={}
    for pre in (False,True):
        old=SCCAAIFI(256,1024,8,normalize_before=pre).eval(); new=GISCCAAIFI(256,1024,8,normalize_before=pre).eval()
        activate(old); new.load_state_dict(old.state_dict())
        x=torch.randn(2,11,256,requires_grad=True); s=torch.randn_like(x,requires_grad=True)
        d1,a1=old.scca_channel(x,s); d2,a2=new.scca_channel(x,s)
        require(torch.equal(d1,d2) and torch.equal(a1,a2), "SCCA detach changed forward")
        d1.square().sum().backward()
        old_grads=dict(src=float(x.grad.abs().sum()),s=float(s.grad.abs().sum()))
        require(all(v>0 for v in old_grads.values()), "Old SCCA not coupled")
        x.grad=None;s.grad=None; d2.square().sum().backward()
        require(x.grad is None and s.grad is None, "GI-SCCA source gradient leak")
        side_grads=grads(new,'scca_')
        new.zero_grad()
        image=torch.randn(1,256,4,5,requires_grad=True)
        old_result=old(image); new_result=new(image)
        require(torch.equal(old_result,new_result), "SCCA full pre/post forward changed")
        (new_result*torch.randn_like(new_result)).sum().backward()
        require(image.grad.abs().sum()>0 and new.ma.in_proj_weight.grad.abs().sum()>0 and new.fc1.weight.grad.abs().sum()>0,
                "AIFI main path detached")
        result['pre' if pre else 'post']=dict(max_abs=0,max_rel=0,old_input_gradients=old_grads,
            isolated_input_gradients="None (no edge)", side_parameter_gradients=side_grads, main_mha_ffn_gradient=True)
    return result


def cbr_check():
    torch.manual_seed(44)
    old=CrackBoundaryRefinement(256,256); new=StableReferenceCBR(256,256)
    activate(old); new.load_state_dict(old.state_dict())
    p=torch.randn(2,256,8,12,requires_grad=True); q=torch.randn(2,13,256,requires_grad=True)
    boxes=(torch.rand(2,13,4)*.6+.15).requires_grad_()
    old(p,q,boxes).sum().backward()
    old_grads=dict(p3=float(p.grad.abs().sum()),query=float(q.grad.abs().sum()))
    p.grad=None;q.grad=None;boxes.grad=None
    y,detail=new(p,q,boxes,True); y.sum().backward()
    require(p.grad is None and q.grad is None, "CBR side input leak")
    require(torch.equal(boxes.grad,torch.ones_like(boxes)), "CBR residual geometry leaks box gradient")
    parameter_grads=grads(new)
    with torch.no_grad():
        # No core_P3 parameter enters SR-CBR; changing the core while holding ref/query/box is structurally irrelevant.
        repeat=new(p,q,boxes,True)[1]['evidence']
        altered=new(p+torch.randn_like(p),q,boxes,True)[1]['evidence']
        require(torch.equal(detail['evidence'],repeat) and not torch.equal(repeat,altered), "Stable evidence sources incorrect")
        ratio=y[...,2:]/boxes[...,2:]
        require(ratio.min()>=.85 and ratio.max()<=1.15, "CBR width/height bound")
        require((detail['displacement'].abs()<=.075*boxes[..., [2,2,3,3]]+1e-7).all(), "CBR side bound")
    return dict(status="passed", ref_query_side_gradient="None (no edge)",box_derivative_max_abs_error=0,
        old_extra_gradients=old_grads, active_parameter_grads=parameter_grads, reference_evidence_sensitive=True,
        ref_change_max_abs=float((repeat-altered).abs().max()),width_height_ratio=[float(ratio.min()),float(ratio.max())])


def run_checks():
    return dict(cscef=cscef_check(),scca=scca_check(),cbr=cbr_check())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True)
    args=p.parse_args();torch.set_num_threads(4);write_json(args.report,run_checks())
