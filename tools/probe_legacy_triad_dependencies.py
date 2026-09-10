"""Reproduce the pre-implementation synthetic derivative audit using preserved original classes."""
import argparse
import torch
from triad_compat import write_json
from ultralytics.nn.modules import CSCEFv51, SCCAAIFI, CrackBoundaryRefinement


def run():
    torch.set_num_threads(2);torch.manual_seed(42)
    def norm(x):return None if x.grad is None else x.grad.abs().sum().item()
    cs=CSCEFv51(256,256);torch.nn.init.normal_(cs.output_projection.weight,std=.01)
    l=torch.randn(1,256,8,8,requires_grad=True);s=torch.randn_like(l,requires_grad=True)
    delta=cs([l,s])-l;delta.square().sum().backward()
    report=dict(scope='synthetic nonzero branch dependency probe, NOT a trained-loss conflict/cosine measurement',
        cscef=dict(lateral_side_grad_l1=norm(l),semantic_grad_l1=norm(s),residual_rms=delta.square().mean().sqrt().item()))
    sc=SCCAAIFI(256,1024,8);torch.nn.init.normal_(sc.scca_o.weight,std=.01)
    x=torch.randn(1,9,256,requires_grad=True);mh=torch.randn_like(x,requires_grad=True)
    d,a=sc.scca_channel(x,mh);d.square().sum().backward()
    report['scca']=dict(src_side_grad_l1=norm(x),mha_side_grad_l1=norm(mh),relation_shape=list(a.shape))
    cb=CrackBoundaryRefinement(256,256);torch.nn.init.normal_(cb.offset_out.weight,std=.05)
    p=torch.randn(1,256,8,8,requires_grad=True);q=torch.randn(1,7,256,requires_grad=True);b=torch.full((1,7,4),.4,requires_grad=True)
    y=cb(p,q,b);(y-b).square().sum().backward()
    report['cbr']=dict(p3_side_grad_l1=norm(p),query_side_grad_l1=norm(q),box_residual_scale_grad_l1=norm(b),rho=cb.rho,normal_fraction=cb.normal_fraction)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);a=p.parse_args();write_json(a.report,run())
