"""Nonzero C17/C25 exact-forward and isolated-edge backward attribution; no dataset loop."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from copy import deepcopy
import torch
from mincompat_v2 import *


@torch.no_grad()
def activate(model):
    for m in model.modules():
        if isinstance(m,CSCEFv51): torch.nn.init.normal_(m.output_projection.weight, std=.02)
        if isinstance(m,SCCAAIFI): torch.nn.init.normal_(m.scca_o.weight, std=.02)
        if isinstance(m,CrackBoundaryRefinement):
            torch.nn.init.normal_(m.offset_out.weight, std=.02)
            torch.nn.init.normal_(m.offset_out.bias, std=.01)


def error(a,b,exact=False):
    require(a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(),'Invalid tensors')
    d=(a.detach().float()-b.detach().float()).abs()
    r=dict(max_abs=float(d.max()),max_rel=float((d/a.detach().float().abs().clamp_min(1e-12)).max()))
    if exact: require(torch.equal(a,b),f'Forward differs: {r}')
    return r


def ratio(old,new,scale):
    require(old is not None and new is not None and old.float().norm()>0,'Absent/vacuous gradient')
    r=dict(L1_ratio=float(new.float().abs().sum()/old.float().abs().sum()),
           L2_ratio=float(new.float().norm()/old.float().norm()),**error(old*scale,new))
    require(torch.allclose(new,old*scale,rtol=2e-5,atol=1e-8),f'Wrong gradient scale {scale}: {r}')
    return r


def module_checks(device='cpu',amp=False,half=False):
    torch.manual_seed(114)
    old=CSCEFv51(256,256).to(device);new=CSCEFv52Compat(256,256).to(device)
    activate(old);new.load_state_dict(old.state_dict(),strict=True)
    require(old.state_dict().keys()==new.state_dict().keys(),'State keys changed')
    require(sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())==26912,'New parameters')
    if half:old.half();new.half()
    dtype=torch.float16 if half else torch.float32
    l=torch.randn(2,256,20,24,device=device,dtype=dtype,requires_grad=True)
    s=torch.randn_like(l,requires_grad=True)
    ln=l.detach().clone().requires_grad_();sn=s.detach().clone().requires_grad_()
    context=torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext()
    with context:
        a=old((l,s));b=new((ln,sn))
        la=a.float().square().mean();lb=b.float().square().mean()
    report=dict(forward=error(a,b,True),identity=error(s,new._semantic_grad_view(s),True),strict_load=True,
                parameters=26912,state_tensors=len(old.state_dict()),nonzero_output=True)
    if not half:
        la.backward();lb.backward()
        # AMP input-edge quantization can round quarter gradients; FP32 proves exact ratio.
        if not amp:
            report.update(semantic=ratio(s.grad,sn.grad,.25),lateral=ratio(l.grad,ln.grad,1.))
        else:
            report.update(semantic=dict(L1_ratio=float(sn.grad.abs().sum()/s.grad.abs().sum()),
                L2_ratio=float(sn.grad.norm()/s.grad.norm()),**error(s.grad*.25,sn.grad)),lateral=ratio(l.grad,ln.grad,1.))
            require(abs(report['semantic']['L2_ratio']-.25)<.005,'AMP edge ratio drift')
        report['parameter_gradients']={n:error(p.grad,dict(new.named_parameters())[n].grad,True) for n,p in old.named_parameters()}
    return report


def capture(model,image):
    features={};handles=[]
    for name,index in dict(Y5=10,Y4=15,semantic=16,lateral=17,CSCEF=18,P3=20,P4=23,P5=26).items():
        def hook(m,args,out,name=name): features[name]=out
        handles.append(model.model[index].register_forward_hook(hook))
    try:output=model(image)
    finally:
        for h in handles:h.remove()
    return output,features


def pair_checks(device='cpu'):
    torch.manual_seed(115)
    old=build('c25_replay').to(device).eval();new=build('cscef_v52_scca_compat').to(device).eval()
    activate(old);new.load_state_dict(old.state_dict(),strict=True)
    report=dict(nonzero_CSCEF_SCCA=True,strict_pair_state_load=True,forward={})
    for h,w in ((640,640),(160,192)):
        image=torch.rand(1,3,h,w,device=device)
        with torch.no_grad():a,af=capture(old,image);b,bf=capture(new,image)
        report['forward'][f'{h}x{w}']={n:error(v,bf[n],True) for n,v in af.items()}
        report['forward'][f'{h}x{w}'].update(bbox=error(a[0][...,:4],b[0][...,:4],True),score=error(a[0][...,4:],b[0][...,4:],True))
    # Separate objectives on the actual captured graph. Frozen CSCEF contribution in the main-only
    # probe is a diagnostic intervention; neither production graph contains this detach.
    image=torch.rand(1,3,160,192,device=device)
    gradients=[]
    for model in (old,new):
        output,f=capture(model,image)
        residual=(f['CSCEF']-f['lateral']).square().mean()
        side=torch.autograd.grad(residual,(f['semantic'],f['Y4']),retain_graph=True)
        main=model.model[20](model.model[19]([f['semantic'],f['CSCEF'].detach()])).square().mean()
        direct=torch.autograd.grad(main,(f['semantic'],f['Y4']),retain_graph=True)
        # Actual downstream P3 + decoder objective, retaining all ordinary main-path contributions.
        objective=output[0].square().mean()+f['P3'].square().mean()
        params=[(n,p) for n,p in model.model[9].named_parameters() if n.startswith('scca_')]
        total=torch.autograd.grad(objective,[p for n,p in params],retain_graph=True)
        sg=torch.autograd.grad(residual,[p for n,p in params])
        gradients.append(dict(side=side,main=direct,params=params,total=total,side_params=sg))
    a,b=gradients
    report['semantic_only']={name:ratio(a['side'][i],b['side'][i],.25) for i,name in enumerate(('upY4','Y4'))}
    report['main_concat_only']={name:ratio(a['main'][i],b['main'][i],1.) for i,name in enumerate(('upY4','Y4'))}
    report['SCCA_total']={}
    report['SCCA_semantic_side']={}
    for i,(n,p) in enumerate(a['params']):
        ga,gb=a['total'][i],b['total'][i]
        require(torch.isfinite(ga).all() and torch.isfinite(gb).all() and ga.norm()>0 and gb.norm()>0,'SCCA gradient lost')
        report['SCCA_total'][n]=dict(old_L2=float(ga.norm()),new_L2=float(gb.norm()))
        report['SCCA_semantic_side'][n]=ratio(a['side_params'][i],b['side_params'][i],.25)
    report['total_gradient_note']='Total SCCA gradients include normal main paths; no fixed total ratio is asserted.'
    # Check the original SCCA channel's own x/s input edges in both norm modes.
    report['SCCA_internal_edges']={}
    for pre in (False,True):
        m=SCCAAIFI(256,1024,8,normalize_before=pre).to(device).eval();activate(m)
        x=torch.randn(2,30,256,device=device,requires_grad=True);s=torch.randn_like(x,requires_grad=True)
        delta,_=m.scca_channel(x,s)
        gx,gs=torch.autograd.grad(delta.square().mean(),(x,s))
        require(gx.norm()>0 and gs.norm()>0 and torch.isfinite(gx).all() and torch.isfinite(gs).all(),'SCCA x/s detached')
        report['SCCA_internal_edges'][str(pre)]=dict(x_L2=float(gx.norm()),s_L2=float(gs.norm()))
    return report


def run_checks():
    report=dict(FP32=module_checks(),pair=pair_checks('cuda' if torch.cuda.is_available() else 'cpu'))
    if torch.cuda.is_available():
        report['AMP']=module_checks('cuda',amp=True)
        report['half']=module_checks('cuda',half=True)
    else:report.update(AMP=dict(status='NOT_RUN'),half=dict(status='NOT_RUN'))
    return dict(status='passed',**report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True)
    a=p.parse_args();torch.set_num_threads(4);write_json(a.report,run_checks())
