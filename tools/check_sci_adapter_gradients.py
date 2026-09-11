"""Non-vacuous SCI forward/routing and exact isolated semantic Jacobian checks."""
from __future__ import annotations
import argparse
from sci_adapter import *


@torch.no_grad()
def activate(model,adapter=True):
    for m in model.modules():
        if isinstance(m,CSCEFv51):torch.nn.init.normal_(m.output_projection.weight,std=.02)
        if isinstance(m,SCCAAIFI):torch.nn.init.normal_(m.scca_o.weight,std=.02)
        if isinstance(m,CrackBoundaryRefinement):
            torch.nn.init.normal_(m.offset_out.weight,std=.02);torch.nn.init.normal_(m.offset_out.bias,std=.01)
        if adapter and isinstance(m,SCIAdapter):
            torch.nn.init.normal_(m.restore.weight,std=.002);torch.nn.init.normal_(m.restore.bias,std=.001)


def error(a,b,exact=False):
    require(a.shape==b.shape and a.dtype==b.dtype and torch.isfinite(a).all() and torch.isfinite(b).all(),'Invalid tensors')
    d=(a.detach().float()-b.detach().float()).abs()
    result=dict(max_abs=float(d.max()),max_rel=float((d/a.detach().float().abs().clamp_min(1e-12)).max()))
    if exact:require(torch.equal(a,b),f'Non-exact tensors: {result}')
    return result


def capture(model,image,adapted=True):
    features={};handles=[]
    roles=dict(Y5=10,Y4=15,semantic=16,lateral=17,CSCEF=19 if adapted else 18,
               P3=21 if adapted else 20,P4=24 if adapted else 23,P5=27 if adapted else 26)
    if adapted:roles['compat']=18
    for name,index in roles.items():
        def hook(m,args,out,name=name):features[name]=out
        handles.append(model.model[index].register_forward_hook(hook))
    def concat(m,args):features['concat_semantic']=args[0][0]
    handles.append(model.model[20 if adapted else 19].register_forward_pre_hook(concat))
    try:output=model(image)
    finally:
        for h in handles:h.remove()
    return output,features


def align_legacy(variant,device='cpu'):
    legacy='c25' if VARIANTS[variant]['scca'] else 'c17'
    old=build(legacy).to(device).eval();activate(old)
    new=build(variant).to(device).eval();state=new.state_dict()
    # Original C17/C25 layers 18..27 shift by exactly one SCI node.
    mapping={i:i if i<18 else i+1 for i in range(28)}
    for k,v in old.state_dict().items():state[common_key(k,mapping)]=v
    new.load_state_dict(state,strict=True)
    return old,new


def equivalence(variant,device='cpu'):
    old,new=align_legacy(variant,device);report={}
    for h,w in ((640,640),(160,192)):
        image=torch.rand(1,3,h,w,device=device)
        with torch.no_grad():a,af=capture(old,image,False);b,bf=capture(new,image)
        record={n:error(v,bf[n],True) for n,v in af.items()}
        record.update(adapter_identity=error(bf['semantic'],bf['compat'],True),
            bbox=error(a[0][...,:4],b[0][...,:4],True),score=error(a[0][...,4:],b[0][...,4:],True))
        require(bf['concat_semantic'] is bf['semantic'],'Main Concat received SCI proxy')
        report[f'{h}x{w}']=record
    return dict(nonzero_original_CSCEF_SCCA=True,common_state_exact=True,comparisons=report)


def routing(variant,device='cpu'):
    _,model=align_legacy(variant,device)
    image=torch.rand(1,3,160,192,device=device)
    with torch.no_grad():
        _,a=capture(model,image)
        activate(model.model[18])
        _,b=capture(model,image)
    for name in ('Y4','semantic','lateral','concat_semantic'):error(a[name],b[name],True)
    require(b['concat_semantic'] is b['semantic'],'Main path changed')
    changes={n:error(a[n],b[n]) for n in ('compat','CSCEF','P3','P4','P5')}
    require(all(v['max_abs']>0 for v in changes.values()),'Nonzero SCI not propagated through CSCEF/PAN')
    y=torch.randn(2,256,10,12,device=device);before=y.clone();version=y._version
    with torch.no_grad():out=model.model[18](y)
    require(y._version==version and torch.equal(y,before) and out.data_ptr()!=y.data_ptr(),'In-place input mutation')
    return dict(status='passed',upstream_exact=True,original_concat_tensor=True,no_inplace=True,changes=changes)


def gradient_checks(variant,device='cpu'):
    old,new=align_legacy(variant,device)
    image=torch.rand(1,3,160,192,device=device)
    grads=[]
    for model,adapted in ((old,False),(new,True)):
        _,f=capture(model,image,adapted)
        loss=(f['CSCEF']-f['lateral']).square().mean()
        grads.append(torch.autograd.grad(loss,(f['semantic'],f['Y4'])))
    zero={}
    for i,name in enumerate(('semantic','Y4')):
        a,b=grads[0][i],grads[1][i];require(a.norm()>0,'Vacuous semantic gradient')
        zero[name]=dict(**error(a,b,True),L2_ratio=float(b.norm()/a.norm()))
    activate(new.model[18])
    _,f=capture(new,image)
    side=(f['CSCEF']-f['lateral']).square().mean()
    gy,gproxy=torch.autograd.grad(side,(f['semantic'],f['compat']),retain_graph=True)
    identity=error(gy,gproxy,True);require(gy.norm()>0,'Identity semantic gradient lost')
    names=[(n,p) for n,p in new.named_parameters() if n in new_keys(new)]
    values=torch.autograd.grad(side,[p for n,p in names])
    trainable={n:float(g.float().norm()) for (n,p),g in zip(names,values)}
    require(all(torch.isfinite(g).all() and g.norm()>0 for g in values),'SCI/SCCA/CSCEF gradient absent')
    # Inspect the isolated correction, not the identity sum. Nonzero restore makes every parameter learn.
    adapter=SCIAdapter().to(device);activate(adapter)
    y=torch.randn(2,256,10,12,device=device,requires_grad=True)
    delta=adapter.residual(y)
    isolated=torch.autograd.grad(delta.square().mean(),[y,*adapter.parameters()],allow_unused=True)
    require(isolated[0] is None and all(g is not None and g.norm()>0 for g in isolated[1:]),'Detach branch incorrect')
    return dict(zero_init_semantic_gradient=zero,nonzero_identity_jacobian=identity,
        nonzero_parameter_gradient_L2=trainable,isolated_input_gradient=None,
        isolated_adapter_gradients=[float(g.norm()) for g in isolated[1:]])


def learning():
    torch.manual_seed(2026);m=SCIAdapter();x=torch.randn(4,256,8,10)
    # Fixed low-rank channel mixing plus per-channel shift. Source never trains.
    u=torch.randn(32,256)*.01;v=torch.randn(256,32)*.01
    target=x+torch.einsum('cr,rd,bdhw->bchw',v,u,x)+torch.linspace(-.1,.1,256)[None,:,None,None]
    opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.0001);losses=[]
    for _ in range(30):
        opt.zero_grad();loss=(m(x)-target).square().mean();loss.backward();opt.step();losses.append(float(loss))
    final=float((m(x)-target).square().mean());require(final<losses[0]*.3,'Synthetic adapter failed to learn')
    return dict(initial_loss=losses[0],final_loss=final,steps=30,losses=losses,
                scope='Synthetic engineering learnability only; no detection-performance evidence')


def run_checks():
    torch.manual_seed(117);device='cuda' if torch.cuda.is_available() else 'cpu'
    report={v:gradient_checks(v,device) for v in ('cscef_v51_sci_control','scca_sci_cscef_v51')}
    return dict(status='passed',device=device,variants=report,synthetic_learning=learning())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args()
    torch.set_num_threads(4);write_json(a.report,run_checks())
