"""Separate process: import an explicitly exported historical tree before ultralytics."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(sys.argv[1])/'ultralytics-main'))
import torch
import ultralytics
from ultralytics.nn.tasks import RTDETRDetectionModel

def probe(module,kind):
    reports={}
    for pre in ([False,True] if kind=='c24' else [False]):
        torch.manual_seed(810);module=module.cpu().float().eval();module.zero_grad(set_to_none=True)
        if kind=='c24':module.normalize_before=pre
        with torch.no_grad():(module.scca_o if kind=='c24' else module.O_proj).weight.normal_(0,.01)
        x=torch.randn(2,256,9,13,requires_grad=True);out=module(x)
        (out*torch.randn_like(out)).mean().backward()
        reports[str(pre)]=dict(output=out.detach(),input_grad=x.grad,
            parameter_grads={n:p.grad for n,p in module.named_parameters()})
    return reports

if __name__=='__main__':
    torch.set_num_threads(4);payload=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
    cfg=str(Path(sys.argv[1])/'ultralytics-main/ultralytics/cfg/models/rt-detr'/payload['yaml'])
    torch.manual_seed(42);initial=RTDETRDetectionModel(cfg,nc=80,verbose=False)
    new={k:v for k,v in initial.state_dict().items() if '.scca_' in k or len(k.split('.'))>3 and k.split('.')[2] in {'B_proj','P','U_mix','U_dw','O_proj'}}
    model=RTDETRDetectionModel(cfg,nc=1,verbose=False).eval();model.load_state_dict(payload['state'],strict=True)
    with torch.no_grad():out=model(payload['image'])
    result=dict(prediction=out,initial_new=new,parameters=sum(p.numel() for p in model.parameters()),states=len(model.state_dict()),import_path=ultralytics.__file__)
    if payload['kind']!='c2':
        cls='SCCAAIFI' if payload['kind']=='c24' else 'LIFDown'
        module=next(m for m in model.modules() if type(m).__name__==cls)
        result['probe']=probe(module,payload['kind'])
    torch.save(result,sys.argv[3])
