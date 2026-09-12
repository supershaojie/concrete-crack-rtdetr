"""Isolated historical imports. Called only by the bounded regression checker."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1])/'ultralytics-main'))
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.patches import torch_load
import ultralytics


def probe(module, kind):
    torch.manual_seed(810)
    module=module.cpu().float().eval()
    with torch.no_grad():
        (module.output_projection if kind=='c17' else module.O_proj).weight.normal_(0,.01)
    x=torch.randn(2,256,9,13,requires_grad=True)
    y=torch.randn(2,256,5,7,requires_grad=True) if kind=='c17' else None
    out=module([x,y]) if kind=='c17' else module(x)
    (out*torch.randn_like(out)).mean().backward()
    result=dict(output=out.detach(),lateral_grad=x.grad,parameter_grads={n:p.grad for n,p in module.named_parameters()})
    if y is not None:
        result['semantic_grad']=y.grad
        confidence=module._compute_structure_confidence(*module._compute_scharr_components(x[:,:32]))
        result['confidence']=confidence
        assert not confidence.requires_grad
    return result


if __name__=='__main__':
    torch.set_num_threads(4)
    payload=torch_load(sys.argv[2],map_location='cpu')
    if len(sys.argv)>4:
        import importlib.util
        fixture=Path(sys.argv[4])
        names=['cscef_v5','cscef_v51'] if payload['kind']=='c17' else ['lif_down']
        for name in names:
            qualified='ultralytics.nn.modules.'+name
            spec=importlib.util.spec_from_file_location(qualified,fixture/(name+'.py'))
            module=importlib.util.module_from_spec(spec);sys.modules[qualified]=module;spec.loader.exec_module(module)
        cls=module.CSCEFv51 if payload['kind']=='c17' else module.LIFDown
        original=cls(256,256)
        # DetectionModel applies this unchanged native BN eps/momentum policy after construction.
        from ultralytics.utils.torch_utils import initialize_weights
        initialize_weights(original)
        original.load_state_dict(payload['module_state'],strict=True)
        torch.save(dict(module=probe(original,payload['kind']),import_path=str(fixture/(names[-1]+'.py'))),sys.argv[3])
        sys.exit(0)
    model=RTDETRDetectionModel(payload['yaml'],nc=1,verbose=False).eval()
    model.load_state_dict(payload['state'],strict=True)
    with torch.no_grad(): out=model(payload['image'])
    result=dict(prediction=out,import_path=ultralytics.__file__)
    if payload['kind']!='c2':
        cls='CSCEFv51' if payload['kind']=='c17' else 'LIFDown'
        module=next(m for m in model.modules() if type(m).__name__==cls)
        result['module']=probe(module,payload['kind'])
    torch.save(result,sys.argv[3])
