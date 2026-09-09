"""Bounded v1 numerical checks. No epoch loop and no full split evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from init_lif_down import (ROOT, MODEL_DIR, VARIANTS, build, controlled_models, is_added,
                           require, runtime, sha256, verify_model, write_json)
from train_lif_down import build_training_model, recipe
from lif_down_topology import locate
from ultralytics.nn.modules import Conv, LIFDown
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load


def seed():
    torch.manual_seed(42)


def compare(a, b, atol=2e-6, rtol=2e-5):
    a, b = a.detach().float(), b.detach().float()
    delta = (a-b).abs()
    row = dict(max_abs_error=float(delta.max()),
               max_rel_error=float((delta / a.abs().clamp_min(1e-8)).max()), atol=atol, rtol=rtol)
    require(torch.allclose(a, b, atol=atol, rtol=rtol), f"Numerical mismatch: {row}")
    return row


def module_checks():
    seed()
    module = LIFDown(256,256).eval()
    base = Conv(256,256,3,2).eval()
    base.load_state_dict({k:v for k,v in module.state_dict().items() if k in base.state_dict()}, strict=True)
    shapes=[]
    for h,w in ((80,80),(79,80),(80,79),(79,81),(1,1)):
        x=torch.randn(2,256,h,w)
        with torch.no_grad():
            a,b=base(x),module(x)
        require(list(b.shape)==[2,256,(h+1)//2,(w+1)//2], 'Odd shape mismatch')
        shapes.append(dict(input=[h,w],output=list(b.shape),equivalence=compare(a,b)))
    # Different channel values reveal grouped-layout errors rather than only checking shape.
    z=torch.tensor([[[[1.,2.],[4.,8.]],[[3.,7.],[11.,19.]]]])
    a,d=module.haar(z)
    require(torch.equal(a.flatten(),torch.tensor([7.5,20.])), 'Haar /2 low band')
    require(torch.equal(d.flatten(),torch.tensor([-2.5,-4.5,1.5,-6.,-10.,2.])), 'Haar detail channel layout')
    odd=torch.tensor([[[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]]])
    a,_=module.haar(odd)
    require(torch.equal(a,torch.tensor([[[[6.,9.],[15.,18.]]]])), 'Odd replicate pad')
    r=torch.tensor([[[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]]])
    expected=torch.tensor([[[[1.,1.75,2.75],[3.25,4.,5.],[6.25,7.,8.]]]])
    require(torch.equal(module.align(r),expected), 'Separable Align hand calculation')
    require(not list(module.align.__dict__), 'Align has no parameters')
    module.train();x=torch.randn(2,256,11,13)
    module(x).square().mean().backward()
    grad=lambda: {n:float(p.grad.norm()) for n,p in module.named_parameters() if n.split('.')[0] in ('B_proj','P','U_mix','U_dw','O_proj')}
    first=grad()
    require(first['O_proj.weight']>0 and all(v==0 for n,v in first.items() if n!='O_proj.weight'), 'Zero-output startup')
    with torch.no_grad(): module.O_proj.weight.add_(module.O_proj.weight.grad,alpha=-.01)
    module.zero_grad(set_to_none=True);module(x).square().mean().backward();second=grad()
    require(all(v>0 for v in second.values()) and all(torch.isfinite(p.grad).all() for p in module.parameters()), 'Lifting branch gradient after O update')
    return dict(shape=shapes,haar_layout='PASSED',odd_replicate='PASSED',align_hand_tensor=expected.tolist(),first_gradient_norms=first,after_O_update_gradient_norms=second)


def real_batch(dataset, size=320):
    import cv2
    images=sorted((dataset/'images/train').glob('*.jpg'))[:2]
    require(len(images)==2, 'Need two real train images')
    tensors,boxes,classes,indices,records=[],[],[],[],[]
    for i,p in enumerate(images):
        im=cv2.imread(str(p));require(im is not None,'Unreadable real image')
        label=dataset/'labels/train'/p.with_suffix('.txt').name
        rows=[list(map(float,line.split())) for line in label.read_text().splitlines() if line.strip()]
        tensors.append(torch.from_numpy(cv2.resize(im,(size,size))[:,:,::-1].copy()).permute(2,0,1).float()/255)
        boxes.extend([r[1:] for r in rows]);classes.extend([[r[0]] for r in rows]);indices.extend([i]*len(rows))
        records.append(dict(image=p.name,image_sha256=sha256(p),label_sha256=sha256(label),instances=len(rows)))
    return dict(img=torch.stack(tensors),bboxes=torch.tensor(boxes).reshape(-1,4),cls=torch.tensor(classes).reshape(-1,1),batch_idx=torch.tensor(indices)),records


def targets(batch):
    return dict(cls=batch['cls'].long().flatten(),bboxes=batch['bboxes'],batch_idx=batch['batch_idx'].long(),
                gt_groups=[int((batch['batch_idx']==i).sum()) for i in range(len(batch['img']))])


def model_checks(reference,target,device,folder,batch):
    a,b=deepcopy(reference).to(device).eval(),deepcopy(target).to(device).eval()
    topology=verify_model(b,zero=True);index=topology['p3_to_p4']['downsample']
    seed();x=torch.randn(1,3,640,640,device=device)
    taps={}
    def capture(name):
        def hook(m,args,out): taps[name]=out.detach()
        return hook
    handles=[a.model[index].register_forward_hook(capture('base')),b.model[index].register_forward_hook(capture('lif'))]
    with torch.no_grad(): ya,yb=a(x),b(x)
    report=dict(downsample=compare(taps['base'],taps['lif']),final=compare(ya[0],yb[0]),
                bbox=compare(ya[0][...,:4],yb[0][...,:4]),score=compare(ya[0][...,4:],yb[0][...,4:]),input=[1,3,640,640])
    for h in handles:h.remove()
    del a,ya,yb,taps;gc.collect()
    if device=='cuda':
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): out=b(x)[0]
        require(torch.isfinite(out).all(),'AMP inference nonfinite')
        report['AMP_inference']=dict(status='PASSED',dtype=str(out.dtype))
    # Nonzero O ensures reload and inference fusion cannot silently drop the residual.
    with torch.no_grad(): b.model[index].O_proj.weight.normal_(std=.01)
    with torch.no_grad(): expected=b(x)[0]
    dest=folder/f'nonzero_{device}.pt';torch.save(dict(model=b),dest)
    c=torch_load(dest,map_location=device)['model']
    require(all(torch.equal(v,c.state_dict()[k]) for k,v in b.state_dict().items()),'Save/load state changed')
    with torch.no_grad(): report['save_load']=compare(expected,c(x)[0])
    c.fuse(verbose=False)
    require(hasattr(c.model[index],'bn') and torch.equal(c.model[index].O_proj.weight,b.model[index].O_proj.weight),'Fusion lost LIF')
    with torch.no_grad(): report['fuse_nonzero']=compare(expected,c(x)[0],atol=2e-5,rtol=2e-4)
    if device=='cuda':
        c.half()
        with torch.no_grad(): out=c(x.half())[0]
        require(out.dtype==torch.float16 and torch.isfinite(out).all(),'True half failed')
        report['model_half']=dict(status='PASSED',dtype=str(out.dtype),shape=list(out.shape))
    del c,x,expected;gc.collect()
    # Native trainer.set_model_attributes supplies nc before the loss is created.
    b.nc = b.model[-1].nc
    b.train();b.zero_grad(set_to_none=True)
    batch={k:v.to(device) for k,v in batch.items()}
    seed();pred=b.predict(batch['img'],batch=targets(batch));loss=b.loss(batch,preds=pred)[0]
    require(pred[-1] is not None and torch.isfinite(loss),'Native DN/loss nonfinite')
    loss.backward()
    require(all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None),'Native backward nonfinite')
    report['DN_loss']=dict(scope='two real training samples, resized smoke only',imgsz=320,batch=2,
                          loss=float(loss),dn_split=pred[-1]['dn_num_split'],criterion=type(b.criterion).__name__,backward='PASSED')
    del pred,loss
    if device=='cuda':
        b.zero_grad(set_to_none=True)
        opt=torch.optim.AdamW(b.parameters(),lr=.0005,weight_decay=.0001)
        with torch.autocast('cuda',dtype=torch.float16): loss=b.loss(batch)[0]
        scaler=torch.cuda.amp.GradScaler(init_scale=128.)
        scaler.scale(loss).backward();scaler.unscale_(opt)
        require(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None),'AMP DN backward nonfinite')
        scaler.step(opt);scaler.update()
        report['AMP_DN']=dict(loss=float(loss),backward='PASSED',update='PASSED',smoke_scaler=128,formal_scaler='native unchanged')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--real-dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    report=dict(runtime=runtime(),status='FAILED',formal_training='NOT_RUN',full_val_test='NOT_RUN',server_batch16='NOT_RUN',
                determinism='TF32 disabled; deterministic algorithms warn_only because native CUDA grid_sample backward is nondeterministic')
    try:
        report['module']=module_checks();print('module checks passed',flush=True)
        _,source,mapping=controlled_models(args.source)
        seed();target,adapt=build_training_model(str(MODEL_DIR/VARIANTS['lif_down'][0]),source,dict(nc=1,channels=3),'lif_down')
        seed();reference=build(nc=1,baseline=True);reference.load(source,verbose=False)
        require(all(torch.equal(v,target.state_dict()[k]) for k,v in reference.state_dict().items()),'Public nc1 state mismatch')
        mapping['nc1_adaptation']=adapt;write_json(ROOT/'docs/lif_down/initialization_mapping.json',mapping)
        report['parameters']=dict(baseline=sum(p.numel() for p in reference.parameters()),lif=sum(p.numel() for p in target.parameters()))
        report['parameters']['delta']=report['parameters']['lif']-report['parameters']['baseline']
        report['topology']=verify_model(target,zero=True)
        report['graph']=[dict(index=i,type=type(m).__name__,source=m.f) for i,m in enumerate(target.model)]
        print(report['graph'],flush=True)
        batch,records=real_batch(args.real_dataset);report['real_samples']=records
        for device in ('cpu','cuda'):
            if device=='cuda' and not torch.cuda.is_available():report['cuda']='NOT_RUN';continue
            report[device]=model_checks(reference,target,device,args.output,batch)
            print(device+' model checks passed',flush=True);gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        resolved,rows=recipe(ROOT/'docs/lif_down/c2_args.yaml','lif_down',Path('/root/autodl-tmp/projects/Crack_RTDETR-lif-down/weights/lif_down_controlled_init.pt'))
        # Document Linux paths even when this audit runs on Windows.
        resolved['model']='/root/autodl-tmp/projects/Crack_RTDETR-lif-down/weights/lif_down_controlled_init.pt'
        for key in ('project','data','save_dir'):
            resolved[key]=resolved[key].replace('\\','/')
        for row in rows:
            row['lif_down']=resolved[row['field']]
            row['changed']=row['C2'] != row['lif_down']
            if not row['changed']:row.pop('reason',None)
        YAML.save(ROOT/'docs/lif_down/resolved_formal_config.yaml',resolved)
        write_json(ROOT/'docs/lif_down/recipe_diff.json',rows)
        report['status']='PASSED'
    finally:
        report['source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in
                                [ROOT/'ultralytics-main/ultralytics/nn/modules/lif_down.py',ROOT/'tools/check_lif_down.py']}
        write_json(args.output/'checks.json',report)
        write_json(ROOT/'docs/lif_down/checks.json',report)


if __name__=='__main__':main()
