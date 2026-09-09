"""Bounded RSC checks: no formal training and no full-dataset evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import math
import os
from pathlib import Path
import random
import subprocess

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from init_rsc_head import ROOT, build, controlled_models, initialize, is_added, require, runtime, sha256, verify_model, write_json
from train_rsc_head import rebuild_audit, recipe, optimizer_groups, build_training_model
from ultralytics.nn.modules.rsc_head import RSCHead, RTDETRDecoderRSC, refinement_descriptor
from ultralytics.nn.modules import RTDETRDecoder
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load


def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)


def compare(a, b, atol=1e-6, rtol=1e-5):
    require(a.shape == b.shape, "Output shape changed")
    delta = (a.detach().float() - b.detach().float()).abs()
    result = dict(max_abs=float(delta.max()), max_rel=float((delta / a.detach().float().abs().clamp_min(1e-8)).max()),
                  atol=atol, rtol=rtol, shape=list(a.shape))
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return result


def scalar_descriptor(history):
    """Independent double-precision scalar oracle, only for these small unit cases."""
    def iou(a, b):
        aw, ah = max(a[2], 0), max(a[3], 0)
        bw, bh = max(b[2], 0), max(b[3], 0)
        iw = max(0, min(a[0]+aw/2, b[0]+bw/2)-max(a[0]-aw/2, b[0]-bw/2))
        ih = max(0, min(a[1]+ah/2, b[1]+bh/2)-max(a[1]-ah/2, b[1]-bh/2))
        inter = iw*ih
        return inter / max(aw*ah+bw*bh-inter, 1e-12)
    states = []
    for a,b in zip(history, history[1:]):
        aw,ah,bw,bh = max(a[2],1e-4),max(a[3],1e-4),max(b[2],1e-4),max(b[3],1e-4)
        states.append([math.tanh(v) for v in ((b[0]-a[0])/aw,(b[1]-a[1])/ah,math.log(bw/aw),math.log(bh/ah))])
    w,h = max(history[2][2],1e-4),max(history[2][3],1e-4)
    return states[0]+states[1]+[a*b for a,b in zip(*states)]+[iou(*history[:2]),iou(*history[1:]),math.tanh(math.log(w/h)/4),math.tanh(math.log(w*h)/8)]


def head_checks():
    seed()
    cases = [
        [[.5,.5,.2,.2]]*3,
        [[.5,.5,.2,.2],[.6,.4,.2,.2],[.7,.3,.2,.2]],
        [[.5,.5,.2,.2],[.5,.5,.4,.1],[.5,.5,.2,.2]],
        [[.5,.5,.2,.2],[.6,.5,.2,.2],[.5,.5,.2,.2]],
        [[.99,0.,.8,.00001],[1.1,0.,.8,.00002],[1.2,0.,.8,.00001]],
        [[.5,.5,0,0],[.5,.5,-.1,1e-9],[.5,.5,1e-9,0]],
    ]
    histories = torch.tensor(cases).transpose(0,1).unsqueeze(1)
    g = refinement_descriptor(histories)
    oracle = torch.tensor([scalar_descriptor(c) for c in cases]).unsqueeze(0)
    result = dict(descriptor=compare(oracle,g,atol=2e-6), cases=["identical","translation","scale_reversal","direction_reversal","thin_outside_image","degenerate"])
    require(g[0,3,8] < 0 and g[0,0,12] == 1 and g[0,5,12] == 0, "Direction/IoU order")
    head = RSCHead(nn.Linear(256,1))
    q = torch.randn(2,503,256,requires_grad=True)
    boxes = [torch.rand(2,503,4,requires_grad=True) for _ in range(3)]
    q0, b0 = q.detach().clone(), [b.detach().clone() for b in boxes]
    result["initial_logits"] = compare(F.linear(q,head.weight,head.bias),head(q,boxes))
    head(q,boxes).square().mean().backward()
    require(q.grad is not None and q.grad.abs().sum()>0 and all(b.grad is None for b in boxes), "Detach boundary")
    require(head.mod_proj.weight.grad.abs().sum()>0, "mod_proj cannot learn")
    result["first_grad_norms"]={n:float(p.grad.norm()) for n,p in head.named_parameters()}
    opt = torch.optim.AdamW(head.parameters(),lr=.001)
    opt.step(); opt.zero_grad(); q.grad=None
    head(q,boxes).square().mean().backward()
    result["second_grad_norms"]={n:float(p.grad.norm()) for n,p in head.named_parameters()}
    require(all(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters()), "Second-step head gradient disconnected")
    require(all(b.grad is None for b in boxes), "Learned head leaks box gradient")
    perm=torch.randperm(503)
    result["query_permutation"]=compare(head(q,boxes)[:,perm],head(q[:,perm],[b[:,perm] for b in boxes]))
    result["descriptor_permutation"]=compare(refinement_descriptor(boxes)[:,perm],refinement_descriptor([b[:,perm] for b in boxes]))
    changed=[b.detach().clone() for b in boxes];changed[2][:,7,:]=torch.tensor([.7,.1,.8,.002])
    mask=torch.arange(503)!=7
    result["other_query_independence"]=compare(refinement_descriptor(boxes)[:,mask],refinement_descriptor(changed)[:,mask])
    effect=float((head(q,boxes)-head(q,changed)).detach().abs().max())
    require(effect>0,"Trajectory does not affect learned classification")
    result["trajectory_logit_effect"]=effect
    require(torch.equal(q,q0) and all(torch.equal(a,b) for a,b in zip(boxes,b0)),"Input mutated")
    for invalid in (boxes[:2],boxes+[boxes[0]]):
        try:refinement_descriptor(invalid)
        except ValueError:pass
        else:raise AssertionError("Invalid history accepted")
    for kwargs in ({"ndl":2},{"ndl":3,"eval_idx":1}):
        try:RTDETRDecoderRSC(nc=1,**kwargs)
        except ValueError:pass
        else:raise AssertionError("Unsupported decoder accepted")
    result.update(query_count=503,inputs_unchanged=True,box_gradients=None,invalid_configs_rejected=True)
    return result


def native_optimizer(model):
    return BaseTrainer.build_optimizer(None,model,name="AdamW",lr=.0005,momentum=.937,decay=.0001)


def prepare(source, folder):
    reference80,target80,mapping=controlled_models(source)
    reference,target=build(nc=1,baseline=True),build(nc=1)
    reference.load(reference80,verbose=False);target.load(target80,verbose=False)
    # Normally supplied by DetectionTrainer.set_model_attributes before native loss.
    reference.nc = target.nc = 1
    audit=rebuild_audit(target80,target,"rsc_head")
    public=reference.state_dict();new=target.state_dict()
    require(all(torch.equal(v,new[k]) for k,v in public.items()),"nc1 adapted public states differ")
    verify_model(target,zero=True)
    mapping.update(nc1_loading=audit,nc1_public_states=len(public),nc1_public_equal=True,
                   nc1_baseline_parameters=sum(p.numel() for p in reference.parameters()),
                   nc1_rsc_parameters=sum(p.numel() for p in target.parameters()))
    opt=native_optimizer(target)
    ids=[id(p) for group in opt.param_groups for p in group['params']]
    require(all(ids.count(id(p))==1 and p.requires_grad for p in target.parameters()),"Optimizer missing/duplicate/frozen parameter")
    mapping['optimizer_groups']=optimizer_groups(target,opt)
    seed()
    _, mapping['native_training_rebuild'] = build_training_model(target80.yaml, target80, dict(nc=1,channels=3), 'rsc_head')
    write_json(folder/'initialization_mapping.json',mapping)
    return reference,target,mapping


def targets(batch):
    return dict(cls=batch['cls'].long().view(-1),bboxes=batch['bboxes'],batch_idx=batch['batch_idx'].long(),
                gt_groups=[int((batch['batch_idx']==i).sum()) for i in range(len(batch['img']))])


def model_checks(reference,target,device,folder):
    a,b=deepcopy(reference).to(device),deepcopy(target).to(device)
    seed();x=torch.rand(2,3,160,160,device=device)
    batch=dict(img=x,cls=torch.zeros(3,1,device=device),batch_idx=torch.tensor([0,0,1],device=device),
               bboxes=torch.tensor([[.4,.5,.2,.05],[.6,.6,.1,.3],[.5,.5,.4,.07]],device=device))
    report={}
    a.eval();b.eval()
    with torch.no_grad():
        outa,outb=a(x),b(x)
    report['eval_prediction']=compare(outa[0],outb[0])
    report['eval_raw']=[compare(u,v) for u,v in zip(outa[1][:4],outb[1][:4])]
    # Native random DN and matching retained; reset full RNG before each forward.
    a.train();b.train()
    seed();pa=a.predict(x,batch=targets(batch))
    seed();pb=b.predict(x,batch=targets(batch))
    report['train_outputs']=[compare(u,v) for u,v in zip(pa[:4],pb[:4])]
    require(pa[-1]['dn_num_split']==pb[-1]['dn_num_split'] and pa[-1]['dn_num_split'][0]>0,"Native DN missing")
    for u,v in zip(pa[-1]['dn_pos_idx'],pb[-1]['dn_pos_idx']):require(torch.equal(u,v),'DN index changed')
    loss_a=a.loss(batch,preds=pa);loss_b=b.loss(batch,preds=pb)
    report['native_dn_loss']=compare(loss_a[0],loss_b[0])
    report['dn_split']=pb[-1]['dn_num_split']
    require(b.criterion.vfl is not None,"VFL disabled")
    loss_b[0].backward()
    report['bbox_gradients']={n:float(p.grad.norm()) for n,p in b.named_parameters() if 'dec_bbox_head' in n and p.grad is not None}
    require(all(any(v>0 for k,v in report['bbox_gradients'].items() if f'dec_bbox_head.{i}.' in k) for i in range(3)),"Original bbox regression lost gradients")
    del pa,pb,loss_a,loss_b,a
    b.zero_grad(set_to_none=True)
    # Nonzero regression in isolated interface test: detects input-reference vs actual b3 mixups.
    head=b.model[-1]
    with torch.no_grad():
        for i,h in enumerate(head.dec_bbox_head):h.layers[-1].bias.copy_(torch.tensor([.02,-.03,.04,-.01],device=device)*(i+1))
        head.dec_score_head[2].mod_proj.weight.normal_(0,.02)
    seen=[]
    handle=head.dec_score_head[2].register_forward_pre_hook(lambda m,args: seen.append(([z.detach().clone() for z in args[1]],args[0].detach().clone())))
    # BN held in eval, only decoder train flag differs; no DN in this interface-only comparison.
    b.eval();head.training=True;head.decoder.train()
    with torch.no_grad():train=b(x)
    train_history=seen.pop()[0]
    report['history_vs_loss_boxes']=[compare(train[0][i],train_history[i]) for i in range(3)]
    head.eval()
    calls=[0,0]
    hooks=[head.dec_score_head[i].register_forward_hook(lambda m,args,out,i=i:calls.__setitem__(i,calls[i]+1)) for i in range(2)]
    with torch.no_grad():evaluation=b(x)
    eval_history=seen.pop()[0]
    report['train_eval_history']=[compare(u,v) for u,v in zip(train_history,eval_history)]
    report['final_history_output']=compare(eval_history[2],evaluation[1][0][0])
    head.export=True
    with torch.no_grad():export=b(x)
    report['export_output']=compare(evaluation[0],export)
    report['eval_export_history']=[compare(u,v) for u,v in zip(eval_history,seen.pop()[0])]
    require(calls==[0,0],"Inference reran auxiliary classification")
    handle.remove()
    for h in hooks:h.remove()
    head.export=False
    # Save/reload whole model and native optimizer after a real update, with learned modulation.
    b.train();opt=native_optimizer(b);seed();loss=b.loss(batch)[0];loss.backward();opt.step();b.eval()
    with torch.no_grad():expected=b(x)[0]
    dest=folder/f'reload_{device}.pt'
    torch.save(dict(model=b,optimizer=opt.state_dict()),dest)
    loaded=torch_load(dest,map_location='cpu');c=loaded['model'].to(device);opt2=native_optimizer(c);opt2.load_state_dict(loaded['optimizer'])
    with torch.no_grad():report['checkpoint_output']=compare(expected,c(x)[0])
    require(all(torch.equal(v,c.state_dict()[k]) for k,v in b.state_dict().items()),'Reload modified learned params')
    require(len(opt.state)==len(opt2.state),'Optimizer state missing')
    for g1,g2 in zip(opt.param_groups,opt2.param_groups):
        for p1,p2 in zip(g1['params'],g2['params']):
            for k,v in opt.state[p1].items():
                if torch.is_tensor(v):require(torch.equal(v.cpu(),opt2.state[p2][k].cpu()),'Optimizer reload mismatch')
    report['optimizer_reload_exact']=True
    report['checkpoint_sha256']=sha256(dest)
    if device=='cuda':
        c.train();c.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():loss=c.loss(batch)[0]
        scaler=torch.cuda.amp.GradScaler(init_scale=128.)
        scaler.scale(loss).backward();scaler.unscale_(opt2)
        require(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in c.parameters() if p.grad is not None),'AMP nonfinite')
        scaler.step(opt2);scaler.update()
        report['amp']=dict(status='PASSED',loss=float(loss),batch=2,imgsz=160,synthetic=True,scaler=float(scaler.get_scale()))
        c.half().eval()
        with torch.no_grad():half=c(x.half())[0]
        require(half.dtype==torch.float16 and torch.isfinite(half).all(),'True half inference failed')
        report['model_half']=dict(status='PASSED',dtype=str(half.dtype),shape=list(half.shape))
    return report


def real_sample_check(model,dataset,folder):
    import cv2
    from rsc_head_results import postprocess,image_record
    images=sorted((dataset/'images/train').glob('*'))[:2]
    require(len(images)==2,'Two real train images required')
    tensors,boxes,classes,indices,records=[],[],[],[],[]
    for i,p in enumerate(images):
        im=cv2.imread(str(p));require(im is not None,'Cannot read real sample')
        label=dataset/'labels/train'/p.with_suffix('.txt').name
        rows=[list(map(float,l.split())) for l in label.read_text().splitlines() if l.strip()]
        tensors.append(torch.from_numpy(cv2.resize(im,(160,160))[:,:,::-1].copy()).permute(2,0,1).float()/255)
        classes.extend([[r[0]] for r in rows]);boxes.extend([r[1:] for r in rows]);indices.extend([i]*len(rows))
        records.append(dict(image=p.name,image_sha256=sha256(p),label_sha256=sha256(label),instances=len(rows),shape=im.shape[:2]))
    batch=dict(img=torch.stack(tensors),bboxes=torch.tensor(boxes).reshape(-1,4),cls=torch.tensor(classes).reshape(-1,1),batch_idx=torch.tensor(indices))
    m=deepcopy(model).train();seed();pred=m.predict(batch['img'],batch=targets(batch));loss=m.loss(batch,preds=pred)[0];loss.backward()
    require(pred[-1] is not None and torch.isfinite(loss),'Real native DN/loss failed')
    m.eval()
    with torch.no_grad():evaluation=m(batch['img'])[0]
    full,_=postprocess(evaluation,160,-float('inf'));selected,_=postprocess(evaluation,160,.001)
    stream=[]
    from ultralytics.utils.ops import xywh2xyxy
    for i in range(2):
        mask=batch['batch_idx']==i
        gt=dict(ori_shape=records[i]['shape'],imgsz=(160,160),im_file=str(images[i]),cls=batch['cls'][mask].view(-1),bboxes=xywh2xyxy(batch['bboxes'][mask])*160)
        row=image_record(full[i],gt,dataset,.001)
        require(len(row['predictions'])==300 and sum(p['used_for_metrics'] for p in row['predictions'])==len(selected[i]['conf']),'Real export alignment')
        stream.append(row)
    write_json(folder/'real_sample_predictions_gt.json',stream)
    return dict(status='PASSED',samples=records,loss=float(loss),dn_split=pred[-1]['dn_num_split'],imgsz=160,batch=2,
                scope='two real train samples native DN/loss backward and same-pass evaluation exporter; no dataset metrics')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--real-dataset',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/rsc_head_checks')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    # Native CUDA grid_sample backward has no deterministic implementation; record warnings honestly.
    torch.use_deterministic_algorithms(True,warn_only=True)
    result=dict(runtime=runtime(),tf32=False,deterministic_algorithms='warn_only (native grid_sample backward)',full_server_preflight='NOT_RUN',
                formal_batch16_memory='NOT_RUN',full_training='NOT_RUN',full_dataset_val_test='NOT_RUN',server_4090_torch212='NOT_RUN')
    result['source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in
        [ROOT/'ultralytics-main/ultralytics/nn/modules/rsc_head.py',ROOT/'ultralytics-main/ultralytics/nn/tasks.py',
         ROOT/'ultralytics-main/ultralytics/nn/modules/__init__.py',ROOT/'tools/init_rsc_head.py',ROOT/'tools/train_rsc_head.py',ROOT/'tools/check_rsc_head.py']}
    try:
        result['head']=head_checks();print('head checks passed',flush=True)
        reference,target,mapping=prepare(args.source,args.output)
        result['parameters']={k:v for k,v in mapping.items() if k.startswith('nc1_')};print('initialization checks passed',flush=True)
        cfg,rows=recipe(ROOT/'docs/rsc_head/c2_args.yaml','rsc_head',Path('/root/autodl-tmp/projects/Crack_RTDETR-rsc-head/weights/rsc_head_controlled_init.pt'))
        YAML.save(args.output/'resolved_formal_config.yaml',cfg);write_json(args.output/'recipe_diff.json',rows)
        for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
            result[device]=model_checks(reference,target,device,args.output);print(device+' model checks passed',flush=True)
        if not torch.cuda.is_available():result['cuda']='NOT_RUN'
        result['real_samples']=real_sample_check(target,args.real_dataset,args.output) if args.real_dataset else 'NOT_RUN'
        write_json(args.output/'initialization_serialization.json',initialize(args.source,args.output/'controlled_init.pt','rsc_head'))
        result['status']='PASSED'
    except BaseException as error:
        result.update(status='FAILED',error=repr(error));raise
    finally:
        write_json(args.output/'checks.json',result)


if __name__=='__main__':main()
