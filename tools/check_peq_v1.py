"""Bounded PEQ algorithm, native-loss, gradient and lifecycle checks; never trains an experiment."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import traceback
import numpy as np
import torch
from peq_v1_common import *
from init_peq_v1 import build
from ultralytics.nn.modules.peq import PrecisionEvidenceQuality, ordinary
from ultralytics.models.utils.peq_loss import quality_targets, quality_loss, threshold_targets, aligned_iou, PEQDetectionLoss
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.rtdetr.peq_train import PEQTrainer, PEQValidator, parameter_partition, corrected_postprocess
from ultralytics.models.rtdetr.peq_model import PEQDetectionModel
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils.patches import torch_load
from ultralytics.utils import YAML
from ultralytics import RTDETR


def close(a,b,atol=2e-6,rtol=2e-5):
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


def sampling_and_targets(device):
    module=PrecisionEvidenceQuality().to(device)
    require(sum(p.numel() for p in module.parameters())==17994,"PEQ parameter budget")
    height,width=4,7
    yy,xx=torch.meshgrid(torch.arange(height),torch.arange(width),indexing="ij")
    values=torch.stack([xx.float()+10*yy,2*(xx.float()+10*yy)+100]).unsqueeze(1).to(device)
    values=values.expand(2,16,height,width).contiguous()
    values[0,1]=0;values[0,1,2,3]=1
    boxes=torch.tensor([[[.5,.5,.9,.8],[.05,.9,.6,.8],[.8,.2,1e-8,.01]],
                        [[.7,.3,.7,.9],[1.1,-.1,.5,.7],[.5,.5,.01,1e-8]]],device=device)
    samples,relative,oob=module.sample(values,boxes)
    expected=torch.zeros_like(samples)
    for image in range(2):
        for q in range(3):
            for j,(u,v) in enumerate(relative.tolist()):
                b=boxes[image,q].tolist()
                x=min(max((b[0]+b[2]*u)*width-.5,0),width-1)
                y=min(max((b[1]+b[3]*v)*height-.5,0),height-1)
                x0,y0=int(x),int(y);x1,y1=min(x0+1,width-1),min(y0+1,height-1)
                ax,ay=x-x0,y-y0
                expected[image,q,j]=((1-ax)*(1-ay)*values[image,:,y0,x0]+ax*(1-ay)*values[image,:,y0,x1]+
                                      (1-ax)*ay*values[image,:,y1,x0]+ax*ay*values[image,:,y1,x1])
    close(samples,expected,atol=2e-5)
    e0,_=module.evidence(values,boxes);e1,_=module.evidence(values,boxes)
    close(e0,e1,atol=0,rtol=0)
    changed=values.clone();changed[1,:,1:3,2:5]+=10
    require(not torch.equal(module.evidence(changed,boxes)[0],e1),"Local visual evidence is unused")
    taus=torch.tensor([.5,.55,.6,.65,.7,.75,.8,.85,.9,.95],device=device)
    target=threshold_targets(taus)
    require(torch.equal(target,torch.tril(torch.ones(10,10,device=device))),">= threshold boundary")
    b=torch.tensor([[[.5,.5,.4,.4],[.2,.2,.1,.1]],[[.5,.5,.4,.4],[.5,.5,.4,.4]]],device=device,requires_grad=True)
    gt=torch.tensor([[.5,.5,.4,.4],[.1,.1,.1,.1],[.5,.5,.4,.4]],device=device,requires_grad=True)
    matches=[(torch.tensor([0]),torch.tensor([1])),(torch.tensor([1]),torch.tensor([2]))]
    targets,mask=quality_targets(b,gt,matches,[2,1])
    require(targets[0,0].sum()==0 and targets[1,1].sum()==10,"Assigned GT must differ from max-IoU proxy")
    require(not targets.requires_grad,"GT targets need detach")
    logits=torch.full((2,2,10),-10000.,device=device,requires_grad=True)
    qloss,details=quality_loss(logits,targets,mask)
    require(torch.isfinite(qloss),"Extreme BCE nonfinite")
    qloss.backward()
    require(b.grad is None and gt.grad is None,"Quality targets leak gradients")
    empty=torch.empty(0,4,device=device)
    t,m=quality_targets(b,empty,[(torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long))]*2,[0,0])
    loss,empty_details=quality_loss(torch.zeros_like(t),t,m)
    close(loss,torch.tensor(np.log(2),device=device,dtype=torch.float32))
    t=torch.ones(1,1,10,device=device);m=torch.ones(1,1,device=device,dtype=torch.bool)
    loss,all_details=quality_loss(torch.zeros_like(t),t,m)
    close(loss,torch.tensor(np.log(2),device=device,dtype=torch.float32))
    for dn in (0,4,202,604):
        tensor=torch.arange((dn+300)*2).reshape(1,dn+300,2)
        close(ordinary(tensor,{"dn_num_split":[dn,300]}),tensor[:,dn:],atol=0,rtol=0)
    try:
        module(torch.zeros(1,256,4,7,device=device),torch.zeros(1,2,256,device=device),
               b[:1],b[:1],torch.zeros(1,2,80,device=device))
        raise AssertionError("nc80 forward accepted")
    except ValueError:
        pass
    try:
        quality_targets(b*float("nan"),empty,[(torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long))]*2,[0,0])
        raise AssertionError("Nonfinite silently ignored")
    except FloatingPointError:
        pass
    return dict(manual_points=54,non_square=[height,width],multiple_images=2,oob_points=int(oob.sum()),
                assigned_gt_not_max_iou=True,empty_group_normalization=True,threshold_equality=True,
                dynamic_dn_splits=[0,4,202,604],nonfinite_fails=True)


def branch_learning(device):
    model=PrecisionEvidenceQuality().to(device)
    features=torch.rand(2,256,7,11,device=device,requires_grad=True)
    query=torch.rand(2,5,256,device=device,requires_grad=True)
    b0=torch.tensor([.5,.5,.7,.6],device=device).expand(2,5,4).clone().requires_grad_()
    b1=(b0.detach()+.01).requires_grad_()
    z=torch.linspace(-20,20,10,device=device).reshape(2,5,1).requires_grad_()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    first=model(features,query,b0,b1,z)
    close(first["s_final"],z.sigmoid(),atol=1e-7)
    target=torch.zeros(2,5,10,device=device)
    target[:,:,::2]=1
    mask=torch.zeros(2,5,device=device,dtype=torch.bool);mask[:,0]=True
    values=[]
    for step in range(2):
        optimizer.zero_grad()
        payload=model(features,query,b0,b1,z)
        loss,_=quality_loss(payload["quality_logits"],target,mask)
        loss.backward()
        require(all(x.grad is None for x in (features,query,b0,b1,z)),"Input gradient escaped detach")
        norms={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
        if step==0:
            require(norms["output.weight"]>0 and norms["p3_proj.weight"]==0,"Zero-init first-step behavior")
        else:
            require(norms["p3_proj.weight"]>0 and norms["query_proj.weight"]>0,"New projections did not learn after output update")
        values.append(norms)
        optimizer.step()
    with torch.no_grad(),torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
        payload=model(features,query,b0,b1,z)
    require(bool((payload["delta"].abs()<=1).all()) and bool(((payload["s_final"]>=0)&(payload["s_final"]<=1)).all()),"Bounded scores")
    return dict(steps=values,amp_or_fp32_finite=True)


def batch_fixture(device):
    return dict(img=torch.rand(2,3,160,160,device=device),bboxes=torch.tensor([[.5,.5,.35,.2],[.3,.4,.15,.3]],device=device),
                cls=torch.zeros(2,1,device=device),batch_idx=torch.zeros(2,device=device,dtype=torch.long))


def native_and_gradients(device):
    model=build(1).to(device).train()
    batch=batch_fixture(device)
    losses=model.loss_components(batch)
    native=sum(v for k,v in losses.items() if k!="loss_peq")
    qloss=losses["loss_peq"]
    native.backward(retain_graph=True)
    gradients={n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters() if ".peq." not in n}
    require(all(p.grad is None for n,p in model.named_parameters() if ".peq." in n),"Native loss trains PEQ")
    model.zero_grad(set_to_none=True)
    native.backward(retain_graph=True)
    repeat_noise={n:float((p.grad-gradients[n]).abs().max()) for n,p in model.named_parameters()
                  if n in gradients and gradients[n] is not None}
    repeat_l2={n:float((p.grad-gradients[n]).float().norm()) for n,p in model.named_parameters()
               if n in gradients and gradients[n] is not None}
    gradient_comparisons={}
    model.zero_grad(set_to_none=True)
    qloss.backward(retain_graph=True)
    require(all(p.grad is None for n,p in model.named_parameters() if ".peq." not in n),"PEQ loss trains detector")
    model.zero_grad(set_to_none=True)
    (native+qloss).backward()
    max_diff=0.
    for n,p in model.named_parameters():
        if ".peq." in n:
            continue
        ref=gradients[n]
        require((ref is None)==(p.grad is None),f"Original gradient route changed: {n}")
        if ref is not None:
            if device.type=="cpu":
                close(p.grad,ref,atol=2e-6,rtol=2e-5)
            else:
                # CUDA grid_sample backward uses atomic accumulation. Compare its
                # native/native repeat noise AND a tight relative vector bound.
                error=float((p.grad-ref).float().norm())
                reference_norm=float(ref.float().norm())
                bound=max(4*repeat_l2[n],1e-6*ref.numel()**.5+2e-5*reference_norm)
                gradient_comparisons[n]=dict(native_repeat_l2=repeat_l2[n],native_plus_peq_l2=error,
                                            reference_l2=reference_norm,bound=bound)
                require(error<=bound,f"CUDA original gradient changed beyond native noise: {n}: {error}, {bound}")
                tolerance=max(2e-5,4*repeat_noise[n],2e-5*float(ref.abs().max()))
                close(p.grad,ref,atol=tolerance,rtol=3e-4)
            max_diff=max(max_diff,float((p.grad-ref).abs().max()))
    original,added=parameter_partition(model)
    references=[]
    for value in gradients.values():
        if value is not None:
            dummy=torch.zeros_like(value,requires_grad=True)
            dummy.grad=value
            references.append(dummy)
    torch.nn.utils.clip_grad_norm_(references,10)
    torch.nn.utils.clip_grad_norm_(original,10);torch.nn.utils.clip_grad_norm_(added,10)
    for n,p in model.named_parameters():
        if n in gradients and gradients[n] is not None:
            close(p.grad,gradients[n],atol=2e-5,rtol=3e-4)
    model.zero_grad(set_to_none=True)
    # Compare every native component/matcher call on the SAME raw forward.
    targets=dict(cls=batch["cls"].long().flatten(),bboxes=batch["bboxes"],batch_idx=batch["batch_idx"],gt_groups=[2,0])
    raw=model.predict(batch["img"],batch=targets)
    dboxes,dscores,eboxes,escores,dn,payload=raw
    dnbox,dboxes=torch.split(dboxes,dn["dn_num_split"],dim=2)
    dnscore,dscores=torch.split(dscores,dn["dn_num_split"],dim=2)
    preds=(torch.cat((eboxes[None],dboxes)),torch.cat((escores[None],dscores)))
    c0=RTDETRDetectionLoss(nc=1,use_vfl=True);c1=PEQDetectionLoss(nc=1,use_vfl=True)
    calls=[[],[]]
    class Recorder(torch.nn.Module):
        def __init__(self,inner,output):
            super().__init__();self.inner=inner;self.output=output
        def forward(self,boxes,scores,*args,**kwargs):
            self.output.append((boxes.detach().clone(),scores.detach().clone()))
            return self.inner(boxes,scores,*args,**kwargs)
    c0.matcher=Recorder(c0.matcher,calls[0]);c1.matcher=Recorder(c1.matcher,calls[1])
    a=c0(preds,targets,dnbox,dnscore,dn)
    b=c1(preds,targets,dnbox,dnscore,dn,payload)
    require(list(b)==list(a)+["loss_peq"],"Changed native summation/insertion order")
    for key in a:
        close(a[key],b[key],atol=0,rtol=0)
    require(len(calls[0])==len(calls[1])==4,"Final/encoder/aux matcher call count changed")
    for ref,actual in zip(calls[0],calls[1]):
        close(ref[0],actual[0],atol=0,rtol=0);close(ref[1],actual[1],atol=0,rtol=0)
    # G>Q is checked through the real Hungarian matcher and global offsets.
    gt=torch.rand(7,4,device=device)*.5+.2
    assignment=c1.matcher.inner(torch.rand(2,3,4,device=device)*.5+.2,torch.zeros(2,3,1,device=device),
                                gt,torch.zeros(7,dtype=torch.long,device=device),[5,2])
    _,mask=quality_targets(torch.ones(2,3,4,device=device)*.5,gt,assignment,[5,2])
    require(int(mask.sum())==5,"G>Q matching/target count")
    return dict(native_components={k:float(v.detach()) for k,v in a.items()},exact_native_components=True,
                final_encoder_aux_matcher_calls=len(calls[0]),native_uncut_gradient_max_difference=max_diff,
                native_repeat_max_difference=max(repeat_noise.values()),cuda_vector_rtol=2e-5,
                cuda_gradient_comparisons=gradient_comparisons,cuda_absolute_per_element_floor=1e-6,
                isolated_quality_backward=True,partitioned_clipping_equal=True,dn_split=dn["dn_num_split"],g_greater_q=True)
def dynamic_dn_and_amp(device):
    model=build(1).to(device).train()
    records=[]
    for count in (0,1,7,301):
        batch=dict(img=torch.rand(2,3,160,160,device=device),
                   cls=torch.zeros(count,1,device=device),
                   bboxes=torch.rand(count,4,device=device)*.4+.2,
                   batch_idx=torch.zeros(count,device=device,dtype=torch.long))
        with torch.no_grad():
            losses=model.loss_components(batch)
        require(all(bool(torch.isfinite(v)) for v in losses.values()),"Dynamic DN/empty batch nonfinite")
        targets=dict(cls=batch["cls"].long().flatten(),bboxes=batch["bboxes"],batch_idx=batch["batch_idx"],gt_groups=[count,0])
        with torch.no_grad():
            raw=model.predict(batch["img"],batch=targets)
        require(raw[5]["quality_logits"].shape==(2,300,10),"PEQ included DN in quality loss")
        model.model[-1].peq.config["enabled"]=False
        with torch.no_grad():
            disabled=model.loss(batch,preds=raw[:5])[0]
        model.model[-1].peq.config["enabled"]=True
        # Use exactly the same forward for disabled/native loss equality.
        with torch.no_grad():
            components=model.loss_components(batch,preds=raw)
        close(disabled,sum(v for k,v in components.items() if k!="loss_peq"),atol=0,rtol=0)
        records.append(dict(gt=count,dn_split=raw[4]["dn_num_split"] if raw[4] else None,
                            ordinary_quality_shape=list(raw[5]["quality_logits"].shape)))
    result=dict(dynamic_dn=records,disabled_native_loss_exact=True)
    if device.type=="cuda":
        model.zero_grad(set_to_none=True)
        batch=batch_fixture(device)
        with torch.autocast(device_type="cuda",enabled=True):
            loss,_=model(batch)
        scaler=torch.cuda.amp.GradScaler(enabled=True,init_scale=128)
        scaler.scale(loss).backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),"AMP model gradients nonfinite")
        result["amp_loss"]=float(loss.detach())
        model.zero_grad(set_to_none=True)
        model.half().eval()
        with torch.no_grad():
            y,raw=model(batch["img"].half())
            val_loss,_=model.loss(batch,(y,raw))
        require(bool(torch.isfinite(y).all()) and bool(torch.isfinite(val_loss)),"Native half validation route nonfinite")
        result["half_validation_loss"]=float(val_loss)
        result["quality_logits_dtype"]=str(raw[5]["quality_logits"].dtype)
    return result


def lifecycle(device, folder):
    if INIT.is_file():
        from ultralytics.models.rtdetr.peq_train import checked_training_model
        initial=torch_load(INIT,map_location="cpu")["model"]
        model,_=checked_training_model(initial.yaml,initial,dict(nc=1,channels=3),verbose=False)
        del initial
        fixture_source=identity(INIT)
    else:
        model=build(1)
        fixture_source={"status":"RANDOM_INTERFACE_FIXTURE_NO_PUBLIC_ASSET"}
    model=model.to(device).eval()
    with torch.no_grad():
        model.model[-1].peq.output.weight.normal_(std=.02)
        model.model[-1].peq.output.bias.copy_(torch.linspace(-.8,.8,10,device=device))
        model.model[-1].cbr.offset_out.weight.normal_(std=.015)
        model.model[-1].cbr.offset_out.bias.fill_(.2)
        model.model[20].O_proj.weight.normal_(std=.002)
    from init_c19_lif_v1 import build as mother_build
    mother=mother_build(nc=1).to(device).eval()
    mother.load_state_dict({k:model.state_dict()[k] for k in mother.state_dict()},strict=True)
    x=torch.rand(2,3,160,160,device=device)
    with torch.no_grad():
        y,raw=model(x);base,base_raw=mother(x)
    close(y[...,:4],base[...,:4],atol=0,rtol=0)
    close(raw[1],base_raw[1],atol=0,rtol=0)
    require(float((y[...,4:]-raw[-1]["s_raw"]).abs().max())>1e-4,"Nonzero PEQ has no scoring effect")
    require(float((raw[-1]["b1"]-raw[-1]["b0"]).abs().max())>1e-6,"Lifecycle CBR must be nonzero")
    model.model[-1].peq.config["enabled"]=False
    with torch.no_grad():
        disabled=model(x)[0]
    close(disabled,base,atol=0,rtol=0)
    model.model[-1].peq.config["enabled"]=True
    fused=AutoBackend(model=deepcopy(model),device=device,fp16=False,fuse=True,verbose=False)
    fused.eval()
    with torch.no_grad():
        fused_y,fused_raw=fused(x)
    # This untrained fixture has zero encoder/decoder box outputs. Unique selected
    # anchors expose candidate identity without hooks or any topk/matcher patch.
    # Reordering near-tied encoder scores is permitted only for identical sets.
    aligned=[];permuted=0
    for i in range(len(y)):
        ids=[tuple(row) for row in raw[2][i].cpu().tolist()]
        other=[tuple(row) for row in fused_raw[2][i].cpu().tolist()]
        require(len(set(ids))==300 and set(ids)==set(other),"Fusion changed the selected candidate set")
        lookup={anchor:j for j,anchor in enumerate(other)}
        order=[lookup[anchor] for anchor in ids]
        permuted+=sum(j!=k for j,k in enumerate(order))
        aligned.append(fused_y[i,order])
    fused_y=torch.stack(aligned)
    close(fused_y,y,atol=3e-5,rtol=2e-4)
    require(hasattr(fused.model.model[20],"bn"),"Original LIF fusion guard lost")
    ema=ModelEMA(model)
    ema.update(model)
    require(any(".peq." in k for k in ema.ema.state_dict()),"EMA lost PEQ")
    trainer=PEQTrainer.__new__(PEQTrainer)
    trainer.model=model
    trainer.ema=ema
    trainer.args=SimpleNamespace(**{**DEFAULT_CFG_DICT,"model":str(MODEL_YAML),"data":str(folder/"data.yaml"),
                                  "save_dir":str(folder),"save_period":-1,"close_mosaic":10})
    trainer.optimizer=trainer.build_optimizer(model,name="AdamW",lr=.0005,momentum=.937,decay=.0001,iterations=10)
    for p in model.parameters():
        p.grad=torch.full_like(p,.001)
    trainer.optimizer.step();trainer.optimizer.zero_grad()
    trainer.scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda",init_scale=128)
    trainer.epoch=1;trainer.best_fitness=trainer.fitness=.1;trainer.metrics={"metrics/mAP50-95(B)":.1}
    trainer.csv=folder/"results.csv"
    trainer.csv.write_text("epoch,metrics/mAP50-95(B)\n1,0.1\n",encoding="utf-8")
    trainer.wdir=folder/"weights";trainer.last=trainer.wdir/"last.pt";trainer.best=trainer.wdir/"best.pt"
    trainer.save_period=-1
    trainer.save_model()
    checkpoint=torch_load(trainer.last,map_location="cpu")
    restored=RTDETR(str(trainer.last))
    require(restored.task_map["detect"]["trainer"] is PEQTrainer and
            restored.task_map["detect"]["validator"] is PEQValidator,"Checkpoint facade lost PEQ routing")
    require(all(torch.equal(v.float().cpu(),restored.model.state_dict()[k].cpu()) for k,v in checkpoint["ema"].state_dict().items()),
            "Native save/load lost weights")
    rebuild=PEQTrainer.__new__(PEQTrainer)
    rebuild.model=str(trainer.last);rebuild.data=dict(nc=1,channels=3);rebuild.args=trainer.args
    ckpt=rebuild.setup_model()
    rebuild.model.to(device);rebuild.ema=ModelEMA(rebuild.model)
    rebuild.optimizer=rebuild.build_optimizer(rebuild.model,name="AdamW",lr=.0005,momentum=.937,decay=.0001,iterations=10)
    rebuild.scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
    rebuild.resume=True;rebuild.epochs=200
    rebuild.resume_training(ckpt)
    require(rebuild.start_epoch==2 and rebuild.ema.updates==checkpoint["updates"] and
            rebuild.scaler.state_dict()==checkpoint["scaler"],"Resume did not restore epoch/scaler/EMA")
    require(len(rebuild.optimizer.state)==len(trainer.optimizer.state),"Resume optimizer state missing")
    # Real facade.predict invokes AutoBackend and the real RTDETR predictor.
    images=[np.zeros((119,231,3),dtype=np.uint8),np.full((187,127,3),100,dtype=np.uint8)]
    predictions=restored.predict(source=images,imgsz=160,device=str(device),conf=.001,max_det=300,save=False,verbose=False)
    require(len(predictions)==2 and all(bool((r.boxes.conf[:-1]>=r.boxes.conf[1:]).all()) for r in predictions),"Predict score sorting")
    # An actual validator lifecycle on a synthetic split, explicitly an interface fixture.
    import cv2
    dataset=folder/"data"
    for split in ("train","val"):
        (dataset/"images"/split).mkdir(parents=True)
        (dataset/"labels"/split).mkdir(parents=True)
        for i,img in enumerate(images):
            cv2.imwrite(str(dataset/"images"/split/f"{i}.png"),img)
            (dataset/"labels"/split/f"{i}.txt").write_text("0 0.5 0.5 0.4 0.3\n",encoding="utf-8")
    YAML.save(folder/"data.yaml",dict(path=str(dataset),train="images/train",val="images/val",names={0:"crack"}))
    val=RTDETR(str(trainer.last))
    metrics=val.val(data=str(folder/"data.yaml"),imgsz=160,batch=2,device=str(device),
                    workers=0,conf=.001,plots=False,save_json=False,project=str(folder),name="validator",
                    exist_ok=False,verbose=False)
    require(np.isfinite(np.asarray(metrics.box.all_ap)).all(),"Real validator failed")
    # The independent read-only loader must agree with native RTDETRDataset, including nonsquare images.
    from peq_v1_eval import batches
    inventory=dataset_identity(folder/"data.yaml",splits=("val",),strict_counts=False)
    ours=next(batches(inventory,"val",2,imgsz=160))
    validator=PEQValidator(args=dict(data=str(folder/"data.yaml"),imgsz=160,batch=2,workers=0,plots=False),save_dir=folder/"loader_check")
    validator.data=dict(path=str(dataset),nc=1,names={0:"crack"},channels=3)
    native_dataset=validator.build_dataset(str(dataset/"images/val"),batch=2)
    native=native_dataset.collate_fn([native_dataset[0],native_dataset[1]])
    close(ours["img"],native["img"],atol=0,rtol=0)
    close(ours["bboxes"],native["bboxes"],atol=1e-7,rtol=1e-7)
    # Full native Trainer constructor/setup on this tiny fixture; no train() call.
    setup_args={**DEFAULT_CFG_DICT,"model":str(trainer.last),"data":str(folder/"data.yaml"),
                "project":str(folder),"name":"trainer_setup","save_dir":str(folder/"trainer_setup"),
                "batch":2,"imgsz":160,"workers":0,"device":str(device),"amp":device.type=="cuda","epochs":2,
                "plots":False,"optimizer":"AdamW","lr0":.0005,"weight_decay":.0001,"freeze":None}
    actual=PEQTrainer(overrides=setup_args)
    actual._setup_train()
    require(actual.peq_loading_audit["common_equal"],"Actual setup lost common initialization")
    old=actual.model.model[20].O_proj.weight
    new=actual.model.model[-1].peq.output.weight
    before_old=old.detach().clone();before_new=new.detach().clone()
    actual.scaler.scale(old.sum()+new.sum()).backward()
    actual.optimizer_step()
    require(not torch.equal(old,before_old) and not torch.equal(new,before_new),"Actual optimizer_step did not update both groups")
    require(actual.ema.updates==1,"Actual optimizer_step did not update EMA")
    # Compare corrected implementation to the mother's independent evaluator using an unsorted mask fixture.
    from c19_lif_v1_results import postprocess as mother_postprocess
    fixture=torch.tensor([[[.1,.1,.1,.1,.2],[.5,.5,.1,.1,.9],[.8,.8,.1,.1,.01]]])
    selected=corrected_postprocess(fixture,160,.1)[0]
    parent,_=mother_postprocess(fixture,160,.1)
    for k in ("bboxes","conf","cls"):
        close(selected[k],parent[0][k],atol=0,rtol=0)
    require(selected["query_ids"].tolist()==[1,0],"Sorted confidence mask/query IDs mismatch")
    try:
        model.model[-1].export=True
        model(x)
        raise AssertionError("Unvalidated export was accepted")
    except NotImplementedError:
        pass
    finally:
        model.model[-1].export=False
    return dict(fixture_source=fixture_source,nonzero_peq=True,nonzero_cbr=True,disabled_exact=True,raw_logits_and_boxes_exact=True,
                autobackend_fuse_max_difference=float((fused_y-y).abs().max()),fusion_reordered_queries=permuted,
                fusion_identical_candidate_sets=True,ema=True,native_save_load=True,
                actual_setup_model=True,full_trainer_constructor_setup=True,actual_optimizer_step=True,
                native_amp_check_messages=actual.peq_amp_check_messages,native_resume_epoch=rebuild.start_epoch,scaler_restored=True,
                optimizer_state_restored=True,actual_predict=True,actual_validator=True,
                native_eval_loader_exact=True,corrected_mask_exact=True,export_explicitly_rejected=True)


def status_checks():
    from peq_v1_runtime import resolve_status
    from peq_v1_common import clean_json
    token="fixture_new"
    process=dict(pid=1234,command=["python",str(ROOT/"tools/peq_v1.py"),"worker","--dispatch",token],
                 state="R",cwd=str(ROOT),start_token="10")
    require(resolve_status({},None,None)["phase"]=="NOT_STARTED","Never-started status")
    state=dict(dispatch=token,phase="RUNNING",pid=1234,start_token="10")
    require(resolve_status(state,dict(dispatch="old",python_exit=0,tee_exit=0),process)["phase"]=="RUNNING","Old exit contaminated current state")
    require(resolve_status(state,None,None)["phase"]=="FAILED","Dead worker status")
    require(resolve_status(state,dict(dispatch=token,python_exit=0,tee_exit=0),None)["phase"]=="COMPLETED","Completed status")
    require(resolve_status(state,dict(dispatch=token,python_exit=1,tee_exit=0),None)["phase"]=="FAILED","Failed status")
    require(resolve_status(state,dict(dispatch=token,python_exit=0,tee_exit=1),None)["phase"]=="FAILED","Tee failure status")
    require(json.loads(json_bytes(dict(loss=float("nan"))))["nonfinite"],"NaN JSON lost flag")
    return dict(fixtures=["NOT_STARTED","RUNNING","COMPLETED","FAILED","STALE_EXIT","TEE_FAILURE"],
                duplicate_dispatch_safe=True,nonfinite_json_null_and_flag=True)


def run_checks(device="cpu", lifecycle_check=True, output=None):
    device=torch.device(device)
    report=dict(status="FAILED",scope="LOCAL_INTERFACE_CHECK",device=str(device),environment=environment(),
                source=source_identity()["sha256"],module_hashes=module_contract(),checks={},started=utc())
    output=Path(output) if output else OUT/f"checks_{device.type}.json"
    if output.exists():
        history=OUT/"check_history"/(output.stem+"_"+sha256(output)[:16]+".json")
        history.parent.mkdir(parents=True,exist_ok=True)
        if not history.exists():
            history.write_bytes(output.read_bytes())
    try:
        with isolated_rng():
            torch.manual_seed(42);torch.set_num_threads(4)
            with tempfile.TemporaryDirectory(prefix="peq_checks_",dir=str(OUT)) as temporary:
                from check_peq_v1_workflows import workflow_checks
                groups=[("workflows",workflow_checks),("sampling_targets",lambda:sampling_and_targets(device)),
                        ("branch_learning",lambda:branch_learning(device)),
                        ("native_and_gradients",lambda:native_and_gradients(device)),
                        ("status",status_checks),
                        ("dynamic_dn_amp",lambda:dynamic_dn_and_amp(device))]
                if lifecycle_check:
                    groups.append(("lifecycle",lambda:lifecycle(device,Path(temporary))))
                for name,check in groups:
                    start=time.monotonic()
                    print("CHECK",name,str(device),flush=True)
                    evidence=check()
                    report["checks"][name]=dict(status="PASS",seconds=time.monotonic()-start,evidence=evidence)
        report["status"]="PASS"
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report["finished"]=utc()
        write_json(output,report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--no-lifecycle",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    result=run_checks(args.device,not args.no_lifecycle,args.output)
    print(result["status"],list(result["checks"]))
