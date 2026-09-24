"""Bounded augmented-train coverage and isolated B16/640 capacity probe."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import traceback
import torch
from pds_v1_common import *
from pds_v1_trainer import PDSTrainer
from ultralytics.nn.modules.pds import labels, decode, dense_loss
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.data.build import build_dataloader
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import autocast


def coverage(model, batch, directory, offset=0, maximum=4):
    """At most four images from each of at most two already-augmented train batches."""
    import cv2
    directory.mkdir(parents=True,exist_ok=True)
    subset={}
    count=min(maximum,len(batch["img"]))
    mask=batch["batch_idx"].flatten()<count
    for k in ("img","bboxes","cls","batch_idx"):
        subset[k]=batch[k][:count] if k=="img" else batch[k][mask]
    # Train-only capture with no DN labels. No retained activations/hook/cross-batch graph.
    with torch.no_grad(),autocast(next(model.parameters()).is_cuda):
        preds,features=model.training_forward(subset["img"],None)
        logits,raw=model.pds_head(*features)
        _,stats=dense_loss(logits,raw,subset,details=True)
    targets=labels(subset)
    hq,wq=raw.shape[-2:]
    h,w=subset["img"].shape[-2:]
    rows=[]
    palette=[(255,140,30),(0,210,255),(200,70,220),(50,220,100)]  # BGR phase colors
    for b in range(count):
        im=(subset["img"][b].detach().float().cpu().permute(1,2,0).numpy()[:,:,::-1]*255).clip(0,255).astype("uint8").copy()
        match=stats["assignment"][b]
        mask_grid=match["excluded"].reshape(hq,wq).cpu().numpy().astype("uint8")
        excluded=cv2.resize(mask_grid,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay=im.copy()
        overlay[excluded]=(35,90,170)
        im=cv2.addWeighted(im,.72,overlay,.28,0)
        # Drawing loops are visualization only, never the assignment implementation.
        for v in range(hq):
            for u in range(wq):
                phase=(v%2)+2*(u%2)
                cv2.circle(im,(round((u+.5)*w/wq),round((v+.5)*h/hq)),0,palette[phase],1)
        for box in targets[b]["pixel"].cpu().tolist():
            cv2.rectangle(im,tuple(round(x) for x in box[:2]),tuple(round(x) for x in box[2:]),(245,245,245),1)
        for ids in match["selected"]:
            for index in ids:
                v,u=divmod(index,wq)
                cv2.circle(im,(round((u+.5)*w/wq),round((v+.5)*h/hq)),3,(0,0,255),1)
        cv2.putText(im,"GT:white positives:red exclusion:tint phase:4 colors",(5,16),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
        path=directory/f"train_aug_{offset+b:02d}.jpg"
        require(cv2.imwrite(str(path),im),"Coverage visualization write failed")
        rows.append(dict(image=str(path),**match["stats"]))
    return rows


def local_coverage(source,dataset,output,device="cpu"):
    from check_pds_v1 import rebuild
    torch.set_num_threads(4)
    _,model,_=rebuild(source)
    model=model.to(device).train()
    args=get_cfg(overrides=recipe())
    args.imgsz=320
    args.workers=0
    args.batch=2
    data=dict(nc=1,names={0:"crack"},channels=3)
    ds=RTDETRDataset(img_path=str(dataset/"images/train"),imgsz=320,batch_size=2,augment=True,
                     hyp=args,rect=False,cache=False,data=data)
    loader=build_dataloader(ds,batch=2,workers=0,shuffle=True,rank=-1)
    rows=[]
    # Exactly two mini-batches, no access to val/test images.
    for i,b in enumerate(loader):
        b["img"]=b["img"].to(device).float()/255
        rows.extend(coverage(model,b,output,offset=2*i,maximum=2))
        if i==1:
            break
    total={k:sum(r[k] for r in rows) for k in ("gt","matched_gt","unmatched_gt","fallback","positives","contested_points")}
    total["matched_fraction"]=total["matched_gt"]/max(total["gt"],1)
    result=dict(status="PASS",scope="two real augmented train mini-batches, B2/320 local coverage only",
                images=rows,totals=total,interpretation="box responsibility points, not crack pixel labels")
    write_json(output/"coverage.json",result)
    return result


def capacity(output):
    torch.set_num_threads(4)
    started=time.monotonic()
    report=dict(status="FAILED",diagnostic_only_scale=128,micro_batches=0,effective_updates=0,
                limit_micro_batches=16,limit_seconds=900,diagnostic_epoch=20,formal_weights_untouched=True,
                trace=[])
    trainer=None
    try:
        require(torch.cuda.is_available(),"Server capacity requires CUDA")
        args=YAML.load(OUT/"train_args.yaml")
        require(args["batch"]==16 and args["imgsz"]==640 and args["nbs"]==64 and args["amp"],"Capacity recipe changed")
        args.update(project=str(output),name="isolated",save_dir=str(output/"isolated"),exist_ok=False)
        trainer=PDSTrainer(overrides=args)
        trainer._setup_train()  # native AMP check and exact optimizer grouping
        report["formal_native_initial_scaler"] = trainer.scaler.state_dict().copy()
        require(trainer.amp and trainer.batch_size==16 and trainer.accumulate==4,"Capacity native setup mismatch")
        trainer.scaler=torch.cuda.amp.GradScaler(enabled=True,init_scale=128)
        trainer.epoch=20
        trainer.model.pds_epoch=20
        trainer.model.train()
        trainer.scheduler.last_epoch=20
        for group in trainer.optimizer.param_groups:
            group["lr"]=group["initial_lr"]*trainer.lf(20)
        trainer.optimizer.zero_grad(set_to_none=True)
        named=dict(trainer.model.named_parameters())
        probe_names=["model.4.blocks.0.branch2a.conv.weight","model.20.O_proj.weight",
                     "model.26.enc_score_head.weight","pds_head.p2_proj.weight","pds_head.box.weight"]
        before={n:named[n].detach().clone() for n in probe_names}
        torch.cuda.reset_peak_memory_stats()
        shapes=[]
        visual=[]
        # This isolated epoch-20 window begins at a clean original accumulate=4 boundary.
        for i,batch in enumerate(trainer.train_loader):
            if i>=16 or time.monotonic()-started>=900:
                break
            trainer._oom_retries=3
            with autocast(trainer.amp):
                batch=trainer.preprocess_batch(batch)
                total,items=trainer.model(batch)
            require(torch.isfinite(total),"Nonfinite total capacity loss")
            trainer.scaler.scale(total).backward()
            shapes.append(list(batch["img"].shape))
            report["micro_batches"]=i+1
            row=dict(micro_batch=i+1,loss=float(total.detach()),main_items=items.tolist(),
                     pds=deepcopy(trainer.model.pds_stats),scale=trainer.scaler.get_scale())
            # Do not run an extra coverage forward in the capacity window (BN/RNG identity).
            report["trace"].append(row)
            if (i+1)%trainer.accumulate==0:
                trainer.optimizer_step()
                report["effective_updates"]=trainer.pds_effective_steps
                report["overflow_skips"]=trainer.pds_overflow_skips
                report["optimizer_diagnostics"]=trainer.pds_step_diagnostics
                if trainer.pds_effective_steps>=2:
                    break
        require(trainer.pds_effective_steps>=1,"No effective optimizer updates within 16 micro-batches/900 s")
        diffs={n:float((named[n].detach()-v).abs().max()) for n,v in before.items()}
        require(all(v>0 and math.isfinite(v) for v in diffs.values()),"Missing public/head real parameter update")
        effective=[r for r in trainer.pds_step_diagnostics if r["effective"]]
        require(all(r["gradients"][k]["finite"] and r["gradients"][k]["norm"]>0 for r in effective for k in ("public","head")),
                "Nonfinite or zero effective public/head gradient")
        report.update(status="PASS",actual_shapes=shapes,parameters=sum(p.numel() for p in trainer.model.parameters()),
                      amp=trainer.amp,optimizer="AdamW",accumulate=trainer.accumulate,nbs=trainer.args.nbs,
                      parameter_changes=diffs,optimizer_groups=trainer.pds_optimizer_groups)
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"]=time.monotonic()-started
        if torch.cuda.is_available():
            report.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        write_json(output/"capacity.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["capacity","coverage"])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--dataset",type=Path,default=MAIN/"datasets/crack_det")
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    if a.mode=="capacity":
        capacity(a.output)
    else:
        local_coverage(a.source,a.dataset,a.output,a.device)
