"""Bounded mother-best phenomenon diagnosis; never scans test or updates weights."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import time

import torch

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.gpc import auxiliary_loss,select_views,paired_targets,matched_consistency,translate
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import init_seeds


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        for pair in row.get("per_gt",[]):
            short=pair["short_side_pixels"]
            group="lt8px" if short<8 else "8to32px" if short<32 else "ge32px"
            groups[group].append(pair)
    result={}
    for name,items in groups.items():
        result[name]=dict(pairs=len(items),mean_abs_pixel_error=sum(abs(v) for p in items for v in p["pixel_error"])/(4*len(items)),
                         mean_abs_normalized_error=sum(abs(v) for p in items for v in p["normalized_error"])/(4*len(items)),
                         mean_iou_change=sum(p["iou_shifted"]-p["iou_original"] for p in items)/len(items))
    return result or {"coverage":"undefined: no paired eligible GT"}


def gradient_groups(loss,model,retain_graph):
    params=[(name,p) for name,p in model.named_parameters() if any(key in name for key in
            ("B_proj","U_mix","U_dw","O_proj",".P.",".cbr.",".dec_bbox_head.2."))]
    gradients=torch.autograd.grad(loss,[p for _,p in params],retain_graph=retain_graph,allow_unused=True) if loss.requires_grad else [None]*len(params)
    groups=defaultdict(float)
    for (name,_),grad in zip(params,gradients):
        if grad is not None:
            group="CBR" if ".cbr." in name else "last_regression_head" if "dec_bbox_head" in name else "LIF"
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("Nonfinite diagnostic gradient "+name)
            groups[group]+=float(grad.detach().float().square().sum())
    return {name:value**.5 for name,value in groups.items()}


def diagnose(prepared):
    from gpc_v1 import PARENT_RUN,OUT,sha,runtime,write
    best=PARENT_RUN/"weights/best.pt"
    report=dict(status="PENDING",identity=prepared["identity"],runtime=runtime(),
                budget=dict(val_images=32,train_batches=4,seconds=900),mother_best=str(best),
                limitation="Finite cross-view regularization; consistent wrong boxes can have zero loss; no AP evidence")
    if not best.is_file():
        return dict(report,reason="Mother best unavailable; diagnostic only, never an initialization replacement")
    if not torch.cuda.is_available():
        return dict(report,reason="CUDA unavailable for fixed B16/640 diagnostic")
    start=time.monotonic();deadline=start+900
    report["mother_best_sha256"]=sha(best)
    args=deepcopy(prepared["identity"]["recipe"])
    stamp=str(time.time_ns());folder=OUT/"diagnostic_runs"/stamp
    args.update(project=str(folder),name="loader",save_dir=str(folder/"loader"))
    init_seeds(42,deterministic=True)
    # Constructor builds datasets/settings only; no training loop or optimizer step.
    trainer=RTDETRTrainer(overrides=args)
    trainer.stride=32
    saved=torch_load(best,map_location="cpu")
    source=saved.get("ema") or saved.get("model")
    from init_c19_lif_v1 import verify_model
    verify_model(source)
    report["mother_checkpoint_git"]=saved.get("git")
    report["mother_checkpoint_epoch"]=saved.get("epoch")
    model=deepcopy(source).float().to(trainer.device)
    require_grad=any(p.requires_grad for p in model.parameters())
    for p in model.parameters():p.requires_grad_(True)
    report["checkpoint_parameters_previously_trainable"]=require_grad
    report["mode"]=dict(val="native inference FP32",train="B16 original training main + B8 BN-eval paired AMP; no optimizer/EMA update")
    model.criterion=RTDETRDetectionLoss(nc=1,use_vfl=True)
    matcher=model.criterion.matcher
    val_loader=trainer.get_dataloader(trainer.data["val"],batch_size=16,rank=-1,mode="val")
    val_rows=[];images_seen=0
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    try:
        for raw in val_loader:
            if images_seen>=32 or time.monotonic()>=deadline:break
            batch=trainer.preprocess_batch(raw)
            count=min(len(batch["img"]),32-images_seen)
            selected=list(range(count))
            # Fixed, non-shuffled first 32 validation images; optional 16px references are explicit.
            for shift in ((8,0),(0,8),(8,8),(16,0),(0,16)):
                if time.monotonic()>=deadline:break
                shifts=[shift]*count
                targets=paired_targets(batch,selected,shifts)
                if not targets["eligible"]:
                    val_rows.append(dict(targets["counts"],pairs=0,shift=shift,loss_gpc=0.0,per_gt=[]))
                    continue
                images=batch["img"][selected]
                moved=torch.stack([translate(x,*shift) for x in images])
                with torch.no_grad():
                    raw_preds=model.predict(torch.cat((images,moved)),batch=None)[1]
                    loss,stats=matched_consistency(raw_preds[0][-1],raw_preds[1][-1],targets,matcher,details=True)
                val_rows.append(dict(stats,shift=shift,images=batch["im_file"][:count]))
            images_seen+=count
        # The train diagnostic starts from a separate pristine mother-best instance.
        del model,val_loader
        model=deepcopy(source).float().to(trainer.device).train()
        for p in model.parameters():p.requires_grad_(True)
        model.criterion=RTDETRDetectionLoss(nc=1,use_vfl=True)
        train_loader=trainer.get_dataloader(trainer.data["train"],batch_size=16,rank=-1,mode="train")
        train_rows=[]
        for index,raw in enumerate(train_loader):
            if index>=4 or time.monotonic()>=deadline:break
            batch=trainer.preprocess_batch(raw)
            if len(batch["img"])!=16:raise RuntimeError("Diagnostic requires original B16")
            with torch.autocast("cuda",enabled=True):
                indices=batch["batch_idx"].long().flatten()
                main_targets=dict(cls=batch["cls"].long().flatten(),bboxes=batch["bboxes"],batch_idx=indices,
                                  gt_groups=[int((indices==i).sum()) for i in range(len(batch["img"]))])
                main_raw=model.predict(batch["img"],batch=main_targets)
                main,_=model.loss(batch,preds=main_raw)
                selected,shifts,derived=select_views(16,batch["im_file"],20,index)
                extra,stats,aux_raw=auxiliary_loss(model,batch,selected,shifts,model.criterion.matcher,details=True,return_predictions=True)
            row=dict(stats,main_loss=float(main.detach()),weighted_gpc=float(extra.detach())*.1,derived_seed=derived,
                     selected=selected,shifts=shifts)
            if aux_raw is not None:
                # A separate GT-linked zero-coordinate-offset diagnostic, not part of GPC or L0.
                # Main B16 BN/DN context differs from the BN-eval auxiliary replay.
                offset=main_raw[-1]["dn_num_split"][0] if main_raw[-1] is not None else 0
                k=len(selected)
                replay_targets=paired_targets(batch,selected,[(0,0)]*k)
                with torch.no_grad():
                    _,mode_stats=matched_consistency(
                        torch.cat((main_raw[0][-1,selected,offset:],aux_raw[0][-1,:k])),
                        torch.cat((main_raw[1][-1,selected,offset:],aux_raw[1][-1,:k])),
                        replay_targets,model.criterion.matcher,details=True)
                row["main_B16_vs_frozen_BN_original_replay"]=mode_stats
            # Two bounded diagnostics only; production still does a single ordinary backward.
            if index==0:
                main_norms=gradient_groups(main,model,retain_graph=True)
                extra_norms=gradient_groups(.1*extra,model,retain_graph=False)
                row["gradient_norms"]=dict(main=main_norms,weighted_gpc=extra_norms,
                    ratio={key:extra_norms.get(key,0)/value if value else None for key,value in main_norms.items()})
            train_rows.append(row)
            del main,extra,batch,main_raw,aux_raw
        complete=images_seen==32 and len(val_rows)==10 and len(train_rows)==4
        nonzero=any(row.get("loss_gpc",0)>1e-12 for row in val_rows+train_rows)
        report.update(status="COMPLETE" if complete else "PENDING",val_images=images_seen,train_batches=len(train_rows),
                      val_rows=val_rows,train_rows=train_rows,val_scale_groups=summarize(val_rows),train_scale_groups=summarize(train_rows),
                      conclusion=("Nonzero response measured; inspect raw scale/coverage/gradient ratios; no AP claim" if nonzero else
                                  "No measurable effect in bounded diagnosis; long training is not recommended") if complete else "Incomplete bounded diagnosis; mechanism remains PENDING",
                      all_required_batches=complete,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-start)
        return report
    except BaseException as error:
        report.update(status="RESOURCE_ERROR" if isinstance(error,torch.cuda.OutOfMemoryError) else "FAIL",error=repr(error),seconds=time.monotonic()-start)
        raise
    finally:
        write(OUT/"diagnose.json",report)
