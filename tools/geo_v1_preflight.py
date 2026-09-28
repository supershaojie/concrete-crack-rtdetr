"""Bounded native lifecycle check. Never starts the formal run or full val/test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from pathlib import Path
import sys
import time
import traceback

from geo_v1_common import (ROOT,OUT,INIT,read_json,write_json,require,now,runtime,sha256,
    recipe,data_config,amp_resources,run_bounded)
import torch
from ultralytics.models.rtdetr.geo_loss import ramp
from ultralytics.models.rtdetr.geo_model import GEOTrainer,GEODetectionModel
from ultralytics.models.rtdetr.geo_val import GEOValidator,EVAL
from ultralytics.utils.torch_utils import autocast,init_seeds
from ultralytics.utils.patches import torch_load


def close_workers(trainer):
    for name in ("train_loader","test_loader"):
        loader=getattr(trainer,name,None)
        iterator=getattr(loader,"iterator",None)
        if iterator is not None and hasattr(iterator,"_shutdown_workers"): iterator._shutdown_workers()


def reload_and_val(folder,checkpoint,local=False):
    args=read_json(folder/"diagnostic_args.json")
    args.update(model=str(checkpoint),resume=str(checkpoint))
    t=GEOTrainer(overrides=args)
    try:
        t._setup_train()
        ckpt=torch_load(checkpoint,map_location="cpu")
        require(type(t.model) is GEODetectionModel,"GEO model specialization lost on native reconstruction")
        require(t.start_epoch==20 and t.model.geo_epoch==20 and ramp(t.start_epoch)==1,"Resume reset GEO epoch")
        require(t.scaler.state_dict()==ckpt["scaler"],"Scaler restore mismatch")
        require(t.ema.updates==ckpt["updates"],"EMA updates mismatch")
        actual=t.optimizer.state_dict()
        require(actual["param_groups"]==ckpt["optimizer"]["param_groups"],"Optimizer groups mismatch")
        for key,state in ckpt["optimizer"]["state"].items():
            for name,value in state.items():
                restored=actual["state"][key][name]
                require(torch.equal(restored.cpu(),value.to(restored.dtype)) if torch.is_tensor(value) else restored==value,
                    f"Optimizer restore mismatch: {key}/{name}")
        require(all(torch.equal(v.cpu(),ckpt["ema"].state_dict()[k].float()) for k,v in t.ema.ema.state_dict().items()),"EMA tensor restore mismatch")
        write_json(folder/"resume.json",dict(status="PASS",epoch=20,ramp=1,optimizer=True,ema=True,
            scaler=t.scaler.state_dict(),scheduler_last_epoch=t.scheduler.last_epoch,
            checkpoint_sha256=sha256(checkpoint),scope="local CPU diagnostic" if local else "server native AMP"))
    finally: close_workers(t)
    del t,ckpt,actual
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    device="0" if torch.cuda.is_available() else "cpu"
    size=1 if local else 16
    validator=GEOValidator(args=dict(EVAL,batch=size,model=str(checkpoint),data=args["data"],split="val",device=device,plots=False),
        save_dir=folder/"one_real_val_batch")
    validator.one_batch=True
    validator.export_identity=dict(diagnostic=True,split="val",checkpoint_sha256=sha256(checkpoint))
    from ultralytics.nn.autobackend import AutoBackend
    original=AutoBackend.warmup; observed=[]
    def warmup(backend,imgsz=(1,3,640,640)):
        handle=backend.model.register_forward_pre_hook(lambda m,a: observed.append(dict(shape=list(a[0].shape),
            finite=bool(torch.isfinite(a[0]).all()),zero=bool(torch.count_nonzero(a[0])==0),dtype=str(a[0].dtype))))
        try: return original(backend,imgsz)
        finally: handle.remove()
    AutoBackend.warmup=warmup
    try: metrics=validator(model=str(checkpoint))
    finally: AutoBackend.warmup=original
    if device!="cpu": require(observed and all(r["finite"] and r["zero"] for r in observed),"Warmup did not use finite zeros")
    require(len(validator.geo_seen)==size and not validator.actual_settings["half"],"Real FP32 validation scope mismatch")
    write_json(folder/"new_process_val.json",dict(status="PASS",warmup=observed,images=len(validator.geo_seen),
        actual_settings=validator.actual_settings,metrics=metrics,export=str(validator.export_path),runtime=runtime(),
        scope=f"one real B{size}/640 FP32 val batch, not full val/test"))


def run(folder,local=False):
    request=read_json(folder/"request.json")
    started=time.monotonic(); deadline=started+request["seconds"]
    report=dict(status="PENDING",started=now(),runtime=runtime(),micro_batches=0,steps=[],updates=[],
        scope="LOCAL B2/160 CPU, no formal capacity claim" if local else "SERVER B16/640 native AMP, <=16 micro-batches, <=900 seconds",
        checks={k:dict(status="PENDING") for k in ("cuda_b16_amp","native_scale","mechanism","resume","new_process_val")})
    trainer=None
    def stage(label):
        print(f"GEO stage={label} elapsed={time.monotonic()-started:.1f}s micro_batches={report['micro_batches']}/{request['micro_batches']}",flush=True)
        write_json(folder/"gpu.json",report)
        require(time.monotonic()<deadline,"Preflight deadline exhausted")
    try:
        stage("native trainer setup")
        init_seeds(42,deterministic=True)
        args=deepcopy(read_json(OUT/"prepare.json")["args"]) if not local else recipe(data_config())[0]
        args.update(project=str(folder),name="diagnostic_train",save_dir=str(folder/"diagnostic_train"),plots=False)
        if local: args.update(batch=2,imgsz=160,workers=0,device="cpu",amp=False)
        write_json(folder/"diagnostic_args.json",args)
        trainer=GEOTrainer(overrides=args); trainer._setup_train()
        if not local:
            require(trainer.batch_size==16 and trainer.args.imgsz==640 and trainer.accumulate==4 and trainer.amp,"Formal recipe drift")
        report["setup"]=dict(args=vars(trainer.args),amp=trainer.geo_amp_evidence,scaler=trainer.scaler.state_dict(),accumulate=trainer.accumulate)
        trainer.epoch=20
        GEOTrainer._geo_epoch_start(trainer)
        trainer.model.train(); trainer.model.geo_sample=True
        trainer.optimizer.zero_grad()
        # e=20 lies beyond original warmup. Keep native loss scaling/accumulation.
        for group in trainer.optimizer.param_groups: group["lr"]=group["initial_lr"]*trainer.lf(20)
        accumulation=trainer.accumulate if not local else 1
        report["diagnostic_accumulation"]=accumulation
        batches=iter(trainer.train_loader); successful=False
        for step in range(request["micro_batches"]):
            stage("augmented train batch")
            batch=trainer.preprocess_batch(next(batches))
            require(tuple(batch["img"].shape)==(args["batch"],3,args["imgsz"],args["imgsz"]),"Actual augmented shape drift")
            report["micro_batches"]+=1
            trainer.model.geo_sample=step<4
            with autocast(trainer.amp): loss,items=trainer.model(batch)
            require(bool(torch.isfinite(loss)),"Nonfinite total loss")
            trainer.scaler.scale(loss).backward()
            stats=trainer.model.criterion.last_diagnostics
            report["steps"].append(dict(step=step,loss=float(loss.detach()),scale=trainer.scaler.get_scale(),diagnostics=stats))
            require(trainer.model.criterion._geo_context is None,"GEO retained graph context")
            report["checks"]["mechanism"]=dict(status="PASS",ramp=ramp(trainer.epoch),
                note="Real augmented batch, finite GEO; zero activation is valid and not a training gate")
            if (step+1)%accumulation==0:
                # Retain just one update sample here even after >2 native-scale attempts.
                trainer.geo_updates=[]
                trainer.optimizer_step()
                update=trainer.geo_updates[-1]; report["updates"].append(update)
                if not update["skipped"] and update["gradients"]["finite"] and update["cbr_parameter_changed"]:
                    successful=True; break
        report["checks"]["native_scale"]=dict(status="PASS" if successful else "PENDING",updates=trainer.geo_update_count,
            note="Original GradScaler only; no lower-scale fallback or retry extension")
        if not local and successful:
            report["checks"]["cuda_b16_amp"]=dict(status="PASS",effective_updates=trainer.geo_update_count,batch=16,imgsz=640,amp=True)
        if not successful:
            report["reason"]="No finite parameter-changing native optimizer update inside boundary. Preserve scale history; no automatic budget extension."
            return report
        stage("native checkpoint save")
        trainer.epoch,trainer.fitness,trainer.best_fitness=19,0.0,0.0
        trainer.save_model(); checkpoint=trainer.last
        report["checkpoint"]=dict(path=str(checkpoint),sha256=sha256(checkpoint))
        close_workers(trainer)
        del trainer,batch,loss,items
        trainer=None; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        stage("fresh process native resume, FP32 warmup and real val")
        command=[sys.executable,str(Path(__file__).resolve()),"--folder",str(folder),"--reload",str(checkpoint)]
        if local: command.append("--local")
        code=run_bounded(command,deadline-time.monotonic(),"checkpoint reload and val")
        require(code==0,"Fresh-process resume/val failed")
        for key in ("resume","new_process_val"): report["checks"][key]=read_json(folder/f"{key}.json",dict(status="PENDING"))
        report["status"]="PASS" if all(v["status"]=="PASS" for k,v in report["checks"].items() if not(local and k=="cuda_b16_amp")) else "PENDING"
        return report
    except BaseException as error:
        report.update(status="FAIL",error=repr(error),traceback=traceback.format_exc(),geometry=getattr(error,"details",None))
        raise
    finally:
        if trainer is not None: close_workers(trainer)
        report["elapsed_seconds"]=time.monotonic()-started
        write_json(folder/"gpu.json",report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,required=True)
    parser.add_argument("--reload",type=Path)
    parser.add_argument("--local",action="store_true",help="Explicit local B2/160 CPU lifecycle only; cannot qualify formal start")
    args=parser.parse_args(); torch.set_num_threads(4)
    if args.reload: reload_and_val(args.folder,args.reload,args.local)
    else:
        report=run(args.folder,args.local)
        if report["status"]!="PASS": raise SystemExit(2)


if __name__=="__main__": main()
