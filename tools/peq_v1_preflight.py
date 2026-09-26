"""One isolated B16/640 AMP preflight, bounded by 16 micro-batches and 900 seconds."""
from __future__ import annotations
import signal
import subprocess
import time
import traceback
import numpy as np
import torch
from peq_v1_common import *
from peq_v1_runtime import require_prepared
from ultralytics.models.rtdetr.peq_train import PEQTrainer, is_peq
from ultralytics.utils import YAML


def preflight(retry=False):
    prepared,args=require_prepared()
    if not torch.cuda.is_available() or not sys.platform.startswith("linux"):
        result=dict(status="PENDING",reason="Formal preflight requires CUDA Linux server, B16/640; local checks are not capacity evidence",
                    identity_key=prepared["identity_key"])
        write_json(OUT/"preflight.json",result)
        return result
    previous=read_json(OUT/"preflight.json")
    if not retry and previous and previous["status"]=="PASS" and previous["identity_key"]==prepared["identity_key"]:
        return previous
    attempt=OUT/"preflight"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    attempt.mkdir(parents=True)
    report_path=attempt/"report.json"
    command=[sys.executable,str(ROOT/"tools/peq_v1.py"),"preflight-worker","--report",str(report_path)]
    started=time.monotonic()
    process=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
    timed_out=False
    try:
        code=process.wait(timeout=900)
    except subprocess.TimeoutExpired:
        # This is the isolated preflight process group just created above, not any training experiment.
        timed_out=True
        os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
        code=124
    result=read_json(report_path,{})
    result.update(identity_key=prepared["identity_key"],subprocess_exit=code,timeout=timed_out,
                  wall_seconds=time.monotonic()-started,attempt=str(attempt))
    if code!=0 or timed_out or result.get("status")!="PASS":
        result["status"]="FAILED"
    write_json(report_path,result)
    write_json(OUT/"preflight.json",result)
    return result


def preflight_worker(report_path):
    started=time.monotonic()
    prepared,args=require_prepared()
    report=dict(status="FAILED",identity_key=prepared["identity_key"],environment=environment(),
                bounds=dict(micro_batches=16,seconds=900),diagnostic_only_scale=128,rows=[],effective_updates=0)
    report_path=Path(report_path)
    try:
        with isolated_rng():
            from check_peq_v1 import run_checks
            report["interface_checks"]=run_checks("cuda:0",True,report_path.parent/"interface_checks.json")
            require(time.monotonic()-started<900,"Preflight time bound reached during algorithm checks")
            diagnostic=dict(args,project=str(report_path.parent),name="capacity",save_dir=str(report_path.parent/"capacity"),
                            workers=args["workers"],plots=False,save=False)
            # The actual Trainer constructor/get_model/setup_model/AMP check and dataloaders.
            trainer=PEQTrainer(overrides=diagnostic)
            trainer._setup_train()
            require(trainer.amp and trainer.batch_size==16 and trainer.args.imgsz==640,"B16/640/native AMP did not survive setup")
            require(type(trainer.optimizer) is torch.optim.AdamW,"Original AdamW is required")
            trainer.scaler=torch.cuda.amp.GradScaler(enabled=True,init_scale=128)
            trainer.model.train()
            trainer.optimizer.zero_grad()
            torch.cuda.reset_peak_memory_stats()
            effective=[0]
            hook=trainer.optimizer.register_step_post_hook(lambda *unused: effective.__setitem__(0,effective[0]+1))
            watch={n:p for n,p in trainer.model.named_parameters() if is_peq(n) or
                   n in ("model.20.O_proj.weight","model.26.cbr.offset_out.weight","model.26.dec_score_head.2.weight")}
            initial={n:p.detach().clone() for n,p in watch.items()}
            iterator=iter(trainer.train_loader)
            nw=max(round(args["warmup_epochs"]*len(trainer.train_loader)),100)
            last_step=-1
            upstream_nonzero=False
            for ni in range(16):
                if time.monotonic()-started>=900:
                    break
                batch=next(iterator)
                trainer.accumulate=max(1,int(np.interp(ni,[0,nw],[1,args["nbs"]/16]).round()))
                for group in trainer.optimizer.param_groups:
                    group["lr"]=float(np.interp(ni,[0,nw],[args["warmup_bias_lr"] if group.get("param_group")=="bias" else 0.,
                                                        group["initial_lr"]*trainer.lf(0)]))
                    if "momentum" in group:
                        group["momentum"]=float(np.interp(ni,[0,nw],[args["warmup_momentum"],args["momentum"]]))
                batch=trainer.preprocess_batch(batch)
                before_scale=float(trainer.scaler.get_scale())
                with torch.autocast(device_type="cuda",enabled=True):
                    losses=trainer.model.loss_components(batch)
                    loss=sum(losses.values()).sum()  # actual pinned RT-DETR reduction, no added batch multiplier
                require(bool(torch.isfinite(loss)),"Nonfinite preflight loss")
                trainer.scaler.scale(loss).backward()
                grads={n:None if p.grad is None else float((p.grad.detach().float()/before_scale).norm()) for n,p in watch.items()}
                if effective[0]>=1 and all(grads.get(n) is not None and math.isfinite(grads[n]) and grads[n]>0 for n in
                                          ("model.26.peq.p3_proj.weight","model.26.peq.query_proj.weight")):
                    upstream_nonzero=True
                previous=effective[0]
                if ni-last_step>=trainer.accumulate:
                    trainer.optimizer_step()
                    last_step=ni
                row=dict(micro_batch=ni+1,shape=list(batch["img"].shape),losses={k:float(v.detach()) for k,v in losses.items()},
                         loss=float(loss.detach()),gradients=grads,accumulate=trainer.accumulate,
                         parameter_group_lr=[g["lr"] for g in trainer.optimizer.param_groups],
                         scale_before=before_scale,scale_after=float(trainer.scaler.get_scale()),
                         overflow_skip=float(trainer.scaler.get_scale())<before_scale,
                         effective_updates=effective[0],updated_this_batch=effective[0]>previous,
                         parameters_changed={n:bool(not torch.equal(p.detach(),initial[n])) for n,p in watch.items()},
                         peak_allocated=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started)
                report["rows"].append(row)
                report.update(effective_updates=effective[0],upstream_nonzero_after_output_update=upstream_nonzero)
                write_json(report_path,report)
                print("preflight",ni+1,"updates",effective[0],"scale",row["scale_after"],"upstream",upstream_nonzero,flush=True)
                if effective[0]>=2 and upstream_nonzero:
                    break
            hook.remove()
            require(effective[0]>=2 and upstream_nonzero,"Bound reached without two effective updates and subsequent upstream PEQ gradients")
            require(any(not torch.equal(watch[n].detach(),initial[n]) for n in watch if not is_peq(n)),"No original parameter update")
            require(any(not torch.equal(watch[n].detach(),initial[n]) for n in watch if is_peq(n)),"No PEQ parameter update")
            report.update(status="PASS",parameters=sum(p.numel() for p in trainer.model.parameters()),
                          peq_parameters=sum(p.numel() for n,p in trainer.model.named_parameters() if is_peq(n)),
                          loading=trainer.peq_loading_audit,amp=True,native_amp_check_messages=trainer.peq_amp_check_messages,
                          scaler=trainer.scaler.state_dict(),
                          native_optimizer_groups=[dict(kind=g.get("param_group"),weight_decay=g["weight_decay"],parameters=len(g["params"]))
                                                   for g in trainer.optimizer.param_groups],
                          seconds=time.monotonic()-started,diagnostic_state_discarded=True)
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(report_path,report)
