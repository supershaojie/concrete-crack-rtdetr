"""Isolated native setup and at most 16 real B16/640 micro-batches, 900s total."""
from __future__ import annotations

from copy import deepcopy
import time
import traceback

from tcr_v1_core import *


def run(folder):
    from tcr_v1_ops import verify_prepared
    from tcr_v1_train import TCRTrainer
    from check_tcr_v1 import algorithm
    from tcr_v1_fusion import diagnose, precision_state, model_identity
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    prepared=verify_prepared()
    report=dict(status="FAILED",fingerprint=prepared["fingerprint"],runtime=runtime(),scope="server B16/640 native-AMP capacity",
                max_micro_batches=16,max_seconds=900,diagnostic_only_scale=128,formal_training="NOT_RUN",effective_updates=0,steps=[],
                precision_observations=dict(before_trainer=precision_state()))
    try:
        require(torch.cuda.is_available(),"CUDA required for server capacity")
        with isolated_rng():
            report["algorithm"]=algorithm("cuda:0")
            args=YAML.load(OUT/"train_args.yaml")
            args.update(project=str(folder),name="native_setup",save_dir=str(folder/"native_setup"))
            trainer=TCRTrainer(overrides=args,evidence=folder)
            report["precision_observations"]["after_trainer_init"]=precision_state()
            trainer._setup_train()
            report["precision_observations"]["after_trainer_setup"]=precision_state()
            require(trainer.amp,"Native check_amp did not enable AMP")
            require(trainer.batch_size==16 and trainer.args.imgsz==640,"Capacity recipe changed")
            report["optimizer_groups"]=optimizer_audit(trainer.model,trainer.optimizer)
            report["native_loading"]=trainer.loading_audit
            trainer.scaler=torch.cuda.amp.GradScaler(init_scale=128,enabled=True)
            trainer.model.train(); trainer.epoch=0; trainer.optimizer.zero_grad()
            trainer._oom_retries=3
            torch.cuda.reset_peak_memory_stats()
            nb=len(trainer.train_loader); nw=max(round(args["warmup_epochs"]*nb),100)
            last_opt=-1; calls=[0]
            handle=trainer.optimizer.register_step_post_hook(lambda *unused: calls.__setitem__(0,calls[0]+1))
            watched={n:p for n,p in trainer.model.named_parameters() if n in NEW or n in {"model.0.conv.weight","model.17.conv.weight","model.20.O_proj.weight","model.26.cbr.offset_out.weight"}}
            trainer.model.criterion=trainer.model.init_criterion()
            components={}
            loss_hook=trainer.model.criterion.register_forward_hook(lambda module,inputs,output: components.update({k:float(v.detach()) for k,v in output.items()}))
            p_after_o=False; finite_old=False
            for ni,raw in enumerate(trainer.train_loader):
                if ni>=16 or time.monotonic()-started>=900: break
                sample=trainer.preprocess_batch(raw)
                require(tuple(sample["img"].shape)==(16,3,640,640),"Real B16/640 not obtained")
                xi=[0,nw]
                trainer.accumulate=max(1,int(np.interp(ni,xi,[1,args["nbs"]/16]).round()))
                for group in trainer.optimizer.param_groups:
                    group["lr"]=np.interp(ni,xi,[args["warmup_bias_lr"] if group.get("param_group")=="bias" else 0.,group["initial_lr"]*trainer.lf(0)])
                    if "momentum" in group: group["momentum"]=np.interp(ni,xi,[args["warmup_momentum"],args["momentum"]])
                module=trainer.model.model[17].tcr; module.capture=True
                with torch.autocast("cuda",dtype=torch.float16):
                    loss,items=trainer.model(sample)
                scale=trainer.scaler.get_scale()
                trainer.scaler.scale(loss).backward()
                grads={n:dict(norm=float(p.grad.detach().float().norm()/scale),finite=bool(torch.isfinite(p.grad).all())) if p.grad is not None else dict(norm=None,finite=True,absent=True) for n,p in watched.items()}
                all_finite=bool(torch.isfinite(loss)) and all(torch.isfinite(p.grad).all().item() for p in trainer.model.parameters() if p.grad is not None)
                row=dict(micro_batch=ni,shape=list(sample["img"].shape),loss=float(loss.detach()),items=items.tolist(),losses=dict(components),
                         finite=all_finite,gradients=grads,scale_before=scale,accumulate=trainer.accumulate,mechanism=module.last_stats,
                         effective_update=False,overflow_skipped=False)
                if ni-last_opt>=trainer.accumulate:
                    before={n:p.detach().clone() for n,p in watched.items()}; old_calls=calls[0]
                    trainer.optimizer_step(); last_opt=ni
                    changes={n:float((p.detach()-before[n]).float().norm()) for n,p in watched.items()}
                    row.update(scale_after=trainer.scaler.get_scale(),optimizer_called=calls[0]>old_calls,
                               overflow_skipped=calls[0]==old_calls,parameter_changes=changes,
                               effective_update=calls[0]>old_calls and any(v>0 for v in changes.values()))
                    if row["effective_update"]: report["effective_updates"]+=1
                if grads["model.17.tcr.P.weight"]["norm"] and grads["model.17.tcr.P.weight"]["finite"]: p_after_o=True
                finite_old=any(g["norm"] and g["finite"] for n,g in grads.items() if n not in NEW)
                report["steps"].append(row)
                report["elapsed_seconds"]=time.monotonic()-started
                write_json(folder/"preflight.json",report)
                require(all_finite,"NaN/Inf in real AMP loss/gradients; report retained")
                if report["effective_updates"]>=2 and p_after_o and finite_old: break
            handle.remove(); loss_hook.remove(); module.capture=False; module.last_stats=None
            require(report["effective_updates"]>=2 and p_after_o and finite_old,"Bound reached without two changed-weight updates and post-O P gradient")
            from check_tcr_v1 import half_validator
            report["native_half_epoch_validator"] = half_validator(trainer.model, folder)
            require(torch.count_nonzero(module.O.weight)>0,"Output projection did not actually update")
            # Check visibility with a controlled nonzero O on a separate copy, so
            # tiny warmup changes cannot be mistaken for a missing inference path.
            trainer_state=model_identity(trainer.model)
            diagnostic_model=deepcopy(trainer.model).eval().float()
            copy_state=model_identity(diagnostic_model)
            require(copy_state["state_sha256"]==trainer_state["state_sha256"],"Diagnostic copy changed weights/BN before O injection")
            with torch.no_grad():
                diagnostic_model.model[17].tcr.O.weight.normal_(0,.01)
                report["fusion_weight_mode"]="isolated copy with artificial nonzero O (std=.01); capacity/gradient records use actual native updates"
            differences=[k for k,v in trainer.model.state_dict().items() if not torch.equal(v,diagnostic_model.state_dict()[k])]
            require(differences==["model.17.tcr.O.weight"],"Diagnostic changed state outside the disclosed O projection")
            report["fusion_copy_audit"]=dict(trainer=trainer_state,before_O=copy_state,
                after_O=model_identity(diagnostic_model),changed_state_keys=differences,
                O_modified_before_either_fused_copy=True)
            x=sample["img"][:1].detach().float().clone()
            report["precision_observations"]["before_fusion"]=precision_state()
            try:
                fusion=diagnose(diagnostic_model,x,folder/"fusion")
                report["fusion"]=dict(status=fusion["status"],strict_accepted=fusion["strict_accepted"],
                    runtime_accepted=fusion["runtime_accepted"],evidence=str(folder/"fusion/precision_diagnostic.json"),
                    acceptance_scope=fusion["acceptance_scope"])
                require(fusion["strict_accepted"],"Strict FP32 fusion failed at unchanged mother tolerances; inspect evidence")
            finally:
                report["precision_observations"]["after_fusion"]=precision_state()
                report["fusion_trainer_unchanged"]=trainer_state==model_identity(trainer.model)
                require(report["fusion_trainer_unchanged"],"Fusion diagnostic changed actual Trainer weights/BN/mode")
            # Do not call an unsuccessful runtime comparison PASS. Capacity and
            # the original FP32 fusion gate have their own explicit scope.
            report.update(status="PASS" if fusion["runtime_accepted"] else "PASS_STRICT_FP32_ONLY",
                          capacity_status="PASS",post_O_P_gradient=True,original_gradients=True,nonzero_fusion=True,
                          parameters=sum(p.numel() for p in trainer.model.parameters()),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        write_json(folder/"preflight.json",report)
    return report
