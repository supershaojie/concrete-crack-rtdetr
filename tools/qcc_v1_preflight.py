"""Single bounded server check, isolated from all formal model/optimizer/RNG state."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from pathlib import Path
import sys
import time
import traceback

from qcc_v1_common import ROOT, OUT, INIT, read_json, write_json, require, now, runtime, sha256
import torch
from ultralytics.models.rtdetr.qcc_loss import ramp
from ultralytics.models.rtdetr.qcc_model import QCCTrainer, gradient_report
from ultralytics.models.rtdetr.qcc_val import QCCValidator, EVAL
from ultralytics.utils.torch_utils import autocast, init_seeds, ModelEMA
from ultralytics.utils.patches import torch_load


def close_workers(trainer):
    for name in ("train_loader", "test_loader"):
        loader = getattr(trainer, name, None)
        iterator = getattr(loader, "iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


def reload_and_val(folder, checkpoint):
    """New interpreter: native resume followed by actual AutoBackend/fuse/warmup/val."""
    args = read_json(folder / "diagnostic_args.json")
    args.update(model=str(checkpoint), resume=str(checkpoint))
    t = QCCTrainer(overrides=args)
    try:
        t._setup_train()
        ckpt = torch_load(checkpoint, map_location="cpu")
        require(t.start_epoch == 20 and ramp(t.start_epoch) == 1, "Native resume reset QCC epoch")
        require(t.model.qcc_epoch == 20, "Restored epoch did not reach actual model")
        require(t.scaler.state_dict() == ckpt["scaler"], "Scaler state did not resume")
        require(t.ema.updates == ckpt["updates"], "EMA update count did not resume")
        actual = t.optimizer.state_dict()
        require(actual["param_groups"] == ckpt["optimizer"]["param_groups"], "Optimizer groups did not resume")
        for key, state in ckpt["optimizer"]["state"].items():
            for name, value in state.items():
                restored = actual["state"][key][name]
                require(torch.equal(restored.cpu(), value.to(restored.dtype)) if torch.is_tensor(value) else restored == value,
                        f"Optimizer state did not resume: {key}/{name}")
        require(all(torch.equal(v.cpu(), ckpt["ema"].state_dict()[k].float()) for k, v in t.ema.ema.state_dict().items()), "EMA tensors did not resume")
        write_json(folder / "resume.json", dict(status="PASS", epoch=t.start_epoch, ramp=ramp(t.start_epoch),
            optimizer=True, ema=True, scaler=t.scaler.state_dict(), checkpoint_sha256=sha256(checkpoint), amp=t.qcc_amp_evidence))
    finally:
        close_workers(t)
    del t, ckpt, actual
    gc.collect(); torch.cuda.empty_cache()
    validator = QCCValidator(args=dict(EVAL, model=str(checkpoint), data=args["data"], split="val", device="0", plots=False),
                             save_dir=folder / "one_real_val_batch")
    validator.one_batch = True
    # Observe the actual input of AutoBackend.warmup without replacing computation.
    from ultralytics.nn.autobackend import AutoBackend
    original = AutoBackend.warmup
    warmup = []
    def observed(backend, imgsz=(1, 3, 640, 640)):
        handle = backend.model.register_forward_pre_hook(lambda m, a: warmup.append(dict(
            shape=list(a[0].shape), finite=bool(torch.isfinite(a[0]).all()), zero=bool(torch.count_nonzero(a[0]) == 0),
            dtype=str(a[0].dtype))))
        try:
            return original(backend, imgsz)
        finally:
            handle.remove()
    AutoBackend.warmup = observed
    try:
        metrics = validator(model=str(checkpoint))
    finally:
        AutoBackend.warmup = original
    require(warmup and all(row["finite"] and row["zero"] for row in warmup), "Actual warmup did not use finite zeros")
    require(len(validator.qcc_seen) == 16 and validator.actual_settings["half"] is False, "Expected real B16 FP32 validation")
    write_json(folder / "new_process_val.json", dict(status="PASS", runtime=runtime(), checkpoint_sha256=sha256(checkpoint),
        warmup=warmup, images=len(validator.qcc_seen), settings=validator.actual_settings, metrics=metrics,
        scope="one real val batch, not full val/test"))


def run(folder):
    request = read_json(folder / "request.json")
    started = time.monotonic()
    deadline = started + request["seconds"]
    report = dict(status="FAIL", runtime=runtime(), started=now(), micro_batches=0, updates=[], steps=[],
                  checks={k: dict(status="PENDING") for k in ("cuda_b16_amp", "native_scale", "mechanism", "resume", "new_process_val")})
    trainer = None
    def stage(name):
        remaining = deadline - time.monotonic()
        print(f"QCC stage={name} elapsed={time.monotonic()-started:.1f}s remaining={remaining:.1f}s micro_batches={report['micro_batches']}/{request['micro_batches']}", flush=True)
        write_json(folder / "gpu.json", report)
        require(remaining > 0, "Preflight global deadline reached")
    try:
        stage("native setup + offline AMP check")
        init_seeds(42, deterministic=True)
        args = deepcopy(read_json(OUT / "prepare.json")["args"])
        args.update(project=str(folder), name="diagnostic_train", save_dir=str(folder / "diagnostic_train"), plots=False)
        write_json(folder / "diagnostic_args.json", args)
        trainer = QCCTrainer(overrides=args)
        trainer._setup_train()
        require(trainer.batch_size == 16 and trainer.args.imgsz == 640 and trainer.accumulate == 4 and trainer.amp, "Formal B16/640/AMP/accumulation changed")
        report["setup"] = dict(rebuild=trainer.qcc_rebuild, groups=trainer.qcc_optimizer_groups, args=vars(trainer.args),
                               amp=trainer.qcc_amp_evidence, initial_scale=trainer.scaler.get_scale())
        cold = deepcopy(trainer.model).cpu()
        torch.cuda.reset_peak_memory_stats()
        successful = False
        arm_limit = min(8, request["micro_batches"])
        for arm in ("native_scale", "diagnostic_init_scale_128"):
            if report["micro_batches"] >= request["micro_batches"]:
                break
            if arm != "native_scale":
                stage("isolated fallback init_scale=128 (formal scaler unchanged)")
                del trainer.model, trainer.optimizer, trainer.ema
                gc.collect(); torch.cuda.empty_cache()
                trainer.model = deepcopy(cold).to(trainer.device)
                trainer.optimizer = trainer.build_optimizer(trainer.model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
                trainer.ema = ModelEMA(trainer.model)
                trainer.scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128)
                trainer.qcc_update_count = 0
                init_seeds(42, deterministic=True)
            trainer.epoch = 20
            QCCTrainer._qcc_epoch_start(trainer)
            trainer.model.train(); trainer.model.qcc_epoch = 20; trainer.model.qcc_sample = True
            trainer.model.criterion = trainer.model.init_criterion()
            for group in trainer.optimizer.param_groups:
                group["lr"] = .0005 * trainer.lf(20)
            trainer.optimizer.zero_grad()
            batches = iter(trainer.train_loader)
            before_update = trainer.model.model[-1].cbr.offset_out.weight.detach().clone()
            for step in range(min(arm_limit, request["micro_batches"] - report["micro_batches"])):
                stage(arm)
                report["micro_batches"] += 1
                trainer.model.qcc_sample = step < 4
                batch = trainer.preprocess_batch(next(batches))
                require(tuple(batch["img"].shape) == (16, 3, 640, 640), "Actual augmented batch is not B16/640")
                holder = {}
                hook = trainer.model.criterion.register_forward_hook(lambda module, inputs, output: holder.update(output))
                tick = time.monotonic()
                try:
                    with autocast(True):
                        loss, items = trainer.model(batch)
                    require(torch.isfinite(loss), "Nonfinite original or QCC loss")
                    if step == 0:
                        named = dict(trainer.model.named_parameters())
                        names = ["model.26.cbr.offset_out.weight", "model.26.dec_bbox_head.2.layers.2.weight", "model.26.dec_score_head.2.weight"]
                        values = torch.autograd.grad(holder["loss_qcc"], [named[n] for n in names], retain_graph=True, allow_unused=True)
                        evidence = {n: dict(present=g is not None, finite=bool(torch.isfinite(g).all()) if g is not None else None,
                                            norm=float(g.float().norm()) if g is not None else None) for n, g in zip(names, values)}
                        require(all(g is None for g in values[:2]), "QCC directly reached CBR or final bbox head")
                        require(values[-1] is not None and torch.isfinite(values[-1]).all() and values[-1].norm() > 0,
                                "QCC final ordinary classification gradient missing/nonfinite/zero")
                        report["checks"]["mechanism"] = dict(status="PASS", epoch=20, ramp=1, gradients=evidence)
                    trainer.scaler.scale(loss).backward()
                finally:
                    hook.remove(); holder.clear()
                diagnostics = dict(trainer.model.criterion.last_diagnostics or {})
                report["steps"].append(dict(arm=arm, step=step, loss=float(loss.detach()), scale=trainer.scaler.get_scale(),
                                            diagnostics=diagnostics, seconds=time.monotonic()-tick))
                if (step + 1) % trainer.accumulate == 0:
                    trainer.optimizer_step()
                    report["updates"].append(dict(arm=arm, **trainer.qcc_update_samples[-1]))
                    if trainer.qcc_update_count and not torch.equal(before_update, trainer.model.model[-1].cbr.offset_out.weight.detach()):
                        successful = True
                        report["effective_update_arm"] = arm
                        break
            report["checks"][arm if arm == "native_scale" else "fallback_scale"] = dict(status="PASS" if successful else "PENDING", updates=trainer.qcc_update_count,
                note="Observed native-scale adaptation within boundary only" if arm == "native_scale" else "Isolated diagnostic scale128; not formal scaler")
            if successful:
                break
        require(successful, "No actual parameter-changing optimizer update within total micro-batch boundary")
        require(all(row["finite"] for row in trainer.qcc_update_samples[-1]["gradients"].values()), "Key update gradients nonfinite")
        require(trainer.qcc_update_samples[-1]["all_gradients_finite"], "An optimizer-update gradient is nonfinite")
        report["checks"]["cuda_b16_amp"] = dict(status="PASS", effective_updates=trainer.qcc_update_count, arm=report["effective_update_arm"],
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(), batch=16, imgsz=640, amp=True)
        report["overhead"] = dict(status="PENDING", note="Per-step diagnostic timings include probes; no controlled formal training overhead estimate")
        stage("native checkpoint save")
        trainer.epoch, trainer.fitness, trainer.best_fitness = 19, 0.0, 0.0
        trainer.ema.ema.qcc_identity = dict(diagnostic=True, commit=request["binding"]["code"]["commit"])
        trainer.save_model()
        checkpoint = trainer.last
        report["checkpoint"] = dict(path=str(checkpoint), sha256=sha256(checkpoint))
        close_workers(trainer)
        del trainer, cold, batch, loss, items, before_update
        trainer = None
        gc.collect(); torch.cuda.empty_cache()
        stage("new Python process: native resume + AutoBackend/fuse/zero warmup + real val")
        from qcc_v1 import run_bounded
        code = run_bounded([sys.executable, str(Path(__file__).resolve()), "--folder", str(folder), "--reload", str(checkpoint)],
                           deadline - time.monotonic(), "new-process lifecycle")
        for key in ("resume", "new_process_val"):
            report["checks"][key] = read_json(folder / f"{key}.json", dict(status="PENDING"))
        require(code == 0, "New-process lifecycle failed")
        report["status"] = "PASS" if all(v["status"] == "PASS" for v in report["checks"].values()) else "PENDING"
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc(), geometry=getattr(error, "details", None))
        raise
    finally:
        if trainer is not None:
            close_workers(trainer)
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(folder / "gpu.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_qccument("--folder", type=Path, required=True)
    parser.add_qccument("--reload", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.reload:
        reload_and_val(args.folder, args.reload)
    else:
        run(args.folder)
