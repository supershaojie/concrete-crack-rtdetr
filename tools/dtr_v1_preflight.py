"""One native-scaler server check, <=16 total training micro-batches / <=900s."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import sys
import time
import traceback

from dtr_v1_common import *
from experiment_runtime import run_bounded
import torch
from ultralytics.models.rtdetr.dtr_loss import ramp
from ultralytics.models.rtdetr.dtr_trainer import DTRTrainer
from ultralytics.models.rtdetr.dtr_model import DTRDetectionModel
from ultralytics.models.rtdetr.dtr_val import DTRValidator, EVAL
from ultralytics.utils.torch_utils import autocast, init_seeds
from ultralytics.utils.patches import torch_load


def close_workers(trainer):
    for key in ("train_loader", "test_loader"):
        iterator = getattr(getattr(trainer, key, None), "iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


def reload_and_val(folder, checkpoint):
    args = read_json(folder / "diagnostic_args.json")
    args.update(model=str(checkpoint), resume=str(checkpoint))
    t = DTRTrainer(overrides=args)
    try:
        t._setup_train()
        ckpt = torch_load(checkpoint, map_location="cpu")
        require(isinstance(t.model, DTRDetectionModel), "Resume lost DTR model")
        require(t.start_epoch == 21 and t.model.dtr_epoch == 21 and ramp(t.start_epoch) == 1, "Resume epoch/schedule reset")
        require(t.scheduler.last_epoch == 20, "Scheduler epoch did not resume")
        require(t.scaler.state_dict() == ckpt["scaler"], "Scaler did not resume")
        require(t.ema.updates == ckpt["updates"], "EMA updates did not resume")
        actual = t.optimizer.state_dict()
        require(actual["param_groups"] == ckpt["optimizer"]["param_groups"], "Optimizer groups did not resume")
        for key, state in ckpt["optimizer"]["state"].items():
            for name, value in state.items():
                restored = actual["state"][key][name]
                require(torch.equal(restored.cpu(), value.to(restored.dtype)) if torch.is_tensor(value) else restored == value,
                        f"Optimizer state mismatch {key}/{name}")
        require(all(torch.equal(v.cpu(), ckpt["ema"].state_dict()[k].float()) for k, v in t.ema.ema.state_dict().items()),
                "EMA tensors did not resume")
        write_json(folder / "resume.json", dict(status="PASS", epoch=21, ramp=1, optimizer=True,
                   scheduler_last_epoch=t.scheduler.last_epoch, ema=True, scaler=t.scaler.state_dict()))
    finally:
        close_workers(t)
    del t, ckpt, actual
    gc.collect()
    torch.cuda.empty_cache()
    validator = DTRValidator(args=dict(EVAL, model=str(checkpoint), data=args["data"], split="val", device="0", plots=False),
                             save_dir=folder / "one_real_val_batch")
    validator.one_batch = True
    validator.export_path = folder / "diagnostic_val_predictions_gt.jsonl.gz"
    validator.export_identity = dict(scope="one diagnostic batch only")
    from ultralytics.nn.autobackend import AutoBackend
    original = AutoBackend.warmup
    warmup = []
    def observed(backend, imgsz=(1, 3, 640, 640)):
        handle = backend.model.register_forward_pre_hook(lambda m, a: warmup.append(dict(
            shape=list(a[0].shape), finite=bool(torch.isfinite(a[0]).all()),
            zero=bool(torch.count_nonzero(a[0]) == 0), dtype=str(a[0].dtype))))
        try:
            return original(backend, imgsz)
        finally:
            handle.remove()
    AutoBackend.warmup = observed
    try:
        metrics = validator(model=str(checkpoint))
    finally:
        AutoBackend.warmup = original
    require(warmup and all(r["finite"] and r["zero"] and r["dtype"] == "torch.float32" for r in warmup),
            "FP32 warmup must use finite zeros")
    require(len(validator.seen_ids) == 16 and validator.actual_settings["half"] is False, "Expected actual B16 FP32 val")
    write_json(folder / "new_process_val.json", dict(status="PASS", images=16, warmup=warmup, metrics=metrics,
               runtime=runtime(), scope="one real val batch only, no test", query_count=validator.query_count))


def run(folder):
    request = read_json(folder / "request.json")
    started = time.monotonic()
    deadline = started + request["seconds"]
    report = dict(status="PENDING", started=now(), micro_batches=0, steps=[], updates=[],
                  checks={k: dict(status="PENDING") for k in ("cuda_b16_amp", "native_scale", "mechanism", "resume", "new_process_val")})
    trainer = None
    def stage(name):
        print(f"DTR stage={name} elapsed={time.monotonic()-started:.1f}s remaining={deadline-time.monotonic():.1f}s "
              f"micro_batches={report['micro_batches']}/{request['micro_batches']}", flush=True)
        write_json(folder / "gpu.json", report)
        require(time.monotonic() < deadline, "Global preflight deadline reached")
    try:
        stage("native setup")
        init_seeds(42, deterministic=True)
        args = deepcopy(read_json(OUT / "prepare.json")["args"])
        args.update(project=str(folder), name="diagnostic_train", save_dir=str(folder / "diagnostic_train"), plots=False)
        write_json(folder / "diagnostic_args.json", args)
        trainer = DTRTrainer(overrides=args)
        trainer._setup_train()
        require(trainer.batch_size == 16 and trainer.args.imgsz == 640 and trainer.accumulate == 4 and trainer.amp,
                "Formal B16/640 AMP/accumulation changed")
        trainer.epoch = 20
        DTRTrainer._epoch_start(trainer)
        trainer.model.train()
        trainer.model.dtr_epoch, trainer.model.dtr_sample = 20, True
        trainer.model.criterion = trainer.model.init_criterion()
        for group in trainer.optimizer.param_groups:
            group["lr"] = trainer.args.lr0 * trainer.lf(20)
        trainer.optimizer.zero_grad()
        report["setup"] = dict(args=vars(trainer.args), runtime=runtime(), amp=trainer.dtr_amp_evidence,
                               native_scale=trainer.scaler.get_scale(), epoch=20, accumulate=trainer.accumulate)
        before = trainer.model.model[-1].cbr.offset_out.weight.detach().clone()
        batches = iter(trainer.train_loader)
        torch.cuda.reset_peak_memory_stats()
        for step in range(request["micro_batches"]):
            stage("native AMP micro-batch")
            batch = trainer.preprocess_batch(next(batches))
            require(tuple(batch["img"].shape) == (16, 3, 640, 640), "Actual augmented input is not B16/640")
            losses = {}
            hook = trainer.model.criterion.register_forward_hook(lambda m, a, output: losses.update(output))
            try:
                with autocast(True):
                    loss, _ = trainer.model(batch)
                require(bool(torch.isfinite(loss)), "Nonfinite L0 or DTR loss")
                if step == 0:
                    names = dict(trainer.model.named_parameters())
                    selected = ["model.26.cbr.offset_out.weight", "model.26.dec_bbox_head.2.layers.2.weight", "model.26.dec_score_head.2.weight"]
                    gradients = torch.autograd.grad(losses["loss_dtr"], [names[n] for n in selected],
                                                    retain_graph=True, allow_unused=True)
                    require(gradients[-1] is None, "DTR reached independent final logits")
                    require(all(g is not None and torch.isfinite(g).all() for g in gradients[:2]), "DTR regression/CBR gradient invalid")
                    report["checks"]["mechanism"] = dict(status="PASS", epoch=20, ramp=1,
                        gradients={n: dict(present=g is not None, norm=float(g.norm()) if g is not None else None)
                                   for n, g in zip(selected, gradients)}, single_extra_backward="first batch only")
                    del gradients
                trainer.scaler.scale(loss).backward()
            finally:
                hook.remove()
                losses.clear()
            report["micro_batches"] += 1
            report["steps"].append(dict(loss=float(loss.detach()), scale=trainer.scaler.get_scale(),
                                        diagnostics=trainer.model.criterion.last_diagnostics))
            trainer.model.dtr_sample = step < 3
            if (step + 1) % trainer.accumulate == 0:
                trainer.optimizer_step()
                report["updates"].append(trainer.dtr_steps[-1])
                if trainer.dtr_update_count and not torch.equal(before, trainer.model.model[-1].cbr.offset_out.weight.detach()):
                    require(trainer.dtr_steps[-1]["gradients_finite"], "Effective update has nonfinite gradients")
                    break
        successful = trainer.dtr_update_count > 0 and not torch.equal(before, trainer.model.model[-1].cbr.offset_out.weight.detach())
        if not successful:
            report["reason"] = ("No effective native-scaler update within the boundary. Initial AMP overflow is recorded, "
                                "not relabeled PASS. No replacement scaler or extra training performed.")
            return report
        report["checks"]["native_scale"] = dict(status="PASS", updates=trainer.dtr_update_count, initial=report["setup"]["native_scale"],
                                                final=trainer.scaler.get_scale(), skipped=sum(s["skipped"] for s in report["updates"]))
        report["checks"]["cuda_b16_amp"] = dict(status="PASS", effective_updates=trainer.dtr_update_count,
            batch=16, imgsz=640, amp=True, peak_allocated=torch.cuda.max_memory_allocated())
        stage("native checkpoint save")
        trainer.fitness = trainer.best_fitness = 0.
        trainer.ema.ema.dtr_identity = "diagnostic_only"
        trainer.save_model()
        checkpoint = trainer.last
        report["checkpoint"] = file_info(checkpoint)
        close_workers(trainer)
        del trainer, batch, loss, before, batches
        trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        stage("new-process resume and finite FP32 real val")
        code = run_bounded([sys.executable, str(Path(__file__).resolve()), "--folder", str(folder), "--reload", str(checkpoint)],
                           deadline - time.monotonic(), "reload/val", ROOT)
        require(code == 0, "New-process resume/val failed")
        for key in ("resume", "new_process_val"):
            report["checks"][key] = read_json(folder / f"{key}.json", dict(status="PENDING"))
        report["status"] = "PASS" if all(v["status"] == "PASS" for v in report["checks"].values()) else "PENDING"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            close_workers(trainer)
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(folder / "gpu.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--reload", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.reload:
        reload_and_val(args.folder, args.reload)
    else:
        result = run(args.folder)
        sys.exit(0 if result["status"] == "PASS" else 2)
