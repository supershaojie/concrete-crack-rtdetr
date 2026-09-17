"""Finite real augmented B16/640 native AMP server gate; never formal training or test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from bfr_p4_lifecycle import (ROOT, MAIN, VARIANTS, paths, identity, recipe, runtime, require,
                             write_json, sha256, trainer_class, optimizer_audit, read_json)
from bfr_p4_lifecycle import server_resource_check
from ultralytics.utils import ASSETS
from ultralytics.utils.torch_utils import ModelEMA, autocast, TORCH_2_4


def amp_resources():
    records = []
    for dest, candidates in (
        (ASSETS / "bus.jpg", [MAIN / "ultralytics-main/ultralytics/assets/bus.jpg", MAIN / "bus.jpg"]),
        (ROOT / "yolo26n.pt", [MAIN / "yolo26n.pt", MAIN / "weights/yolo26n.pt"]),
    ):
        if not dest.is_file():
            found = next((p for p in candidates if p.is_file()), None)
            if found is None:
                raise FileNotFoundError("PENDING existing native AMP resource: " + str(dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as target, found.open("rb") as source:
                shutil.copyfileobj(source, target)
        records.append({"path": str(dest), "sha256": sha256(dest)})
    return records


def new_scaler(enabled=True):
    return torch.amp.GradScaler("cuda", enabled=enabled) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=enabled)


def native_resume(trainer, variant, folder):
    """Use native save_model and resume_training, including native half EMA/state storage."""
    from init_bfr_p4 import build_training_model
    trainer.epoch = 0
    trainer.fitness = trainer.best_fitness = 0.0
    trainer.save_model()
    checkpoint = torch.load(trainer.last, map_location="cpu", weights_only=False)
    require(checkpoint["epoch"] == 0 and checkpoint.get("optimizer") and checkpoint.get("scaler") is not None,
            "Native checkpoint lacks optimizer/scaler/epoch")
    quantized = deepcopy(checkpoint["ema"]).float()
    require(torch.count_nonzero(quantized.model[22].bfr.wo.weight) > 0, "Native EMA checkpoint lost learned BFR")
    restored, loading = build_training_model(quantized.yaml, quantized, trainer.data, variant)
    restored.to(trainer.device)
    ResumeTrainer = trainer_class(variant)
    resumed = ResumeTrainer.__new__(ResumeTrainer)
    resumed.args = SimpleNamespace(**vars(trainer.args))
    resumed.args.model = str(trainer.last)
    resumed.model = restored
    resumed.optimizer = trainer.build_optimizer(restored, name="AdamW", lr=trainer.args.lr0,
                                                 momentum=trainer.args.momentum, decay=trainer.args.weight_decay)
    resumed.scaler = new_scaler(enabled=trainer.scaler.is_enabled())
    resumed.ema = ModelEMA(restored)
    resumed.resume, resumed.epochs = True, 200
    resumed.resume_training(checkpoint)
    require(resumed.start_epoch == 1, "Native resume epoch mismatch")
    require(resumed.scaler.state_dict() == checkpoint["scaler"], "Native resume scaler mismatch")
    reference = checkpoint["optimizer"]
    actual = resumed.optimizer.state_dict()
    require(reference["param_groups"] == actual["param_groups"], "Native resume optimizer groups mismatch")
    require(set(reference["state"]) == set(actual["state"]), "Native resume optimizer state keys mismatch")
    for key, saved in reference["state"].items():
        for name, value in saved.items():
            after = actual["state"][key][name]
            require(torch.equal(value.to(after.device, dtype=after.dtype), after)
                    if torch.is_tensor(value) else value == after, "Native resume optimizer state mismatch")
    for name, before in quantized.state_dict().items():
        require(torch.equal(before, restored.state_dict()[name].cpu()), "Native resume learned tensor mismatch: " + name)
    require(resumed.ema.updates == checkpoint["updates"], "Native resume EMA update count mismatch")
    for name, before in quantized.state_dict().items():
        require(torch.equal(before, resumed.ema.ema.state_dict()[name].cpu()), "Native resume EMA tensor mismatch: " + name)
    result = {"status": "PASSED", "native_save_model": True, "native_resume_training": True,
              "epoch": resumed.start_epoch, "optimizer": "exact saved quantization", "scaler": "exact",
              "learned_parameters": "exact saved half quantization converted to FP32", "ema_updates": resumed.ema.updates,
              "checkpoint_sha256": sha256(trainer.last), "checkpoint": str(trainer.last), "trainer_loading": loading}
    del restored, resumed, checkpoint, quantized
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scope": "server_B16_640_native_AMP", "runtime": runtime(),
              "formal_training": "NOT_STARTED", "final_test": "NOT_RUN",
              "budget": {"capacity_max_batches": args.max_batches, "target_effective_updates": 2,
                         "engineering": "check_bfr_p4.py fixed CPU/CUDA engineering budgets; no val/test"}}
    dest = args.output / "preflight.json"
    trainer = None
    try:
        if not torch.cuda.is_available():
            raise FileNotFoundError("PENDING CUDA unavailable")
        report["identity"] = identity(args.variant, args.source, args.initialized, args.data)
        report["amp_resources"] = amp_resources()
        report["gpu_before"] = server_resource_check()
        from check_bfr_p4_math import run_checks
        report["mathematics"] = {"cpu": run_checks("cpu"), "cuda": run_checks("cuda"), "status": "PASSED"}
        write_json(dest, report)
        # Full engineering diagnostics are rerun at this exact code identity on the server.
        engineering = args.output / "engineering"
        with (args.output / "engineering.log").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-u", str(ROOT / "tools/check_bfr_p4.py"),
                            "--variant", args.variant, "--source", str(args.source),
                            "--dataset", report["identity"]["data"]["root"], "--output", str(engineering),
                            "--devices", "cpu", "cuda"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        check = read_json(engineering / "checks.json")
        require(check["status"] == "PASSED_LOCAL" and all(check["modes"][m]["status"] == "PASSED"
                for m in ("cpu_fp32", "cuda_fp32", "cuda_amp")), "Server engineering diagnostics failed/pending")
        report["engineering"] = {"status": "PASSED", "report": str(engineering / "checks.json"),
                                 "sha256": sha256(engineering / "checks.json")}
        report["gpu_before_capacity"] = server_resource_check()
        overrides, _ = recipe(args.variant, args.initialized, args.data)
        overrides.update(project=str(args.output), name="disposable_train", save_dir=str(args.output / "disposable_train"))
        Trainer = trainer_class(args.variant, args.output / "trainer_loading.json")
        trainer = Trainer(overrides=overrides)
        trainer._setup_train()
        require(trainer.amp and trainer.args.batch == 16 and trainer.args.imgsz == 640, "Native AMP/B16/640 changed")
        effective = vars(trainer.args)
        changed = {k: [v, effective.get(k)] for k, v in overrides.items()
                   if type(v) is not type(effective.get(k)) or v != effective.get(k)}
        require(not changed, "Native disposable trainer recipe drift: " + repr(changed))
        require(type(trainer.optimizer) is torch.optim.AdamW, "Expected native AdamW")
        report["optimizer"] = optimizer_audit(trainer.model, trainer.optimizer)
        report["augmented_train_args"] = vars(trainer.args)
        trainer.model.train()
        trainer.optimizer.zero_grad()
        trainer.epoch = 0
        rows, updates, observed = [], [0], []
        original_step = trainer.optimizer.step
        def recorded_step(*values, **options):
            grads = {n: float(p.grad.detach().float().norm()) if p.grad is not None else 0.0
                     for n, p in trainer.model.named_parameters() if n.startswith("model.22.bfr.")}
            require(all(math.isfinite(v) for v in grads.values()), "Effective update has nonfinite BFR gradients")
            observed.append(grads)
            result = original_step(*values, **options)
            updates[0] += 1
            return result
        trainer.optimizer.step = recorded_step
        dn = []
        def record_dn(module, inputs, output):
            meta = output[-1]
            dn.append({"dn_num_split": meta.get("dn_num_split") if meta else None,
                       "dn_num_group": meta.get("dn_num_group") if meta else None})
        hook = trainer.model.model[-1].register_forward_hook(record_dn)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        last_step = -1
        nw = max(round(trainer.args.warmup_epochs * len(trainer.train_loader)), 100)
        try:
            for i, batch in enumerate(trainer.train_loader):
                if i >= args.max_batches:
                    break
                # Identical epoch-0 warmup and accumulation to the native parent trainer.
                trainer.accumulate = max(1, int(np.interp(i, [0, nw], [1, trainer.args.nbs / 16]).round()))
                for group in trainer.optimizer.param_groups:
                    group["lr"] = float(np.interp(i, [0, nw], [trainer.args.warmup_bias_lr
                        if group.get("param_group") == "bias" else 0.0, group["initial_lr"] * trainer.lf(0)]))
                    if "momentum" in group:
                        group["momentum"] = float(np.interp(i, [0, nw], [trainer.args.warmup_momentum, trainer.args.momentum]))
                before, count = float(trainer.scaler.get_scale()), updates[0]
                with autocast(True):
                    batch = trainer.preprocess_batch(batch)
                    require(batch["img"].shape == (16, 3, 640, 640) and len(batch["cls"]) > 0,
                            "Capacity batch must be real augmented B16/640 with GT")
                    losses, items = trainer.model(batch)
                    loss = losses.sum()
                require(torch.isfinite(loss), "Nonfinite real detection loss")
                trainer.scaler.scale(loss).backward()
                attempted = i - last_step >= trainer.accumulate
                if attempted:
                    trainer.optimizer_step()
                    last_step = i
                row = {"batch": i, "loss": float(loss.detach()), "loss_items": items.detach().cpu().tolist(),
                       "gt": len(batch["cls"]), "dn": dn[-1], "scale_before": before,
                       "scale_after": float(trainer.scaler.get_scale()), "optimizer_attempted": attempted,
                       "effective_update": updates[0] > count, "scaler_skipped": attempted and updates[0] == count,
                       "wo_nonzero": bool(torch.count_nonzero(trainer.model.model[22].bfr.wo.weight))}
                require(row["dn"]["dn_num_group"] is not None, "GT/DN path was not exercised")
                rows.append(row)
                # Warmup's very first step has zero weight lr: require a post-nonzero upstream update too.
                upstream = ("wd.weight", "gn.weight", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
                startup = any(all(g.get("model.22.bfr." + n, 0) > 0 for n in upstream) for g in observed)
                report["capacity"] = {"batch": 16, "imgsz": 640, "amp": True, "actual_batches": len(rows),
                    "effective_optimizer_updates": updates[0], "steps": rows, "effective_gradients": observed,
                    "gradient_startup": "PASSED" if startup and row["wo_nonzero"] else "PENDING",
                    "gn_bias_exception": "DC-only zero-gradient direction; parameter remains in native optimizer"}
                write_json(dest, report)
                if updates[0] >= 2 and startup and row["wo_nonzero"]:
                    break
        finally:
            hook.remove()
            trainer.optimizer.step = original_step
        torch.cuda.synchronize()
        report["capacity"].update(elapsed_seconds=time.perf_counter() - started,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
        require(updates[0] >= 2 and report["capacity"]["gradient_startup"] == "PASSED",
                "Finite batch budget exhausted without stable effective updates and BFR gradient startup")
        require(observed[0]["model.22.bfr.wo.weight"] > 0, "First effective W_o gradient missing")
        report["native_resume"] = native_resume(trainer, args.variant, args.output)
        # Reuse candidate-aware fusion diagnostics on the learned augmented-batch model.
        from check_bfr_p4 import fusion_trace, strict_tf32
        x = batch["img"][:1].detach()
        trainer.model.eval()
        lifecycle = {"default_fp32": fusion_trace(trainer.model, x, "cuda", "fp32")}
        with strict_tf32() as flags:
            lifecycle["strict_fp32"] = fusion_trace(trainer.model, x, "cuda", "fp32")
            lifecycle["tf32_original"] = list(flags)
        lifecycle["half"] = fusion_trace(trainer.model, x, "cuda", "half")
        lifecycle["amp"] = fusion_trace(trainer.model, x, "cuda", "amp")
        lifecycle["status"] = "PASSED"
        report["learned_lifecycle"] = lifecycle
        require(report["identity"] == identity(args.variant, args.source, args.initialized, args.data),
                "Code/data/init identity changed during preflight")
        report["status"] = "PASSED"
    except FileNotFoundError as error:
        report.update(status="PENDING", error=repr(error))
        raise
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        write_json(dest, report)
    print(json.dumps({"status": report["status"], "report": str(dest)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_bfr_p4_v1")
    for flag in ("source", "initialized", "data", "output"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--max-batches", type=int, default=16, help="Fixed finite capacity budget, 2..32 (default 16)")
    args = parser.parse_args()
    require(2 <= args.max_batches <= 32, "Budget must be declared between 2 and 32 batches")
    p = paths(args.variant)
    for key in ("source", "initialized", "data"):
        if getattr(args, key) is None:
            setattr(args, key, p[key])
    if args.output is None:
        args.output = p["metadata"] / "server_preflight"
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
