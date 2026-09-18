"""Disposable native B16/640/AMP train-data preflight, <=16 batches, >=2 real updates.

The native training loop, warmup, accumulation, optimizer, augmentation and scaler
are preserved. A private callback exception stops before epoch validation/saving.
This command cannot dispatch formal training, val or test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from init_gra import ROOT, VARIANTS, SOURCE_SHA256, require, sha256, runtime, write_json, verify_model
from train_gra import (MAIN, RecordingTrainer, recipe, inventory, optimizer_coverage,
                       source_manifest, verify_initialization, ensure_amp_resources)


class BudgetReached(Exception):
    """Intentional finite stop, never a completed epoch or formal result."""


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def preflight(args):
    output = args.output.resolve()
    require(not output.exists(), "Existing preflight evidence protected; choose a new directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "FAILED", "scope": "real train data, original online augmentation, disposable native trainer",
              "variant": args.variant, "batch": 16, "imgsz": 640, "amp": True, "batch_budget": args.max_batches,
              "target_effective_updates": 2, "effective_updates": 0, "optimizer_attempts": 0, "batches": [],
              "formal_training": "NOT_STARTED", "final_test": "NOT_RUN", "full_val": "NOT_RUN",
              "runtime": runtime(), "source_manifest": source_manifest(), "source_sha256": SOURCE_SHA256,
              "created": datetime.now(timezone.utc).isoformat()}
    handles, trainer = [], None
    try:
        require(1 <= args.max_batches <= 16, "Preflight hard limit is 16 batches")
        require(torch.cuda.is_available(), "PENDING: CUDA unavailable; B16/640/native AMP not executed")
        require(args.initialized.is_file() and args.source.is_file() and args.data.is_file() and args.audit.is_file(),
                "PENDING: controlled initialization/public source/data/audit unavailable")
        verify_initialization(args.variant, args.initialized, args.audit, args.source)
        report.update(initialized_sha256=sha256(args.initialized), data_sha256=sha256(args.data),
                      dataset_inventory=inventory(args.data), amp_resources=ensure_amp_resources())
        try:
            query = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            report["gpu_concurrent_load"] = query.stdout
        except (OSError, subprocess.TimeoutExpired) as error:
            report["gpu_concurrent_load"] = repr(error)
        report["timing_note"] = "Observed current concurrent GPU load; not an exclusive-GPU benchmark"
        train_args, differences = recipe(args.variant, args.initialized, args.data)
        # Only disposable output identity changes. Epochs remain 200; no artificial scheduler shortening.
        train_args.update(project=str(output), name="disposable_native_train", save_dir=str(output / "disposable_native_train"))
        report["train_args"] = train_args
        report["recipe_diff"] = differences
        report["temporary_output_only"] = ["project", "name", "save_dir"]

        class BoundedTrainer(RecordingTrainer):
            gra_variant = args.variant
            gra_metadata = output

            def preprocess_batch(self, batch):
                data = super().preprocess_batch(batch)
                self.gra_batch = {"input_shape": list(data["img"].shape), "input_dtype": str(data["img"].dtype),
                                  "device": str(data["img"].device), "GT": len(data["cls"]),
                                  "gt_per_image": torch.bincount(data["batch_idx"].long(), minlength=len(data["img"])).tolist(),
                                  "image_files": [str(p) for p in data.get("im_file", [])]}
                return data

            def optimizer_step(self):
                scale = self.scaler.get_scale()
                grads = {n: float(p.grad.detach().float().norm() / scale) for n, p in self.model.named_parameters()
                         if n.startswith("model.18.") and p.grad is not None}
                before = report["effective_updates"]
                super().optimizer_step()  # native unscale, clip=10, scaler.step/update, zero_grad, EMA
                report["optimizer_attempts"] += 1
                self.gra_step = {"grad_norms_before_clip_unscaled": {k: v if torch.isfinite(torch.tensor(v)) else str(v) for k, v in grads.items()},
                                 "scale_before": scale, "scale_after": self.scaler.get_scale(),
                                 "skipped": report["effective_updates"] == before,
                                 "effective_update_count": report["effective_updates"]}

        trainer = BoundedTrainer(overrides=train_args)
        torch.cuda.reset_peak_memory_stats(trainer.device)

        def decoder_hook(_module, _inputs, result):
            require(all(torch.isfinite(t).all() for t in result if isinstance(t, torch.Tensor)), "Nonfinite native Decoder output")
            trainer.gra_dn = json_value(result[-1]) if isinstance(result, tuple) and isinstance(result[-1], dict) else None
            trainer.gra_decoder_shapes = [list(t.shape) for t in result if isinstance(t, torch.Tensor)]

        def offset_hook(_module, _inputs, result):
            delta = .25 * result.detach().float().tanh()
            trainer.gra_offset = {"max_abs_source_pixels": float(delta.abs().max()), "mean_abs_source_pixels": float(delta.abs().mean()),
                                  "finite": bool(torch.isfinite(delta).all()), "saturation_fraction": float((delta.abs() > .249).float().mean())}
            require(trainer.gra_offset["finite"] and trainer.gra_offset["max_abs_source_pixels"] <= .25, "Invalid offset")

        def setup(t):
            verify_model(t.model, args.variant, zero=True)
            require(t.amp and t.batch_size == 16 and t.args.imgsz == 640, "Native preflight disabled AMP/changed B16/640")
            require(t.train_loader.dataset.augment and not t.train_loader.dataset.rect, "Real online train augmentation missing")
            report["optimizer_coverage"] = optimizer_coverage(t.model, t.optimizer)
            report["native_scaler"] = type(t.scaler).__module__ + "." + type(t.scaler).__name__
            report["initial_scaler"] = t.scaler.state_dict()
            handles.append(t.optimizer.register_step_post_hook(lambda *_: report.__setitem__("effective_updates", report["effective_updates"] + 1)))
            handles.append(t.model.model[26].register_forward_hook(decoder_hook))
            handles.append(t.model.model[18].offset.register_forward_hook(offset_hook))

        def begin_batch(t):
            t._oom_retries = 3  # forbid automatic B16->B8 fallback
            t.gra_step = None
            t.gra_batch_start = time.perf_counter()

        def end_batch(t):
            require(bool(torch.isfinite(t.loss).all()), "Nonfinite native forward/loss")
            torch.cuda.synchronize(t.device)
            row = dict(t.gra_batch, index=len(report["batches"]), epoch=int(t.epoch),
                       loss=float(t.loss.detach()), loss_items=t.loss_items.detach().cpu().tolist(),
                       accumulate=int(t.accumulate), DN=t.gra_dn, decoder_shapes=t.gra_decoder_shapes,
                       optimizer_step=t.gra_step, scaler_scale=t.scaler.get_scale(), offset=t.gra_offset,
                       seconds=time.perf_counter() - t.gra_batch_start,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(t.device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(t.device))
            report["batches"].append(row)
            write_json(output / "progress.json", report)
            if report["effective_updates"] >= 2 or len(report["batches"]) >= args.max_batches:
                raise BudgetReached()

        trainer.add_callback("on_train_start", setup)
        trainer.add_callback("on_train_batch_start", begin_batch)
        trainer.add_callback("on_train_batch_end", end_batch)
        try:
            trainer.train()
        except BudgetReached:
            pass
        require(report["effective_updates"] >= 2, "FAILED: <=16-batch budget exhausted before 2 effective optimizer updates")
        offset = trainer.model.model[18].offset
        report["learned_offset_head"] = {n: {"max_abs": float(v.detach().abs().max()), "nonzero": int(torch.count_nonzero(v))}
                                         for n, v in offset.named_parameters()}
        require(any(v["nonzero"] for v in report["learned_offset_head"].values()), "Offset head did not learn")
        require(all(torch.isfinite(p).all() for p in trainer.model.parameters()), "Nonfinite learned model")
        report["final_scaler"] = trainer.scaler.state_dict()
        report["status"] = "PASSED"
    except BaseException as error:
        report["error"] = repr(error)
        if "PENDING:" in str(error):
            report["status"] = "PENDING"
        raise
    finally:
        for handle in handles:
            handle.remove()
        report["seconds"] = time.perf_counter() - started
        report["stopped_before_epoch_validation_and_save"] = True
        write_json(output / "preflight.json", report)
        print(json.dumps({"status": report["status"], "report": str(output / "preflight.json"),
                          "batches": len(report["batches"]), "effective_updates": report["effective_updates"]}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_gra_v1")
    p.add_argument("--initialized", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--data", type=Path, default=MAIN / "configs/crack_autodl.yaml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-batches", type=int, default=16)
    args = p.parse_args()
    os.chdir(ROOT)  # check_amp resolves its already verified local model from cwd
    preflight(args)


if __name__ == "__main__":
    main()
