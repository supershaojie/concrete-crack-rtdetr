"""Bounded native Trainer capacity check: unchanged 200-epoch recipe, at most 16 batches, no val/test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from tempfile import TemporaryDirectory

import torch

from dcc_common import ROOT, VARIANTS, require, sha256, write_json, runtime, verify_model, build_training_model, is_added
from train_dcc import (DEFAULT_MAIN, recipe, code_identity, dataset_identity, verify_initialization,
                       optimizer_coverage, disable_oom_retry)
from ultralytics import RTDETR
from dcc_checkpoint import DCCCheckpointTrainer
from dcc_acceptance import CONTRACT_VERSION
from ultralytics.utils import ASSETS


class BudgetComplete(Exception):
    """Normal finite-check stop, before native epoch validation/final_eval."""


def run(args):
    require(1 <= args.max_batches <= 16 and 2 <= args.target_updates <= args.max_batches,
            "Budget must be <=16 batches and target >=2 effective updates")
    out = args.output.resolve()
    require(not out.exists(), "Previous preflight output protected: " + str(out))
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="PENDING", variant=args.variant, runtime=runtime(), code_identity=code_identity(),
                  report_kind="native_capacity", contract_version=CONTRACT_VERSION,
                  formal_training="NOT_STARTED", final_test="NOT_RUN", full_validation="NOT_RUN",
                  init_sha256=sha256(args.init) if args.init.is_file() else None,
                  capacity=dict(batch=16, imgsz=640, AMP=True, observed_batches=0, effective_updates=0,
                                optimizer_calls=0, native_optimizer_steps=0), steps=[], batches=[])
    started = time.perf_counter()
    current = {}
    handles = []
    trainer_instance = []
    try:
        missing = []
        if not torch.cuda.is_available():
            missing.append("CUDA")
        if not args.init.is_file():
            missing.append("controlled initialization")
        data = args.data or args.main / "configs/crack_autodl.yaml"
        if not data.is_file():
            missing.append("real dataset configuration")
        # Reuse only pre-existing trusted AMP resources. Missing assets are a
        # dependency, never a reason to download/upgrade or disable native AMP.
        for destination, candidates in (
            (ASSETS / "bus.jpg", [args.main / "ultralytics-main/ultralytics/assets/bus.jpg", args.main / "bus.jpg"]),
            (ROOT / "yolo26n.pt", [args.main / "yolo26n.pt", args.main / "weights/yolo26n.pt"]),
        ):
            if not destination.is_file():
                existing = next((p for p in candidates if p.is_file()), None)
                if existing:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as stream, existing.open("rb") as source:
                        shutil.copyfileobj(source, stream)
                else:
                    missing.append("existing native AMP resource " + destination.name)
            if destination.is_file():
                report.setdefault("amp_resources", []).append(dict(path=str(destination), sha256=sha256(destination)))
        if missing:
            report["pending"] = missing
            return report
        verify_initialization(args.init, args.variant)
        report["dataset_identity"] = dataset_identity(data)
        train_args, differences = recipe(args.variant, args.init, args.main, data, args.parent_args)
        # Only output identity is temporary. Epoch count, optimizer, warmup, accumulate,
        # augmentation, AMP, workers, B16 and 640 are exactly the formal recipe.
        train_args.update(project=str(out), name="native_capacity", save_dir=str(out / "native_capacity"))
        report["recipe"] = train_args
        report["recipe_differences"] = differences
        require(args.workers == train_args["workers"], "Capacity workers must match the full recipe (8)")
        torch.cuda.reset_peak_memory_stats()

        class CapacityTrainer(DCCCheckpointTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model, audit = build_training_model(cfg, weights, self.data, args.variant)
                report["trainer_rebuild"] = audit
                return model

            def preprocess_batch(self, batch):
                batch = super().preprocess_batch(batch)
                current.clear()
                current.update(batch=report["capacity"]["observed_batches"],
                               shape=list(batch["img"].shape), GT=int(batch["cls"].shape[0]),
                               files=list(map(str, batch.get("im_file", []))))
                require(current["shape"] == [16, 3, 640, 640] and current["GT"] > 0,
                        "Capacity batch must be real B16/640 with valid GT")
                return batch

            def optimizer_step(self):
                branch = {n: p for n, p in self.model.named_parameters() if is_added(n)}
                before = {n: p.detach().clone() for n, p in branch.items()}
                scale = float(self.scaler.get_scale())
                gradients = {}
                for n, p in branch.items():
                    finite = p.grad is not None and bool(torch.isfinite(p.grad).all())
                    # Scaler Inf skips are valid evidence; JSON must preserve them
                    # instead of crashing while serializing Inf/NaN before recovery.
                    gradients[n] = dict(finite=finite,
                        norm=float((p.grad.detach().double() / scale).norm()) if finite else None,
                        nonfinite_count=int((~torch.isfinite(p.grad)).sum()) if p.grad is not None else None)
                steps_before = sum(float(v.get("step", 0)) for v in self.optimizer.state.values())
                # Native clipping, scaler.step/update, zero_grad and EMA are untouched.
                super().optimizer_step()
                steps_after = sum(float(v.get("step", 0)) for v in self.optimizer.state.values())
                applied = steps_after > steps_before
                changed = {n: bool(not torch.equal(before[n], p.detach())) for n, p in branch.items()}
                effective = applied and any(changed.values())
                report["capacity"]["optimizer_calls"] += 1
                report["capacity"]["native_optimizer_steps"] += int(applied)
                report["capacity"]["effective_updates"] += int(effective)
                report["steps"].append(dict(batch=current.get("batch"), applied=applied, effective=effective,
                    skipped=not applied, scale_before=scale, scale_after=float(self.scaler.get_scale()),
                    gradients=gradients, changed=changed, accumulate=self.accumulate,
                    learning_rates=[float(g["lr"]) for g in self.optimizer.param_groups],
                    new_norms={n: float(p.detach().float().norm()) for n, p in branch.items()}))

        model = RTDETR(str(args.init))
        model.add_callback("on_train_batch_start", disable_oom_retry)

        def setup(trainer):
            trainer_instance.append(trainer)
            verify_model(trainer.model, args.variant, zero=True)
            require(trainer.amp is True and trainer.scaler.is_enabled(), "Native check disabled AMP; capacity invalid")
            require(type(trainer.optimizer) is torch.optim.AdamW, "Expected original AdamW")
            report["optimizer_coverage"] = optimizer_coverage(trainer.model, trainer.optimizer)
            report["actual_recipe"] = vars(trainer.args).copy()
            require(all(vars(trainer.args).get(k) == v for k, v in train_args.items()), "Effective recipe changed")
            require(trainer.train_loader.dataset.augment and not trainer.train_loader.dataset.rect,
                    "Native real online augmentation absent")
            report["augmentation"] = repr(trainer.train_loader.dataset.transforms)

            def loss_finite(module, inputs, output):
                if isinstance(output, tuple) and torch.is_tensor(output[0]):
                    require(bool(torch.isfinite(output[0]).all()), "Nonfinite native forward detection loss")
            handles.append(trainer.model.register_forward_hook(loss_finite))

            def dn_record(module, inputs, output):
                if isinstance(output, tuple) and len(output) >= 5:
                    metadata = output[-1]
                    current["DN"] = {k: v for k, v in (metadata or {}).items()
                                     if isinstance(v, (str, int, float, list, tuple, type(None))) and k != "dn_pos_idx"}
            handles.append(trainer.model.model[-1].register_forward_hook(dn_record))

        def batch_end(trainer):
            require(bool(torch.isfinite(trainer.loss)), "Nonfinite native loss")
            current.update(loss=float(trainer.loss.detach()), loss_items=trainer.loss_items.detach().float().cpu().tolist(),
                           accumulate=trainer.accumulate, scaler_scale=float(trainer.scaler.get_scale()))
            report["batches"].append(dict(current))
            report["capacity"]["observed_batches"] += 1
            write_json(out / "preflight.json", report)
            if report["capacity"]["effective_updates"] >= args.target_updates or report["capacity"]["observed_batches"] >= args.max_batches:
                raise BudgetComplete()

        model.add_callback("on_train_start", setup)
        model.add_callback("on_train_batch_end", batch_end)
        try:
            model.train(trainer=CapacityTrainer, **train_args)
            raise RuntimeError("Native training escaped the finite batch budget")
        except BudgetComplete:
            pass
        trainer = trainer_instance[0]
        require(report["capacity"]["effective_updates"] >= args.target_updates,
                "Native GradScaler did not achieve the required effective DCC updates within 16 batches")
        branch = {n: p for n, p in trainer.model.named_parameters() if is_added(n)}
        require(all(torch.isfinite(p).all() for p in branch.values()), "Nonfinite learned DCC parameter")
        require(all(any(step["applied"] and step["gradients"][n]["finite"] and
                        (step["gradients"][n]["norm"] or 0) > 0 for step in report["steps"]) for n in branch),
                "Not all DCC projections received later finite nonzero gradients")
        # Exercise diagnostic serialization, retaining only its hash and size.
        # A mid-epoch budget stop is never a resumable formal checkpoint.
        # DCC_PREFLIGHT_DETACH_BEFORE_SAVE_V1
        for handle in handles:
            handle.remove()
        handles.clear()
        with TemporaryDirectory(prefix="diagnostic_", dir=out) as temporary:
            diagnostic = Path(temporary) / "learned_diagnostic.pt"
            torch.save({"model": deepcopy(trainer.model).cpu(), "optimizer": trainer.optimizer.state_dict(),
                        "scaler": trainer.scaler.state_dict(), "ema": deepcopy(trainer.ema.ema).cpu(),
                        "epoch": trainer.epoch, "batches": report["capacity"]["observed_batches"],
                        "kind": "bounded_mid_epoch_diagnostic_NOT_FORMAL_RESUME"}, diagnostic)
            report["learned_diagnostic_sha256"] = sha256(diagnostic)
            report["learned_diagnostic_bytes"] = diagnostic.stat().st_size
        report["learned_diagnostic_retained"] = False
        report["checkpoint_policy"] = "optimizer_fp32_v1"
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        for handle in handles:
            handle.remove()
        report["elapsed_seconds"] = time.perf_counter() - started
        if torch.cuda.is_available():
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["gpu"] = torch.cuda.get_device_name()
        report["capacity"]["status"] = report["status"]
        write_json(out / "preflight.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, default="cbr_lif_dcc_v1")
    p.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    p.add_argument("--data", type=Path)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--parent-args", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-batches", type=int, default=16)
    p.add_argument("--target-updates", type=int, default=2)
    p.add_argument("--workers", type=int, default=8, help="Must equal original recipe; no capacity fallback")
    args = p.parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in ("status", "capacity", "formal_training", "final_test")}, indent=2))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
