"""Actual RTDETR.train reconstruction and <=16-batch native B16/640 AMP audit."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import torch
from blc_common import (ROOT, KEYS, VARIANTS, BLCTrainer, require, paths, recipe, sha256,
                        torch_load, stamp, write_json, evidence_context)
from blc_lifecycle import groups, lifecycle
from ultralytics import RTDETR


class ProbeComplete(Exception):
    pass


class ReconstructionProbe(BLCTrainer):
    def train(self):
        # Model.train has already instantiated the real Trainer, checked data,
        # and called the actual get_model/load path. No batch is executed here.
        self.probe_audit = self.blc_rebuild_audit
        raise ProbeComplete()


def api_reconstruction(variant, folder):
    p = paths(variant)
    wrapper = RTDETR(str(p["init"]))
    args, _ = recipe(variant)
    args.update(project=str(folder), name="actual_train_api_rebuild", save_dir=str(folder/"actual_train_api_rebuild"))
    try:
        wrapper.train(trainer=ReconstructionProbe, **args)
    except ProbeComplete:
        pass
    else:
        raise RuntimeError("Reconstruction probe did not stop before training")
    result = wrapper.trainer.probe_audit
    before, after = torch_load(p["init"], map_location="cpu")["model"].state_dict(), wrapper.trainer.model.state_dict()
    require(all(torch.equal(v, after[k]) for k, v in before.items()), "Actual Model.train overwrote controlled nc1 init")
    require(wrapper.trainer.data["nc"] == 1, "Actual data nc mismatch")
    return dict(status="PASSED", actual_RTDERT_train_api=True, batch_count=0, all_states_exact=True, audit=result)


class BudgetComplete(Exception):
    pass


def avoid_oom_resize(trainer):
    trainer._oom_retries = 3


class BoundedTrainer(BLCTrainer):
    """Native loop/AMP/accumulation/optimizer; stop only at batch boundaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blc_batches, self.blc_steps, self.blc_gt = [], [], []
        self.blc_effective = 0
        self.blc_started = time.monotonic()
        self.add_callback("on_train_batch_start", avoid_oom_resize)
        self.add_callback("on_train_batch_end", budget_end)

    def preprocess_batch(self, batch):
        out = super().preprocess_batch(batch)
        require(tuple(out["img"].shape) == (16, 3, 640, 640), "Capacity batch/image changed")
        require(len(out["cls"]) > 0, "Capacity requires valid GT")
        self.blc_gt.append(len(out["cls"]))
        return out

    def optimizer_step(self):
        # Read scaled gradients; native super() alone unscales/clips/steps/updates EMA.
        before = {n: p.detach().clone() for n, p in self.model.named_parameters() if n in KEYS}
        scale_before = self.scaler.get_scale()
        norms = {n: float((p.grad.detach().float()/scale_before).norm()) if p.grad is not None else None
                 for n, p in self.model.named_parameters() if n in KEYS}
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters())
        if finite and torch.count_nonzero(before["model.5.blc.Wo.weight"]) == 0:
            require(norms["model.5.blc.Wo.weight"] is not None and norms["model.5.blc.Wo.weight"] > 0,
                    "First finite native AMP backward did not start Wo")
            require(all(value == 0 for name, value in norms.items() if "Wo." not in name),
                    "Upstream gradient should remain zero while Wo is zero")
        native_steps = [float(s["step"]) for s in self.optimizer.state.values() if "step" in s]
        previous_step = max(native_steps, default=0)
        super().optimizer_step()
        new_step = max([float(s["step"]) for s in self.optimizer.state.values() if "step" in s], default=0)
        delta = {n: float((p-before[n]).norm()) for n, p in self.model.named_parameters() if n in KEYS}
        applied = new_step > previous_step
        effective = applied and delta["model.5.blc.Wo.weight"] > 0
        self.blc_effective += int(effective)
        self.blc_steps.append(dict(batch=len(self.blc_batches)+1, scale_before=scale_before,
                                  scale_after=self.scaler.get_scale(), scaler_skipped=not applied,
                                  optimizer_state_step=new_step, effective_update=effective,
                                  all_gradients_finite=finite, new_gradient_norms=norms, parameter_delta=delta))


def budget_end(trainer):
    loss = float(trainer.loss)
    require(torch.isfinite(trainer.loss), "Nonfinite native loss")
    trainer.blc_batches.append(dict(batch=len(trainer.blc_batches)+1, loss=loss, gt=trainer.blc_gt[-1],
                                    accumulation=trainer.accumulate, amp=trainer.amp))
    upstream = any(s["effective_update"] and s["all_gradients_finite"] and
                   all(v is not None and v > 0 for v in s["new_gradient_norms"].values()) for s in trainer.blc_steps)
    if (trainer.blc_effective >= 2 and upstream) or len(trainer.blc_batches) >= 16:
        raise BudgetComplete()


def bounded_call(wrapper, args):
    try:
        wrapper.train(trainer=BoundedTrainer, **args)
    except BudgetComplete:
        pass
    else:
        raise RuntimeError("Bounded run reached unexpected normal completion")
    trainer = wrapper.trainer
    require(trainer.amp and trainer.batch_size == 16 and trainer.args.imgsz == 640, "Native AMP/capacity settings changed")
    require(len(trainer.blc_batches) <= 16 and trainer.blc_effective >= 2, "Budget exhausted before two effective updates")
    require(any(s["effective_update"] and s["all_gradients_finite"] and
                all(v is not None and v > 0 for v in s["new_gradient_norms"].values()) for s in trainer.blc_steps),
            "No evidence of effective upstream BLC gradients")
    return trainer


class OneBatchLoader:
    """One real val batch, retaining the native dataset/validator path."""
    def __init__(self, loader):
        self.loader, self.dataset = loader, loader.dataset
    def __len__(self):
        return 1
    def __iter__(self):
        yield next(iter(self.loader))


def capacity(variant, folder, report):
    p = paths(variant)
    initial_hash = sha256(p["init"])
    args, _ = recipe(variant)
    args.update(project=str(folder), name="bounded_start", save_dir=str(folder/"bounded_start"))
    torch.cuda.reset_peak_memory_stats()
    wrapper = RTDETR(str(p["init"]))
    try:
        trainer = bounded_call(wrapper, args)
    finally:
        t = getattr(wrapper, "trainer", None)
        if t is not None:
            report["capacity"] = dict(status="PENDING", batches=getattr(t, "blc_batches", []),
                                       steps=getattr(t, "blc_steps", []), effective_updates=getattr(t, "blc_effective", 0),
                                       batch=16, imgsz=640, AMP=getattr(t, "amp", None),
                                       peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                                       seconds=time.monotonic()-getattr(t, "blc_started", time.monotonic()))
    report["capacity"]["status"] = "PASSED"
    report["optimizer_groups"] = groups(trainer.model, trainer.optimizer)
    report["lifecycle"] = lifecycle(trainer.model, trainer.optimizer, trainer.ema, variant,
                                      vars(trainer.args), trainer.device, trainer.scaler)
    # Real epoch-validator invocation using its half EMA path, bounded to one
    # existing val batch; this is not independent/final val or test.
    old = trainer.validator.dataloader
    try:
        trainer.validator.dataloader = OneBatchLoader(old)
        metrics = trainer.validator(trainer=trainer)
        require(trainer.validator.args.half, "Native CUDA AMP epoch val did not use half EMA")
        require(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()), "Nonfinite epoch validation")
        report["native_half_ema_epoch_val"] = dict(status="PASSED", batches=1, metrics=metrics, half=True)
    finally:
        trainer.validator.dataloader = old
    # Save a real incomplete, bounded training checkpoint using native policy.
    # Its epoch=0 records the interrupted first epoch; resume starts epoch=1.
    # Diagnostic only: never a formal controlled initialization or final result.
    trainer.fitness = trainer.best_fitness = 0.
    trainer.save_model()
    checkpoint = torch_load(trainer.last, map_location="cpu")
    report["bounded_checkpoint"] = dict(path=str(trainer.last), sha256=sha256(trainer.last), epoch=checkpoint["epoch"],
                                        incomplete_epoch=True, native_resume_next_epoch=checkpoint["epoch"]+1,
                                        source="real bounded updates; diagnostic only")
    # Native resume shares this SAME 16-batch total budget. A resumed optimizer
    # may accumulate four batches, so reserve eight batches for that phase.
    remaining = 16 - len(trainer.blc_batches)
    if remaining <= 0:
        report["native_resume"] = dict(status="PENDING", reason="Total 16-batch budget exhausted; no extra retries")
        report["formal_init_untouched"] = sha256(p["init"]) == initial_hash
        return report
    resume_wrapper = RTDETR(str(trainer.last))
    resume_start = time.monotonic()
    # The native resume API restores its own recorded output path. Preserve the
    # first checkpoint/report before this diagnostic continuation writes any file.
    class ResumeBounded(BoundedTrainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.callbacks["on_train_batch_end"] = [resume_end]
        def _setup_train(self):
            super()._setup_train()
            require(self.start_epoch == checkpoint["epoch"] + 1, "Native resume epoch lost")
            require(self.ema.updates == checkpoint["updates"], "Native resume EMA updates lost")
            require(self.scaler.state_dict() == checkpoint["scaler"], "Native resume scaler lost")
            require(groups(self.model, self.optimizer) == report["optimizer_groups"], "Resume optimizer grouping differs")
            # Compare every moment/step against native dtype promotion.
            for file_group, live_group in zip(checkpoint["optimizer"]["param_groups"], self.optimizer.param_groups):
                for ident, param in zip(file_group["params"], live_group["params"]):
                    for key, value in checkpoint["optimizer"]["state"].get(ident, {}).items():
                        actual = self.optimizer.state[param][key]
                        require(torch.equal(actual.cpu(), value.cpu().to(actual.dtype)) if torch.is_tensor(value) else actual == value,
                                "Native resume optimizer state lost")
            before = checkpoint["ema"].float().state_dict()
            require(all(torch.equal(v, self.model.state_dict()[k].cpu()) for k, v in before.items()), "Native resume model state lost")
    def resume_end(t):
        require(torch.isfinite(t.loss), "Nonfinite resumed loss")
        t.blc_batches.append(dict(loss=float(t.loss), gt=t.blc_gt[-1]))
        if t.blc_effective >= 1 or len(t.blc_batches) >= remaining:
            raise BudgetComplete()
    try:
        resume_wrapper.train(trainer=ResumeBounded, resume=True)
    except BudgetComplete:
        resumed = resume_wrapper.trainer
        require(resumed.blc_effective >= 1, "No effective resumed update within total 16-batch budget")
        report["native_resume"] = dict(status="PASSED", epoch=resumed.start_epoch, batches=resumed.blc_batches,
                                        steps=resumed.blc_steps, effective_updates=resumed.blc_effective,
                                        seconds=time.monotonic()-resume_start)
    else:
        raise RuntimeError("Native bounded resume did not stop")
    require(sha256(p["init"]) == initial_hash, "Preflight changed formal controlled_init")
    report["formal_init_untouched"] = True
    return report
