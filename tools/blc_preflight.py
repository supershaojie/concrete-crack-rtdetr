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
from blc_probe_io import temporary_probe, retained_size
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
        self.blc_seen, self.blc_limit = 0, 16
        self.blc_started = time.monotonic()
        self.add_callback("on_train_batch_start", budget_start)
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


def budget_start(trainer):
    avoid_oom_resize(trainer)
    require(trainer.blc_seen < trainer.blc_limit, "Training batch budget exceeded")
    trainer.blc_seen += 1


def budget_end(trainer):
    loss = float(trainer.loss)
    trainer.blc_batches.append(dict(batch=len(trainer.blc_batches)+1, loss=loss, gt=trainer.blc_gt[-1],
                                    accumulation=trainer.accumulate, amp=trainer.amp))
    require(torch.isfinite(trainer.loss), "Nonfinite native loss")
    upstream = any(s["effective_update"] and s["all_gradients_finite"] and
                   all(v is not None and v > 0 for v in s["new_gradient_norms"].values()) for s in trainer.blc_steps)
    if (trainer.blc_effective >= 2 and upstream) or trainer.blc_seen >= trainer.blc_limit:
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
        self.batches = 0
    def __len__(self):
        return 1
    def __iter__(self):
        require(self.batches == 0, "One-batch validator was iterated twice")
        batch = next(iter(self.loader))
        self.batches += 1
        yield batch


def capacity(variant, folder, report):
    """One temporary start/resume run; preserve small diagnostics even on failure."""
    p = paths(variant)
    initial_hash = sha256(p["init"])
    active = []
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    report.update(stage="capacity", batch_budget=16, validation_batch_budget=1,
                  capacity=dict(status="NOT_RUN"), lifecycle=dict(status="NOT_RUN"),
                  native_half_ema_epoch_val=dict(status="NOT_RUN"), native_resume=dict(status="NOT_RUN"),
                  probe_output_overrides=dict(plots=False))
    try:
        with temporary_probe(folder, "bounded-start-resume", report) as temporary:
            try:
                return _capacity(variant, temporary, report, active)
            finally:
                # The native loop was interrupted before teardown; release only
                # these probe loaders, including workers holding temporary files.
                for wrapper in active:
                    trainer = getattr(wrapper, "trainer", None)
                    for name in ("train_loader", "test_loader"):
                        iterator = getattr(getattr(trainer, name, None), "iterator", None)
                        shutdown = getattr(iterator, "_shutdown_workers", None)
                        if shutdown:
                            shutdown()
    except BaseException as error:
        report.update(exception_stage=report["stage"], probe_error=repr(error))
        for key in ("capacity", "lifecycle", "native_half_ema_epoch_val", "native_resume"):
            if report[key]["status"] == "RUNNING":
                report[key].update(status="FAILED", error=repr(error))
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[:2]
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[2:4]
        torch.use_deterministic_algorithms(flags[4], warn_only=flags[5])
        report["formal_init_untouched"] = sha256(p["init"]) == initial_hash
        report["total_attempted_training_batches"] = sum(getattr(getattr(w, "trainer", None), "blc_seen", 0) for w in active)
        report["retained_bytes_before_report"] = retained_size(folder)
        report["retention_target_bytes"] = 10 * 1024 * 1024
        require(report["formal_init_untouched"], "Preflight changed formal controlled_init")


def _capacity(variant, folder, report, active):
    p = paths(variant)
    args, _ = recipe(variant)
    args.update(project=str(folder), name="bounded_start", save_dir=str(folder/"bounded_start"), plots=False)
    torch.cuda.reset_peak_memory_stats()
    wrapper = RTDETR(str(p["init"]))
    active.append(wrapper)
    report["capacity"]["status"] = "RUNNING"
    try:
        trainer = bounded_call(wrapper, args)
    finally:
        t = getattr(wrapper, "trainer", None)
        if t is not None:
            report["capacity"].update(batches=getattr(t, "blc_batches", []), attempted_batches=getattr(t, "blc_seen", 0),
                                      steps=getattr(t, "blc_steps", []), effective_updates=getattr(t, "blc_effective", 0),
                                      batch=16, imgsz=640, AMP=getattr(t, "amp", None),
                                      peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                                      seconds=time.monotonic()-getattr(t, "blc_started", time.monotonic()))
    report["capacity"]["status"] = "PASSED"
    report["optimizer_groups"] = groups(trainer.model, trainer.optimizer)
    # Save once using native policy. Lifecycle and real resume consume this same
    # incomplete-epoch checkpoint; all native last/best files are temporary.
    report["stage"] = "native_save"
    trainer.fitness = trainer.best_fitness = 0.
    trainer.save_model()
    checkpoint = torch_load(trainer.last, map_location="cpu")
    report["bounded_checkpoint"] = dict(path=str(trainer.last), sha256=sha256(trainer.last), epoch=checkpoint["epoch"],
                                        incomplete_epoch=True, native_resume_next_epoch=checkpoint["epoch"]+1,
                                        source="real bounded updates; temporary input to lifecycle and native resume")
    report["stage"] = "lifecycle"
    lifecycle(trainer.model, trainer.optimizer, trainer.ema, variant, vars(trainer.args), trainer.device,
              trainer.scaler, checkpoint_path=trainer.last, report=report["lifecycle"])
    report["stage"] = "native_half_ema_epoch_val"
    report["native_half_ema_epoch_val"] = dict(status="RUNNING", batches=0)
    old = trainer.validator.dataloader
    ema_dtype = next(trainer.ema.ema.parameters()).dtype
    try:
        one_batch = OneBatchLoader(old)
        trainer.validator.dataloader = one_batch
        metrics = trainer.validator(trainer=trainer)
        report["native_half_ema_epoch_val"].update(batches=one_batch.batches, metrics=metrics,
                                                   half=trainer.validator.args.half)
        require(one_batch.batches == 1, "Native validation must consume exactly one real batch")
        require(trainer.validator.args.half, "Native CUDA AMP epoch val did not use half EMA")
        require(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()), "Nonfinite epoch validation")
        report["native_half_ema_epoch_val"]["status"] = "PASSED"
    finally:
        report["native_half_ema_epoch_val"]["batches"] = getattr(trainer.validator.dataloader, "batches", 0)
        trainer.validator.dataloader = old
        trainer.ema.ema.to(dtype=ema_dtype)
    remaining = 16 - trainer.blc_seen
    report["stage"] = "native_resume"
    if remaining <= 0:
        report["native_resume"] = dict(status="PENDING", reason="Total 16-batch budget exhausted; no extra retries")
        return report
    report["native_resume"] = dict(status="RUNNING", batches=[], steps=[], effective_updates=0)
    resume_wrapper = RTDETR(str(trainer.last))
    active.append(resume_wrapper)
    resume_start = time.monotonic()
    class ResumeBounded(BoundedTrainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.blc_limit = remaining
            self.callbacks["on_train_batch_end"] = [resume_end]
        def _setup_train(self):
            super()._setup_train()
            require(self.start_epoch == checkpoint["epoch"] + 1, "Native resume epoch lost")
            require(self.ema.updates == checkpoint["updates"], "Native resume EMA updates lost")
            require(self.scaler.state_dict() == checkpoint["scaler"], "Native resume scaler lost")
            require(groups(self.model, self.optimizer) == report["optimizer_groups"], "Resume optimizer grouping differs")
            for file_group, live_group in zip(checkpoint["optimizer"]["param_groups"], self.optimizer.param_groups):
                for ident, param in zip(file_group["params"], live_group["params"]):
                    for key, value in checkpoint["optimizer"]["state"].get(ident, {}).items():
                        actual = self.optimizer.state[param][key]
                        require(torch.equal(actual.cpu(), value.cpu().to(actual.dtype)) if torch.is_tensor(value) else actual == value,
                                "Native resume optimizer state lost")
                        if torch.is_tensor(value) and value.numel():
                            require(actual.data_ptr() != value.data_ptr(), "Resume optimizer shares file storage")
            before = checkpoint["ema"].float().state_dict()
            for restored in (self.model, self.ema.ema):
                require(all(torch.equal(v, restored.state_dict()[k].cpu()) for k, v in before.items()),
                        "Native resume model/EMA state lost")
            report["native_resume"]["restored_state"] = dict(epoch=True, optimizer_moments_steps=True,
                                                               scaler=True, ema=True, updates=True, shared_storage=False)
    def resume_end(t):
        t.blc_batches.append(dict(loss=float(t.loss), gt=t.blc_gt[-1]))
        require(torch.isfinite(t.loss), "Nonfinite resumed loss")
        if t.blc_effective >= 1 or t.blc_seen >= remaining:
            raise BudgetComplete()
    try:
        try:
            resume_wrapper.train(trainer=ResumeBounded, resume=True)
        except BudgetComplete:
            resumed = resume_wrapper.trainer
            require(resumed.blc_effective >= 1, "No effective resumed update within total 16-batch budget")
            report["native_resume"]["status"] = "PASSED"
        else:
            raise RuntimeError("Native bounded resume did not stop")
    finally:
        resumed = getattr(resume_wrapper, "trainer", None)
        if resumed is not None:
            report["native_resume"].update(epoch=resumed.start_epoch, batches=resumed.blc_batches, steps=resumed.blc_steps,
                                            attempted_batches=resumed.blc_seen, effective_updates=resumed.blc_effective,
                                            seconds=time.monotonic()-resume_start)
        report["total_attempted_training_batches"] = trainer.blc_seen + getattr(resumed, "blc_seen", 0)
        require(report["total_attempted_training_batches"] <= 16, "Combined training batch budget exceeded")
    report["stage"] = "complete"
    return report
