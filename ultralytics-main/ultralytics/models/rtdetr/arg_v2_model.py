"""Importable ARG model/trainer wrappers; native model construction and optimization."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math
import os
import time

import torch

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import unwrap_model
from .arg_v2_loss import ARGv2DetectionLoss, ARGGeometryError, FORMULA, ramp, aggregate_samples


def strict_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"value": None, "nonfinite": str(value)}
    if isinstance(value, dict):
        return {str(k): strict_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_value(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(strict_value(value), ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class ARGv2DetectionModel(RTDETRDetectionModel):
    def init_criterion(self):
        return ARGv2DetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        if not isinstance(getattr(self, "criterion", None), ARGv2DetectionLoss):
            self.criterion = self.init_criterion()
        self.criterion.epoch = getattr(self, "arg_epoch", 0)
        self.criterion.enabled = self.training
        self.criterion.sample = self.training and getattr(self, "arg_sample", False)
        try:
            return super().loss(batch, preds)
        except ARGGeometryError as error:
            error.details["image_files"] = list(batch.get("im_file", []))
            raise


def groups(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [{"names": [names[id(p)] for p in g["params"]],
             **{k: v for k, v in g.items() if k != "params"}} for g in optimizer.param_groups]


def gradient_report(model, scale=1.0):
    report = {}
    for name, param in model.named_parameters():
        if any(key in name for key in ("cbr.offset_out", "dec_bbox_head.2.layers.2", "model.20.O_proj")):
            grad = None if param.grad is None else param.grad.detach().float() / scale
            report[name] = dict(present=grad is not None, finite=bool(torch.isfinite(grad).all()) if grad is not None else None,
                                norm=float(grad.norm()) if grad is not None else None)
    return report


class ARGv2Trainer(RTDETRTrainer):
    """No alternative forward, optimizer, scheduler, EMA or scaler implementation."""
    def __init__(self, *args, **kwargs):
        self.arg_output = None
        self.arg_update_count = 0
        self.arg_epoch_samples = []
        self.arg_update_samples = []
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._arg_epoch_start)
        self.add_callback("on_train_batch_start", self._arg_batch_start)
        self.add_callback("on_train_batch_end", self._arg_batch_end)
        self.add_callback("on_fit_epoch_end", self._arg_epoch_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Construct the mother in an isolated RNG scope, then execute the SAME
        # native reconstruction at the original RNG position. No new head init.
        with torch.random.fork_rng(devices=[]):
            mother = super().get_model(deepcopy(cfg), weights, verbose=False)
        model = super().get_model(cfg, weights, verbose=verbose)
        assert set(mother.state_dict()) == set(model.state_dict())
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in mother.state_dict().items())
        assert sum(p.numel() for p in model.parameters()) == 20149765
        self.arg_rebuild = dict(equal_states=len(model.state_dict()), trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                native_reconstruction=True, formula=FORMULA)
        # Python-only specialization after native construction/loading. No modules,
        # parameters, buffers or RNG calls are added by this assignment.
        model.__class__ = ARGv2DetectionModel
        model.arg_epoch = 0
        return model

    def build_optimizer(self, model, *args, **kwargs):
        optimizer = super().build_optimizer(model, *args, **kwargs)
        self.arg_optimizer_groups = groups(model, optimizer)
        optimizer.register_step_post_hook(self._arg_step_done)
        return optimizer

    def _arg_step_done(self, optimizer, args, kwargs):
        self.arg_update_count += 1

    def _setup_train(self):
        import logging
        from ultralytics.utils import LOGGER
        messages = []
        class AMPLog(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                if "AMP:" in message:
                    messages.append(message)
        handler = AMPLog()
        LOGGER.addHandler(handler)
        try:
            super()._setup_train()
        finally:
            LOGGER.removeHandler(handler)
        if self.args.amp and not self.amp:
            raise RuntimeError("Native AMP check failed; ARG forbids silently disabling AMP")
        if self.args.amp and not any("checks passed" in m for m in messages):
            raise RuntimeError(f"Native AMP check did not explicitly pass: {messages}")
        self.arg_amp_evidence = messages
        if self.arg_output:
            write_json(Path(self.arg_output) / "actual_setup.json", dict(rebuild=self.arg_rebuild,
                optimizer_groups=self.arg_optimizer_groups, args=vars(self.args), start_epoch=self.start_epoch,
                scaler=self.scaler.state_dict(), ema_updates=self.ema.updates))

    @staticmethod
    def _arg_epoch_start(trainer):
        trainer.arg_epoch_samples, trainer.arg_update_samples = [], []
        trainer.arg_batch = 0
        trainer.arg_epoch_update_start = trainer.arg_update_count
        trainer.arg_epoch_started = time.monotonic()
        unwrap_model(trainer.model).arg_epoch = trainer.epoch

    @staticmethod
    def _arg_batch_start(trainer):
        # Preserve B16: the mother's first-epoch OOM batch-halving is disallowed.
        trainer._oom_retries = 3
        unwrap_model(trainer.model).arg_sample = trainer.arg_batch < 4

    @staticmethod
    def _arg_batch_end(trainer):
        criterion = getattr(unwrap_model(trainer.model), "criterion", None)
        if trainer.arg_batch < 4 and getattr(criterion, "last_diagnostics", None):
            trainer.arg_epoch_samples.append(dict(criterion.last_diagnostics))
        trainer.arg_batch += 1

    @staticmethod
    def _arg_epoch_end(trainer):
        if not trainer.arg_output or getattr(trainer, "arg_finalizing", False):
            return
        samples = trainer.arg_epoch_samples
        row = dict(epoch=trainer.epoch, ramp=ramp(trainer.epoch), sampled_batches=len(samples),
                   sampling="first four training batches only; scalar loss ratios are not gradient ratios",
                   aggregate=aggregate_samples(samples), samples=samples, updates=trainer.arg_update_samples,
                   protected_count=sum(s["protected_count"] for s in samples),
                   optimizer_updates_this_epoch=trainer.arg_update_count - trainer.arg_epoch_update_start,
                   optimizer_updates_this_process=trainer.arg_update_count, seconds=time.monotonic() - trainer.arg_epoch_started,
                   fitness=float(trainer.fitness), stop=bool(trainer.stop))
        folder = Path(trainer.arg_output)
        with (folder / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(strict_value(row), allow_nan=False) + "\n")
        write_json(folder / "progress.json", row)

    def optimizer_step(self):
        sample = len(self.arg_update_samples) < 2
        if sample:
            model = unwrap_model(self.model)
            parameter = model.model[-1].cbr.offset_out.weight
            before = parameter.detach().clone()
            scale, updates = self.scaler.get_scale(), self.arg_update_count
            gradients = gradient_report(model, scale)
        super().optimizer_step()
        if sample:
            self.arg_update_samples.append(dict(scale_before=scale, scale_after=self.scaler.get_scale(),
                skipped=self.arg_update_count == updates, updates=self.arg_update_count,
                cbr_parameter_changed=not torch.equal(before, parameter.detach()), gradients=gradients))

    def final_eval(self):
        # Called by native training ONLY after the loop's legal stopping condition.
        self.arg_finalizing = True
        folder = Path(self.arg_output) if self.arg_output else None
        if folder:
            write_json(folder / "training_completed.json", dict(status="TRAINING_COMPLETED", epoch=self.epoch,
                reason="epochs" if self.epoch + 1 >= self.epochs else "native_patience", best=str(self.best), last=str(self.last)))
        try:
            # Same native checkpoint stripping, before computing the immutable best hash.
            # A single FP32 evaluation now writes metrics, all queries/GT and val lock.
            from ultralytics.utils import RANK, LOCAL_RANK
            from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first
            from arg_v2_common import OUT, RUN, require
            from arg_v2_evaluation import evaluate
            require(folder and folder.resolve() == OUT.resolve() and self.best.resolve() == (RUN / "weights/best.pt").resolve(),
                    "Final evaluation must use this v2 experiment's output and best")
            with torch_distributed_zero_first(LOCAL_RANK):
                if RANK in {-1, 0}:
                    ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                    if self.best.exists():
                        strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
            report = evaluate("val", during_final_eval=True)
            self.metrics = {k: v for k, v in report["metrics"].items() if k != "fitness"}
            self.run_callbacks("on_fit_epoch_end")
        except BaseException as error:
            if folder:
                write_json(folder / "final_eval.json", dict(status="FINAL_EVAL_FAILED", error=repr(error)))
            raise
        if folder:
            write_json(folder / "final_eval.json", dict(status="PASS", metrics=self.metrics,
                val_lock=str(folder / "val_lock.json"), report=report["report"],
                checkpoint_sha256=report["checkpoint_sha256"], predictions=report["predictions"]))
