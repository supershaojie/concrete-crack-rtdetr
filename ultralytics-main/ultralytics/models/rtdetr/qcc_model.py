"""Importable native RT-DETR specialization; QCC adds no inference tensors."""
from __future__ import annotations
from copy import deepcopy
import json
import logging
from pathlib import Path
import time

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model
from .qcc_loss import QCCDetectionLoss, FORMULA, ramp
from .qcc_io import write_json, strict_value, aggregate


class QCCDetectionModel(RTDETRDetectionModel):
    def init_criterion(self):
        return QCCDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        if not isinstance(getattr(self, "criterion", None), QCCDetectionLoss):
            self.criterion = self.init_criterion()
        self.criterion.epoch = getattr(self, "qcc_epoch", 0)
        self.criterion.enabled = self.training and getattr(self, "qcc_enabled", True)
        self.criterion.train(self.training)
        self.criterion.sample = self.training and getattr(self, "qcc_sample", False)
        return super().loss(batch, preds)


def groups(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [{"names": [names[id(p)] for p in group["params"]],
             **{k: v for k, v in group.items() if k != "params"}} for group in optimizer.param_groups]


def gradient_report(model, scale=1.0):
    report = {}
    for name, p in model.named_parameters():
        if any(s in name for s in ("cbr.offset_out", "dec_bbox_head.2.layers.2", "dec_score_head.2", "model.20.O_proj")):
            grad = None if p.grad is None else p.grad.detach().float() / scale
            report[name] = dict(present=grad is not None, finite=bool(torch.isfinite(grad).all()) if grad is not None else None,
                                norm=float(grad.norm()) if grad is not None else None)
    return report


class QCCTrainer(RTDETRTrainer):
    def __init__(self, *args, **kwargs):
        self.qcc_output = None
        self.qcc_update_count = self.qcc_attempts = self.qcc_skips = 0
        self.qcc_epoch_samples, self.qcc_update_samples = [], []
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._qcc_epoch_start)
        self.add_callback("on_train_batch_start", self._qcc_batch_start)
        self.add_callback("on_train_batch_end", self._qcc_batch_end)
        self.add_callback("on_fit_epoch_end", self._qcc_epoch_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.data["nc"] != 1:
            raise NotImplementedError("QCC v1 supports nc=1 only")
        # Ported native reconstruction audit from ef9cb7e ARG reliability tooling.
        # The audit fork leaves the production constructor at its original RNG state.
        with torch.random.fork_rng(devices=[]):
            mother = super().get_model(deepcopy(cfg), weights, verbose=False)
        model = super().get_model(cfg, weights, verbose=verbose)
        assert set(mother.state_dict()) == set(model.state_dict())
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in mother.state_dict().items())
        assert sum(p.numel() for p in model.parameters()) == 20149765
        self.qcc_rebuild = dict(equal_states=len(model.state_dict()), parameters=20149765,
                                native_reconstruction=True, formula=FORMULA)
        model.__class__ = QCCDetectionModel  # importable Python class; no module/tensor/RNG changes
        model.qcc_epoch, model.qcc_enabled = 0, True
        return model

    def build_optimizer(self, model, *args, **kwargs):
        optimizer = super().build_optimizer(model, *args, **kwargs)
        self.qcc_optimizer_groups = groups(model, optimizer)
        optimizer.register_step_post_hook(self._qcc_step_done)
        return optimizer

    def _qcc_step_done(self, optimizer, args, kwargs):
        self.qcc_update_count += 1

    def _setup_train(self):
        messages = []
        class AMPLog(logging.Handler):
            def emit(self, record):
                if "AMP:" in record.getMessage():
                    messages.append(record.getMessage())
        handler = AMPLog()
        LOGGER.addHandler(handler)
        try:
            super()._setup_train()
        finally:
            LOGGER.removeHandler(handler)
        if self.world_size > 1:
            raise NotImplementedError("QCC v1 delivery validates single GPU only; DDP is unverified")
        if self.args.amp and (not self.amp or not any("checks passed" in x for x in messages)):
            raise RuntimeError(f"Native AMP did not explicitly pass; recipe must remain unchanged: {messages}")
        self.qcc_amp_evidence = messages
        unwrap_model(self.model).qcc_epoch = self.start_epoch
        if self.qcc_output:
            write_json(Path(self.qcc_output) / f"setup_{time.time_ns()}.json", dict(rebuild=self.qcc_rebuild,
                optimizer_groups=self.qcc_optimizer_groups, args=vars(self.args), start_epoch=self.start_epoch,
                scaler=self.scaler.state_dict(), ema_updates=self.ema.updates, amp_evidence=messages))

    @staticmethod
    def _qcc_epoch_start(trainer):
        trainer.qcc_epoch_samples, trainer.qcc_update_samples = [], []
        trainer.qcc_batch = 0
        trainer.qcc_epoch_started = time.monotonic()
        unwrap_model(trainer.model).qcc_epoch = trainer.epoch

    @staticmethod
    def _qcc_batch_start(trainer):
        trainer._oom_retries = 3  # fail with B16, never silently halve the formal batch
        unwrap_model(trainer.model).qcc_sample = trainer.qcc_batch < 4

    @staticmethod
    def _qcc_batch_end(trainer):
        criterion = getattr(unwrap_model(trainer.model), "criterion", None)
        if trainer.qcc_batch < 4 and getattr(criterion, "last_diagnostics", None):
            trainer.qcc_epoch_samples.append(dict(criterion.last_diagnostics))
        trainer.qcc_batch += 1

    @staticmethod
    def _qcc_epoch_end(trainer):
        if not trainer.qcc_output or getattr(trainer, "qcc_finalizing", False):
            return
        row = dict(epoch=trainer.epoch, ramp=ramp(trainer.epoch), lambda_qcc=0.10,
            sampling="first 4 training micro-batches per epoch; scalar ratios are not gradient ratios",
            sampled_batches=len(trainer.qcc_epoch_samples), aggregate=aggregate(trainer.qcc_epoch_samples),
            samples=trainer.qcc_epoch_samples, updates=trainer.qcc_update_samples,
            optimizer_updates=trainer.qcc_update_count, optimizer_attempts=trainer.qcc_attempts, amp_skips=trainer.qcc_skips,
            seconds=time.monotonic() - trainer.qcc_epoch_started, fitness=float(trainer.fitness), stop=bool(trainer.stop))
        folder = Path(trainer.qcc_output)
        with (folder / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(strict_value(row), allow_nan=False) + "\n")
        write_json(folder / "progress.json", row)
        # This callback runs after the native stopper decision AND checkpoint
        # save. Record the legal end before the separate final_eval can fail.
        if trainer.stop:
            write_json(folder / "training_completed.json", dict(status="TRAINING_COMPLETED", epoch=trainer.epoch,
                reason="epochs" if trainer.epoch + 1 >= trainer.epochs else "native_patience",
                evidence="native stop decision after save_model, on_fit_epoch_end", best=str(trainer.best), last=str(trainer.last),
                optimizer_updates_this_process=trainer.qcc_update_count, completed_at=time.time()))

    def optimizer_step(self):
        sample = len(self.qcc_update_samples) < 2
        updates, scale = self.qcc_update_count, self.scaler.get_scale()
        if sample:
            model = unwrap_model(self.model)
            parameter = model.model[-1].cbr.offset_out.weight
            before = parameter.detach().clone()
            gradients = gradient_report(model, scale)
            finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        super().optimizer_step()
        self.qcc_attempts += 1
        self.qcc_skips += int(self.qcc_update_count == updates)
        if sample:
            self.qcc_update_samples.append(dict(scale_before=scale, scale_after=self.scaler.get_scale(),
                skipped=self.qcc_update_count == updates, updates=self.qcc_update_count,
                cbr_parameter_changed=not torch.equal(before, parameter.detach()), gradients=gradients, all_gradients_finite=finite))

    def final_eval(self):
        # Native calls this only after the legal training-loop stopping condition.
        # Keep native stripping, but perform its necessary final val through the
        # same export-and-lock operation as finish (no second inference).
        from ultralytics.utils.torch_utils import strip_optimizer
        from .qcc_io import sha256
        self.qcc_finalizing = True
        if not self.qcc_output:
            raise RuntimeError("QCC final_eval requires its experiment evidence directory")
        folder = Path(self.qcc_output)
        if not (folder / "training_completed.json").exists():
            write_json(folder / "training_completed.json", dict(status="TRAINING_COMPLETED", epoch=self.epoch,
                reason="epochs" if self.epoch + 1 >= self.epochs else "native_patience", best=str(self.best), last=str(self.last),
                optimizer_updates_this_process=self.qcc_update_count, completed_at=time.time()))
        try:
            # Record best selection before stripping epoch/optimizer from checkpoints.
            from ultralytics.utils.patches import torch_load
            selected = torch_load(self.best, map_location="cpu")
            write_json(folder / "best_selection.json", dict(epoch=selected["epoch"], fitness=selected["best_fitness"],
                rule="native val mAP50-95; save when best_fitness == fitness (latest tie)", pre_strip_sha256=sha256(self.best)))
            del selected
            ckpt = strip_optimizer(self.last) if self.last.exists() else {}
            strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
            del ckpt
            from tools.qcc_v1 import evaluate
            report = evaluate("val", in_worker=True)
            self.metrics = dict(report["metrics"])
            self.metrics.pop("fitness", None)
            write_json(folder / "final_eval.json", dict(status="PASS", val_lock=str(folder / "val_lock.json")))
            self.run_callbacks("on_fit_epoch_end")
        except BaseException as error:
            write_json(folder / "final_eval.json", dict(status="FINAL_EVAL_FAILED", error=repr(error)))
            raise
