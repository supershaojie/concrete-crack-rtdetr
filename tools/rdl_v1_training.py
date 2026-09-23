"""RDL v1 真实 Trainer：原类型/初始化/优化器，按真实 epoch 显式启用附加损失。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch
from init_c19_lif_v1 import build_training_model, require, verify_model
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.rdl import CONFIG, RDLDetectionLoss, ramp
from ultralytics.utils.torch_utils import unwrap_model


def configure(model, completed_epochs=0):
    """Retain the audited native model instance; only add ordinary Python metadata."""
    verify_model(model)
    old = getattr(model, "rdl_config", CONFIG)
    require(old == CONFIG, "Checkpoint RDL version/config mismatch")
    keys = tuple(model.state_dict())
    names = tuple(n for n, _ in model.named_parameters())
    model.rdl_config = dict(CONFIG)
    model.rdl_epoch = int(completed_epochs)
    ramp(model.rdl_epoch)
    if hasattr(model, "criterion"):
        # A native criterion may have been created by an audit; it has no learned state.
        require(not tuple(model.criterion.parameters()) and not tuple(model.criterion.buffers()), "Stateful criterion")
        if not isinstance(model.criterion, RDLDetectionLoss):
            model.criterion = model.init_criterion()
        model.criterion.set_epoch(model.rdl_epoch)
    require(tuple(model.state_dict()) == keys and tuple(n for n, _ in model.named_parameters()) == names,
            "RDL must add zero state/parameters")
    return model


def epoch_start(trainer):
    model = unwrap_model(trainer.model)
    require(model.rdl_config == CONFIG, "RDL was lost during Trainer reconstruction")
    model.rdl_epoch = int(trainer.epoch)
    if hasattr(model, "criterion"):
        require(isinstance(model.criterion, RDLDetectionLoss), "Wrong actual criterion")
        model.criterion.set_epoch(model.rdl_epoch)
    if trainer.ema:
        trainer.ema.ema.rdl_epoch = model.rdl_epoch
        trainer.ema.ema.rdl_config = dict(CONFIG)
    trainer.rdl_totals = dict(batches=0, active_batches=0, scaler_overflow_steps=0)


def batch_start(trainer):
    trainer._oom_retries = 3  # Original Trainer otherwise silently halves B16 after OOM.


def batch_end(trainer):
    if not torch.isfinite(trainer.loss).all():
        raise FloatingPointError("RDL run: nonfinite total loss; automatic recovery is disabled")
    stats = unwrap_model(trainer.model).criterion.last_stats
    totals = trainer.rdl_totals
    totals["batches"] += 1
    totals["active_batches"] += int(stats["active"])
    for key in ("L_out", "L_in", "weighted", "M", "a_protected", "e_protected", "outside_sides", "saturated_targets"):
        totals[key] = totals.get(key, 0) + stats.get(key, 0)


def epoch_end(trainer):
    totals = dict(trainer.rdl_totals)
    totals.update(e=int(trainer.epoch), weight=.05 * ramp(int(trainer.epoch)), validation_loss="L0 only")
    totals["reduction"] = "sums over batches; divide loss sums by active_batches for batch means"
    path = Path(trainer.save_dir) / "rdl_epochs.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(totals, allow_nan=False) + "\n")


class RDLTrainer(RTDETRTrainer):
    """Importable Trainer; native model type also keeps original evaluation audits valid."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", epoch_start)
        self.add_callback("on_train_batch_start", batch_start)
        self.add_callback("on_train_batch_end", batch_end)
        self.add_callback("on_train_epoch_end", epoch_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.resume:
            require(weights is not None and getattr(weights, "rdl_config", None) == CONFIG,
                    "Resume requires this RDL checkpoint, not a baseline/other experiment")
            model = super().get_model(cfg, weights, verbose)
            verify_model(model, zero=False)
            before, after = weights.float().state_dict(), model.state_dict()
            require(before.keys() == after.keys() and all(torch.equal(v.to(after[k].device), after[k]) for k, v in before.items()),
                    "Resume Trainer reconstruction changed weights")
            audit = dict(resume=True, state_exact=True, added_parameters=0)
        else:
            require(weights is not None, "RDL formal training requires the verified controlled init")
            model, audit = build_training_model(cfg, weights, self.data)
        self.rdl_loading_audit = audit
        return configure(model, getattr(weights, "rdl_epoch", 0))

    def resume_training(self, ckpt):
        if self.resume:
            require(ckpt is not None and 0 <= ckpt.get("epoch", -1) < self.epochs - 1,
                    "Cannot resume stripped/completed/wrong-epoch checkpoint")
            saved = ckpt.get("ema") if ckpt.get("ema") is not None else ckpt.get("model")
            require(getattr(saved, "rdl_config", None) == CONFIG, "Missing checkpoint RDL version")
            require(getattr(saved, "rdl_epoch", None) == ckpt["epoch"], "Checkpoint epoch metadata differs")
        super().resume_training(ckpt)
        if self.resume:
            self.model.rdl_epoch = int(self.start_epoch)
            if hasattr(self.model, "criterion"):
                self.model.criterion.set_epoch(self.start_epoch)
            if self.ema:
                configure(self.ema.ema, self.start_epoch)

    def optimizer_step(self):
        # Native GradScaler may skip overflowing scaled updates and lower its scale.
        # Record this explicitly, but preserve the mother's exact optimizer/scaler steps.
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        require(grads, "Missing training gradients")
        if not torch.stack([g.isfinite().all() for g in grads]).all():
            if not self.amp:
                raise FloatingPointError("Nonfinite FP32 gradients")
            self.rdl_totals['scaler_overflow_steps'] += 1
            print(f"RDL native AMP scaled-gradient overflow: e={self.epoch}, scale={self.scaler.get_scale()}; native scaler handles this step", flush=True)
        return super().optimizer_step()

    def _handle_nan_recovery(self, epoch):
        if self.loss is not None and not torch.isfinite(self.loss).all():
            raise FloatingPointError("Nonfinite loss; preserve failure, no automatic checkpoint recovery")
        if self.fitness is not None and not torch.isfinite(torch.as_tensor(self.fitness)):
            raise FloatingPointError("Nonfinite validation fitness")
        return False
