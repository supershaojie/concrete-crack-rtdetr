"""Importable CEA trainer; native backward/accumulation/scaler/clip/EMA ordering."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.cea import CEAConfig, CEADetectionModel
from ultralytics.utils.torch_utils import strip_optimizer


class CEABudgetStop(Exception):
    """A bounded development run ended; this is not formal training completion."""


def strict_load(model, weights):
    """Native class adaptation, with an exact whitelist and strict final loading."""
    before, fresh = weights.float().state_dict(), model.state_dict()
    if set(before) != set(fresh):
        raise ValueError("CEA/model checkpoint parameter keys differ")
    prefix = f"model.{len(model.model) - 1}."
    allowed = {prefix + k for k in ("denoising_class_embed.weight", "enc_score_head.weight", "enc_score_head.bias")}
    allowed |= {prefix + f"dec_score_head.{i}.{s}" for i in range(3) for s in ("weight", "bias")}
    changed = {k for k in before if before[k].shape != fresh[k].shape}
    expected = allowed if weights.model[-1].nc == 80 and model.model[-1].nc == 1 else set()
    if changed != expected:
        raise ValueError(f"Unexpected checkpoint shape adaptation: {sorted(changed)}")
    model.load_state_dict({k: fresh[k] if k in changed else v for k, v in before.items()}, strict=True)
    return dict(loaded=len(before) - len(changed), class_adaptation=sorted(changed), missing=[], unexpected=[])


class CEATrainer(RTDETRTrainer):
    def __init__(self, *args, cea_config=None, cea_identity=None, cea_budget=None, **kwargs):
        self.cea_config = asdict(CEAConfig(**(cea_config or {})))
        self.cea_identity = cea_identity
        self.cea_budget = cea_budget
        self.cea_started = time.monotonic()
        self.cea_steps = []
        self.cea_batches = 0
        super().__init__(*args, **kwargs)
        if self.world_size > 1 or self.data["nc"] != 1 or self.args.compile:
            raise ValueError("CEA v1 supports single GPU, nc=1, compile=false only")
        self.add_callback("on_train_batch_start", self._cea_batch_start)
        self.add_callback("on_train_batch_end", self._cea_batch_end)
        self.add_callback("on_train_epoch_start", self._cea_epoch_start)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CEADetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"],
                                  verbose=verbose, cea_config=self.cea_config)
        if weights is not None:
            if getattr(self, "resume", False):
                if getattr(weights, "cea_config", None) != self.cea_config or getattr(weights, "cea_identity", None) != self.cea_identity:
                    raise ValueError("Resume CEA configuration/run identity mismatch")
            self.cea_loading = strict_load(model, weights)
        model.cea_identity = deepcopy(self.cea_identity)
        return model

    def _setup_train(self):
        super()._setup_train()
        if self.args.amp and not self.amp:
            raise RuntimeError("AMP check disabled AMP; stop instead of changing the formal recipe")

    def _cea_epoch_start(self, trainer):
        self.model.cea_epoch = 20 if self.cea_budget else int(self.epoch)
        self.model.cea_log_path = str(self.save_dir / "cea_batches.jsonl")
        if self.ema:
            self.ema.ema.cea_epoch = int(self.epoch)
            self.ema.ema.cea_identity = deepcopy(self.cea_identity)

    def _cea_batch_start(self, trainer):
        # Native OOM handler checks this counter before reducing batch. Set on every
        # batch because native epoch completion resets it; no shared GPU gate.
        self._oom_retries = 3
        if self.cea_budget and (self.cea_batches >= self.cea_budget["batches"] or
                               time.monotonic() - self.cea_started >= self.cea_budget["seconds"]):
            raise CEABudgetStop("bounded microbatch/time budget")

    def _cea_batch_end(self, trainer):
        # Undo the retry marker on successful batches so native for/else handling
        # never restarts a successful partial epoch. Next batch reinstates it.
        self._oom_retries = 0
        self.cea_batches += 1
        if not torch.isfinite(self.loss).all():
            raise FloatingPointError("Nonfinite training loss")
        if self.cea_budget and (self.cea_batches >= self.cea_budget["batches"] or
                               time.monotonic() - self.cea_started >= self.cea_budget["seconds"]):
            raise CEABudgetStop("bounded microbatch/time budget")

    def optimizer_step(self):
        if not self.cea_budget:
            return super().optimizer_step()
        # Preflight only: inspect gradients before native unscale/clip. Production
        # never calls autograd.grad or a second backward for diagnostics.
        scale_before = float(self.scaler.get_scale())
        gradients_finite = all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None)
        key = self.model.model[-1].cbr.offset_out.weight
        old = key.detach().clone()
        step_before = max((float(s["step"]) for s in self.optimizer.state.values() if "step" in s), default=0.)
        super().optimizer_step()
        step_after = max((float(s["step"]) for s in self.optimizer.state.values() if "step" in s), default=0.)
        finite = all(torch.isfinite(p).all() for p in self.model.parameters())
        row = dict(batch=self.cea_batches, scale_before=scale_before, scale_after=float(self.scaler.get_scale()),
                   optimizer_step_before=step_before, optimizer_step_after=step_after,
                   effective_update=step_after > step_before, skipped=step_after == step_before,
                   gradients_finite=bool(gradients_finite), parameters_finite=bool(finite),
                   offset_out_changed=not torch.equal(old, key),
                   accumulate=self.accumulate, lr=[g["lr"] for g in self.optimizer.param_groups])
        self.cea_steps.append(row)
        if not finite:
            raise FloatingPointError("Nonfinite parameters after optimizer step")

    def save_model(self):
        self.ema.ema.cea_config = deepcopy(self.cea_config)
        self.ema.ema.cea_identity = deepcopy(self.cea_identity)
        self.ema.ema.cea_epoch = int(self.epoch)
        self.ema.ema.cea_log_path = None
        super().save_model()
        row = dict(epoch_zero_based=int(self.epoch), best_fitness=float(self.best_fitness) if self.best_fitness is not None else None,
                   latest_fitness=float(self.fitness) if self.fitness is not None else None, selected_by="native training val mAP50-95",
                   cea_identity=self.cea_identity)
        if self.best_fitness is not None and self.best_fitness == self.fitness:
            (self.save_dir / "best_selection.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        (self.save_dir / "latest_epoch.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    def final_eval(self):
        # Native training val/best selection already ran. Independent FP32 val/test
        # are explicit commands. Preserve native stripped-checkpoint completion.
        for path in (self.last, self.best):
            if path.exists():
                strip_optimizer(path)
        row = dict(status="TRAINING_COMPLETE", epoch_zero_based=int(self.epoch),
                   epochs_completed=int(self.epoch) + 1, early_stopped=int(self.epoch) + 1 < self.epochs,
                   best_training_val_map50_95=float(self.best_fitness), best=str(self.best),
                   independent_fp32_test="PENDING", cea_identity=self.cea_identity)
        (self.save_dir / "training_complete.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False, indent=2))
