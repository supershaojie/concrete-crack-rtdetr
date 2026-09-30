"""Native single-GPU training loop with explicit GPC batch context and scalar logs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.gpc import GPCConfig, GPCDetectionModel, finite
from ultralytics.utils.patches import torch_load


class PreflightStop(Exception):
    """Bounded exit before native end-of-epoch validation or final_eval."""


class GPCTrainer(RTDETRTrainer):
    def __init__(self, *args, algorithm=None, identity=None, preflight=False, **kwargs):
        self.algorithm = asdict(GPCConfig.from_dict(algorithm))
        self.gpc_identity = identity or {}
        self.gpc_preflight = preflight
        self.gpc_batch_index = 0
        self.gpc_rows = [] if preflight else None  # JSON scalars only; no graph/tensor cache
        self.gpc_updates = []
        self.gpc_deadline = time.monotonic() + 900 if preflight else None
        super().__init__(*args, **kwargs)
        if self.world_size > 1 or self.data["nc"] != 1 or self.args.compile:
            raise ValueError("GPC v1 supports one GPU, nc=1, compile=false")
        self.add_callback("on_train_epoch_start", self._gpc_epoch)
        self.add_callback("on_train_batch_start", self._gpc_batch_start)
        self.add_callback("on_train_batch_end", self._gpc_batch_end)
        self.add_callback("on_train_start", self._gpc_start)
        self.add_callback("on_fit_epoch_end", self._gpc_fit_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        audit = None
        if weights is not None and weights.model[-1].nc == 80:
            # Exactly the parent's audited public-source mapping and native nc80 -> nc1 adaptation.
            from init_c19_lif_v1 import build_training_model
            native, audit = build_training_model(cfg, weights, self.data)
            with torch.random.fork_rng(devices=[]):
                model = GPCDetectionModel(cfg, ch=self.data["channels"], nc=1, verbose=False, gpc_config=self.algorithm)
            model.load_state_dict(native.state_dict(), strict=True)
        else:
            model = GPCDetectionModel(cfg, ch=self.data["channels"], nc=1, verbose=verbose, gpc_config=self.algorithm)
            if weights is not None:
                saved = getattr(weights, "gpc_config", None)
                if self.resume and (saved != self.algorithm or getattr(weights, "gpc_identity", None) != self.gpc_identity):
                    raise ValueError("Resume checkpoint GPC configuration/identity mismatch")
                model.load_state_dict(weights.float().state_dict(), strict=True)
        model.gpc_identity = deepcopy(self.gpc_identity)
        if audit is not None:
            self._write_json(self.save_dir / "nc1_loading.json", audit)
        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        metrics = {}
        # Only this scalar dictionary is held until on_train_batch_end; features never leave loss().
        self.gpc_batch_metrics = metrics
        batch["gpc_context"] = dict(epoch=20 if self.gpc_preflight else int(self.epoch),
                                    batch_index=self.gpc_batch_index, seed=int(self.args.seed),
                                    image_ids=list(batch["im_file"]), metrics=metrics)
        return batch

    @staticmethod
    def _write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(path)

    @staticmethod
    def _gpc_start(trainer):
        if trainer.device.type == "cuda" and trainer.args.amp and not trainer.amp:
            raise RuntimeError("Native AMP check disabled AMP; formal recipe must remain unchanged")
        expected = trainer.gpc_identity.get("recipe")
        if expected:
            permitted = {"project", "name", "save_dir"} if trainer.gpc_preflight else set()
            if trainer.resume:
                permitted |= {"model", "resume"}
            actual = vars(trainer.args)
            differences = {k: [value, actual.get(k)] for k, value in expected.items() if
                           k not in permitted and (type(value) is not type(actual.get(k)) or value != actual.get(k))}
            if differences:
                raise RuntimeError("Actual native recipe differs: " + json.dumps(differences))
        trainer._write_json(trainer.save_dir / "gpc_metadata.json", dict(
            algorithm=trainer.algorithm, identity=trainer.gpc_identity,
            model_class=type(trainer.model).__module__ + "." + type(trainer.model).__name__,
            criterion_class="ultralytics.models.utils.loss.RTDETRDetectionLoss",
            actual_args=vars(trainer.args), start_epoch=trainer.start_epoch,
            parameters=sum(p.numel() for p in trainer.model.parameters()),
            optimizer=type(trainer.optimizer).__name__, amp=bool(trainer.amp)))

    @staticmethod
    def _gpc_fit_end(trainer):
        if trainer.stop and not trainer.gpc_preflight:
            trainer._write_json(trainer.save_dir / "training_finished.json", dict(
                epoch_zero_based=int(trainer.epoch), epochs_completed=int(trainer.epoch)+1,
                early_stopped=int(trainer.epoch)+1 < trainer.epochs,
                best_fitness=float(trainer.best_fitness), identity=trainer.gpc_identity,
                note="Training loop complete; native final_eval may still be running or fail separately"))

    @staticmethod
    def _gpc_epoch(trainer):
        trainer.gpc_batch_index = 0

    @staticmethod
    def _gpc_batch_start(trainer):
        # Native retry is checked inside its OOM handler; never reduce B16 to fit.
        trainer._oom_retries = 3
        if trainer.gpc_preflight and time.monotonic() >= trainer.gpc_deadline:
            raise PreflightStop("900 second budget reached")

    @staticmethod
    def _gpc_batch_end(trainer):
        # Prevent native for/else's OOM-recovery branch from mistaking the guard for a retry.
        trainer._oom_retries = 0
        row = dict(epoch=int(trainer.epoch), batch_index=trainer.gpc_batch_index,
                   gpc_epoch=20 if trainer.gpc_preflight else int(trainer.epoch),
                   **trainer.gpc_batch_metrics, total_loss=float(trainer.loss.detach()),
                   accumulate=int(trainer.accumulate), scale=float(trainer.scaler.get_scale()))
        with (trainer.save_dir / "gpc_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        trainer.gpc_batch_metrics = None
        trainer.gpc_batch_index += 1
        if trainer.gpc_preflight:
            trainer.gpc_rows.append(row)
            if len(trainer.gpc_rows) >= 16 or time.monotonic() >= trainer.gpc_deadline:
                raise PreflightStop("16 microbatch / 900 second budget reached")

    def optimizer_step(self):
        if not self.gpc_preflight:
            return super().optimizer_step()
        # Observe native optimizer/GradScaler/clip/EMA once; do not execute an extra step.
        def steps():
            return sum(float(s.get("step", 0)) for s in self.optimizer.state.values())
        before = steps()
        scale_before = float(self.scaler.get_scale())
        gradient_norms = {}
        gradients_finite = True
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                valid = bool(torch.isfinite(parameter.grad).all())
                gradients_finite &= valid
                if any(key in name for key in ("O_proj", "cbr.offset_out", "dec_bbox_head.2")):
                    gradient_norms[name] = float(parameter.grad.detach().double().norm()) / scale_before if valid else None
        anchor = self.model.model[-1].dec_bbox_head[-1].layers[-1].bias
        old = anchor.detach().clone()
        super().optimizer_step()
        for name, parameter in self.model.named_parameters():
            finite("updated parameter " + name, parameter)
        self.gpc_updates.append(dict(effective=steps() > before, scale_before=scale_before,
                                    scale_after=float(self.scaler.get_scale()), gradient_norms=gradient_norms,
                                    gradients_finite=gradients_finite,
                                    auxiliary_images=self.gpc_batch_metrics.get("auxiliary_images",0),
                                    final_regression_bias_max_change=float((anchor.detach() - old).abs().max()),
                                    ema_updates=self.ema.updates))

    def save_model(self):
        # Parent checkpoint already serializes the EMA model, optimizer, scaler, epoch and updates.
        self.ema.ema.gpc_config = deepcopy(self.algorithm)
        self.ema.ema.gpc_identity = deepcopy(self.gpc_identity)
        super().save_model()
        if self.best_fitness == self.fitness:
            self._write_json(self.save_dir / "best_selection.json", dict(
                epoch_zero_based=int(self.epoch), fitness=float(self.fitness),
                metrics=self.metrics, selector="native_training_val_mAP50_95", identity=self.gpc_identity))

    def resume_training(self, ckpt):
        if ckpt is not None and self.resume:
            saved_model = ckpt.get("ema") or ckpt.get("model")
            if ckpt.get("optimizer") is None or ckpt.get("scaler") is None or ckpt.get("epoch", -1) < 0:
                raise ValueError("Cannot resume a stripped/incomplete checkpoint")
            if ckpt["epoch"] + 1 >= self.epochs:
                raise ValueError("Training already completed; use val/test, not resume")
            if getattr(saved_model, "gpc_config", None) != self.algorithm or getattr(saved_model, "gpc_identity", None) != self.gpc_identity:
                raise ValueError("Checkpoint belongs to another GPC run/recipe")
        return super().resume_training(ckpt)
