"""PDS-specific native Trainer extension: audits, diagnostics and complete resume."""
from __future__ import annotations
from copy import deepcopy
import io
import json
import os
import random
import time
import numpy as np
import torch

from pds_v1_common import (CONFIG, OUT, ROOT, build_training_model, require, verify_model,
                           native_view, state_audit, write_json, environment, git, utc)
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.pds import PDSDetectionModel
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def gradient_report(model, scale=1.):
    groups = {}
    for label, prefix in (("public", "model."), ("head", "pds_head.")):
        params = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        norm_sq = 0.
        nonfinite = []
        for name, p in params:
            g = p.grad.detach().float() / scale
            if not torch.isfinite(g).all():
                nonfinite.append(name)
            norm_sq += float(g.square().sum())
        groups[label] = dict(tensors=len(params), finite=not nonfinite,
                             nonfinite_names=nonfinite, norm=norm_sq ** .5)
    groups["global_norm"] = (groups["public"]["norm"] ** 2 + groups["head"]["norm"] ** 2) ** .5
    groups["clip_max_norm"] = 10.
    return groups


def validate_resume(ckpt):
    required = ("model", "ema", "optimizer", "scaler", "scheduler", "stopper", "rng", "pds_resume")
    require(all(ckpt.get(k) is not None for k in required), "Incomplete checkpoint/deploy is not resumable")
    meta = ckpt["pds_resume"]
    require(meta["config"] == CONFIG and meta["epoch"] == ckpt["epoch"], "PDS definition/epoch mismatch")
    require(meta["epoch_boundary"] and not meta["completed"] and 0 <= ckpt["epoch"] < 199,
            "Only unfinished PDS epoch-boundary checkpoints can resume")
    for key in ("model", "ema"):
        model = ckpt[key]
        require(type(model) is PDSDetectionModel, f"Missing PDS wrapper/head in {key}")
        require(len(model.pds_head.state_dict()) == 11, "Incomplete auxiliary head")
        verify_model(native_view(model))
        for name, value in model.state_dict().items():
            require(torch.isfinite(value).all(), f"Nonfinite checkpoint {key}.{name}")
    return meta


class PDSTrainer(RTDETRTrainer):
    def __init__(self, *args, **kwargs):
        self.pds_effective_steps = 0
        self.pds_overflow_skips = 0
        self.pds_step_diagnostics = []
        self.pds_dispatch = os.environ.get("PDS_DISPATCH")
        self.pds_events = OUT / (f"events_{self.pds_dispatch}.jsonl" if self.pds_dispatch else "diagnostic_events.jsonl")
        self._pds_batch_in_epoch = 0
        self._pds_started = False
        super().__init__(*args, **kwargs)
        require(not self.args.compile and self.world_size <= 1 and not self.ddp
                and int(os.environ.get("WORLD_SIZE", "1")) == 1,
                "PDS v1 supports single device, compile=false only")
        self.add_callback("on_train_epoch_start", self._epoch_start)
        self.add_callback("on_train_batch_start", self._batch_start)
        self.add_callback("on_train_batch_end", self._batch_end)
        self.add_callback("on_fit_epoch_end", self._epoch_end)

    def event(self, phase, **kwargs):
        row = dict(time=utc(), dispatch=self.pds_dispatch, pid=os.getpid(), phase=phase, **kwargs)
        self.pds_events.parent.mkdir(parents=True, exist_ok=True)
        with self.pds_events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False, default=str) + "\n")
        if self.pds_dispatch:
            write_json(OUT / "state.json", row)

    def get_model(self, cfg=None, weights=None, verbose=True):
        require(self.data["nc"] == 1, "PDS F is single-class")
        if isinstance(weights, PDSDetectionModel):
            native = RTDETRTrainer.get_model(self, cfg=cfg, weights=native_view(weights), verbose=False)
            verify_model(native)
            model = PDSDetectionModel.attach(native)
            model.load_state_dict(weights.state_dict(), strict=True)
            self.pds_rebuild_audit = state_audit(native_view(weights), model)
        else:
            require(weights is not None, "PDS requires controlled unified initial weights")
            native, report = build_training_model(cfg, weights, self.data)
            reference = deepcopy(native)
            model = PDSDetectionModel.attach(native)
            self.pds_rebuild_audit = dict(state_audit(reference, model), native_nc_adaptation=report)
        require(sum(p.numel() for p in model.parameters()) == 20160586, "PDS train parameter count differs")
        return model

    def setup_model(self):
        if self.resume:
            ckpt = torch_load(self.args.model, map_location="cpu")
            validate_resume(ckpt)
            require(ckpt["pds_resume"]["source_sha"] == git("rev-parse", "HEAD"), "Resume source SHA changed")
            self.model = self.get_model(cfg=ckpt["model"].yaml, weights=ckpt["model"], verbose=False)
            return ckpt
        return super().setup_model()

    def build_optimizer(self, model, *args, **kwargs):
        optimizer = super().build_optimizer(model, *args, **kwargs)
        names = {id(p): name for name, p in model.named_parameters()}
        flat = [id(p) for group in optimizer.param_groups for p in group["params"]]
        require(len(flat) == len(set(flat)) and set(flat) == set(names), "Optimizer missing/duplicate parameters")
        self.pds_optimizer_groups = [
            dict(index=i, lr=g["lr"], decay=g["weight_decay"], kind=g.get("param_group"),
                 parameters=[names[id(p)] for p in g["params"]])
            for i, g in enumerate(optimizer.param_groups)]
        def stepped(_optimizer, _args, _kwargs):
            self.pds_effective_steps += 1
        optimizer.register_step_post_hook(stepped)
        return optimizer

    def _setup_train(self):
        self.event("SETTING_UP")
        super()._setup_train()
        require(not self.args.amp or self.amp, "Native AMP check disabled AMP; fixed PDS recipe must fail")
        require(self.batch_size == self.args.batch, "Batch changed during setup")
        write_json(self.save_dir / "pds_setup.json",
                   dict(runtime=environment(), config=CONFIG, rebuild=self.pds_rebuild_audit,
                        optimizer_groups=self.pds_optimizer_groups, amp=self.amp,
                        initial_scaler=self.scaler.state_dict(), clipping="native global max_norm=10"))

    @staticmethod
    def _epoch_start(trainer):
        trainer.model.pds_epoch = trainer.epoch
        trainer._pds_batch_in_epoch = 0

    @staticmethod
    def _batch_start(trainer):
        trainer._oom_retries = 3  # native branch then raises; it cannot halve batch
        trainer.model.pds_epoch = trainer.epoch

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if not self._pds_started and self.pds_dispatch:
            self._pds_started = True
            print("PDS_TRAINING_RUNNING", flush=True)
            self.event("RUNNING", epoch=self.epoch, image_shape=list(batch["img"].shape))
        return batch

    @staticmethod
    def _batch_end(trainer):
        trainer._pds_batch_in_epoch += 1
        stats = trainer.model.pds_stats
        trainer.event("RUNNING", epoch=trainer.epoch, batch=trainer._pds_batch_in_epoch,
                      effective_steps=trainer.pds_effective_steps, overflow_skips=trainer.pds_overflow_skips,
                      pds=stats)

    @staticmethod
    def _epoch_end(trainer):
        trainer.event("RUNNING", completed_epochs=trainer.epoch + 1,
                      early_stop=bool(trainer.stop and trainer.epoch + 1 < trainer.epochs))

    def optimizer_step(self):
        before, scale = self.pds_effective_steps, self.scaler.get_scale()
        row = dict(epoch=getattr(self, "epoch", None), scale_before=scale,
                   gradients=gradient_report(self.model, scale))
        super().optimizer_step()  # native unscale, global clip=10, step, update, zero, EMA
        skipped = self.pds_effective_steps == before
        self.pds_overflow_skips += int(skipped)
        row.update(scale_after=self.scaler.get_scale(), effective=not skipped,
                   effective_steps=self.pds_effective_steps)
        self.pds_step_diagnostics.append(row)
        self.pds_step_diagnostics = self.pds_step_diagnostics[-16:]
        write_json(self.save_dir / "pds_optimizer_latest.json", row)

    def validate(self):
        before = self.ema.ema.pds_head.calls
        result = super().validate()
        require(self.ema.ema.pds_head.calls == before, "PDS executed during per-epoch validation")
        self.event("RUNNING", stage="epoch_validation_complete", epoch=self.epoch, pds_calls=0)
        return result

    def _handle_nan_recovery(self, epoch):
        # Native recovery silently reloads EMA into raw weights and loses partial
        # accumulation. A PDS checkpoint requires the explicit complete resume path.
        if self.loss is not None and not torch.isfinite(self.loss).all():
            raise FloatingPointError("Nonfinite main/PDS training loss; preserve checkpoint and traceback")
        if self.fitness is not None and not np.isfinite(self.fitness):
            raise FloatingPointError("Nonfinite validation fitness")
        return False

    def save_model(self):
        model = unwrap_model(self.model)
        # Full raw FP32 train state + EMA; never derive resumed raw weights from EMA.
        ckpt = dict(epoch=self.epoch, best_fitness=self.best_fitness,
                    model=deepcopy(model).float(), ema=deepcopy(self.ema.ema).float(),
                    updates=self.ema.updates, optimizer=deepcopy(self.optimizer.state_dict()),
                    scaler=self.scaler.state_dict(), scheduler=self.scheduler.state_dict(),
                    stopper=vars(self.stopper).copy(), rng=rng_state(), train_args=vars(self.args).copy(),
                    train_metrics={**self.metrics, "fitness": self.fitness},
                    train_results=self.read_results_csv(), date=utc(),
                    pds_resume=dict(config=deepcopy(CONFIG), epoch=self.epoch, epoch_boundary=True,
                                    completed=bool(self.stop), last_opt_step=getattr(self, "_last_opt_step", -1),
                                    accumulate=self.accumulate, effective_steps=self.pds_effective_steps,
                                    overflow_skips=self.pds_overflow_skips, source_sha=git("rev-parse", "HEAD"),
                                    pending_gradients={n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                                                       if p.grad is not None}))
        buffer = io.BytesIO()
        torch.save(ckpt, buffer)
        payload = buffer.getvalue()
        self.wdir.mkdir(parents=True, exist_ok=True)
        targets = [self.last]
        if self.best_fitness == self.fitness:
            targets.append(self.best)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            targets.append(self.wdir / f"epoch{self.epoch}.pt")
        for path in targets:
            tmp = path.with_suffix(".pt.tmp")
            tmp.write_bytes(payload)
            os.replace(tmp, path)

    def resume_training(self, ckpt):
        if ckpt is None or not self.resume:
            return
        meta = validate_resume(ckpt)
        # The original loads optimizer/scaler/EMA, sets start_epoch and closes mosaic.
        super().resume_training(ckpt)
        self.scheduler.load_state_dict(ckpt["scheduler"])
        vars(self.stopper).update(ckpt["stopper"])
        self.model.pds_epoch = self.start_epoch
        self._resume_last_opt_step = meta["last_opt_step"]
        self.accumulate = meta["accumulate"]
        self.pds_effective_steps = meta["effective_steps"]
        self.pds_overflow_skips = meta["overflow_skips"]
        self._pds_pending = meta["pending_gradients"]
        self._pds_rng = ckpt["rng"]

    def _restore_pending_gradients(self):
        if hasattr(self, "_pds_pending"):
            named = dict(self.model.named_parameters())
            require(set(self._pds_pending) <= set(named), "Pending gradient keys changed")
            for name, grad in self._pds_pending.items():
                require(named[name].shape == grad.shape, "Pending gradient shape mismatch")
                named[name].grad = grad.to(named[name].device, dtype=named[name].dtype)
            del self._pds_pending
            restore_rng(self._pds_rng)
            del self._pds_rng

    def final_eval(self):
        # Preserve complete training best/last; native strip_optimizer destroys resume state.
        if self.best.exists():
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False
            self.metrics = self.validator(model=self.best)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")
