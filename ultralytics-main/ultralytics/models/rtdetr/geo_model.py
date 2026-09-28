"""Importable GEO specialization of the native mother model and trainer."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

import torch

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import unwrap_model
from .geo_loss import GEODetectionLoss, GEOGeometryError, FORMULA, ramp


def strict_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return dict(value=None, nonfinite=str(value))
    if isinstance(value, dict):
        return {str(k): strict_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_value(v) for v in value]
    return value


def write_json(path, value):
    """Atomic evidence persistence; utility derived from ARG ef9cb7e, no algorithm reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(strict_value(value), ensure_ascii=False, indent=2, allow_nan=False, default=str)+"\n", encoding="utf-8")
    os.replace(tmp, path)


class GEODetectionModel(RTDETRDetectionModel):
    def init_criterion(self):
        return GEODetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        if not isinstance(getattr(self, "criterion", None), GEODetectionLoss):
            self.criterion = self.init_criterion()
        self.criterion.epoch = getattr(self, "geo_epoch", 0)
        self.criterion.enabled = self.training and getattr(self, "geo_enabled", True)
        self.criterion.sample = self.training and getattr(self, "geo_sample", False)
        try:
            return super().loss(batch, preds)
        except GEOGeometryError as error:
            error.details["image_files"] = list(batch.get("im_file", []))
            raise


def gradient_report(model, scale=1.0):
    values = [p.grad.detach().float() / scale for p in model.parameters() if p.grad is not None]
    return dict(present=len(values), finite=all(bool(torch.isfinite(v).all()) for v in values),
                max_abs=max((float(v.abs().max()) for v in values), default=0.0))


def aggregate(samples):
    """Count-weighted summary, never an unweighted mean of unequal batches."""
    counts = ("M", "images", "gt_count", "neighbor_count", "selected_count", "outside_edges",
              "active_matches_1e8", "active_matches_001", "E_sum", "E_count", "gt_with_overlap", "match_mean_sum")
    out = {k: sum(s.get(k, 0) for s in samples) for k in counts}
    out["raw_geo"] = out["match_mean_sum"] / max(out["M"], 1)
    out["weighted"] = .2 * samples[0]["ramp"] * out["raw_geo"] if samples else 0.0
    out["E_mean"] = out["E_sum"] / max(out["E_count"], 1)
    out["E_max"] = max((s.get("E_max", 0) for s in samples), default=0)
    out["gt_original_overlap_ratio"] = out["gt_with_overlap"] / max(out["gt_count"], 1)
    out["original_losses_image_weighted"] = {k: sum(s["original_losses"][k]*s["images"] for s in samples)/max(out["images"], 1)
        for k in (samples[0]["original_losses"] if samples else {})}
    out["finite"] = all(s["finite"] for s in samples)
    out["geometry_skipped"] = all(s.get("geometry_skipped", False) for s in samples)
    if out["geometry_skipped"]:
        for k in ("raw_geo", "E_mean", "E_max", "gt_original_overlap_ratio"):
            out[k] = None  # skipped geometry is not an observed zero overflow
    return out


class GEOTrainer(RTDETRTrainer):
    def __init__(self, *args, **kwargs):
        self.geo_output = None
        self.geo_update_count = 0
        self.geo_samples, self.geo_updates = [], []
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._geo_epoch_start)
        self.add_callback("on_train_batch_start", self._geo_batch_start)
        self.add_callback("on_train_batch_end", self._geo_batch_end)
        self.add_callback("on_fit_epoch_end", self._geo_epoch_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.data["nc"] != 1:
            raise ValueError("GEO v1 supports nc=1 only")
        # Exactly the mother's construction/load and RNG sequence, including nc80->1.
        model = super().get_model(cfg, weights, verbose)
        from ultralytics.utils import YAML
        mother = YAML.load(Path(__file__).resolve().parents[2]/"cfg/models/rt-detr/rtdetr-resnet18-lite-cbr-lif-down.yaml")
        if any(model.yaml[k] != mother[k] for k in ("backbone", "head", "scales")):
            raise ValueError("GEO topology differs from the original mother YAML")
        from ultralytics.nn.modules import RTDETRDecoderCBR, LIFDown
        head = model.model[-1]
        if type(head) is not RTDETRDecoderCBR or sum(isinstance(m, LIFDown) for m in model.modules()) != 1:
            raise ValueError("GEO requires original CBR + LIF-Down mother")
        if sum(p.numel() for p in model.parameters()) != 20149765:
            raise ValueError("Unexpected mother parameter inventory")
        if head.num_queries != 300 or len(head.decoder.layers) != 3 or head.cbr.rho != .10:
            raise ValueError("Unexpected mother decoder geometry")
        model.__class__ = GEODetectionModel  # Python-only, importable, no new tensors/RNG
        model.geo_epoch = 0
        return model

    def build_optimizer(self, model, *args, **kwargs):
        optimizer = super().build_optimizer(model, *args, **kwargs)
        optimizer.register_step_post_hook(self._geo_step_done)
        return optimizer

    def _geo_step_done(self, optimizer, args, kwargs):
        self.geo_update_count += 1

    def _build_train_pipeline(self):
        if self.batch_size != self.args.batch:
            raise RuntimeError("Batch drift during setup")
        if getattr(self, "_geo_initial_batch", self.batch_size) != self.batch_size:
            raise RuntimeError("GEO refuses automatic batch reduction")
        self._geo_initial_batch = self.batch_size
        return super()._build_train_pipeline()

    def _setup_train(self):
        import logging
        from ultralytics.utils import LOGGER
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
        self.geo_amp_evidence = messages
        if self.args.amp and (not self.amp or not any("checks passed" in s for s in messages)):
            raise RuntimeError(f"Native AMP must explicitly pass without fallback: {messages}")
        unwrap_model(self.model).geo_epoch = self.start_epoch
        if self.geo_output:
            names = {id(p): n for n, p in self.model.named_parameters()}
            groups = [{**{k: v for k, v in g.items() if k != "params"}, "names": [names[id(p)] for p in g["params"]]}
                      for g in self.optimizer.param_groups]
            write_json(Path(self.geo_output)/"actual_setup.json", dict(args=vars(self.args), start_epoch=self.start_epoch,
                optimizer_groups=groups, scaler=self.scaler.state_dict(), ema_updates=self.ema.updates, amp=messages))

    @staticmethod
    def _geo_epoch_start(t):
        t.geo_samples, t.geo_updates, t.geo_batch = [], [], 0
        t.geo_epoch_started = time.monotonic()
        unwrap_model(t.model).geo_epoch = t.epoch

    @staticmethod
    def _geo_batch_start(t):
        t._oom_retries = 3  # disable native first-epoch automatic B16 halving
        unwrap_model(t.model).geo_sample = t.geo_batch < 4

    @staticmethod
    def _geo_batch_end(t):
        criterion = getattr(unwrap_model(t.model), "criterion", None)
        if t.geo_batch < 4 and getattr(criterion, "last_diagnostics", None):
            t.geo_samples.append(deepcopy(criterion.last_diagnostics))
        t.geo_batch += 1

    @staticmethod
    def _geo_epoch_end(t):
        if not t.geo_output:
            return
        row = dict(epoch_zero=t.epoch, epoch_display=t.epoch+1, ramp=ramp(t.epoch),
            scope="first at most 4 real training micro-batches; no extra forward/backward",
            sampled_batches=len(t.geo_samples), totals=aggregate(t.geo_samples), samples=t.geo_samples,
            updates=t.geo_updates, optimizer_updates=t.geo_update_count, fitness=float(t.fitness),
            seconds=time.monotonic()-t.geo_epoch_started)
        folder = Path(t.geo_output)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/"mechanism.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(strict_value(row), allow_nan=False)+"\n")
        write_json(folder/"progress.json", row)

    def optimizer_step(self):
        sample = len(self.geo_updates) < 2
        if sample:
            model = unwrap_model(self.model)
            parameter = model.model[-1].cbr.offset_out.weight
            before = parameter.detach().clone()
            scale, updates = self.scaler.get_scale(), self.geo_update_count
            gradients = gradient_report(model, scale)
        super().optimizer_step()
        if sample:
            self.geo_updates.append(dict(scale_before=scale, scale_after=self.scaler.get_scale(),
                skipped=self.geo_update_count == updates, updates=self.geo_update_count,
                cbr_parameter_changed=not torch.equal(before, parameter.detach()), gradients=gradients))

    def final_eval(self):
        # Native epoch validation/fitness/best selection is unchanged. Formal FP32
        # evaluation is deferred to explicit finish, which exports all queries once.
        # Keeping full last also preserves optimizer/scaler/EMA for failure recovery.
        if self.geo_output:
            write_json(Path(self.geo_output)/"training_completed.json", dict(status="TRAINING_COMPLETED",
                epoch_zero=self.epoch, epoch_display=self.epoch+1, epochs_budget=self.epochs,
                reason="epochs" if self.epoch+1 >= self.epochs else "native_patience",
                best=str(self.best), last=str(self.last), formal_evaluation="PENDING: run finish",
                best_rule="native val mAP50-95; later equal-fitness epoch overwrites best"))
