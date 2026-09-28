"""Native trainer with Python-only DTR wiring and bounded mechanism records.

Reliability patterns: ARG ef9cb7e (native reconstruction, AMP audit, OOM guard,
step observation). No ARG algorithm, fallback scaler, or training state is used.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import unwrap_model
from .dtr_loss import ramp, require
from .dtr_model import DTRDetectionModel


def aggregate(samples):
    active = [s for s in samples if not s["skipped"]]
    result = dict(sampled_batches=len(samples), active_batches=len(active),
                  skipped_batches=len(samples) - len(active), M=0, with_dn=0, without_dn=0, K={})
    for s in active:
        for k in ("M", "with_dn", "without_dn"):
            result[k] += s[k]
        for k, v in s["K"].items():
            result["K"][k] = result["K"].get(k, 0) + v
    count = result["with_dn"]
    result["reference_edge_count"] = 4 * count
    for key in ("radius_over_scale_sum", "error_over_scale_sum", "excess_sum", "active_count"):
        sums = [sum(s[key][i] for s in active) for i in range(4)]
        result[key] = sums
        result[key.replace("_sum", "") + "_per_edge_mean"] = [v / count for v in sums] if count else None
    result["active_fraction"] = sum(result["active_count"]) / (4 * count) if count else None
    # Losses are batch means; keep sums + count instead of averaging unequal M.
    result["raw_dtr_numerator"] = sum(s["raw_dtr"] * 4 * max(s["M"], 1) for s in active)
    result["raw_dtr_all_matches_mean"] = result["raw_dtr_numerator"] / (4 * result["M"]) if result["M"] else 0.
    result["weighted_dtr_numerator"] = sum(s["weighted_dtr"] * 4 * max(s["M"], 1) for s in active)
    result["weighted_dtr_all_matches_mean"] = result["weighted_dtr_numerator"] / (4 * result["M"]) if result["M"] else 0.
    return result


class DTRTrainer(RTDETRTrainer):
    def __init__(self, *args, **kwargs):
        self.dtr_output = None
        self.dtr_identity = None
        self.dtr_update_count = 0
        self.dtr_samples, self.dtr_steps = [], []
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._epoch_start)
        self.add_callback("on_train_batch_start", self._batch_start)
        self.add_callback("on_train_batch_end", self._batch_end)
        self.add_callback("on_fit_epoch_end", self._epoch_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        require(self.data["nc"] == 1, "v1 supports only nc=1")
        model = super().get_model(cfg, weights, verbose)
        # No second initialization and no RNG consumption.
        model.__class__ = DTRDetectionModel
        model.dtr_epoch = 0
        return model

    def build_optimizer(self, model, *args, **kwargs):
        optimizer = super().build_optimizer(model, *args, **kwargs)
        optimizer.register_step_post_hook(self._step_done)
        return optimizer

    def _step_done(self, optimizer, args, kwargs):
        self.dtr_update_count += 1

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
        if self.args.amp:
            require(self.amp and any("checks passed" in m for m in messages),
                    f"native AMP check must explicitly pass: {messages}")
        self.dtr_amp_evidence = messages
        model = unwrap_model(self.model)
        require(isinstance(model, DTRDetectionModel), "actual trainer model lost DTR class")
        require(sum(p.numel() for p in model.parameters()) == 20149765, "unfused parameter count changed")
        model.dtr_epoch = self.start_epoch
        if self.dtr_identity:
            model.dtr_identity = self.dtr_identity
            self.ema.ema.dtr_identity = self.dtr_identity
        if self.dtr_output:
            from experiment_runtime import write_json
            names = {id(p): n for n, p in model.named_parameters()}
            write_json(Path(self.dtr_output) / "actual_setup.json", dict(args=vars(self.args),
                model_class=f"{type(model).__module__}.{type(model).__name__}", start_epoch=self.start_epoch,
                scaler=self.scaler.state_dict(), ema_updates=self.ema.updates, amp=messages,
                groups=[dict(names=[names[id(p)] for p in g["params"]],
                             **{k: v for k, v in g.items() if k != "params"}) for g in self.optimizer.param_groups]))

    @staticmethod
    def _epoch_start(t):
        t.dtr_batch, t.dtr_samples, t.dtr_steps = 0, [], []
        t.dtr_epoch_started = time.monotonic()
        unwrap_model(t.model).dtr_epoch = t.epoch

    @staticmethod
    def _batch_start(t):
        t._oom_retries = 3  # preserve original B16, never silently halve on OOM
        unwrap_model(t.model).dtr_sample = t.dtr_batch < 4

    @staticmethod
    def _batch_end(t):
        criterion = getattr(unwrap_model(t.model), "criterion", None)
        if t.dtr_batch < 4 and getattr(criterion, "last_diagnostics", None):
            t.dtr_samples.append(deepcopy(criterion.last_diagnostics))
        t.dtr_batch += 1

    @staticmethod
    def _epoch_end(t):
        if not t.dtr_output:
            return
        from experiment_runtime import now, write_json
        row = dict(time=now(), epoch_zero_based=t.epoch, displayed_epoch=t.epoch + 1, ramp=ramp(t.epoch),
                   sampling="at most first 4 training micro-batches; no additional forward/backward",
                   aggregate=aggregate(t.dtr_samples), samples=t.dtr_samples, optimizer_steps=t.dtr_steps,
                   effective_updates=t.dtr_update_count, seconds=time.monotonic() - t.dtr_epoch_started,
                   fitness=float(t.fitness), stopped=bool(t.stop))
        path = Path(t.dtr_output)
        with (path / "mechanism.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(path / "progress.json", row)

    def optimizer_step(self):
        observed = len(self.dtr_steps) < 4
        if observed:
            scale, updates = self.scaler.get_scale(), self.dtr_update_count
            parameter = unwrap_model(self.model).model[-1].cbr.offset_out.weight
            before = parameter.detach().clone()
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            finite = all(bool(torch.isfinite(g).all()) for g in grads)
        super().optimizer_step()
        if observed:
            self.dtr_steps.append(dict(scale_before=scale, scale_after=self.scaler.get_scale(),
                skipped=updates == self.dtr_update_count, gradients_finite=finite,
                parameter_changed=not torch.equal(before, parameter.detach()), updates=self.dtr_update_count))

    def final_eval(self):
        # Native loop has reached epochs/patience. Defer its extra val and stripping
        # to explicit finish: first formal FP32 val exports all queries exactly once.
        # Epoch validator, fitness and best selection are untouched.
        if self.dtr_output:
            from experiment_runtime import write_json, now
            write_json(Path(self.dtr_output) / "training_completed.json", dict(status="TRAINING_COMPLETED",
                ended=now(), epoch_zero_based=self.epoch, displayed_epoch=self.epoch + 1,
                reason="epochs" if self.epoch + 1 >= self.epochs else "native_patience",
                best=str(self.best), last=str(self.last), formal_eval="PENDING: run finish",
                checkpoint_stripped=False))
