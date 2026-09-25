"""Experiment Trainer: native optimization, audited reconstruction, explicit epoch propagation."""
from __future__ import annotations

from copy import deepcopy
import json
import time

import numpy as np
import torch

from bmc_v1_common import ROOT, OUT, require, write, research, utc
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.bmc import BMCDetectionModel
from ultralytics.utils.torch_utils import unwrap_model


def compare_states(before, after):
    a, b = before.state_dict(), after.state_dict()
    require(a.keys() == b.keys(), "State inventory changed")
    rows = [dict(key=k, shape=list(v.shape), equal=v.shape == b[k].shape and torch.equal(v, b[k])) for k, v in a.items()]
    require(all(r["equal"] for r in rows), "Constructor/reload disturbed mother state")
    require(dict(before.named_parameters()).keys() == dict(after.named_parameters()).keys(), "Parameter names changed")
    return dict(states=len(rows), parameters=sum(p.numel() for p in after.parameters()), new_parameters=0,
                all_equal=True, rows=rows)


class EpochCollector:
    def __init__(self, folder=None, dispatch=None):
        self.folder = folder
        self.dispatch = dispatch
        self.step = 0
        self.epoch_attempts = {}
        self.reset(None)

    def reset(self, epoch):
        if epoch is not None:
            self.epoch_attempts[epoch] = self.epoch_attempts.get(epoch, 0) + 1
        self.epoch, self.batches, self.state = epoch, 0, None
        self.counts = {k: 0 for k in ("matched", "disagreement", "changed", "components", "changed_components",
                      "background_to_positive", "target_swap", "fallback_g_gt_q")}
        self.ratios, self.losses, self.events = [], {}, []
        self.representatives = {}
        self.seconds = 0.
        self.logging_seconds = 0.
        self.epoch_start = time.perf_counter()

    @property
    def wants_detail(self):
        return self.step % 100 == 0 or len(self.representatives) < 2

    def record(self, epoch, report, loss):
        started = time.perf_counter()
        if self.epoch != epoch:
            self.reset(epoch)
        self.state = report["state"]
        self.batches += 1
        self.seconds += report.get("consensus_seconds", 0.)
        self.counts["matched"] += report.get("matched_gt", 0)
        for image in report["images"]:
            for k in self.counts:
                self.counts[k] += image[k]
            self.ratios.extend(image["ratios"])
            for event in image["events"]:
                kind = "accepted" if event["accepted"] else "rejected"
                if kind not in self.representatives:
                    self.representatives[kind] = dict(step=self.step, image=image["image"],
                                                     gt_offset=image["gt_offset"], **event)
            # Exact epoch quantiles on this dataset; fail loudly if future scope exceeds bound.
            require(len(self.ratios) <= 2_000_000, "BMC diagnostic epoch bound exceeded")
        for k, v in loss.items():
            self.losses[k] = self.losses.get(k, 0.) + float(v)
        if self.step % 100 == 0:
            event = dict(dispatch=self.dispatch, epoch=epoch, attempt=self.epoch_attempts[epoch], step=self.step, state=self.state,
                         images=[{k: v for k, v in r.items() if k != "ratios"} for r in report["images"][:16]])
            self.events.append(event)
            self.events = self.events[-8:]
            if self.folder:
                path = self.folder / "events.jsonl"
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.step += 1
        self.logging_seconds += time.perf_counter()-started

    def summary(self):
        c = self.counts
        active = self.state == "ENABLED"
        ratio = lambda a, b: a / b if active and b else None
        visible = c if active else {k: v if k == "matched" else None for k, v in c.items()}
        return dict(dispatch=self.dispatch, epoch=self.epoch, attempt=self.epoch_attempts.get(self.epoch),
                    state=self.state, batches=self.batches, **visible,
                    disagreement_rate=ratio(c["disagreement"], c["matched"]),
                    changed_fraction=ratio(c["changed"], c["matched"]),
                    disagreement_accepted=ratio(c["changed"], c["disagreement"]),
                    null_reason="not computed during warmup/disabled or denominator zero",
                    ratio_quantiles=dict(zip(("min", "p25", "p50", "p75", "p95", "max"),
                        np.quantile(self.ratios, [0, .25, .5, .75, .95, 1]).tolist())) if self.ratios else None,
                    decoder2_loss_mean={k: v / self.batches for k, v in self.losses.items()} if self.batches else {},
                    representatives=self.representatives,
                    additional_consensus_seconds=self.seconds, additional_collection_seconds=self.logging_seconds,
                    timing_scope="Measured consensus and collection/IO sections; no claim of a paired baseline slowdown measurement",
                    epoch_wall_seconds=time.perf_counter()-self.epoch_start)

    def finish(self):
        result = self.summary()
        if self.folder:
            write(self.folder / f"epoch_{self.epoch:03d}_{self.dispatch}_attempt{self.epoch_attempts[self.epoch]}.json", result)
        return result


class BMCTrainer(RTDETRTrainer):
    def __init__(self, *args, **kwargs):
        self.bmc_config = research()
        self.audit_folder = None
        self.mechanism_collector = None
        self.dispatch = None
        super().__init__(*args, **kwargs)
        require(self.world_size <= 1, "BMC v1 formal contract supports single GPU device=0 only")
        self.add_callback("on_train_epoch_start", self._bmc_epoch_start)
        self.add_callback("on_train_batch_start", self._bmc_batch_start)
        self.add_callback("on_train_epoch_end", self._bmc_epoch_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Replay the native constructor/load in an isolated RNG scope, not a post-audit repair.
        with torch.random.fork_rng(devices=[]):
            control = RTDETRTrainer.get_model(self, cfg=deepcopy(cfg), weights=weights, verbose=False)
        model = BMCDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        if weights is not None:
            before, after = weights.float().state_dict(), model.state_dict()
            require(before.keys() == after.keys(), "Unexpected source state keys")
            allowed = {f"model.26.{k}" for k in ("denoising_class_embed.weight", "enc_score_head.weight", "enc_score_head.bias")}
            allowed |= {f"model.26.dec_score_head.{i}.{k}" for i in range(3) for k in ("weight", "bias")}
            changed = {k for k in before if before[k].shape != after[k].shape}
            require(changed == (allowed if weights.model[-1].nc != self.data["nc"] else set()), "Unexpected class adaptation")
            # Strict merge explicitly retains only documented native class-adaptation tensors.
            model.load_state_dict({k: after[k] if k in changed else v for k, v in before.items()}, strict=True)
        self.initialization_audit = compare_states(control, model)
        model.bmc_config = self.bmc_config
        if self.audit_folder:
            write(self.audit_folder / "trainer_initialization.json", self.initialization_audit)
        return model

    def setup_model(self):
        ckpt = super().setup_model()
        model = unwrap_model(self.model)
        model.nc = self.data["nc"]  # needed before native set_model_attributes/EMA construction
        model.set_bmc_epoch(ckpt["epoch"] + 1 if self.resume and ckpt else 0)
        model.criterion = model.init_criterion()
        return ckpt

    def resume_training(self, ckpt):
        super().resume_training(ckpt)  # native optimizer/scaler/EMA/epoch restore
        model = unwrap_model(self.model)
        model.set_bmc_epoch(self.start_epoch)
        if self.ema:
            self.ema.ema.set_bmc_epoch(self.start_epoch)
        if self.resume:
            require(ckpt["optimizer"] is not None and ckpt["scaler"] is not None and ckpt["ema"] is not None,
                    "Resume requires native optimizer/scaler/EMA, not stripped inference weights")
            self.resume_audit = dict(checkpoint_epoch=ckpt["epoch"], start_epoch=self.start_epoch,
                criterion_epoch=model.criterion.epoch, bmc_active_at_train=self.bmc_config.enabled and self.start_epoch >= 20,
                ema_updates=self.ema.updates, scaler=self.scaler.state_dict(),
                optimizer_state_count=len(self.optimizer.state), optimizer_restored=True,
                ema_exact=all(torch.equal(v.cpu(), self.ema.ema.state_dict()[k].cpu()) for k, v in ckpt["ema"].float().state_dict().items()))
            if self.audit_folder:
                write(self.audit_folder / f"resume_{utc()}.json", self.resume_audit)

    @staticmethod
    def _bmc_epoch_start(trainer):
        model = unwrap_model(trainer.model)
        model.set_bmc_epoch(trainer.epoch)
        if trainer.ema:
            trainer.ema.ema.set_bmc_epoch(trainer.epoch)
        model.criterion.collector = trainer.mechanism_collector
        if trainer.mechanism_collector:
            trainer.mechanism_collector.reset(trainer.epoch)

    @staticmethod
    def _bmc_batch_start(trainer):
        trainer._oom_retries = 3  # native exception path raises; never silently lowers B16

    @staticmethod
    def _bmc_epoch_end(trainer):
        if trainer.mechanism_collector:
            trainer.mechanism_collector.finish()
