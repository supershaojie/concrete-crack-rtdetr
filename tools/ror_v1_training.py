"""ROR 专用 Trainer；标准模型类型、原优化器和推理协议保持不变。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch
from init_c19_lif_v1 import build_training_model, native_rebuild, verify_model, require
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.ror import RORLoss, CONFIG
from ultralytics.utils.torch_utils import unwrap_model


def install(model, identity):
    verify_model(model)
    before = set(model.state_dict())
    names = [n for n, _ in model.named_parameters()]
    model.criterion = RORLoss(nc=model.model[-1].nc)
    model.ror_v1 = deepcopy(identity)
    require(set(model.state_dict()) == before, "ROR introduced state_dict keys")
    require([n for n, _ in model.named_parameters()] == names, "ROR introduced parameters")
    return model


def epoch_start(trainer):
    model = unwrap_model(trainer.model)
    require(type(model.criterion) is RORLoss and model.criterion.config == CONFIG, "ROR criterion lost")
    model.criterion.set_epoch(int(trainer.epoch))
    model.criterion.reset_statistics()
    # Native EMA updates state_dict only; synchronize scalar metadata explicitly.
    if trainer.ema:
        trainer.ema.ema.criterion.set_epoch(int(trainer.epoch))
        trainer.ema.ema.ror_v1 = deepcopy(model.ror_v1)


def epoch_end(trainer):
    criterion = unwrap_model(trainer.model).criterion
    row = dict(e=int(trainer.epoch), weight=criterion.ror_weight,
               ror_skipped=criterion.ror_weight==0,
               validation_loss="L0 only; no ROR forward", **criterion.statistics)
    path = Path(trainer.ror_output) / "training_ror.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def batch_start(trainer):
    trainer._oom_retries = 3  # native fallback must not reduce formal batch16


def training_start(trainer):
    model = unwrap_model(trainer.model)
    verify_model(model, zero=not trainer.resume)
    require(bool(trainer.amp), "Native AMP was disabled; formal recipe cannot change")
    expected = dict(trainer.ror_plan["args"])
    if trainer.resume:
        expected.update(model=str(trainer.ror_last), resume=str(trainer.ror_last))
    actual = vars(trainer.args)
    diff = {k: [v, actual.get(k)] for k, v in expected.items()
            if type(v) is not type(actual.get(k)) or v != actual.get(k)}
    require(not diff, "Actual 109-field recipe differs: " + str(diff))
    require(type(trainer.optimizer) is torch.optim.AdamW, "Expected original AdamW")
    ids = [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
    require(all(p.requires_grad and ids.count(id(p)) == 1 for p in model.parameters()), "Frozen/missing/duplicate parameters")
    from init_c19_lif_v1 import write_json
    write_json(Path(trainer.ror_output) / "actual_training_setup.json",
               dict(args=actual, start_epoch=trainer.start_epoch, model_type=type(model).__name__,
                    criterion=type(model.criterion).__name__, identity=model.ror_v1, parameters=sum(p.numel() for p in model.parameters())))


class RORTrainer(RTDETRTrainer):
    """Use only with an independently verified plan. get_model is the actual rebuild boundary."""

    def __init__(self, *args, ror_plan=None, **kwargs):
        require(ror_plan is not None, "RORTrainer requires verified experiment plan")
        self.ror_plan = deepcopy(ror_plan)
        self.ror_output = ror_plan["output"]
        self.ror_last = Path(ror_plan["args"]["save_dir"]) / "weights/last.pt"
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_start", training_start)
        self.add_callback("on_train_epoch_start", epoch_start)
        self.add_callback("on_train_epoch_end", epoch_end)
        self.add_callback("on_train_batch_start", batch_start)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.resume:
            require(weights is not None and getattr(weights, "ror_v1", None) == self.ror_plan["identity"],
                    "Resume checkpoint ROR identity differs")
            model = native_rebuild(cfg, weights, self.data["nc"], self.data["channels"])
            verify_model(model)
        else:
            require(weights is not None, "Verified controlled initialization is required")
            model, report = build_training_model(cfg, weights, self.data)
            from init_c19_lif_v1 import write_json
            write_json(Path(self.ror_output) / "nc1_loading.json", report)
        return install(model, self.ror_plan["identity"])

    def resume_training(self, ckpt):
        if self.resume:
            require(ckpt is not None and 0 <= ckpt["epoch"] < self.epochs - 1, "No resumable ROR epoch")
            saved = ckpt.get("ema") or ckpt.get("model")
            require(getattr(saved, "ror_v1", None) == self.ror_plan["identity"], "Resume metadata mismatch")
            require(type(saved.criterion) is RORLoss and saved.criterion.config == CONFIG, "Resume criterion/version mismatch")
            require(saved.criterion.completed_epochs == ckpt["epoch"], "Saved ROR schedule/epoch mismatch")
            require(ckpt.get("optimizer") is not None and ckpt.get("scaler") is not None, "Stripped checkpoint cannot resume")
        super().resume_training(ckpt)
        if self.resume:
            unwrap_model(self.model).criterion.set_epoch(int(self.start_epoch))
            self.ema.ema.criterion.set_epoch(int(self.start_epoch))

    def optimizer_step(self):
        before = self.scaler.get_scale()
        result = super().optimizer_step()  # preserve native unscale, clip, skip-on-overflow and scale update
        after = self.scaler.get_scale()
        if after < before:
            path = Path(self.ror_output) / "amp_overflows.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(e=int(self.epoch),scale_before=before,scale_after=after,
                    event="native GradScaler skipped optimizer step; recipe unchanged"))+"\n")
        return result

    def _handle_nan_recovery(self, epoch):
        require(self.loss is None or bool(torch.isfinite(self.loss).all()), "Nonfinite loss; automatic restart forbidden")
        require(self.fitness is None or bool(torch.isfinite(torch.tensor(self.fitness))), "Nonfinite fitness")
        return False
