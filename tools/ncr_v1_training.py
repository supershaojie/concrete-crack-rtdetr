"""Importable native trainer adapter: audited initialization, real epoch, bounded diagnostics."""
from __future__ import annotations

import json

import torch

from ncr_v1_common import NCR, paths, recipe, require, write
from init_c19_lif_v1 import build_training_model, verify_model
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.ncr import configure_ncr
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import strip_optimizer, unwrap_model


def set_epoch(trainer):
    """Called after native resume_training has restored trainer.epoch/start_epoch."""
    for model in (unwrap_model(trainer.model), trainer.ema.ema if trainer.ema else None):
        if model is not None:
            model.ncr_epoch = int(trainer.epoch)
            model.ncr_collect_diagnostics = False
    trainer._ncr_batch = 0


def before_batch(trainer):
    trainer._oom_retries = 3  # disable native batch-halving retries; no GPU-wide lock
    unwrap_model(trainer.model).ncr_collect_diagnostics = trainer._ncr_batch == 0


def after_batch(trainer):
    if trainer._ncr_batch == 0:
        model = unwrap_model(trainer.model)
        row = dict(epoch=int(trainer.epoch), batch=0, **(model.criterion.last_diagnostics or {}))
        with (paths()["output"]/"training_diagnostics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        model.ncr_collect_diagnostics = False
    trainer._ncr_batch += 1


def training_setup(trainer):
    p = paths()
    expected, _ = recipe()
    actual = vars(trainer.args).copy()
    ignored = {"resume", "model"} if trainer.resume else set()
    differences = {k: [v, actual.get(k)] for k, v in expected.items()
                   if k not in ignored and (type(actual.get(k)) is not type(v) or actual.get(k) != v)}
    require(not differences, "Native training config drift: "+str(differences))
    require(trainer.amp and trainer.batch_size == 16 and trainer.args.imgsz == 640, "AMP/B16/640 changed")
    model = unwrap_model(trainer.model)
    verify_model(model, zero=not trainer.resume)
    require(model.ncr_config == NCR and type(trainer.optimizer) is torch.optim.AdamW, "Loss/optimizer changed")
    ids = [id(v) for g in trainer.optimizer.param_groups for v in g["params"]]
    require(all(ids.count(id(v)) == 1 for v in model.parameters()), "Missing/duplicated optimizer parameter")
    YAML.save(p["output"]/"actual_train_args.yaml", actual)
    write(p["output"]/"training_setup.json", dict(status="PASSED", amp=bool(trainer.amp),
          start_epoch=trainer.start_epoch, recipe_differences=differences, ncr=model.ncr_config,
          parameters=sum(v.numel() for v in model.parameters()), optimizer=type(trainer.optimizer).__name__,
          scheduler=type(trainer.scheduler).__name__, accumulate=trainer.accumulate,
          groups=[dict(lr=g["lr"], weight_decay=g["weight_decay"], parameters=len(g["params"])) for g in trainer.optimizer.param_groups]))


class NCRTrainer(RTDETRTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_start", training_setup)
        self.add_callback("on_train_epoch_start", set_epoch)
        self.add_callback("on_train_batch_start", before_batch)
        self.add_callback("on_train_batch_end", after_batch)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.resume:
            require(getattr(weights, "ncr_config", None) == NCR, "Resume checkpoint is not NCR v1")
            model = super().get_model(cfg, weights, verbose)
            verify_model(model)
        else:
            model, loading = build_training_model(cfg, weights, self.data)
            write(paths()["output"]/"nc1_loading.json", loading)
        return configure_ncr(model, NCR)

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = "giou_loss", "cls_loss", "l1_loss", "ncr_loss"
        # Preserve parent epoch validation/fitness/early stopping exactly.
        return validator

    def final_eval(self):
        # Native epoch validation already selects best. Defer independent forward
        # to finish so its FIRST formal FP32 val also exports all regular queries.
        ckpt = strip_optimizer(self.last) if self.last.exists() else {}
        if self.best.exists():
            strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        write(paths()["output"]/"formal_eval_deferred.json", dict(status="NOT_RUN", command="bash tools/ncr_v1.sh finish",
              reason="Independent FP32 val/test and synchronous all-query export belong to finish"))
