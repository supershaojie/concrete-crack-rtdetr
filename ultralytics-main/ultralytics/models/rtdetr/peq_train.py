# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Native AdamW/scaler lifecycle with two disjoint gradient clipping sets."""
from __future__ import annotations
from copy import copy, deepcopy
from pathlib import Path
import logging
import torch
from .train import RTDETRTrainer
from .val import RTDETRValidator
from .peq_model import PEQDetectionModel
from .peq_stats import PEQStatistics
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import ops, LOGGER


def is_peq(name):
    return name.startswith("model.26.peq.")


def parameter_partition(model):
    original, added = [], []
    for name, parameter in model.named_parameters():
        (added if is_peq(name) else original).append(parameter)
    if not added or len({id(p) for p in original+added}) != len(list(model.parameters())):
        raise ValueError("PEQ clipping sets must be disjoint and exhaustive")
    return original, added


def class_adaptation_keys():
    prefix = "model.26."
    return {prefix+"denoising_class_embed.weight", prefix+"enc_score_head.weight", prefix+"enc_score_head.bias"} | {
        prefix+f"dec_score_head.{i}.{kind}" for i in range(3) for kind in ("weight", "bias")}


def checked_training_model(cfg, weights, data, verbose=False):
    """Native constructor RNG and strict nc adaptation, audited against the pinned mother."""
    mother_cfg = deepcopy(cfg) if isinstance(cfg, dict) else None
    if mother_cfg is None:
        from ultralytics.utils import YAML
        mother_cfg = YAML.load(cfg)
    if mother_cfg["head"][-1][2] != "RTDETRDecoderCBRPEQ":
        raise ValueError("PEQ Trainer cannot rebuild a non-PEQ architecture")
    mother_cfg["head"][-1][2] = "RTDETRDecoderCBR"
    mother_cfg.pop("peq", None)
    with torch.random.fork_rng(devices=[]):
        mother = RTDETRDetectionModel(deepcopy(mother_cfg), nc=data["nc"], ch=data["channels"], verbose=False)
        if weights is not None:
            # Auditing mother uses its real native loading path; only the target is strictly loaded below.
            mother.load(weights)
    model = PEQDetectionModel(deepcopy(cfg), nc=data["nc"], ch=data["channels"], verbose=verbose)
    report = dict(allowed_class_adaptation=[], unexpected=[], missing=[], common_equal=True)
    if weights is not None:
        source = weights.float().state_dict()
        target = model.state_dict()
        if set(source) != set(target):
            raise ValueError(f"PEQ strict reload state mismatch: missing={set(target)-set(source)}, extra={set(source)-set(target)}")
        changed = {k for k in source if source[k].shape != target[k].shape}
        allowed = class_adaptation_keys() if weights.model[-1].nc != data["nc"] else set()
        if changed != allowed:
            raise ValueError(f"Unexpected nc adaptation: {changed}")
        model.load_state_dict({k: target[k] if k in changed else source[k] for k in target}, strict=True)
        report["allowed_class_adaptation"] = [dict(name=k, before=list(source[k].shape), after=list(target[k].shape))
                                               for k in sorted(changed)]
        report["loaded_exact"] = len(source)-len(changed)
        if any(not torch.equal(source[k], model.state_dict()[k]) for k in source if k not in changed):
            raise ValueError("PEQ reload changed compatible weights")
    public = mother.state_dict()
    if len(public) != 552 or any(not torch.equal(value, model.state_dict()[key]) for key, value in public.items()):
        raise ValueError("PEQ Trainer common values differ from native mother constructor/loading")
    report["common_states"] = len(public)
    report["added_states"] = sorted(set(model.state_dict())-set(public))
    if not all(is_peq(k) for k in report["added_states"]):
        raise ValueError("Unexpected added state")
    return model, report


def corrected_postprocess(preds, imgsz, conf, max_det=300):
    """The mother corrected_sorted_conf_mask_v1 algorithm; attach the SAME sorted mask."""
    preds = preds[0] if isinstance(preds, (list, tuple)) else preds
    output = []
    for pred in preds:
        boxes = ops.xywh2xyxy(pred[:, :4] * imgsz)  # no in-place mutation of the raw/payload
        score, cls = pred[:, 4:].max(-1)
        order = score.argsort(descending=True)
        rows = torch.cat((boxes, score[:, None], cls[:, None]), -1)[order]
        mask = rows[:, 4] > conf
        rows = rows[mask][:max_det]
        output.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5],
                           query_ids=order[mask][:max_det]))
    return output


class PEQValidator(RTDETRValidator):
    def postprocess(self, preds):
        return corrected_postprocess(preds, self.args.imgsz, self.args.conf, self.args.max_det)


class PEQTrainer(RTDETRTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model, self.peq_loading_audit = checked_training_model(cfg, weights, self.data, verbose)
        return model

    def get_validator(self):
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return PEQValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def _setup_train(self):
        messages = []
        class AMPTrace(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                if "AMP" in message:
                    messages.append(message)
        handler = AMPTrace()
        LOGGER.addHandler(handler)
        try:
            super()._setup_train()
        finally:
            LOGGER.removeHandler(handler)
        self.peq_amp_check_messages = messages
        if self.args.amp and (not self.amp or not any("checks passed" in m for m in messages)):
            raise RuntimeError(f"Native check_amp did not actually pass (skipped is not PASS): {messages}")
        self.peq_optimizer_calls = 0
        self.model.peq_stats = PEQStatistics()
        if self.ema:
            self.ema.ema.peq_stats = PEQStatistics()
        ids = [id(p) for group in self.optimizer.param_groups for p in group["params"]]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.model.parameters()}:
            raise ValueError("Every original and PEQ parameter must occur once in native AdamW")
        if not all(p.requires_grad for p in self.model.parameters()):
            raise ValueError("PEQ experiment cannot freeze detector or PEQ parameters")
        parameter_partition(self.model)
        # Parent silently lowers batch on OOM. This experiment must fail at the requested B/size.
        self._oom_retries = 3

    def optimizer_step(self):
        if not self.model.model[-1].peq.config["enabled"]:
            return super().optimizer_step()
        self.scaler.unscale_(self.optimizer)
        original, added = parameter_partition(self.model)
        self.peq_optimizer_calls = getattr(self, "peq_optimizer_calls", 0) + 1
        stats = self.model.peq_stats
        if stats is not None:
            stats.gradient_event(self.model, self.peq_optimizer_calls)
        self.peq_clip_norms = (
            float(torch.nn.utils.clip_grad_norm_(original, max_norm=10.0)),
            float(torch.nn.utils.clip_grad_norm_(added, max_norm=10.0)),
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
