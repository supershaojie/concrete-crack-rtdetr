"""CEA v1: batch-local counterfactual labels for the existing CBR score weight.

No trainable state, hooks, cached activations, extra views or inference operators.
The teacher and student intentionally share the original FP32 score precision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ultralytics.models.utils.loss import DETRLoss, RTDETRDetectionLoss
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.metrics import bbox_iou


@dataclass(frozen=True)
class CEAConfig:
    method: str = "cea_v1"
    enabled: bool = True
    loss_weight: float = 0.10
    candidate_mix: float = 0.50
    advantage_eps: float = 1e-8
    scope: str = "final_decoder_ordinary_matched_queries"
    aux_gradient_scope: str = "existing_cbr_score_weight_only"
    teacher_precision: str = "float32"
    student_score_precision: str = "float32"
    tie_policy: str = "current_first_then_position_index"
    ramp_start_epoch: int = 5
    ramp_full_epoch: int = 20

    def __post_init__(self):
        if self.method != "cea_v1":
            raise ValueError("Only cea_v1 is supported")
        expected = {"candidate_mix": .5, "advantage_eps": 1e-8,
                    "scope": "final_decoder_ordinary_matched_queries",
                    "aux_gradient_scope": "existing_cbr_score_weight_only",
                    "teacher_precision": "float32", "student_score_precision": "float32",
                    "tie_policy": "current_first_then_position_index",
                    "ramp_start_epoch": 5, "ramp_full_epoch": 20}
        if any(getattr(self, k) != v for k, v in expected.items()) or self.loss_weight not in (0., .1):
            raise ValueError("CEA v1 constants are fixed; changes need a separate version")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")

    def ramp(self, epoch):
        return max(0., min(1., (int(epoch) - 5) / 15))

    def coefficient(self, epoch):
        return self.loss_weight * self.ramp(epoch) if self.enabled else 0.


def residual_from_sides(d):
    """Signed image-axis displacement, L/R/T/B -> normalized cxcywh residual."""
    dl, dr, dt, db = d.unbind(-1)
    return torch.stack(((dl + dr) / 2, (dt + db) / 2, dr - dl, db - dt), -1)


def _finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite CEA {name}")


@torch.no_grad()
def counterfactual_teacher(b0, hidden, p0, gt, offset_parameters):
    """Return detached [M,side,candidate,...] teacher data; never call Hungarian."""
    with torch.autocast(device_type=b0.device.type, enabled=False):
        b0, h, p0, gt = (x.detach().float() for x in (b0, hidden, p0, gt))
        if b0.shape[0] == 0:
            raise ValueError("Empty matches must bypass the candidate network")
        for name, value in (("b0", b0), ("hidden", h), ("p0", p0), ("GT", gt)):
            _finite(name, value)
        if (b0[:, 2:] <= 0).any() or (gt[:, 2:] <= 0).any():
            raise ValueError("CEA requires positive original and GT widths/heights; no coordinate clamp")
        eye3 = torch.eye(3, device=p0.device)
        pi = torch.cat((p0.unsqueeze(-2), .5 * p0.unsqueeze(-2) + .5 * eye3), dim=-2)
        wh, bh, wo, bo = (v.detach().float() for v in offset_parameters)
        pooled = (pi.unsqueeze(-1) * h.unsqueeze(-3)).sum(-2)
        u = F.linear(F.silu(F.linear(pooled, wh, bh)), wo, bo).squeeze(-1).tanh()
        displacement = .10 * b0[:, [2, 2, 3, 3], None] * u
        baseline_d = displacement[..., 0]
        # [M, replaced side, candidate, displacement side]. Only one side changes.
        d = baseline_d[:, None, None, :] + (
            displacement - baseline_d[..., None]
        )[..., None] * torch.eye(4, device=b0.device)[None, :, None, :]
        boxes = b0[:, None, None, :] + residual_from_sides(d)
        truth = gt[:, None, None, :].expand_as(boxes)
        giou = bbox_iou(boxes.reshape(-1, 4), truth.reshape(-1, 4), xywh=True, GIoU=True)
        giou = giou.reshape(boxes.shape[:-1])  # eliminate [N,1] before adding L1
        energy = 5 * (boxes - truth).abs().sum(-1) + 2 * (1 - giou)
        _finite("candidate boxes", boxes)
        _finite("energy", energy)
        choice = energy.argmin(-1)  # stable first occurrence; current is candidate 0
        selected_e = energy.gather(-1, choice[..., None]).squeeze(-1)
        improvement = energy[..., 0] - selected_e
        raw_a = improvement / (energy[..., 0] + 1e-8)
        _finite("advantage", raw_a)
        out_of_bounds = (raw_a < -1e-6) | (raw_a > 1 + 1e-6)
        if out_of_bounds.any():
            raise FloatingPointError(f"CEA advantage outside roundoff bounds: {raw_a[out_of_bounds].tolist()}")
        target = pi.gather(-2, choice[..., None, None].expand(-1, -1, 1, 3)).squeeze(-2)
        return dict(target=target, advantage=raw_a.clamp(0, 1), choice=choice,
                    improvement=improvement, energy=energy, candidates=boxes, distributions=pi,
                    anchor=boxes[:, 0, 0], boundary_roundoff_count=((raw_a < 0) | (raw_a > 1)).sum())


def weighted_kl(logits, target, advantage):
    """sum A KL(t||p)/(4M), including zero targets and zero advantages."""
    target, advantage = target.detach(), advantage.detach()
    logp = F.log_softmax(logits, dim=-1)
    # xlogy(0,0)=0. Teacher has no graph; student logp stays finite even at underflow.
    kl = (torch.xlogy(target, target) - target * logp).sum(-1)
    weighted = advantage * kl
    return weighted.mean(), kl, weighted


def cea_loss(b0, hidden, p_forward, b1, gt, live_score_weight, offset_parameters):
    """Only live_score_weight retains a gradient. All outputs are current-call values."""
    if len(b0) == 0:
        zero = live_score_weight.float().sum() * 0
        return zero, dict(matches=0, score_probability_max_abs=0., anchor_main_max_abs=0.)
    with torch.autocast(device_type=b0.device.type, enabled=False):
        h = hidden.detach().float()
        z = F.linear(h, live_score_weight.float(), bias=None).squeeze(-1)
        p = z.softmax(-1)
        _finite("student logits", z)
        probability_error = (p.detach() - p_forward.detach().float()).abs().max()
        # Declared before tests: FP32 score equivalence tolerance (not AMP-box tolerance).
        if probability_error > 1e-6:
            raise RuntimeError(f"CEA score differs from same-forward CBR: {probability_error.item()}")
        teacher = counterfactual_teacher(b0, h, p.detach(), gt, offset_parameters)
        loss, kl, weighted = weighted_kl(z, teacher["target"], teacher["advantage"])
        _finite("loss", loss)
        teacher.update(matches=len(b0), kl=kl.detach(), weighted_kl=weighted.detach(),
                       score_probability_max_abs=float(probability_error),
                       anchor_main_max_abs=float((teacher["anchor"] - b1.detach().float()).abs().max()))
        return loss, teacher


def quantiles(value):
    if value.numel() == 0:
        return []
    return torch.quantile(value.detach().float().flatten(), value.new_tensor([0., .25, .5, .75, .9, .99, 1.])).cpu().tolist()


def summarize(details):
    """Scalar-only production log; never store activations in model/criterion state."""
    row = {k: details[k] for k in ("matches", "score_probability_max_abs", "anchor_main_max_abs")}
    m = row["matches"]
    row.update(edges=4 * m, quantile_levels=[0, .25, .5, .75, .9, .99, 1])
    if m:
        positive = details["improvement"] > 0
        row.update(improved_edges=int(positive.sum()), improved_gt=int(positive.any(-1).sum()),
                   choice_counts=torch.bincount(details["choice"].flatten(), minlength=4).cpu().tolist(),
                   boundary_roundoff_count=int(details["boundary_roundoff_count"]))
        for key in ("advantage", "improvement", "kl", "weighted_kl"):
            row[key + "_quantiles"] = quantiles(details[key])
    return row


def matched_context(context, dn_meta, matches, gt):
    """Use the exact decoder DN split, then original ordinary-query and global GT indices."""
    ordinary = {}
    for key in ("before", "hidden", "aggregation_weights", "after"):
        value = context[key]
        if dn_meta is not None:
            _, value = torch.split(value, dn_meta["dn_num_split"], dim=1)
        ordinary[key] = value
    idx, target_idx = DETRLoss._get_index(matches)
    return tuple(ordinary[k][idx] for k in ("before", "hidden", "aggregation_weights", "after")) + (gt[target_idx],)


class CEADetectionLoss(RTDETRDetectionLoss):
    """Ordinary forward is inherited verbatim; opt-in path explicitly returns final matches."""

    def forward_with_matches(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        boxes, scores = preds
        self.device = boxes.device
        gt, cls, groups = batch["bboxes"], batch["cls"], batch["gt_groups"]
        matches = self.matcher(boxes[-1], scores[-1], gt, cls, groups)
        losses = self._get_loss(boxes[-1], scores[-1], gt, cls, groups, match_indices=matches)
        if self.aux_loss:
            losses.update(self._get_loss_aux(boxes[:-1], scores[:-1], gt, cls, groups, match_indices=None))
        if dn_meta is not None:
            assert len(groups) == len(dn_meta["dn_pos_idx"])
            dn_matches = self.get_dn_match_indices(dn_meta["dn_pos_idx"], dn_meta["dn_num_group"], groups)
            losses.update(DETRLoss.forward(self, dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=dn_matches))
        else:
            losses.update({f"{k}_dn": torch.tensor(0., device=self.device) for k in list(losses)})
        return losses, matches


class CEADetectionModel(RTDETRDetectionModel):
    """Same module tree/parameter keys; CEA is a loss-time opt-in only."""

    def __init__(self, *args, cea_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cea_config = asdict(CEAConfig(**(cea_config or {})))
        self.cea_epoch = 0
        self.cea_log_path = None
        self.cea_identity = None

    def init_criterion(self):
        return CEADetectionLoss(nc=self.nc, use_vfl=True)

    def cea_components(self, batch):
        """Explicit single-pass API for active training and isolated diagnosis."""
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        img, batch_idx = batch["img"], batch["batch_idx"]
        targets = dict(cls=batch["cls"].to(img.device, dtype=torch.long).view(-1),
                       bboxes=batch["bboxes"].to(img.device),
                       batch_idx=batch_idx.to(img.device, dtype=torch.long).view(-1),
                       gt_groups=[int((batch_idx == i).sum()) for i in range(img.shape[0])])
        preds, context = self.predict(img, batch=targets, return_cbr_context=True)
        boxes, scores, enc_boxes, enc_scores, dn_meta = preds if self.training else preds[1]
        dn_boxes = dn_scores = None
        if dn_meta is not None:
            dn_boxes, boxes = torch.split(boxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, scores = torch.split(scores, dn_meta["dn_num_split"], dim=2)
        boxes = torch.cat((enc_boxes.unsqueeze(0), boxes))
        scores = torch.cat((enc_scores.unsqueeze(0), scores))
        losses, matches = self.criterion.forward_with_matches((boxes, scores), targets, dn_boxes, dn_scores, dn_meta)
        cbr = self.model[-1].cbr
        matched = matched_context(context, dn_meta, matches, targets["bboxes"])
        extra, details = cea_loss(*matched, cbr.score.weight,
                                 (cbr.offset_hidden.weight, cbr.offset_hidden.bias,
                                  cbr.offset_out.weight, cbr.offset_out.bias))
        # Add only after ordinary/aux/DN complete. Diagnostics never enter this dict.
        losses["loss_cea"] = CEAConfig(**self.cea_config).coefficient(self.cea_epoch) * extra
        details["raw_loss"] = float(extra.detach())
        details["weighted_loss"] = float(losses["loss_cea"].detach())
        details["matched_images"] = sum(bool(len(src)) for src, _ in matches)
        if len(matched[0]):
            image_idx = DETRLoss._get_index(matches)[0][0].to(details["choice"].device)
            improved_gt = (details["improvement"] > 0).any(-1)
            details["improved_images"] = int(image_idx[improved_gt].unique().numel())
        else:
            details["improved_images"] = 0
        return losses, details, matches

    def loss(self, batch, preds=None):
        cfg = CEAConfig(**self.cea_config)
        if not self.training or cfg.coefficient(self.cea_epoch) == 0:
            return super().loss(batch, preds)
        if preds is not None:
            raise ValueError("Active CEA needs the same-forward explicit context; compile is unsupported")
        losses, details, _ = self.cea_components(batch)
        if self.cea_log_path:
            row = dict(epoch=self.cea_epoch, ramp=cfg.ramp(self.cea_epoch), coefficient=cfg.coefficient(self.cea_epoch),
                       raw_loss=details["raw_loss"], weighted_loss=details["weighted_loss"],
                       matched_images=details["matched_images"], improved_images=details["improved_images"],
                       original_losses={k: float(v.detach()) for k, v in losses.items() if k != "loss_cea"},
                       **summarize(details))
            with Path(self.cea_log_path).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        return sum(losses.values()), torch.stack([losses[k].detach() for k in ("loss_giou", "loss_class", "loss_bbox")])
