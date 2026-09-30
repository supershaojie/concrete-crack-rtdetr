# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""GPC v1: training-only GT-linked phase consistency, without new parameters.

All tensors live in the current call. The native detection loss, prediction,
decoder, DN, matcher and parameter names are unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json

import torch
from torch import nn

from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


@dataclass(frozen=True)
class GPCConfig:
    method: str = "gpc_v1"
    enabled: bool = True
    loss_weight: float = 0.10
    aux_source_images: int = 4
    aux_pair_layout: str = "originals_then_shifted"
    aux_bn: str = "eval_shared_running_statistics"
    aux_gradients: str = "both_views"
    aux_dn: str = "disabled"
    aux_detection_loss: str = "disabled"
    shift_pixels: int = 8
    shift_choices_xy: tuple = ((-8, -8), (-8, 0), (-8, 8), (0, -8), (0, 8), (8, -8), (8, 0), (8, 8))
    pad_value_01: float = 114 / 255
    coordinate_format: str = "normalized_xyxy"
    scale_floor_pixels: float = 1.0
    huber_delta: float = 1.0
    ramp_start_epoch: int = 5
    ramp_full_epoch: int = 20

    @classmethod
    def from_dict(cls, values=None):
        values = dict(values or {})
        if "shift_choices_xy" in values:
            values["shift_choices_xy"] = tuple(tuple(x) for x in values["shift_choices_xy"])
        result = cls(**values)
        default = cls()
        if any(len(pair) != 2 or any(type(x) is not int for x in pair) for pair in result.shift_choices_xy):
            raise ValueError("GPC shift choices require integer pixel pairs")
        for key, value in asdict(result).items():
            expected = getattr(default, key)
            if type(value) is not type(expected):
                raise ValueError(f"GPC config type changed: {key}")
            if key not in {"enabled", "loss_weight"} and value != expected:
                raise ValueError(f"GPC v1 fixed configuration changed: {key}")
        if result.loss_weight not in (0.0, 0.10):
            raise ValueError("GPC v1 permits the fixed 0.10 weight or an explicit off test (0.0)")
        return result

    def ramp(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("A real, nonnegative zero-based epoch is required")
        return max(0.0, min(1.0, (epoch - self.ramp_start_epoch) / (self.ramp_full_epoch - self.ramp_start_epoch)))


def finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"GPC nonfinite {name}, shape={tuple(value.shape)}")


def select_views(batch_size, image_ids, epoch, batch_index, seed=42, config=GPCConfig()):
    """Local CPU generator; never advances Python, NumPy, torch or CUDA global RNG."""
    if len(image_ids) != batch_size or type(batch_index) is not int or batch_index < 0:
        raise ValueError("GPC requires ordered image identifiers and epoch batch_index")
    config.ramp(epoch)
    payload = json.dumps(["gpc_v1", int(seed), epoch, batch_index, [str(x) for x in image_ids]],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu").manual_seed(derived)
    selected = torch.randperm(batch_size, generator=generator)[:min(config.aux_source_images, batch_size)].tolist()
    choices = torch.randint(len(config.shift_choices_xy), (len(selected),), generator=generator).tolist()
    return selected, [config.shift_choices_xy[i] for i in choices], derived


def translate(image, dx, dy, pad_value=114 / 255):
    """Integer CHW copy: positive x/right, positive y/down, no interpolation/wrap."""
    h, w = image.shape[-2:]
    if type(dx) is not int or type(dy) is not int:
        raise ValueError("GPC shifts must be integer pixels")
    result = torch.full_like(image, pad_value)
    x0, x1, y0, y1 = max(0, -dx), min(w, w - dx), max(0, -dy), min(h, h - dy)
    if x1 > x0 and y1 > y0:
        result[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx] = image[..., y0:y1, x0:x1]
    return result


def paired_targets(batch, selected, shifts):
    """Flatten auxiliary labels in row order; retain explicit global-to-parent maps."""
    h, w = batch["img"].shape[-2:]
    if len(selected) != len(shifts) or len(set(selected)) != len(selected) or any(i < 0 or i >= len(batch["img"]) for i in selected):
        raise ValueError("Invalid auxiliary source/shift mapping")
    device = batch["img"].device
    boxes = batch["bboxes"].detach().to(device=device, dtype=torch.float32)
    classes = batch["cls"].detach().to(device=device, dtype=torch.long).flatten()
    batch_idx = batch["batch_idx"].detach().to(device=device).flatten()
    finite("GT", boxes)
    if bool((classes != 0).any()):
        raise ValueError("GPC v1 supports nc=1 only")
    originals, shifted, eligible = [], [], {}
    counts = dict(source_gt=0, invalid_gt=0, eligible_gt=0, clipped_gt=0, vanished_gt=0)
    for source, (dx, dy) in zip(selected, shifts):
        gt = xywh2xyxy(boxes[batch_idx == source])
        delta = gt.new_tensor([dx / w, dy / h, dx / w, dy / h])
        orig_rows, shift_rows = [], []
        for j, box in enumerate(gt):
            counts["source_gt"] += 1
            parent = (source, j)
            if not bool((box[2:] > box[:2]).all()):
                counts["invalid_gt"] += 1
                continue
            moved = box + delta
            complete = bool(((box >= 0) & (box <= 1) & (moved >= 0) & (moved <= 1)).all())
            # Original labels are unchanged; only shifted matching labels are clipped.
            orig_rows.append((parent, box))
            clipped = moved.clamp(0, 1)
            if bool((clipped[2:] > clipped[:2]).all()):
                shift_rows.append((parent, clipped))
                if complete:
                    eligible[parent] = (box, delta)
                    counts["eligible_gt"] += 1
                else:
                    counts["clipped_gt"] += 1
            else:
                counts["vanished_gt"] += 1
        originals.append(orig_rows)
        shifted.append(shift_rows)
    rows = originals + shifted
    flat = [box for row in rows for _, box in row]
    gt_xyxy = torch.stack(flat) if flat else boxes.new_empty((0, 4))
    return dict(bboxes=xyxy2xywh(gt_xyxy), cls=classes.new_zeros(len(flat)),
                gt_groups=[len(row) for row in rows], parents=[parent for row in rows for parent, _ in row],
                eligible=eligible, selected=list(selected), shifts=list(shifts), hw=(h, w), counts=counts)


@contextmanager
def paired_forward_scope(model, device):
    """BN statistics only; affine/input grads and train-mode decoder stay active."""
    if not model.training or not model.model[-1].training:
        raise RuntimeError("GPC auxiliary forward requires the model and decoder in training mode")
    modules = [(m, m.training) for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if any(m.running_mean is None for m, _ in modules):
        raise RuntimeError("GPC requires BN running statistics")
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            for m, _ in modules:
                m.training = False
            yield
        finally:
            for m, training in modules:
                m.training = training


def consistency_xyxy(original, shifted, gt, delta, hw):
    """FP32 Huber mean over four coordinates and actual paired GTs; both views live."""
    with torch.autocast(device_type=original.device.type, enabled=False):
        original, shifted = original.float(), shifted.float()
        gt, delta = gt.detach().float(), delta.detach().float()
        for name, value in (("original prediction", original), ("shifted prediction", shifted), ("GT", gt), ("shift", delta)):
            finite(name, value)
        if len(original) == 0:
            return original.sum() * 0.0 + shifted.sum() * 0.0, original
        h, w = hw
        wh = (gt[:, 2:] - gt[:, :2]).clamp_min(gt.new_tensor([1 / w, 1 / h]))
        scale = wh.repeat(1, 2)
        residual = (original - (shifted - delta)) / scale
        finite("normalized residual", residual)
        a = residual.abs()
        loss = torch.where(a <= 1, 0.5 * residual.square(), a - 0.5).mean()
        finite("loss", loss)
        return loss, residual


def matched_consistency(boxes, scores, targets, matcher, details=False):
    """Use the unchanged matcher separately per auxiliary row, then join GT IDs."""
    k = len(targets["selected"])
    if boxes.shape[:2] != scores.shape[:2] or boxes.shape[0] != 2 * k or scores.shape[-1] != 1:
        raise ValueError("Unexpected auxiliary final prediction shape")
    finite("aux boxes", boxes)
    finite("aux logits", scores)
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        matches = matcher(boxes.detach().float(), scores.detach().float(), targets["bboxes"],
                          targets["cls"], targets["gt_groups"])
    by_row = []
    offset = 0
    for row, ((queries, flat_gt), size) in enumerate(zip(matches, targets["gt_groups"])):
        q, g = queries.tolist(), flat_gt.tolist()
        if len(set(q)) != len(q) or len(set(g)) != len(g) or any(i < offset or i >= offset + size for i in g):
            raise RuntimeError(f"Invalid unique Hungarian mapping in auxiliary row {row}")
        by_row.append({targets["parents"][j]: i for i, j in zip(q, g)})
        offset += size
    original, shifted, gt, deltas, ids = [], [], [], [], []
    for row, source in enumerate(targets["selected"]):
        for parent, (box, delta) in targets["eligible"].items():
            if parent[0] == source and parent in by_row[row] and parent in by_row[k + row]:
                original.append(boxes[row, by_row[row][parent]])
                shifted.append(boxes[k + row, by_row[k + row][parent]])
                gt.append(box)
                deltas.append(delta)
                ids.append(parent)
    stats = dict(targets["counts"], pairs=len(ids), missing_matches=len(targets["eligible"]) - len(ids))
    stats.update(clipped_fraction=stats["clipped_gt"] / stats["source_gt"] if stats["source_gt"] else None,
                 vanished_fraction=stats["vanished_gt"] / stats["source_gt"] if stats["source_gt"] else None,
                 missing_match_fraction=stats["missing_matches"] / stats["eligible_gt"] if stats["eligible_gt"] else None)
    if not ids:
        stats.update(loss_gpc=0.0, coverage=None if not stats["source_gt"] else 0.0)
        return boxes.sum() * 0.0, stats
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        left = xywh2xyxy(torch.stack(original).float())
        right = xywh2xyxy(torch.stack(shifted).float())
        gt, delta = torch.stack(gt), torch.stack(deltas)
        loss, residual = consistency_xyxy(left, right, gt, delta, targets["hw"])
    stats.update(loss_gpc=float(loss.detach()), coverage=len(ids) / stats["source_gt"],
                 mean_abs_normalized=float(residual.detach().abs().mean()))
    if details:
        from ultralytics.utils.metrics import bbox_iou
        h, w = targets["hw"]
        pixel_error = (left - (right - delta)) * left.new_tensor([w, h, w, h])
        iou0 = bbox_iou(left.detach(), gt, xywh=False).flatten()
        iou1 = bbox_iou(right.detach(), gt + delta, xywh=False).flatten()
        stats["per_gt"] = [dict(parent_gt_id=list(parent), pixel_error=pe, normalized_error=u,
                                 short_side_pixels=float(((g[2:] - g[:2]) * g.new_tensor([w, h])).min()),
                                 iou_original=float(a), iou_shifted=float(b))
                           for parent, pe, u, g, a, b in zip(ids, pixel_error.detach().tolist(), residual.detach().tolist(), gt, iou0, iou1)]
    return loss, stats


def auxiliary_loss(model, batch, selected, shifts, matcher, config=GPCConfig(), details=False, return_predictions=False):
    targets = paired_targets(batch, selected, shifts)
    if not targets["eligible"]:
        stats = dict(targets["counts"], pairs=0,
                    missing_matches=0, loss_gpc=0.0, auxiliary_images=0, skipped="no_eligible_gt")
        count = stats["source_gt"]
        stats.update(coverage=0.0 if count else None, clipped_fraction=stats["clipped_gt"] / count if count else None,
                     vanished_fraction=stats["vanished_gt"] / count if count else None, missing_match_fraction=None)
        result = batch["img"].new_zeros((), dtype=torch.float32), stats
        return (*result, None) if return_predictions else result
    images = batch["img"]
    finite("preprocessed images", images)
    if bool((images < 0).any()) or bool((images > 1).any()):
        raise ValueError("GPC expects preprocessed float images in [0,1]")
    original = images[selected]
    moved = torch.stack([translate(image, dx, dy, config.pad_value_01) for image, (dx, dy) in zip(original, shifts)])
    auxiliary = torch.cat((original, moved))
    with paired_forward_scope(model, images.device):
        raw = model.predict(auxiliary, batch=None)
    if raw[-1] is not None:
        raise RuntimeError("GPC auxiliary DN must be disabled via batch=None")
    loss, stats = matched_consistency(raw[0][-1], raw[1][-1], targets, matcher, details)
    stats.update(auxiliary_images=len(auxiliary), aux_dn_none=True)
    # Optional diagnostic return is consumed in this same batch, never stored on a module.
    return (loss, stats, raw) if return_predictions else (loss, stats)


class GPCDetectionModel(RTDETRDetectionModel):
    """Persistable top-level subclass; inherited tensor inference/export are native."""

    def __init__(self, cfg="rtdetr-resnet18-lite-cbr-lif-down.yaml", ch=3, nc=1, verbose=True, gpc_config=None):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if nc != 1:
            raise ValueError("GPC v1 requires nc=1")
        self.nc = nc  # Native Trainer also sets this; support direct standalone loss calls.
        self.gpc_config = asdict(GPCConfig.from_dict(gpc_config))
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout) and module.p != 0 or isinstance(module, nn.MultiheadAttention) and module.dropout != 0:
                raise ValueError(f"Unexpected stochastic layer: {name}")

    def loss(self, batch, preds=None):
        config = GPCConfig.from_dict(self.gpc_config)
        # Inactive calls use the original function directly, including validator loss(preds=...).
        if preds is not None or not self.training or not config.enabled or config.loss_weight == 0:
            return super().loss(batch, preds)
        context = batch.get("gpc_context")
        if context is None:
            raise RuntimeError("GPC training requires explicit epoch/batch_index/seed/image_ids context")
        ramp = config.ramp(context["epoch"])
        main, main_items = super().loss(batch, preds)
        if ramp == 0:
            context.get("metrics", {}).update(ramp=0.0, loss_gpc=0.0, weighted_gpc=0.0, auxiliary_images=0)
            return main, main_items
        selected, shifts, seed = select_views(len(batch["img"]), context["image_ids"], context["epoch"],
                                              context["batch_index"], context["seed"], config)
        extra, stats = auxiliary_loss(self, batch, selected, shifts, self.criterion.matcher, config)
        weighted = config.loss_weight * ramp * extra
        finite("main loss", main)
        context.get("metrics", {}).update(stats, ramp=ramp, weighted_gpc=float(weighted.detach()),
                                         main_loss=float(main.detach()), derived_seed=seed, selected=selected, shifts=shifts)
        return main + weighted, main_items
