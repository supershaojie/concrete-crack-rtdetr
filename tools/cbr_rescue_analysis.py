"""Independent tensor observations. No model, training or native postprocess edits."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from pathlib import Path
import sys

import numpy as np
import torch

from cbr_rescue_common import ROOT, require
sys.path.insert(0, str(ROOT / "ultralytics-main"))
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.modules.cscef_v51 import CSCEFv51
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou

EPS = 1e-9
IOU_TOL = 1e-6
SIDES = ("left", "right", "top", "bottom")


def raw_tensor(prediction):
    raw = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
    require(raw.ndim == 3 and raw.shape[-1] >= 5, f"Expected B,Q,4+C probabilities: {raw.shape}")
    require(bool(torch.isfinite(raw).all()), "Nonfinite prediction.")
    require(bool(((raw[..., 4:] >= 0) & (raw[..., 4:] <= 1)).all()), "Scores must already be probabilities.")
    return raw


class MaskObserver:
    def __init__(self, conf=.001):
        self.conf = conf
        self.images = self.queries = self.low_images = self.low_queries = 0
        self.different_images = self.different_queries = 0
        self.minimum_score = None
        self.query_counts = set()
        self.examples = []

    def observe(self, raw, images):
        raw = raw_tensor(raw)
        score = raw[..., 4:].max(-1).values
        order = score.argsort(-1, descending=True)
        old = score > self.conf
        aligned = score.gather(-1, order) > self.conf
        difference = old != aligned
        low = ~old
        self.images += score.shape[0]
        self.queries += score.numel()
        self.query_counts.add(score.shape[1])
        value = float(score.min())
        self.minimum_score = value if self.minimum_score is None else min(self.minimum_score, value)
        self.low_images += int(low.any(-1).sum())
        self.low_queries += int(low.sum())
        self.different_images += int(difference.any(-1).sum())
        self.different_queries += int(difference.sum())
        for bi in difference.any(-1).nonzero().flatten().tolist():
            if len(self.examples) >= 16:
                break
            indices = difference[bi].nonzero().flatten()
            self.examples.append({"image": str(images[bi]), "sorted_positions": indices.tolist(),
                "query_indices": order[bi, indices].tolist(), "unsorted_scores": score[bi, indices].tolist(),
                "sorted_scores": score[bi, order[bi, indices]].tolist(),
                "old_keep": old[bi, indices].tolist(), "aligned_keep": aligned[bi, indices].tolist()})

    def report(self):
        return {**vars(self), "query_counts": sorted(self.query_counts),
                "difference_definition": "sorted-position old mask XOR score.gather(order)>conf; not set cardinality",
                "history_affected_this_mode": self.different_queries > 0}


class AlignedValidator(RTDETRValidator):
    def postprocess(self, preds):
        raw = raw_tensor(preds).clone()  # never share scaling storage with another route
        outputs = []
        for row in raw:
            boxes = ops.xywh2xyxy(row[:, :4] * self.args.imgsz)
            score, cls = row[:, 4:].max(-1)
            order = score.argsort(descending=True)
            keep = order[score[order] > self.args.conf]
            outputs.append({"bboxes": boxes[keep], "conf": score[keep], "cls": cls[keep].to(score.dtype)})
        return outputs


def distribution(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    require(bool(np.isfinite(a).all()), "Nonfinite statistic.")
    if not a.size:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p25": None, "p75": None, "p95": None,
                "min": None, "max": None}
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                **{f"p{q:02}": float(np.percentile(a, q)) for q in (5, 25, 75, 95)},
                min=float(a.min()), max=float(a.max()))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or (list(rows[0]) if rows else ["no_rows"]))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def cscef_observer(model, selected_indices, images, rows):
    """Observe actual lateral/output and derive confidence from actual normalized lateral."""
    handles, projected = [], {}
    modules = [(name, m) for name, m in model.named_modules() if isinstance(m, CSCEFv51)]
    require(len(modules) == 1, "Expected exactly one CSCEFv51 identified by type.")
    name, module = modules[0]
    def lateral_hook(m, args, result):
        projected["l"] = result[selected_indices].detach()
    def output_hook(m, args, result):
        lateral = args[0][0][selected_indices].detach().float()
        output = result[selected_indices].detach().float()
        l = projected.pop("l")
        gx, gy = m._compute_scharr_components(l)
        confidence = m._compute_structure_confidence(gx, gy)
        weight = m.output_projection.weight.detach().float()
        tensors = (lateral, output, confidence, weight)
        require(all(bool(torch.isfinite(t).all()) for t in tensors), "Nonfinite CSCEF observation.")
        rms = lambda x: x.square().mean((1, 2, 3)).sqrt()
        lateral_rms, output_rms, residual_rms = rms(lateral), rms(output), rms(output - lateral)
        for j, bi in enumerate(selected_indices):
            rows.append({"image": str(images[bi]), "module": name, "lateral_rms": float(lateral_rms[j]),
                         "output_rms": float(output_rms[j]), "residual_rms": float(residual_rms[j]),
                         "residual_ratio": float(residual_rms[j] / (lateral_rms[j] + EPS)), "eps": EPS,
                         "structure_confidence_spatial_mean": float(confidence[j].mean()),
                         "projection_weight_l2": float(weight.norm()), "nonfinite_values": 0})
    try:
        handles.append(module.lateral_norm.register_forward_hook(lateral_hook))
        handles.append(module.register_forward_hook(output_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        projected.clear()


def fixed_pairs(before_xyxy, gt_xyxy, candidates, pred_classes, gt_classes, min_iou=0.0):
    """Deterministic greedy maximum-IoU one-to-one match ONCE on before boxes.

    Threshold 0 retains zero-overlap pairs for visibility; pairs <.5 are flagged.
    Ties: ascending GT then query index. Empty GT/candidates returns no pairs.
    """
    if len(gt_xyxy) == 0 or len(candidates) == 0:
        return []
    iou = box_iou(gt_xyxy, before_xyxy[candidates]).cpu().numpy()
    edges = [(float(iou[g, j]), g, int(q)) for g in range(len(gt_xyxy)) for j, q in enumerate(candidates)
             if float(iou[g, j]) >= min_iou and int(pred_classes[q]) == int(gt_classes[g])]
    used_g, used_q, pairs = set(), set(), []
    for value, g, q in sorted(edges, key=lambda x: (-x[0], x[1], x[2])):
        if g not in used_g and q not in used_q:
            used_g.add(g)
            used_q.add(q)
            pairs.append((g, q, value))
    return pairs


def cbr_rows(details, raw, gt_xyxy, gt_classes, image):
    """One image, normalized xywh; signed sides ordered L,R,T,B."""
    d = {k: v.detach().cpu().float() for k, v in details.items()}
    before, after = d["before"], d["after"]
    require(before.shape == after.shape == (300, 4), "Expected all 300 CBR queries.")
    require(all(bool(torch.isfinite(v).all()) for v in d.values()), "Nonfinite CBR diagnostic.")
    require(bool((before[:, 2:] > 0).all() and (after[:, 2:] > 0).all()), "Nonpositive CBR box size.")
    score, cls = raw.detach().cpu()[:, 4:].max(-1)
    bxy, axy = ops.xywh2xyxy(before), ops.xywh2xyxy(after)
    scale = before[:, [2, 2, 3, 3]]
    relative = d["displacement"] / scale
    actual = (axy - bxy)[:, [0, 2, 1, 3]]
    torch.testing.assert_close(actual, d["displacement"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(d["aggregation_weights"].sum(-1), torch.ones((300, 4)), atol=1e-6, rtol=1e-5)
    center = after[:, :2] - before[:, :2]
    rows = []
    for q in range(len(before)):
        row = {"image": image, "query": q, "score": float(score[q]), "class": int(cls[q]),
               "score_ge_025": int(score[q] >= .25),
               "area_ratio": float(after[q, 2:].prod() / before[q, 2:].prod()),
               "center_dx": float(center[q, 0]), "center_dy": float(center[q, 1]),
               "center_dx_over_width": float(center[q, 0] / before[q, 2]),
               "center_dy_over_height": float(center[q, 1] / before[q, 3]),
               "center_distance": float(center[q].norm()),
               "bbox_short_side": float(before[q, 2:].min())}
        for route, boxes in (("before", before), ("after", after)):
            row.update({f"{route}_{axis}": float(boxes[q, j]) for j, axis in enumerate(("cx", "cy", "w", "h"))})
        for j, side in enumerate(SIDES):
            t, r = float(d["tanh_offsets"][q, j]), float(relative[q, j])
            row.update({f"{side}_displacement": float(d["displacement"][q, j]), f"{side}_relative": r,
                        f"{side}_abs_relative": abs(r), f"{side}_tanh": t, f"{side}_abs_tanh": abs(t),
                        f"{side}_saturated": int(abs(t) >= .95), f"{side}_positive": int(t > 0),
                        f"{side}_negative": int(t < 0), f"{side}_zero": int(t == 0)})
            row.update({f"{side}_weight_{k}": float(d["aggregation_weights"][q, j, k]) for k in range(3)})
        rows.append(row)
    pair_rows, image_rows = [], []
    gt_xyxy, gt_classes = gt_xyxy.cpu(), gt_classes.cpu()
    for scope, candidates in (("all", list(range(300))), ("score_ge_025", (score >= .25).nonzero().flatten().tolist())):
        pairs = fixed_pairs(bxy, gt_xyxy, candidates, cls, gt_classes)
        image_rows.append({"image": image, "scope": scope, "gt": len(gt_xyxy), "candidates": len(candidates),
                           "matched": len(pairs), "unmatched_gt": len(gt_xyxy) - len(pairs),
                           "unmatched_candidates": len(candidates) - len(pairs), "empty_gt": int(len(gt_xyxy) == 0),
                           "no_candidates": int(len(candidates) == 0)})
        for g, q, old_iou in pairs:
            new_iou = float(box_iou(gt_xyxy[g:g+1], axy[q:q+1])[0, 0])
            delta = new_iou - old_iou
            row = {"image": image, "scope": scope, "gt": g, "query": q, "score": float(score[q]),
                   "before_iou": old_iou, "after_iou": new_iou, "iou_delta": delta,
                   "low_iou_before_lt_05": int(old_iou < .5), "improved": int(delta > IOU_TOL),
                   "worsened": int(delta < -IOU_TOL), "unchanged": int(abs(delta) <= IOU_TOL)}
            old_err, new_err = (bxy[q] - gt_xyxy[g]).abs(), (axy[q] - gt_xyxy[g]).abs()
            for side, j in zip(SIDES, (0, 2, 1, 3)):
                row.update({f"{side}_error_before": float(old_err[j]), f"{side}_error_after": float(new_err[j]),
                            f"{side}_error_delta": float(new_err[j] - old_err[j])})
            pair_rows.append(row)
    return rows, pair_rows, image_rows


def numeric_summary(rows, excluded=()):
    if not rows:
        return {"n_rows": 0, "statistics": {}}
    return {"n_rows": len(rows), "statistics": {k: distribution([r[k] for r in rows]) for k, v in rows[0].items()
            if isinstance(v, (int, float)) and k not in excluded}}
