"""PDS v1: fixed, single-class phase-preserving dense auxiliary supervision."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

PHASES = ((0, 0), (1, 0), (0, 1), (1, 1))
CONFIG = dict(version="pds_v1", experiment="F", taps=[4, 19], channels=[64, 256],
              phases=PHASES, width=32, groups=4, norm_eps=1e-5, seed=424003,
              cls_std=.01, cls_prior=.01, box_std=.001, box_prior=.1,
              center_scale=.1, epsilon=1e-7, quality_epsilon=1e-6,
              max_candidates=128, K=4, alpha=.25, gamma=2, negative_weight=.25,
              l1_weight=5., giou_weight=2., lambda_pds=.25,
              ramp_start=5, ramp_length=15, parameters=10821, state_keys=11)


def ramp(epoch):
    return min(max((int(epoch) - 5) / 15., 0.), 1.)


def finite(value, where):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"PDS nonfinite at {where}")
    return value


def split_phases(x):
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("PDS requires even P2 spatial dimensions")
    return torch.stack([x[..., dy::2, dx::2] for dy, dx in PHASES], 1)


def scatter_phases(x):
    """B,phase,C,H,W -> B,C,2H,2W; no pixel_shuffle convention assumption."""
    b, t, c, h, w = x.shape
    if t != 4:
        raise ValueError("PDS needs four phases")
    out = x.new_empty(b, c, 2 * h, 2 * w)
    for phase, (dy, dx) in enumerate(PHASES):
        out[..., dy::2, dx::2] = x[:, phase]
    return out


class PDSHead(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(CONFIG["seed"])
            self.p2_proj = nn.Conv2d(64, 32, 1, bias=False)
            self.p3_proj = nn.Conv2d(256, 32, 1, bias=False)
            self.norm1 = nn.GroupNorm(4, 32, eps=1e-5)
            self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
            self.norm2 = nn.GroupNorm(4, 32, eps=1e-5)
            self.cls = nn.Conv2d(32, 1, 1)
            self.box = nn.Conv2d(32, 4, 1)
            nn.init.normal_(self.cls.weight, std=.01)
            nn.init.constant_(self.cls.bias, math.log(.01 / .99))
            nn.init.normal_(self.box.weight, std=.001)
            with torch.no_grad():
                self.box.bias.copy_(torch.tensor([0., 0., math.log(.1 / .9), math.log(.1 / .9)]))
        self.calls = 0

    @staticmethod
    def conv(layer, x):
        return F.conv2d(x, layer.weight.float(), None if layer.bias is None else layer.bias.float(),
                        layer.stride, layer.padding, layer.dilation, layer.groups)

    @staticmethod
    def norm(layer, x):
        return F.group_norm(x, 4, layer.weight.float(), layer.bias.float(), 1e-5)

    def forward(self, f2, f3):
        if (f2.ndim != 4 or f3.ndim != 4 or f2.shape[0] != f3.shape[0]
                or f2.shape[1] != 64 or f3.shape[1] != 256
                or f2.shape[-2:] != (2 * f3.shape[-2], 2 * f3.shape[-1])):
            raise ValueError(f"PDS tap mismatch: P2={tuple(f2.shape)}, Neck P3={tuple(f3.shape)}")
        self.calls += 1
        with torch.autocast(device_type=f2.device.type, enabled=False):
            b, _, h, w = f3.shape
            p2 = split_phases(f2.float()).flatten(0, 1)
            s = self.conv(self.p3_proj, f3.float())[:, None]
            z = self.conv(self.p2_proj, p2).reshape(b, 4, 32, h, w) + s
            z = F.silu(self.norm(self.norm1, z.flatten(0, 1)))
            t = F.silu(self.norm(self.norm2, self.conv(self.dw, z)))
            logits = scatter_phases(self.conv(self.cls, t).reshape(b, 4, 1, h, w))
            raw = scatter_phases(self.conv(self.box, t).reshape(b, 4, 4, h, w))
            return finite(logits, "head.logits"), finite(raw, "head.raw_boxes")


def xywh_to_xyxy(box):
    return torch.cat((box[..., :2] - box[..., 2:] / 2, box[..., :2] + box[..., 2:] / 2), -1)


def xyxy_to_xywh(box):
    return torch.cat(((box[..., :2] + box[..., 2:]) / 2, box[..., 2:] - box[..., :2]), -1)


def overlap(pred, target, giou=False):
    """Aligned/broadcast IoU; candidate pairs only, never GT x all grid."""
    lo = torch.maximum(pred[..., :2], target[..., :2])
    hi = torch.minimum(pred[..., 2:], target[..., 2:])
    intersection = (hi - lo).clamp(min=0).prod(-1)
    union = ((pred[..., 2:] - pred[..., :2]).clamp(min=0).prod(-1)
             + (target[..., 2:] - target[..., :2]).clamp(min=0).prod(-1) - intersection)
    iou = intersection / (union + 1e-7)
    if not giou:
        return iou
    enclosing = (torch.maximum(pred[..., 2:], target[..., 2:])
                 - torch.minimum(pred[..., :2], target[..., :2])).clamp(min=0).prod(-1)
    return iou - (enclosing - union) / (enclosing + 1e-7)


def reference_grid(h, w, device):
    v, u = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack(((u.flatten().float() + .5) / w, (v.flatten().float() + .5) / h), -1)


def decode(raw):
    b, c, h, w = raw.shape
    if c != 4:
        raise ValueError("PDS boxes must have four channels")
    raw = raw.float().flatten(2).transpose(1, 2)
    points = reference_grid(h, w, raw.device)
    return finite(torch.cat((points[None] + .1 * raw[..., :2], raw[..., 2:].sigmoid()), -1),
                  "decoded_boxes"), points


@torch.no_grad()
def labels(batch):
    """Copy augmented normalized cxcywh GT, clip to canvas, count invalid boxes."""
    for key in ("img", "batch_idx", "cls", "bboxes"):
        if key not in batch or not isinstance(batch[key], torch.Tensor):
            raise ValueError(f"PDS missing tensor batch.{key}")
    image, idx, cls, boxes = (batch[k] for k in ("img", "batch_idx", "cls", "bboxes"))
    if image.ndim != 4 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("PDS expects BCHW image and Nx4 normalized cxcywh bboxes")
    n = len(boxes)
    if idx.shape not in ((n,), (n, 1)) or cls.shape not in ((n,), (n, 1)):
        raise ValueError("PDS batch_idx/cls lengths or shapes invalid")
    idx, cls = idx.flatten(), cls.flatten()
    if (not torch.isfinite(idx).all() or not torch.isfinite(cls).all()
            or (idx != idx.round()).any() or (idx < 0).any()
            or (idx >= len(image)).any() or (cls != 0).any()):
        raise ValueError("PDS requires integral in-batch indices and nc=1, cls=0")
    # The audited RTDETR loader supplies normalized xywh; do not invent a numeric
    # coordinate cutoff that would stop valid clipping/skipping of malformed GT.
    if (not boxes.is_floating_point() or batch.get("bbox_format", "xywh") not in ("xywh", "cxcywh")
            or batch.get("normalized", True) is not True):
        raise ValueError("PDS requires floating normalized cxcywh labels from the native loader")
    boxes, idx = boxes.detach().float().to(image.device).clone(), idx.to(image.device)
    h, w = image.shape[-2:]
    scale = boxes.new_tensor([w, h, w, h])
    pixel = xywh_to_xyxy(boxes) * scale
    pixel[:, 0::2].clamp_(0, w)
    pixel[:, 1::2].clamp_(0, h)
    valid = (torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 0).all(-1)
             & (pixel[:, 2:] > pixel[:, :2]).all(-1))
    result = []
    for b in range(len(image)):
        mask = idx == b
        keep = mask & valid
        result.append(dict(pixel=pixel[keep], xywh=xyxy_to_xywh(pixel[keep] / scale),
                           original_index=torch.where(keep)[0], invalid=int((mask & ~valid).sum())))
    return result


def inside(points, box):
    return ((points >= box[:2]) & (points < box[2:])).all(-1)


@torch.no_grad()
def assign(logits, boxes, points, gt, height, width, grid_h, grid_w):
    """Deterministic capped ranking and four-round greedy allocation for one image."""
    pixel = points.detach() * points.new_tensor([width, height])
    sx, sy = width / grid_w, height / grid_h
    pools, fallback, counts = [], [], []
    exclude = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    pred_xyxy = xywh_to_xyxy(boxes.detach())
    scale = boxes.new_tensor([width, height, width, height])
    for box in gt["pixel"]:
        ids = torch.where(inside(pixel, box))[0]
        counts.append(len(ids))
        fallback.append(len(ids) == 0)
        if len(ids) == 0:
            center, wh = (box[:2] + box[2:]) / 2, box[2:] - box[:2]
            denom = torch.maximum(wh, wh.new_tensor([sx, sy]))
            ids = (((pixel - center) / denom).square().sum(-1)).argmin().reshape(1)
        elif len(ids) > 128:
            positions = torch.arange(128, device=ids.device) * (len(ids) - 1) // 127
            ids = ids[positions]
        quality = logits.detach()[ids].sigmoid().sqrt() * (overlap(pred_xyxy[ids], box / scale) + 1e-6).square()
        finite(quality, "assignment.quality")
        # At most 128 items/GT: exact tie rule on torch 2.1 and 2.7; no pixel loop.
        ranked = sorted(zip(quality.cpu().tolist(), ids.cpu().tolist()), key=lambda q: (-q[0], q[1]))
        pools.append([i for _, i in ranked])
        wh = box[2:] - box[:2]
        margin = torch.maximum(.1 * wh, wh.new_tensor([sx, sy]))
        expanded = torch.cat((box[:2] - margin, box[2:] + margin))
        expanded[0::2].clamp_(0, width)
        expanded[1::2].clamp_(0, height)
        exclude |= inside(pixel, expanded)
    area = (gt["pixel"][:, 2:] - gt["pixel"][:, :2]).prod(-1).cpu().tolist()
    original = gt["original_index"].cpu().tolist()
    order = sorted(range(len(pools)), key=lambda j: (len(pools[j]), area[j], original[j]))
    used, selected = set(), [[] for _ in pools]
    for _ in range(4):
        for j in order:
            point = next((i for i in pools[j] if i not in used), None)
            if point is not None:
                selected[j].append(point)
                used.add(point)
    positive = torch.zeros_like(exclude)
    if used:
        positive[list(used)] = True
    stats = dict(gt=len(pools), invalid_gt=gt["invalid"], matched_gt=sum(bool(s) for s in selected),
                 unmatched_gt=sum(not s for s in selected), positives=len(used), fallback=sum(fallback),
                 inside_covered_gt=sum(c > 0 for c in counts), candidates=sum(map(len, pools)),
                 contested_points=sum(map(len, pools)) - len(set(i for p in pools for i in p)),
                 negatives=int((~exclude & ~positive).sum()), ignored=int((exclude & ~positive).sum()))
    return dict(selected=selected, negative=~exclude & ~positive, excluded=exclude, pools=pools,
                fallback=fallback, inside_counts=counts, stats=stats)


def dense_loss(logits, raw, batch, details=False):
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        boxes, points = decode(raw)
        hq, wq = raw.shape[-2:]
        h, w = batch["img"].shape[-2:]
        targets = labels(batch)
        if logits.shape != (len(targets), 1, hq, wq):
            raise ValueError("PDS logits/batch/grid mismatch")
        zs = logits.flatten(1)
        rows, diagnostics = [], []
        for b, gt in enumerate(targets):
            match = assign(zs[b], boxes[b], points, gt, h, w, hq, wq)
            zero = zs[b].sum() * 0  # no artificial regression gradient
            cls_pos, l1, giou = zero, zero, zero
            for j, indices in enumerate(match["selected"]):
                if not indices:
                    continue
                z, pred = zs[b, indices], boxes[b, indices]
                cls_pos = cls_pos + (.25 * (1 - z.sigmoid()).square() * F.softplus(-z)).mean()
                l1 = l1 + (pred - gt["xywh"][j]).abs().sum(-1).mean()
                giou = giou + (1 - overlap(xywh_to_xyxy(pred), xywh_to_xyxy(gt["xywh"][j]), True)).mean()
            neg = zs[b, match["negative"]]
            negative = (.75 * neg.sigmoid().square() * F.softplus(neg)).sum()
            m = max(match["stats"]["matched_gt"], 1)
            rows.append(torch.stack(((cls_pos + .25 * negative) / m, l1 / m, giou / m)))
            diagnostics.append(match if details else match["stats"])
        terms = finite(torch.stack(rows).mean(0), "loss.terms")
        loss = finite(terms[0] + 5 * terms[1] + 2 * terms[2], "loss.total")
        stats = dict(cls=float(terms[0].detach()), l1=float(terms[1].detach()),
                     giou=float(terms[2].detach()), raw=float(loss.detach()),
                     grid=[hq, wq], images=len(targets), assignment=diagnostics)
        return loss, stats
