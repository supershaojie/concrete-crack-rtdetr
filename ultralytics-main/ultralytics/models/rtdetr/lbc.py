"""LBC v1: training-only local background-referenced feature supervision.

The public predict method and native RT-DETR criterion are inherited unchanged.
No live activation is retained on a module, hook, or global variable.
"""
from __future__ import annotations

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.tasks import RTDETRDetectionModel

LBC_CONFIG = dict(version="lbc_v1", feature_layer=5, channels=128, rank=32, seed=424001,
                  eps=1e-6, margin=0.20, temperature=0.20, beta=0.25, weight=0.05,
                  delay=5, ramp=15, candidates=128, topk=4, min_background=16)
HEAD_KEYS = {"lbc_head.proj.weight", "lbc_head.prototype"}


def require(ok, message):
    if not ok:
        raise ValueError("LBC v1: " + message)


def ramp(epoch):
    return max(0.0, min(1.0, (int(epoch) - 5) / 15.0))


class LBCHead(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(424001)
            self.proj = nn.Conv2d(128, 32, 1, bias=False)
            self.prototype = nn.Parameter(torch.randn(32))

    def forward(self, f3):
        require(f3.ndim == 4 and f3.shape[1] == 128, "missing/incorrect backbone S3 feature")
        with torch.autocast(device_type=f3.device.type, enabled=False):
            require(torch.isfinite(f3).all(), "nonfinite S3 input")
            require(torch.isfinite(self.proj.weight).all() and torch.isfinite(self.prototype).all(),
                    "nonfinite auxiliary parameters")
            q = self.proj(f3.float())
            require(torch.isfinite(q).all(), "nonfinite projected feature")
            q = F.normalize(q, p=2, dim=1, eps=1e-6)
            v = F.normalize(self.prototype, p=2, dim=0, eps=1e-6)
            z = (q * v[None, :, None, None]).sum(1)
            require(torch.isfinite(z).all(), "nonfinite evidence score")
            return z


def validate_batch(batch):
    require(all(k in batch for k in ("img", "bboxes", "cls", "batch_idx")), "missing batch labels/image")
    img, boxes, cls, idx = (batch[k] for k in ("img", "bboxes", "cls", "batch_idx"))
    require(all(isinstance(t, torch.Tensor) for t in (img, boxes, cls, idx)), "batch values must be tensors")
    require(img.ndim == 4 and img.shape[0] > 0, "expected BCHW images")
    require(boxes.ndim == 2 and boxes.shape[1] == 4, "expected normalized cxcywh [N,4]")
    require(cls.shape in ((len(boxes),), (len(boxes), 1)) and idx.shape == (len(boxes),),
            "inconsistent cls/batch_idx shapes")
    require(torch.isfinite(idx).all() and torch.isfinite(cls).all(), "nonfinite batch_idx/cls")
    require((idx == idx.long()).all() and ((idx >= 0) & (idx < len(img))).all(), "invalid image indices")
    require((cls == 0).all(), "only nc=1 class zero is supported")


def sample_indices(indices):
    """Unique, deterministic, ascending row-major subsampling; consumes no RNG."""
    if indices.numel() <= 128:
        return indices
    positions = torch.div(torch.arange(128, device=indices.device) * (indices.numel() - 1),
                          127, rounding_mode="floor")
    return indices[positions]


@torch.no_grad()
def regions(batch, feature_hw):
    validate_batch(batch)
    img = batch["img"]
    height, width = img.shape[-2:]
    hf, wf = map(int, feature_hw)
    require(hf > 0 and wf > 0, "empty S3 grid")
    sx, sy = width / wf, height / hf
    boxes = batch["bboxes"].detach().to(img.device, dtype=torch.float32)
    idx = batch["batch_idx"].detach().to(img.device, dtype=torch.long)
    xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)
    xyxy *= boxes.new_tensor([width, height, width, height])
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, height)
    valid = torch.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1) & (xyxy[:, 2:] > xyxy[:, :2]).all(1)
    xs = (torch.arange(wf, device=img.device, dtype=torch.float32) + .5) * sx
    ys = (torch.arange(hf, device=img.device, dtype=torch.float32) + .5) * sy
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x, y = xx.flatten(), yy.flatten()

    def inside(box):
        return (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])

    def expand(box, cells, factor):
        wh = box[2:] - box[:2]
        delta = torch.maximum(wh * factor, wh.new_tensor([cells * sx, cells * sy]))
        out = torch.cat((box[:2] - delta, box[2:] + delta))
        out[[0, 2]] = out[[0, 2]].clamp(0, width)
        out[[1, 3]] = out[[1, 3]].clamp(0, height)
        return out

    stats = dict(gt=len(boxes), valid_gt=int(valid.sum()), invalid_box=int((~valid).sum()),
                 empty_positive=0, insufficient_background=0, pairs=0)
    pairs, geometry = [], []
    for image in range(len(img)):
        ids = torch.where(valid & (idx == image))[0]
        exclusion = torch.zeros(hf * wf, device=img.device, dtype=torch.bool)
        expanded = []
        for j in ids:
            e = expand(xyxy[j], 1, .1)
            exclusion |= inside(e)
            expanded.append(e)
        for j, exclusion_box in zip(ids, expanded):
            box = xyxy[j]
            p = torch.where(inside(box))[0]
            outer = expand(box, 4, .5)
            n = torch.where(inside(outer) & ~exclusion)[0]
            reason = None
            if p.numel() == 0:
                stats["empty_positive"] += 1
                reason = "empty_positive"
            elif n.numel() < 16:
                stats["insufficient_background"] += 1
                reason = "insufficient_background"
            row = dict(image=image, gt_index=int(j), box=box, exclusion=exclusion_box, outer=outer,
                       positive_count=p.numel(), negative_count=n.numel(), skip=reason)
            if reason is None:
                ps, ns = sample_indices(p), sample_indices(n)
                row.update(positive=ps, negative=ns, k=min(4, ps.numel(), ns.numel()))
                pairs.append(row)
            geometry.append(row)
    stats["pairs"] = len(pairs)
    return pairs, stats, geometry


def score_loss(z, pairs, capture=False):
    require(torch.isfinite(z).all(), "nonfinite scores")
    require(bool(pairs), "score_loss requires a nonempty pair list")
    with torch.autocast(device_type=z.device.type, enabled=False):
        positives, negatives, points = [], [], []
        for row in pairs:
            values = z[row["image"]].float().flatten()
            pos_top = values[row["positive"]].topk(row["k"])
            neg_top = values[row["negative"]].topk(row["k"])
            positives.append(pos_top.values.mean())
            negatives.append(neg_top.values.mean())
            if capture:
                points.append(dict(image=row['image'], gt_index=row['gt_index'],
                    positive=row['positive'][pos_top.indices].tolist(), negative=row['negative'][neg_top.indices].tolist()))
        pos, neg = torch.stack(positives), torch.stack(negatives)
        pair = F.softplus((.20 + neg - pos) / .20)
        bg = F.softplus(neg / .20)
        total = (pair + .25 * bg).mean()
        require(torch.isfinite(total), "nonfinite auxiliary loss")
        stats = dict(raw_pair=float(pair.detach().mean()), raw_bg=float(bg.detach().mean()),
                     raw_total=float(total.detach()), s_pos=float(pos.detach().mean()),
                     s_neg=float(neg.detach().mean()), gap=float((pos-neg).detach().mean()),
                     positive_candidates=sum(r["positive"].numel() for r in pairs),
                     negative_candidates=sum(r["negative"].numel() for r in pairs),
                     positive_positions=sum(r["positive_count"] for r in pairs),
                     negative_positions=sum(r["negative_count"] for r in pairs))
        if capture:
            stats['selected_points'] = points
        return total, stats


class LBCDetectionModel(RTDETRDetectionModel):
    """Explicit stable subclass built from an already audited native model."""
    def __init__(self, native):
        require(type(native) is RTDETRDetectionModel, "audit a native model before wrapping")
        nn.Module.__init__(self)
        # Copy the native Module state without nesting it or renaming its keys.
        self.__dict__.update(deepcopy(native.__dict__))
        self.lbc_head = LBCHead()
        self.lbc_config = deepcopy(LBC_CONFIG)
        self.lbc_epoch = 0
        self.lbc_last = {}

    def predict_with_s3(self, x, targets):
        y, f3 = [], None
        for m in self.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i == 5:
                f3 = x
        require(f3 is not None and f3.shape[1] == 128, "S3 traversal wiring missing")
        head = self.model[-1]
        return head([y[j] for j in head.f], targets), f3

    def loss(self, batch, preds=None):
        # Validator passes predictions: report only native L0, never another forward.
        if not self.training or preds is not None:
            return super().loss(batch, preds)
        r = ramp(self.lbc_epoch)
        self.lbc_last = dict(epoch=self.lbc_epoch, ramp=r, weight=.05*r, pairs=0, disabled=r == 0)
        if r == 0:
            result = super().loss(batch, preds)
            require(torch.isfinite(result[0]).all(), "nonfinite native L0")
            return result
        validate_batch(batch)
        img, idx = batch["img"], batch["batch_idx"]
        targets = dict(cls=batch["cls"].to(img.device, dtype=torch.long).view(-1),
                       bboxes=batch["bboxes"].to(img.device), batch_idx=idx.to(img.device, dtype=torch.long).view(-1),
                       gt_groups=[(idx == i).sum().item() for i in range(len(img))])
        original, f3 = self.predict_with_s3(img, targets)
        l0, items = super().loss(batch, original)
        require(torch.isfinite(l0).all(), "nonfinite native L0")
        require(torch.isfinite(f3).all(), "nonfinite S3 input")
        pairs, stats, _ = regions(batch, f3.shape[-2:])
        self.lbc_last.update(stats, feature_hw=list(f3.shape[-2:]))
        if not pairs:
            return l0, items  # Do not materialize head gradients or erase accumulated ones.
        raw, stats = score_loss(self.lbc_head(f3), pairs, capture=getattr(self, 'lbc_capture_points', False))
        self.lbc_last.update(stats, weighted=float(raw.detach()) * .05*r)
        return l0 + .05 * r * raw, items
