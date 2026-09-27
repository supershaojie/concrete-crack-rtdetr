"""RMD v1: detached residual matching of the original positive DN box loss.

No model parameter, decoder operation, RNG draw, or inference interface is added.
Reference tensors live only in the current predict return's private DN metadata.
"""
from __future__ import annotations

from copy import deepcopy
import inspect

import torch

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.loss import DETRLoss, RTDETRDetectionLoss
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.torch_utils import unwrap_model

FORMULA = "rmd_v1_eta0.50_tau0.50_eps1e-7"
ETA, TAU, EPS = 0.50, 0.50, 1e-7
CONTEXT = "_rmd_v1_initial_reference"


def require(condition, message):
    if not condition:
        raise RuntimeError("RMD v1: " + message)


def ramp(epoch):
    """epoch is the number of completed epochs, never a saved ramp value."""
    return max(0.0, min(1.0, (int(epoch) - 5) / 15.0))


def finite_boxes(boxes, name):
    require(bool(torch.isfinite(boxes).all()), name + " contains NaN/Inf")
    require(bool((boxes[..., 2:] > 0).all()), name + " contains nonpositive width/height")


@torch.no_grad()
def centered_weights(similarity, identities, r):
    """identities are global GT identities, already disambiguated by image."""
    with torch.autocast(device_type=similarity.device.type, enabled=False):
        s = similarity.detach().float()
        require(s.ndim == 1 and identities.shape == s.shape, "similarity/identity shape mismatch")
        require(bool(torch.isfinite(s).all()) and bool(((s >= 0) & (s <= 1)).all()), "invalid similarity")
        if not s.numel():
            return s.clone()
        _, inverse, counts = torch.unique(identities, return_inverse=True, return_counts=True)
        sums = s.new_zeros(counts.numel()).scatter_add_(0, inverse, s)
        return 1.0 + ETA * r * (s - (sums / counts)[inverse])


def distribution(values):
    x = values.detach().float().cpu()
    if not x.numel():
        return {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": None, "max": None, "quantiles": None}
    require(bool(torch.isfinite(x).all()), "nonfinite diagnostic values")
    return dict(count=x.numel(), sum=float(x.sum()), sum_sq=float(x.square().sum()),
                min=float(x.min()), max=float(x.max()),
                quantiles=torch.quantile(x, x.new_tensor([0, .25, .5, .75, 1])).tolist())


@torch.no_grad()
def residual_weights(initial, batch, normal_match, dn_match, dn_meta, r):
    """Return weights in DETRLoss._get_index(dn_match) order, plus scalar evidence."""
    with torch.autocast(device_type=initial.device.type, enabled=False):
        a = initial.detach().float()
        gt = batch["bboxes"].detach().float()
        groups = batch["gt_groups"]
        split, k = dn_meta["dn_num_split"], int(dn_meta["dn_num_group"])
        require(k >= 1 and len(split) == 2 and split[1] == 300, "invalid native DN split/group")
        require(a.shape == (len(groups), sum(split), 4), "captured decoder input does not match DN split")
        require(len(normal_match) == len(dn_match) == len(groups), "image index count mismatch")
        require(sum(groups) == len(gt), "GT group total mismatch")
        parts, s_parts, ids, unmatched = [], [], [], 0
        offset = 0
        for image, (n, normal, dn) in enumerate(zip(groups, normal_match, dn_match)):
            q, g = (t.to(a.device, dtype=torch.long) for t in normal)
            p, dg = (t.to(a.device, dtype=torch.long) for t in dn)
            require(len(p) == n * k and len(p) == len(dg), f"positive DN count mismatch image={image}")
            require(len(p.unique()) == len(p), f"duplicate positive DN index image={image}")
            require(bool(((p >= 0) & (p < split[0])).all()), "positive DN index out of range")
            require(len(q) == len(g) and len(g.unique()) == len(g) and len(q.unique()) == len(q), "invalid ordinary matching")
            require(bool(((q >= 0) & (q < split[1])).all()), "ordinary query index out of range")
            require(bool(((g >= offset) & (g < offset + n)).all()), "ordinary GT crossed image identity")
            require(bool(((dg >= offset) & (dg < offset + n)).all()), "DN GT crossed image identity")
            if n:
                counts = torch.bincount(dg - offset, minlength=n)
                require(bool((counts == k).all()), "positive DN does not repeat each GT K times")
                finite_boxes(gt[offset:offset + n], f"GT image={image}")
                require(bool((batch["batch_idx"][offset:offset+n] == image).all()), "GT groups disagree with native image identity")
                finite_boxes(a[image, p], f"positive initial DN image={image}")
            lookup = torch.full((n,), -1, dtype=torch.long, device=a.device)
            lookup[g - offset] = q
            mapped = lookup[dg - offset]
            valid = mapped >= 0
            weights = a.new_ones(len(p))
            similarities = a.new_zeros(int(valid.sum()))
            if valid.any():
                truth, noisy = gt[dg[valid]], a[image, p[valid]]
                ordinary = a[image, split[0] + mapped[valid]]
                finite_boxes(noisy, f"positive initial DN image={image}")
                finite_boxes(ordinary, f"matched initial ordinary image={image}")

                def rho(box):
                    return torch.cat(((box[:, :2] - truth[:, :2]) / (truth[:, 2:] + EPS),
                                      torch.log((box[:, 2:] + EPS) / (truth[:, 2:] + EPS))), -1)

                residual = rho(noisy) - rho(ordinary)
                require(bool(torch.isfinite(residual).all()), "nonfinite signed residual")
                similarities = torch.exp(-residual.square().sum(-1) / (2 * TAU**2))
                weights[valid] = centered_weights(similarities, dg[valid], r)
            # Unmatched GTs are explicitly kept at one, never rematched.
            unmatched += n - len(g)
            parts.append(weights)
            s_parts.append(similarities)
            ids.append(dg)
            offset += n
        w = torch.cat(parts)
        s = torch.cat(s_parts)
        require(bool(torch.isfinite(w).all()) and bool(((w >= .5) & (w <= 1.5)).all()), "weight bound violated")
        delta = (w - 1).abs()
        stats = dict(gt_groups=list(groups), gt_occurrences=sum(groups), K=k, positives=len(w),
                     unmatched_gt=unmatched, similarity=distribution(s), weight=distribution(w),
                     abs_delta_sum=float(delta.sum()), abs_delta_ge_001=int((delta >= .01).sum()),
                     mean_abs_delta=float(delta.mean()) if len(w) else 0.0,
                     std_weight=float(w.std(unbiased=False)) if len(w) else 0.0,
                     fraction_abs_delta_ge_001=float((delta >= .01).float().mean()) if len(w) else 0.0)
        return w.detach(), stats


def weighted_box_loss(pred, truth, weights, gains):
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, g, w = pred.float(), truth.float(), weights.detach().float()
        finite_boxes(p, "DN regression prediction")
        finite_boxes(g, "DN regression GT")
        require(len(p) == len(g) == len(w), "regression weight alignment mismatch")
        n = max(len(w), 1)
        return {"loss_bbox_dn": gains["bbox"] * (w * (p - g).abs().sum(-1)).sum() / n,
                "loss_giou_dn": gains["giou"] * (w * (1 - bbox_iou(p, g, xywh=True, GIoU=True).squeeze(-1))).sum() / n}


class RMDLoss(RTDETRDetectionLoss):
    """Keep all native classification/matching terms; replace positive DN box numerators."""
    epoch = 0
    enabled = False
    expect_dn = True

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
        result = super()._get_loss_bbox(pred_bboxes, gt_bboxes, postfix)
        if postfix == "_dn" and hasattr(self, "_layer_log"):
            self._layer_log.append({"original": {k: float(v.detach()) for k, v in result.items()}})
        return result

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        r = ramp(self.epoch) if self.enabled else 0.0
        self._layer_log = []  # scalar diagnostics only; removed even if an exception occurs
        self.statistics = None
        try:
            active = r > 0 and sum(batch["gt_groups"]) > 0 and self.expect_dn
            if active:
                require(dn_meta is not None and dn_bboxes is not None and dn_scores is not None,
                        "expected DN metadata/predictions missing during active training")
                require(CONTEXT in dn_meta, "initial-reference capture missing during active training")
            if not active:
                total = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
                k = int(dn_meta["dn_num_group"]) if dn_meta is not None else 0
                stats = dict(gt_groups=list(batch["gt_groups"]), gt_occurrences=sum(batch["gt_groups"]),
                             K=k, positives=sum(batch["gt_groups"]) * k, residuals_computed=False)
            else:
                boxes, scores = preds
                self.device = boxes.device
                gt, cls, groups = batch["bboxes"], batch["cls"], batch["gt_groups"]
                # Only the final, already CBR-refined ordinary layer reuses this match.
                final_match = self.matcher(boxes[-1], scores[-1], gt, cls, groups)
                total = self._get_loss(boxes[-1], scores[-1], gt, cls, groups, match_indices=final_match)
                if self.aux_loss:
                    total.update(self._get_loss_aux(boxes[:-1], scores[:-1], gt, cls, groups))
                dn_match = self.get_dn_match_indices(dn_meta["dn_pos_idx"], dn_meta["dn_num_group"], groups)
                require(dn_bboxes.shape[:3] == dn_scores.shape[:3] and
                        dn_bboxes.shape[2] == dn_meta["dn_num_split"][0], "DN prediction split mismatch")
                w, stats = residual_weights(dn_meta[CONTEXT], batch, final_match, dn_match, dn_meta, r)
                if bool((w == 1).all()):
                    dn_loss = DETRLoss.forward(self, dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=dn_match)
                else:
                    idx, gt_idx = self._get_index(dn_match)

                    def layer(index):
                        # Native soft-IoU VFL labels/detach and native negative DN classification.
                        row = self._get_loss(dn_bboxes[index], dn_scores[index], gt, cls, groups,
                                             postfix="_dn", match_indices=dn_match)
                        replacement = weighted_box_loss(dn_bboxes[index][idx], gt[gt_idx], w, self.loss_gain)
                        with torch.no_grad():
                            fp32_unit = weighted_box_loss(dn_bboxes[index][idx], gt[gt_idx], torch.ones_like(w), self.loss_gain)
                        self._layer_log[-1]["fp32_unit_minus_native"] = {
                            key: float(value-row[key].detach()) for key, value in fp32_unit.items()}
                        row.update(replacement)  # replacement, never add original + new
                        self._layer_log[-1]["weighted"] = {key: float(v.detach()) for key, v in replacement.items()}
                        return row

                    dn_loss = layer(-1)
                    if self.aux_loss:
                        aux = torch.zeros(3, device=self.device)
                        for i in range(len(dn_bboxes) - 1):
                            row = layer(i)
                            for j, key in enumerate(("class", "bbox", "giou")):
                                aux[j] += row[f"loss_{key}_dn"]
                        dn_loss.update({f"loss_{key}_aux_dn": aux[j] for j, key in enumerate(("class", "bbox", "giou"))})
                total.update(dn_loss)
                stats["residuals_computed"] = True
                for key, value in total.items():
                    require(bool(torch.isfinite(value).all()), "nonfinite active training loss: " + key)
            for i, row in enumerate(self._layer_log):
                row["layer"] = "final" if i == 0 else f"aux_{i - 1}"
                row.setdefault("weighted", row["original"].copy())
            self.statistics = dict(stats, epoch=int(self.epoch), r=r, dn_layers=self._layer_log,
                                   ordinary={k: float(v.detach()) for k, v in total.items() if not k.endswith("_dn")},
                                   losses={k: float(v.detach()) for k, v in total.items()},
                                   optimized_total=float(sum(total.values()).detach()))
            return total
        finally:
            del self._layer_log


class RMDDetectionModel(RTDETRDetectionModel):
    """Parameter-free training wrapper; inference executes the original model verbatim."""
    rmd_epoch = 0

    def init_criterion(self):
        return RMDLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        if not isinstance(getattr(self, "criterion", None), RMDLoss):
            self.criterion = self.init_criterion()
        self.criterion.epoch = int(self.rmd_epoch)
        self.criterion.enabled = bool(self.training)
        self.criterion.expect_dn = self.model[-1].num_denoising > 0
        return super().loss(batch, preds)

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        kwargs = dict(profile=profile, visualize=visualize, batch=batch, augment=augment, embed=embed)
        capture = (self.training and ramp(self.rmd_epoch) > 0 and batch is not None and
                   sum(batch["gt_groups"]) > 0 and self.model[-1].num_denoising > 0)
        if not capture:
            return super().predict(x, **kwargs)
        initial = []

        def read_reference(module, args):
            # Mirror decoder's EXACT sigmoid dtype before FP32 residual arithmetic.
            initial.append(args[1].detach().sigmoid())

        handle = self.model[-1].decoder.register_forward_pre_hook(read_reference)
        try:
            result = super().predict(x, **kwargs)
            require(len(initial) == 1 and len(result) == 5, "expected one original decoder forward")
            require(result[-1] is not None, "expected original DN metadata")
            return (*result[:-1], {**result[-1], CONTEXT: initial[0]})
        finally:
            handle.remove()
            initial.clear()


def promote(model):
    require(type(model) in (RTDETRDetectionModel, RMDDetectionModel), "unexpected model class")
    model.__class__ = RMDDetectionModel
    model.rmd_epoch = 0
    if hasattr(model, "criterion"):
        del model.criterion
    return model


def sync_epoch(trainer):
    model = unwrap_model(trainer.model)
    require(isinstance(model, RMDDetectionModel), "Trainer rebuilt a non-RMD model")
    model.rmd_epoch = int(trainer.epoch)
    if trainer.ema is not None:
        trainer.ema.ema.rmd_epoch = int(trainer.epoch)


class RMDTrainer(RTDETRTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", sync_epoch)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Fork only constructor RNG for the native comparison, not a training forward.
        with torch.random.fork_rng(devices=[]):
            baseline = super().get_model(deepcopy(cfg), weights, verbose=False)
        model = super().get_model(cfg, weights, verbose=verbose)
        before = baseline.state_dict()
        require(set(before) == set(model.state_dict()), "native reconstruction state keys differ")
        require(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()), "native reconstruction values differ")
        self.rmd_rebuild_audit = dict(native_states=len(before), all_values_exact=True,
                                      parameters=sum(p.numel() for p in model.parameters()),
                                      trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                      added_parameters=0, criterion_source=inspect.getfile(RMDLoss))
        return promote(model)
