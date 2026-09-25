"""BMC v1: training-only, per-GT budget, whole alternating-component assignment.

No trainable state; the native matcher owns all cost arithmetic and SciPy calls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import torch

from .loss import DETRLoss, RTDETRDetectionLoss
from ultralytics.nn.tasks import RTDETRDetectionModel


@dataclass(frozen=True)
class BMCConfig:
    method: str = "bmc_v1"
    enabled: bool = True
    target_decoder_layer: int = 2
    target_loss_slot: int = 2
    source: str = "final_cbr_native_assignment"
    budget_iqr_fraction: float = 0.02
    iqr_floor: float = 0.001
    warmup_epochs: int = 20

    def __post_init__(self):
        if (self.method, self.target_decoder_layer, self.target_loss_slot, self.source) != (
            "bmc_v1", 2, 2, "final_cbr_native_assignment"
        ):
            raise ValueError("BMC v1 layer/source contract changed")
        if self.budget_iqr_fraction < 0 or self.iqr_floor <= 0 or self.warmup_epochs < 0:
            raise ValueError("Invalid BMC budget/floor/warmup")


def _validate(pair, q, g, offset):
    query, target = (x.detach().cpu() for x in pair)
    if query.dtype != torch.long or target.dtype != torch.long or query.ndim != 1 or target.ndim != 1:
        raise ValueError("Matching indices must be one-dimensional int64 tensors")
    if len(query) != min(q, g) or len(target) != len(query):
        raise ValueError(f"Incomplete matching: Q={q}, G={g}, offset={offset}, pair={pair}")
    if (len(query.unique()) != len(query) or len(target.unique()) != len(target)
            or (query < 0).any() or (query >= q).any()
            or (target < offset).any() or (target >= offset + g).any()):
        raise ValueError(f"Invalid/cross-image matching: Q={q}, G={g}, offset={offset}, pair={pair}")
    return query, target - offset


@torch.no_grad()
def bounded_consensus(a2, a3, costs, gt_groups, config=BMCConfig(), *, detail=False):
    """Return loss indices and detached diagnostics; unchanged images retain original objects/order."""
    start = time.perf_counter()
    if not (len(a2) == len(a3) == len(costs) == len(gt_groups)):
        raise ValueError("Matching batch lengths differ")
    output, rows, offset = [], [], 0
    for image, (p2, p3, cost, g) in enumerate(zip(a2, a3, costs, gt_groups)):
        if cost.ndim != 2 or cost.shape[1] != g or cost.requires_grad:
            raise ValueError("Expected detached Q x local-GT native cost")
        c = cost.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(c).all():
            raise ValueError(f"Nonfinite C2 for image={image}, offset={offset}")
        q = c.shape[0]
        i2, j2 = _validate(p2, q, g, offset)
        i3, j3 = _validate(p3, q, g, offset)
        row = dict(image=image, gt_offset=offset, matched=min(q, g), disagreement=0, changed=0,
                   components=0, changed_components=0, background_to_positive=0, target_swap=0,
                   fallback_g_gt_q=int(g > q), ratios=[], events=[])
        if g == 0 or g > q:
            output.append(p2)
            rows.append(row)
            offset += g
            continue
        x2, x3 = torch.empty(g, dtype=torch.long), torch.empty(g, dtype=torch.long)
        x2[j2], x3[j3] = i2, i3
        quartiles = torch.quantile(c, torch.tensor([.25, .75]), dim=0, interpolation="linear")
        scale = (quartiles[1] - quartiles[0]).clamp_min(config.iqr_floor)
        delta = c[x3, torch.arange(g)] - c[x2, torch.arange(g)]
        allowed = delta <= config.budget_iqr_fraction * scale
        parent = list(range(q + g))  # query 0..Q-1, GT Q..Q+G-1

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for gt in range(g):
            for query in (int(x2[gt]), int(x3[gt])):
                a, b = root(query), root(q + gt)
                parent[max(a, b)] = min(a, b)
        groups = {}
        for gt in range(g):
            groups.setdefault(root(q + gt), []).append(gt)
        result = x2.clone()
        for component in groups.values():
            accept = bool(allowed[component].all())
            changed = [gt for gt in component if x2[gt] != x3[gt]]
            if accept:
                result[component] = x3[component]
            row["changed_components"] += int(accept and bool(changed))
            if detail and changed and len(row["events"]) < 8:
                subset = component[:8]
                row["events"].append(dict(size=len(component), accepted=accept, gt=subset,
                    a2=x2[subset].tolist(), a3=x3[subset].tolist(),
                    delta=delta[subset].tolist(), scale=scale[subset].tolist(),
                    max_ratio=float((delta[component] / scale[component]).max())))
        changed_mask = result != x2
        row.update(disagreement=int((x2 != x3).sum()), changed=int(changed_mask.sum()),
                   components=len(groups), ratios=(delta / scale).tolist())
        previous = set(x2.tolist())
        row["background_to_positive"] = sum(int(v) not in previous for v in result[changed_mask])
        row["target_swap"] = row["changed"] - row["background_to_positive"]
        if changed_mask.any():
            if not bool((delta[changed_mask] <= config.budget_iqr_fraction * scale[changed_mask]).all()):
                raise AssertionError("Changed GT exceeded its budget")
            order = result.argsort(stable=True)
            pair = (result[order], torch.arange(g)[order] + offset)
            _validate(pair, q, g, offset)
            output.append(pair)
        else:
            output.append(p2)
        rows.append(row)
        offset += g
    return output, dict(images=rows, consensus_seconds=time.perf_counter() - start)


class BMCLoss(RTDETRDetectionLoss):
    """Only the ordinary loss stack slot 2 may change; DN always uses native indices."""

    def __init__(self, nc=1, config=None, epoch=None):
        super().__init__(nc=nc, use_vfl=True)
        self.config = config or BMCConfig()
        self.epoch = epoch
        self.collector = None
        self.route_observer = None  # optional explicit verification interface, never a global hook
        self.last_report = None
        self._loss_call = 0

    def __getstate__(self):
        state = super().__getstate__().copy()
        for key in ("collector", "route_observer", "last_report"):
            state[key] = None
        return state

    @property
    def active(self):
        return self.training and self.config.enabled and self.epoch is not None and self.epoch >= self.config.warmup_epochs

    def _get_loss(self, *args, **kwargs):
        result = super()._get_loss(*args, **kwargs)
        if not kwargs.get("postfix", ""):
            self._loss_call += 1
            if self._loss_call == 4:  # native call order: final, encoder, decoder1, decoder2
                self._second_loss = {k: v.detach() for k, v in result.items()}
        return result

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        self._loss_call, self._second_loss = 0, {}
        if self.training and self.epoch is None:
            raise RuntimeError("Trainer must set the real zero-based BMC epoch before training loss")
        boxes, scores = preds
        if self.training and "batch_idx" in batch:
            expected = torch.repeat_interleave(torch.arange(len(batch["gt_groups"]), device=boxes.device),
                torch.tensor(batch["gt_groups"], device=boxes.device))
            if not torch.equal(batch["batch_idx"].reshape(-1).to(expected), expected):
                raise ValueError("Targets are not grouped in batch image order; global GT offsets would be wrong")
        if self.training and (boxes.shape[0] != 4 or scores.shape[:3] != boxes.shape[:3] or boxes.shape[2] != 300):
            raise AssertionError("Expected [encoder, decoder1, decoder2, final CBR] x B x 300")
        if not self.active:
            result = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            self.last_report = dict(state="EVAL_NATIVE" if not self.training else
                ("DISABLED_CONFIG" if not self.config.enabled else "DISABLED_WARMUP"), images=[])
            if self.training:
                self.last_report["matched_gt"] = sum(min(boxes.shape[2], g) for g in batch["gt_groups"])
        else:
            self.device = boxes.device
            gt, cls, groups = batch["bboxes"], batch["cls"], batch["gt_groups"]
            # The native main match is computed once and immediately used for the main loss.
            a3, _ = self.matcher(boxes[3], scores[3], gt, cls, groups, return_costs=True)
            if self.route_observer:
                self.route_observer("ordinary_3", a3)
            result = self._get_loss(boxes[3], scores[3], gt, cls, groups, match_indices=a3)
            aux = torch.zeros(3, device=self.device)  # preserve native summation order/dtype
            for slot in range(3):
                if slot == 2:
                    a2, costs = self.matcher(boxes[slot], scores[slot], gt, cls, groups, return_costs=True)
                    indices, self.last_report = bounded_consensus(a2, a3, costs, groups, self.config,
                        detail=bool(self.collector and self.collector.wants_detail))
                    self.last_report["state"] = "ENABLED"
                else:
                    indices = self.matcher(boxes[slot], scores[slot], gt, cls, groups)
                if self.route_observer:
                    self.route_observer(f"ordinary_{slot}", indices)
                layer = self._get_loss(boxes[slot], scores[slot], gt, cls, groups, match_indices=indices)
                for i, key in enumerate(("class", "bbox", "giou")):
                    aux[i] += layer[f"loss_{key}"]
            result.update({f"loss_{key}_aux": aux[i] for i, key in enumerate(("class", "bbox", "giou"))})
            if dn_meta is not None:
                if dn_bboxes.shape[0] != 3 or dn_scores.shape[:3] != dn_bboxes.shape[:3]:
                    raise AssertionError("Expected three independent native DN layers")
                dn_indices = self.get_dn_match_indices(dn_meta["dn_pos_idx"], dn_meta["dn_num_group"], groups)
                if self.route_observer:
                    self.route_observer("dn_all", dn_indices)
                result.update(DETRLoss.forward(self, dn_bboxes, dn_scores, batch,
                                              postfix="_dn", match_indices=dn_indices))
            else:
                result.update({f"{key}_dn": torch.tensor(0., device=self.device) for key in list(result)})
        if self.training and self.collector:
            self.collector.record(self.epoch, self.last_report, self._second_loss)
        return result


class BMCDetectionModel(RTDETRDetectionModel):
    """Native YAML/forward/CBR, with an experiment-owned training criterion."""

    def set_bmc_epoch(self, epoch):
        self.bmc_epoch = int(epoch)
        if hasattr(self, "criterion"):
            self.criterion.epoch = self.bmc_epoch

    def init_criterion(self):
        head = self.model[-1]
        if (len(head.decoder.layers), head.num_queries, head.f) != (3, 300, [19, 22, 25]):
            raise AssertionError("BMC requires the original three-layer CBR/LIF topology")
        return BMCLoss(nc=self.nc, config=getattr(self, "bmc_config", BMCConfig()),
                       epoch=getattr(self, "bmc_epoch", None)).train(self.training)

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        self.criterion.train(self.training)  # validator EMA loss is always native, even at epoch >=20
        return super().loss(batch, preds)
