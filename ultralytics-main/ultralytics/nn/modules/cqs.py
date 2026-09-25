# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""CQS v1: preserve native top-k core, then select complementary ordinary queries."""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .cbr import RTDETRDecoderCBR


CQS_CONFIG = dict(method="cqs_v1", enabled=True, num_queries=300, candidate_pool=600,
                  core_queries=240, complement_queries=60, beta=0.5, iou_power=2,
                  positive_cosine_power=2, feature_norm_eps=1e-12, iou_union_eps=1e-12,
                  selection_precision="float32")


def finite_or_raise(**tensors):
    """A single batch check; infinity *anchor logits* are intentionally not inspected."""
    checks = torch.stack([torch.isfinite(v).all() for v in tensors.values()])
    if not bool(checks.all()):
        context = {k: dict(shape=list(v.shape), dtype=str(v.dtype), device=str(v.device),
                           nonfinite=int((~torch.isfinite(v)).sum())) for k, v in tensors.items()}
        raise FloatingPointError(f"CQS nonfinite selection inputs: {context}")


def candidate_pool(logits, num_queries=300, pool_size=600):
    """Native dtype/topk/order first; explicitly remove those indices before extra topk."""
    if logits.ndim != 3 or logits.shape[-1] != 1:
        raise ValueError("CQS v1 requires nc=1 encoder logits [B,N,1]")
    n = logits.shape[1]
    if n < num_queries:
        raise ValueError(f"CQS requires N >= {num_queries}; received N={n}")
    finite_or_raise(encoder_logits=logits)
    scores = logits.max(-1).values  # identical operation and dtype to the parent
    native = torch.topk(scores, num_queries, dim=1).indices
    excluded = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, native, True)
    count = min(pool_size - num_queries, n - num_queries)
    # -inf is an exact exclusion, not a finite heuristic score.
    extra = torch.topk(scores.masked_fill(excluded, -torch.inf), count, dim=1).indices
    return native, torch.cat((native, extra), dim=1)


def pairwise_relation(features, boxes):
    """Detached FP32 ordinary IoU squared times positive cosine squared, per pair."""
    with torch.no_grad(), torch.autocast(device_type=features.device.type, enabled=False):
        f, b = features.detach().float(), boxes.detach().float()
        finite_or_raise(features=f, decoded_boxes=b)
        u = F.normalize(f, p=2, dim=-1, eps=1e-12)
        cosine = torch.bmm(u, u.transpose(1, 2)).clamp(-1, 1).clamp_min(0)
        low, high = b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2
        size = (torch.minimum(high[:, :, None], high[:, None, :]) -
                torch.maximum(low[:, :, None], low[:, None, :])).clamp_min(0)
        intersection = size[..., 0] * size[..., 1]
        wh = (high - low).clamp_min(0)
        area = wh[..., 0] * wh[..., 1]
        union = area[:, :, None] + area[:, None, :] - intersection
        iou = (intersection / union.clamp_min(1e-12)).clamp(0, 1)
        relation = iou.square() * cosine.square()
        finite_or_raise(relation=relation)
        return relation


def greedy_select(scores, relation, core=240, queries=300, beta=0.5):
    """Exact batched sequential greedy; argmax chooses the first pool index on ties."""
    if not 0 < core <= queries <= scores.shape[1] or not 0 <= beta <= 1:
        raise ValueError("Invalid CQS diagnostic selection dimensions/beta")
    with torch.no_grad(), torch.autocast(device_type=scores.device.type, enabled=False):
        scores, relation = scores.detach().float(), relation.detach().float()
        finite_or_raise(scores=scores, relation=relation)
        bs, n = scores.shape
        chosen = torch.zeros_like(scores, dtype=torch.bool)
        chosen[:, :core] = True
        redundancy = relation[:, :, :core].amax(-1)
        ids = [torch.arange(core, device=scores.device).expand(bs, -1)]
        discounts = []
        for _ in range(queries - core):
            factor = 1 - beta * redundancy
            index = (scores * factor).masked_fill(chosen, -torch.inf).argmax(dim=1, keepdim=True)
            ids.append(index)
            discounts.append(factor.gather(1, index))
            chosen.scatter_(1, index, True)
            update = relation.gather(2, index[:, None, :].expand(bs, n, 1)).squeeze(-1)
            redundancy = torch.maximum(redundancy, update)
        factors = torch.cat(discounts, 1) if discounts else scores[:, :0]
        return torch.cat(ids, 1), factors


def gather_positions(values, indices):
    return values.gather(1, indices[..., None].expand(-1, -1, values.shape[-1]))


class RTDETRDecoderCBRCQS(RTDETRDecoderCBR):
    """Unmodified CBR/DN/decoder forward, with an always-on ordinary query selector.

    No added parameters or buffers. The fixed configuration is pickled with the
    model, and class construction restores it when rebuilding from the new YAML.
    nc80 construction is allowed for the parent's controlled ImageNet source;
    every actual CQS selection requires nc1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cqs_config = dict(CQS_CONFIG)
        self.cqs_calls = 0
        self._cqs_sink = None

    def __getstate__(self):
        state = super().__getstate__().copy()
        state['_cqs_sink'] = None  # never serialize diagnostic activations/consumers
        return state

    @contextmanager
    def capture_selection(self, full=False):
        """Explicit bounded diagnostics; full tensors only while inside this context."""
        old, rows = self._cqs_sink, []
        self._cqs_sink = (rows, full)
        try:
            yield rows
        finally:
            self._cqs_sink = old

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        if self.nc != 1:
            raise ValueError(f"CQS v1 selection requires nc=1, received {self.nc}")
        cfg = self.cqs_config
        if not cfg['enabled'] or cfg['beta'] == 0:
            # Strict degradation: no candidate extension, no FP32 re-ranking.
            return super()._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        if cfg != CQS_CONFIG or self.num_queries != 300:
            raise ValueError(f"CQS v1 fixed research configuration changed: {cfg}")
        bs = feats.shape[0]
        if feats.shape[1] < self.num_queries:
            raise ValueError(f"CQS requires N >= 300; received N={feats.shape[1]}")
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes
        features = self.enc_output(self.valid_mask * feats)
        logits = self.enc_score_head(features)
        native, pool = candidate_pool(logits)
        pool_features = gather_positions(features, pool)
        pool_anchors = gather_positions(self.anchors.expand(bs, -1, -1), pool)
        # The 600-candidate bbox forward stays on the original differentiable AMP path.
        pool_box_logits = self.enc_bbox_head(pool_features) + pool_anchors
        pool_boxes = pool_box_logits.sigmoid()
        pool_logits = gather_positions(logits, pool)
        with torch.no_grad(), torch.autocast(device_type=feats.device.type, enabled=False):
            scores = pool_logits.detach().float().squeeze(-1).sigmoid()
            relation = pairwise_relation(pool_features, pool_boxes)
            selected, factors = greedy_select(scores, relation)
        ids = pool.gather(1, selected)
        top_features = gather_positions(pool_features, selected)
        refer_bbox = gather_positions(pool_box_logits, selected)
        enc_bboxes = gather_positions(pool_boxes, selected)
        enc_scores = gather_positions(pool_logits, selected)
        self.cqs_calls += 1
        if self._cqs_sink is not None:
            rows, full = self._cqs_sink
            extra = (selected >= self.num_queries).sum(1)
            duplicates = (ids.sort(1).values[:, 1:] == ids.sort(1).values[:, :-1]).sum(1)
            row = dict(pool_size=pool.shape[1], core_queries=240, complement_queries=60,
                       extra_count=extra.detach(), overlap=(300-extra).float()/300,
                       discount_factors=factors.detach(), duplicates=duplicates.detach(),
                       nonfinite=0, representative_indices=ids[:2, -8:].detach())
            if full:
                row.update(native=native.detach(), pool=pool.detach(), selected=selected.detach(),
                           indices=ids.detach(), features=pool_features.detach(),
                           anchors=pool_anchors.detach(), box_logits=pool_box_logits.detach(),
                           boxes=pool_boxes.detach(), logits=pool_logits.detach(), relation=relation,
                           shapes=shapes)
            rows.append(row)
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
        return embeddings, refer_bbox, enc_bboxes, enc_scores
