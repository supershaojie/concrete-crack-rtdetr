# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""PEQ v1: detached local evidence, ten bounded logit residuals, unchanged CBR."""
from __future__ import annotations

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .cbr import RTDETRDecoderCBR

PEQ_CONFIG = dict(
    method="peq_v1", enabled=True, feature_channels=16, query_projection_channels=32,
    grid_size=3, pool_hidden=16, fusion_hidden=64,
    iou_thresholds=[round(.50 + .05 * i, 2) for i in range(10)],
    delta_limit=1.0, quality_loss_weight=1.0, matched_group_weight=.5, unmatched_group_weight=.5,
    geometry_eps=.000001, geometry_delta_clip=2.0, context_logit_clip=10.0,
    sampling_mode="bilinear", padding_mode="border", align_corners=False, extra_epoch_warmup=0,
)


def finite(*tensors):
    if not all(bool(torch.isfinite(t).all()) for t in tensors):
        raise FloatingPointError("PEQ received/produced non-finite values")


def configuration(config=None):
    result = deepcopy(PEQ_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Unknown PEQ fields: {sorted(unknown)}")
        result.update(deepcopy(config))
    if type(result["enabled"]) is not bool or any(
        result[k] != v for k, v in PEQ_CONFIG.items() if k != "enabled"
    ):
        raise ValueError("PEQ v1 has a fixed research contract; only enabled can be changed")
    return result


def ordinary(tensor, dn_meta, dim=1, count=300):
    """Use the actual DN split, including zero-DN and dynamic large-GT batches."""
    if dn_meta is not None:
        split = dn_meta["dn_num_split"]
        if len(split) != 2 or split[1] != count or sum(split) != tensor.shape[dim]:
            raise ValueError("PEQ DN split/query identity mismatch")
        tensor = torch.split(tensor, split, dim=dim)[1]
    if tensor.shape[dim] != count:
        raise ValueError(f"Expected {count} ordinary queries, got {tensor.shape[dim]}")
    return tensor


class PrecisionEvidenceQuality(nn.Module):
    """17,994 parameters. Inputs detach before all new trainable projections."""

    def __init__(self, channels=256, query_dim=256):
        super().__init__()
        if (channels, query_dim) != (256, 256):
            raise ValueError("PEQ v1 requires 256-dimensional P3 and queries")
        self.config = configuration()
        with torch.random.fork_rng(devices=[]):
            self.p3_proj = nn.Conv2d(256, 16, 1, bias=False)
            self.query_proj = nn.Linear(256, 32, bias=False)
            self.pool_hidden = nn.Linear(18, 16, bias=True)
            self.pool_score = nn.Linear(16, 1, bias=False)
            self.fusion = nn.Linear(73, 64, bias=True)
            self.output = nn.Linear(64, 10, bias=True)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        relative = torch.tensor([(u, v) for v in (-1/3, 0, 1/3) for u in (-1/3, 0, 1/3)])
        self.register_buffer("relative", relative, persistent=False)

    def sample(self, values, boxes):
        with torch.autocast(device_type=values.device.type, enabled=False):
            b = boxes.detach().float()
            # Recreate FP32 constants: model.half() must not round +/-1/3 permanently.
            positions = b.new_tensor([(u, v) for v in (-1/3, 0, 1/3) for u in (-1/3, 0, 1/3)])
            points = b[..., None, :2] + b[..., None, 2:] * positions
            grid = 2 * points - 1
            finite(values, grid)
            samples = F.grid_sample(values.float(), grid, mode="bilinear",
                                    padding_mode="border", align_corners=False).permute(0, 2, 3, 1)
            oob = ((points < 0) | (points > 1)).any(-1)
        return samples, positions, oob

    def evidence(self, values, boxes):
        sampled, positions, oob = self.sample(values, boxes)
        positions = positions.expand(*sampled.shape[:-1], 2)
        joined = torch.cat((sampled, positions), -1).to(self.pool_hidden.weight.dtype)
        logits = self.pool_score(F.silu(self.pool_hidden(joined)))
        with torch.autocast(device_type=values.device.type, enabled=False):
            weights = logits.float().softmax(-2)
            evidence = (weights * sampled.float()).sum(-2)
        return evidence, oob

    def forward(self, p3, query, b0, b1, z):
        if z.shape[-1] != 1:
            raise ValueError("PEQ forward requires nc=1; multi-class averaging is forbidden")
        finite(p3, query, b0, b1, z)
        values = self.p3_proj(p3.detach().to(self.p3_proj.weight.dtype))
        projected_query = F.silu(self.query_proj(query.detach().to(self.query_proj.weight.dtype)))
        e0, oob0 = self.evidence(values, b0)
        e1, oob1 = self.evidence(values, b1)
        with torch.autocast(device_type=p3.device.type, enabled=False):
            before, after, context = b0.detach().float(), b1.detach().float(), z.detach().float()
            scale = before[..., [2, 3, 2, 3]].clamp_min(1e-6)
            geometry = torch.cat((after, ((after - before) / scale).clamp(-2, 2)), -1)
            h = torch.cat((e1, e1 - e0, projected_query.float(), geometry, context.clamp(-10, 10)), -1)
        u = self.output(F.silu(self.fusion(h.to(self.fusion.weight.dtype))))
        with torch.autocast(device_type=p3.device.type, enabled=False):
            delta = u.float().tanh()
            logits = context + delta
            probabilities = logits.sigmoid()
            score = probabilities.mean(-1, keepdim=True)
            finite(logits, score)
        return dict(quality_logits=logits, delta=delta.detach(), s_final=score,
                    s_raw=context.sigmoid(), b0=before, b1=after,
                    evidence_difference=(e1-e0).detach(), oob0=oob0, oob1=oob1)


class RTDETRDecoderCBRPEQ(RTDETRDecoderCBR):
    """Original same-pass decoder/CBR scheduling plus an ordinary-query-only payload."""

    def __init__(self, nc=80, ch=(256, 256, 256), hd=256, nq=300, ndp=4, nh=8, ndl=3, **kwargs):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, **kwargs)
        if (nq, ndl) != (300, 3):
            raise ValueError("PEQ v1 requires 300 queries and three decoder layers")
        self.peq = PrecisionEvidenceQuality(ch[0], hd)

    def forward(self, x, batch=None):
        if not self.peq.config["enabled"]:
            return super().forward(x, batch)
        return self._forward_peq(x, batch)

    def _forward_peq(self, x, batch):
        from ultralytics.models.utils.ops import get_cdn_group

        if self.export:
            raise NotImplementedError("PEQ v1 supports eager PyTorch .pt/AutoBackend only; export backends are unvalidated")
        if self.nc != 1:
            raise ValueError("PEQ actual forward requires nc=1")
        p3 = x[0]
        if len(x) != 3 or any(p3.shape[-2] < f.shape[-2] or p3.shape[-1] < f.shape[-1] for f in x[1:]):
            raise ValueError("CBR expects the YAML decoder inputs ordered P3, P4, P5.")
        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch, self.nc, self.num_queries, self.denoising_class_embed.weight,
            self.num_denoising, self.label_noise_ratio, self.box_noise_scale, self.training,
        )
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        dec_bboxes, dec_scores, final_query = self.decoder(
            embed, refer_bbox, feats, shapes, self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, attn_mask=attn_mask, return_final_query=True,
        )
        before = dec_bboxes[-1]
        # The ORIGINAL CBR is applied to ordinary and DN queries, exactly once.
        refined, _ = self.cbr(p3, final_query, before, return_diagnostics=True)
        payload = self.peq(
            p3, ordinary(final_query, dn_meta), ordinary(before, dn_meta),
            ordinary(refined, dn_meta), ordinary(dec_scores[-1], dn_meta),
        )
        dec_bboxes = torch.cat((dec_bboxes[:-1], refined.unsqueeze(0)), dim=0)
        raw = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, payload
        if self.training:
            return raw
        return torch.cat((refined, payload["s_final"]), -1), raw

    def forward_with_diagnostics(self, x, batch=None):
        if not self.peq.config["enabled"]:
            return super().forward_with_diagnostics(x, batch)
        result = self._forward_peq(x, batch)
        raw = result if self.training else result[1]
        return result, raw[-1]
