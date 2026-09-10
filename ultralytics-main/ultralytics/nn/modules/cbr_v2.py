# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Stable boundary evidence and isolated conditions, with the native box path intact."""
import torch
from torch.nn import functional as F
from .cbr import CrackBoundaryRefinement
from .head import RTDETRDecoder

__all__ = ("StableReferenceCBR", "RTDETRDecoderCBRv2")


class StableReferenceCBR(CrackBoundaryRefinement):
    """C19's 36-sample signed boundary mechanism at fixed rho=.075."""

    def __init__(self, p3_channels, query_dim):
        super().__init__(p3_channels, query_dim)
        self.rho = 0.075

    def forward(self, p3_ref, query, boxes, return_diagnostics=False):
        if p3_ref.ndim != 4 or query.ndim != 3 or boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("SR-CBR expects BCHW reference, BQD query and BQ4 boxes.")
        if query.shape[:2] != boxes.shape[:2] or p3_ref.shape[0] != boxes.shape[0]:
            raise ValueError("SR-CBR batch/query dimensions disagree.")
        values = self.p3_proj(p3_ref.detach())
        query_condition, geometry_condition = query.detach(), boxes.detach()
        batch, count = boxes.shape[:2]
        with torch.autocast(device_type=p3_ref.device.type, enabled=False):
            grid = self.sampling_grid(geometry_condition)
            sampled = F.grid_sample(values.float(), grid.reshape(batch, count, 36, 2),
                                    mode="bilinear", padding_mode="border", align_corners=False)
            sampled = sampled.permute(0, 2, 3, 1).reshape(batch, count, 4, 3, 3, 64)
            evidence = torch.cat((sampled[..., 1, :], sampled[..., 0, :] - sampled[..., 2, :]), -1)
        local = self.evidence_proj(evidence.to(self.evidence_proj.weight.dtype))
        condition = self.query_proj(query_condition.to(self.query_proj.weight.dtype))[:, :, None, None, :]
        geometry = self.geometry_proj(geometry_condition[..., 2:].to(self.geometry_proj.weight.dtype))[:, :, None, None, :]
        hidden = F.silu(local + condition + geometry + self.side_embed.weight[None, None, :, None, :])
        with torch.autocast(device_type=p3_ref.device.type, enabled=False):
            weights = self.score(hidden.to(self.score.weight.dtype)).float().softmax(dim=-2)
            pooled = (weights * hidden.float()).sum(dim=-2)
        t = self.offset_out(F.silu(self.offset_hidden(pooled.to(self.offset_hidden.weight.dtype)))).squeeze(-1)
        with torch.autocast(device_type=p3_ref.device.type, enabled=False):
            original = boxes.float()  # the direct main-box derivative remains the identity
            unit = t.float().tanh()
            # Width/height scaling is also a condition: detach it, not the residual.
            displacement = self.rho * geometry_condition.float()[..., [2, 2, 3, 3]] * unit
            dl, dr, dt, db = displacement.unbind(-1)
            residual = torch.stack(((dl + dr) / 2, (dt + db) / 2, dr - dl, db - dt), -1)
            refined = original + residual
        if return_diagnostics:
            return refined, dict(before=original, after=refined, tanh_offsets=unit, displacement=displacement,
                                 aggregation_weights=weights.squeeze(-1), evidence=evidence)
        return refined


class RTDETRDecoderCBRv2(RTDETRDecoder):
    """Three core encoder levels plus a fourth, exclusively boundary-reference input."""

    def __init__(self, nc=80, ch=(256, 256, 256, 256), hd=256, nq=300, ndp=4, nh=8, ndl=3, **kwargs):
        if len(ch) != 4:
            raise ValueError("CBRv2 requires channels for [core_P3, core_P4, core_P5, stable_P3_ref].")
        super().__init__(nc, ch[:3], hd, nq, ndp, nh, ndl, **kwargs)
        if self.decoder.eval_idx != ndl - 1:
            raise ValueError("CBRv2 requires the native final-layer eval_idx.")
        self.cbr = StableReferenceCBR(ch[3], hd)

    def forward(self, x, batch=None):
        return self._forward_cbr(x, batch, False)

    def forward_with_diagnostics(self, x, batch=None):
        return self._forward_cbr(x, batch, True)

    def _forward_cbr(self, x, batch, diagnostics):
        from ultralytics.models.utils.ops import get_cdn_group

        if len(x) != 4:
            raise ValueError("CBRv2 requires exactly four feature inputs.")
        core, p3_ref = x[:3], x[3]
        if core[0].shape[0] != p3_ref.shape[0] or core[0].shape[-2:] != p3_ref.shape[-2:]:
            raise ValueError("Stable P3 reference must match core P3 batch/spatial size.")
        if any(core[i].shape[-2] != 2 * core[i+1].shape[-2] or
               core[i].shape[-1] != 2 * core[i+1].shape[-1] for i in range(2)):
            raise ValueError("CBRv2 core inputs must be P3/P4/P5 in decreasing spatial order.")
        feats, shapes = self._get_encoder_input(core)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch, self.nc, self.num_queries, self.denoising_class_embed.weight,
            self.num_denoising, self.label_noise_ratio, self.box_noise_scale, self.training,
        )
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        dec_bboxes, dec_scores, final_query = self.decoder(
            embed, refer_bbox, feats, shapes, self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, attn_mask=attn_mask, return_final_query=True,
        )
        result = self.cbr(p3_ref, final_query, dec_bboxes[-1], return_diagnostics=diagnostics)
        refined, details = result if diagnostics else (result, None)
        dec_bboxes = torch.cat((dec_bboxes[:-1], refined.unsqueeze(0)), dim=0)
        raw = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            result = raw
        else:
            y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
            result = y if self.export else (y, raw)
        return (result, details) if diagnostics else result
