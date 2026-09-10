# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""C19 candidate: query-conditioned refinement of horizontal detection-box sides.

Coordinate conventions checked against DySample in the supplied RTDETR module
archive; this is a new box head, not a transplant of its upsampling operation.
See docs/C19_CBR.md for the exact source trace and experimental limitations.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .head import RTDETRDecoder


class CrackBoundaryRefinement(nn.Module):
    """36 fixed samples/query, signed inside-outside evidence, bounded xywh residual."""

    def __init__(self, p3_channels: int, query_dim: int):
        super().__init__()
        self.rho = 0.10
        self.normal_fraction = 0.10
        # Do not consume the RNG used by C2's subsequent model/head construction.
        # All new tensors are constructed on CPU; no CUDA RNG is consumed here.
        with torch.random.fork_rng(devices=[]):
            self.p3_proj = nn.Conv2d(p3_channels, 64, 1)
            self.query_proj = nn.Linear(query_dim, 64)
            self.evidence_proj = nn.Linear(128, 64)
            self.side_embed = nn.Embedding(4, 64)
            self.geometry_proj = nn.Linear(2, 64)
            self.score = nn.Linear(64, 1, bias=False)  # a shared scalar bias would cancel in softmax
            self.offset_hidden = nn.Linear(64, 64)
            self.offset_out = nn.Linear(64, 1)
            nn.init.zeros_(self.offset_out.weight)
            nn.init.zeros_(self.offset_out.bias)

    def sampling_grid(self, boxes):
        """FP32 grid [B,Q,side,position,inside/edge/outside,xy], side=L,R,T,B.

        Normalized image coordinates u map to 2*u-1: feature pixel centers are
        (j+.5)/W and (i+.5)/H with align_corners=False, including rectangular maps.
        Only the sampling geometry is detached. No box clamp/epsilon is applied.
        """
        boxes = boxes.detach().float()
        center, wh = boxes[..., :2], boxes[..., 2:]
        along = boxes.new_tensor([-.25, 0., .25])
        inner = boxes.new_tensor([1., 0., -1.]) * self.normal_fraction
        left = torch.stack(((-.5 + inner)[None, :].expand(3, 3), along[:, None].expand(3, 3)), -1)
        right = torch.stack(((.5 - inner)[None, :].expand(3, 3), along[:, None].expand(3, 3)), -1)
        top = torch.stack((along[:, None].expand(3, 3), (-.5 + inner)[None, :].expand(3, 3)), -1)
        bottom = torch.stack((along[:, None].expand(3, 3), (.5 - inner)[None, :].expand(3, 3)), -1)
        relative = torch.stack((left, right, top, bottom))
        return (center[..., None, None, None, :] + wh[..., None, None, None, :] * relative) * 2 - 1

    def forward(self, p3, query, boxes, return_diagnostics=False):
        values = self.p3_proj(p3)
        batch, count = boxes.shape[:2]
        with torch.autocast(device_type=p3.device.type, enabled=False):
            grid = self.sampling_grid(boxes)
            # PyTorch 2.1.2 CPU/CUDA half grid_sample is handled locally in FP32.
            sampled = F.grid_sample(values.float(), grid.reshape(batch, count, 36, 2),
                                    mode="bilinear", padding_mode="border", align_corners=False)
            sampled = sampled.permute(0, 2, 3, 1).reshape(batch, count, 4, 3, 3, 64)
            evidence = torch.cat((sampled[..., 1, :], sampled[..., 0, :] - sampled[..., 2, :]), -1)
        dtype = self.evidence_proj.weight.dtype
        local = self.evidence_proj(evidence.to(dtype))
        condition = self.query_proj(query.to(self.query_proj.weight.dtype))[:, :, None, None, :]
        geometry = self.geometry_proj(boxes.detach()[..., 2:].to(self.geometry_proj.weight.dtype))[:, :, None, None, :]
        # Nonlinearity precedes scores: query changes relative position scores,
        # rather than adding a softmax-invariant scalar constant to every position.
        hidden = F.silu(local + condition + geometry + self.side_embed.weight[None, None, :, None, :])
        with torch.autocast(device_type=p3.device.type, enabled=False):
            weights = self.score(hidden.to(self.score.weight.dtype)).float().softmax(dim=-2)
            pooled = (weights * hidden.float()).sum(dim=-2)
        t = self.offset_out(F.silu(self.offset_hidden(pooled.to(self.offset_hidden.weight.dtype)))).squeeze(-1)
        with torch.autocast(device_type=p3.device.type, enabled=False):
            original = boxes.float()  # keep the main original-box gradient path
            unit = t.float().tanh()
            displacement = self.rho * original[..., [2, 2, 3, 3]] * unit
            dl, dr, dt, db = displacement.unbind(-1)
            residual = torch.stack(((dl + dr) / 2, (dt + db) / 2, dr - dl, db - dt), -1)
            refined = original + residual
        if return_diagnostics:
            return refined, {"before": original, "after": refined, "tanh_offsets": unit,
                             "displacement": displacement, "aggregation_weights": weights.squeeze(-1)}
        return refined


class RTDETRDecoderCBR(RTDETRDecoder):
    """Preserve all C2 parameter paths; refine only the returned final decoder box."""

    def __init__(self, nc=80, ch=(256, 256, 256), hd=256, nq=300, ndp=4, nh=8, ndl=3, **kwargs):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, **kwargs)
        if self.decoder.eval_idx != ndl - 1:
            raise ValueError("C19 requires the original final-layer eval_idx.")
        # Decoder input slot 0 is Neck P3 in the YAML's from list, not a model layer number.
        self.cbr = CrackBoundaryRefinement(ch[0], hd)

    def forward(self, x, batch=None):
        return self._forward_cbr(x, batch, False)

    def forward_with_diagnostics(self, x, batch=None):
        """Explicit opt-in return; no hooks, mutable caches, or training state switches."""
        return self._forward_cbr(x, batch, True)

    def _forward_cbr(self, x, batch, diagnostics):
        from ultralytics.models.utils.ops import get_cdn_group

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
        refined, details = self.cbr(p3, final_query, dec_bboxes[-1], return_diagnostics=True)
        dec_bboxes = torch.cat((dec_bboxes[:-1], refined.unsqueeze(0)), dim=0)
        raw = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            result = raw
        else:
            y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
            result = y if self.export else (y, raw)
        return (result, details) if diagnostics else result
