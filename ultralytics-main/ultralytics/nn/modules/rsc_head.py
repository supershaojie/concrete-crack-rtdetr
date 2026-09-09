# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Same-query ordered refinement state conditions only the final classification projection."""

import torch
from torch import nn
from torch.nn import functional as F

from .head import RTDETRDecoder
from .transformer import DeformableTransformerDecoder
from .utils import inverse_sigmoid


def aligned_iou(a, b):
    """Ordinary aligned IoU, no Q x Q matrix or image-boundary clipping (FP32 inputs)."""
    awh, bwh = a[..., 2:].clamp_min(0), b[..., 2:].clamp_min(0)
    alo, ahi = a[..., :2] - awh / 2, a[..., :2] + awh / 2
    blo, bhi = b[..., :2] - bwh / 2, b[..., :2] + bwh / 2
    intersection = (torch.minimum(ahi, bhi) - torch.maximum(alo, blo)).clamp_min(0).prod(-1, keepdim=True)
    union = (ahi - alo).clamp_min(0).prod(-1, keepdim=True) + (bhi - blo).clamp_min(0).prod(-1, keepdim=True) - intersection
    return intersection / union.clamp_min(1e-12)


def refinement_descriptor(box_history):
    """Return [B,Q,16]: u2,u3,u2*u3,IoU12,IoU23,aspect3,area3; detach only boxes."""
    if len(box_history) != 3:
        raise ValueError("RSC requires exactly three ordered post-regression boxes")
    boxes = [b.detach().float() for b in box_history]
    if any(b.ndim != 3 or b.shape[-1] != 4 or b.shape != boxes[0].shape for b in boxes):
        raise ValueError("RSC history must contain three equal [B,Q,4] tensors")
    safe = [b[..., 2:].clamp_min(1e-4) for b in boxes]
    changes = [torch.cat(((boxes[i][..., :2] - boxes[i-1][..., :2]) / safe[i-1],
                          torch.log(safe[i] / safe[i-1])), -1).tanh() for i in (1, 2)]
    u2, u3 = changes
    w3, h3 = safe[2].split(1, -1)
    return torch.cat((u2, u3, u2 * u3, aligned_iou(boxes[0], boxes[1]), aligned_iou(boxes[1], boxes[2]),
                      (torch.log(w3 / h3) / 4).tanh(), (torch.log(w3 * h3) / 8).tanh()), -1)


class RSCHead(nn.Linear):
    """Keep original weight/bias keys and objects; add 8,736 parameters for hd=256."""

    def __init__(self, original):
        # Do not call Linear.__init__: it would draw/reset a second classification projection.
        nn.Module.__init__(self)
        if original.in_features != 256 or original.bias is None:
            raise ValueError("RSC v1 requires a biased 256-channel classification projection")
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight, self.bias = original.weight, original.bias
        # Preserve caller RNG and every subsequent public initialization draw.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.sem_proj = nn.Linear(256, 16)
            self.geo_proj = nn.Linear(16, 16)
            self.mod_proj = nn.Linear(16, 256)
            for layer in (self.sem_proj, self.geo_proj):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.mod_proj.weight)
            nn.init.zeros_(self.mod_proj.bias)

    def forward(self, q, box_history):
        g = refinement_descriptor(box_history)
        if q.shape[:-1] != g.shape[:-1] or q.shape[-1] != 256:
            raise ValueError("RSC query/history shape mismatch")
        q32 = q.float()
        rms = q32 / (q32.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        s = F.silu(self.sem_proj(rms.to(self.sem_proj.weight.dtype))).float()
        t = self.geo_proj(g.to(self.geo_proj.weight.dtype)).float().tanh()
        delta = 0.5 * self.mod_proj((s * t).to(self.mod_proj.weight.dtype)).float().tanh()
        q_cls = (q32 + q32 * delta).to(q.dtype)
        return F.linear(q_cls, self.weight, self.bias)


class DeformableTransformerDecoderRSC(DeformableTransformerDecoder):
    """C2 regression graph verbatim; forward-local post-regression history for RSC."""

    def __init__(self, original):
        nn.Module.__init__(self)
        self.layers = original.layers
        self.num_layers, self.hidden_dim, self.eval_idx = original.num_layers, original.hidden_dim, original.eval_idx
        self._validate()

    def _validate(self):
        if self.num_layers != 3 or self.eval_idx != 2 or self.hidden_dim != 256:
            raise ValueError("RSC v1 supports only three Decoder layers, hd=256 and final eval_idx=2")

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head, pos_mlp,
                attn_mask=None, padding_mask=None):
        self._validate()
        output = embed
        dec_bboxes, dec_cls, history = [], [], []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))
            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
            if self.training:
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
                history.append(dec_bboxes[-1])  # Exactly the auxiliary-gradient box entering loss.
                dec_cls.append(score_head[i](output, history) if i == 2 else score_head[i](output))
            else:
                history.append(refined_bbox)
                if i == self.eval_idx:
                    dec_cls.append(score_head[i](output, history))
                    dec_bboxes.append(refined_bbox)
                    break
            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox
        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class RTDETRDecoderRSC(RTDETRDecoder):
    """Independent C2 Decoder; all public modules/keys remain in their original locations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = DeformableTransformerDecoderRSC(self.decoder)
        self.dec_score_head[2] = RSCHead(self.dec_score_head[2])
