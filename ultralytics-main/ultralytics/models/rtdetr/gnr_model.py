"""Importable GNR model wrapper. Original modules, parameters and inference path.

The opt-in forward below mirrors the pinned mother's CBR forward exactly and
returns its existing final_query as an explicit value. No hooks or tensor caches.
"""
from __future__ import annotations

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.utils.ops import get_cdn_group
from .gnr_loss import GNRDetectionLoss, FORMULA, ramp, require, summarize


def forward_with_features(model, images, targets):
    """One forward, identical head math to cbr.py at a0459d6; explicit [B,DN+N,256] h."""
    x, saved = images, []
    for module in model.model[:-1]:
        if module.f != -1:
            x = saved[module.f] if isinstance(module.f, int) else [x if j == -1 else saved[j] for j in module.f]
        x = module(x)
        saved.append(x if module.i in model.save else None)
    head = model.model[-1]
    x = [saved[j] for j in head.f]
    p3 = x[0]
    require(len(x) == 3 and all(p3.shape[-2] >= f.shape[-2] and p3.shape[-1] >= f.shape[-1] for f in x[1:]), "CBR input order")
    linear = head.dec_score_head[-1]
    require(isinstance(linear, torch.nn.Linear) and linear.bias is not None and linear.out_features == 1, "last classifier must be biased nc1 Linear")
    feats, shapes = head._get_encoder_input(x)
    dn_embed, dn_bbox, mask, meta = get_cdn_group(targets, head.nc, head.num_queries, head.denoising_class_embed.weight,
                                                head.num_denoising, head.label_noise_ratio, head.box_noise_scale, head.training)
    embed, reference, enc_b, enc_z = head._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
    dec_b, dec_z, h = head.decoder(embed, reference, feats, shapes, head.dec_bbox_head, head.dec_score_head,
                                   head.query_pos_head, attn_mask=mask, return_final_query=True)
    refined, _ = head.cbr(p3, h, dec_b[-1], return_diagnostics=True)
    dec_b = torch.cat((dec_b[:-1], refined.unsqueeze(0)), dim=0)
    return (dec_b, dec_z, enc_b, enc_z, meta), h


def targets_from_batch(batch):
    image, indices = batch["img"], batch["batch_idx"]
    return {"cls": batch["cls"].to(image.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(image.device),
            "batch_idx": indices.to(image.device, dtype=torch.long).view(-1),
            "gt_groups": [(indices == i).sum().item() for i in range(len(image))]}


def criterion_inputs(raw, features):
    dec_b, dec_z, enc_b, enc_z, meta = raw
    dn_b = dn_z = None
    if meta is not None:
        dn_b, dec_b = torch.split(dec_b, meta["dn_num_split"], dim=2)
        dn_z, dec_z = torch.split(dec_z, meta["dn_num_split"], dim=2)
        _, features = torch.split(features, meta["dn_num_split"], dim=1)
    require(features.shape[:2] == dec_z.shape[1:3], "ordinary/DN feature slicing mismatch")
    return (torch.cat((enc_b.unsqueeze(0), dec_b)), torch.cat((enc_z.unsqueeze(0), dec_z))), features, dn_b, dn_z, meta


class GNRDetectionModel(RTDETRDetectionModel):
    gnr_epoch = 0
    gnr_beta = .5
    gnr_formula = FORMULA

    def init_criterion(self):
        return GNRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        # Validation loss and all warmup/off cases use the mother's entire path.
        if not self.training or self.gnr_beta == 0 or ramp(self.gnr_epoch) == 0:
            return super().loss(batch, preds)
        require(preds is None, "active GNR needs explicit same-forward features; compile/precomputed predictions unsupported")
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        targets = targets_from_batch(batch)
        raw, features = forward_with_features(self, batch["img"], targets)
        inputs, features, dn_b, dn_z, meta = criterion_inputs(raw, features)
        losses, details = self.criterion(inputs, targets, dn_b, dn_z, meta, features=features,
                                         epoch=self.gnr_epoch, beta=self.gnr_beta, return_details=True)
        # JSON values only; no graph, activation, or tensor retained across batches.
        self.gnr_diagnostics = summarize(details)
        return sum(losses.values()), torch.as_tensor([losses[k].detach() for k in ("loss_giou", "loss_class", "loss_bbox")], device=batch["img"].device)
