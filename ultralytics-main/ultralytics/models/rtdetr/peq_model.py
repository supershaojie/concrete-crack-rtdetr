# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""PEQ-specific model/raw parsing; the original five-item API is untouched."""
from __future__ import annotations
from copy import deepcopy
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.modules.peq import RTDETRDecoderCBRPEQ, configuration
from ultralytics.models.utils.peq_loss import PEQDetectionLoss


class PEQDetectionModel(RTDETRDetectionModel):
    def __init__(self, cfg="rtdetr-resnet18-lite-cbr-lif-down-peq-v1.yaml", ch=3, nc=None, verbose=True):
        super().__init__(deepcopy(cfg), ch, nc, verbose)
        if not isinstance(self.model[-1], RTDETRDecoderCBRPEQ):
            raise ValueError("PEQ model requires the registered PEQ head")
        self.nc = self.yaml["nc"]
        self.yaml["peq"] = configuration(self.yaml.get("peq"))
        self.model[-1].peq.config = deepcopy(self.yaml["peq"])
        self.peq_stats = None

    def init_criterion(self):
        return PEQDetectionLoss(nc=self.nc, use_vfl=True)

    def loss_components(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        img, batch_idx = batch["img"], batch["batch_idx"]
        targets = dict(cls=batch["cls"].to(img.device, dtype=torch.long).view(-1),
                       bboxes=batch["bboxes"].to(img.device),
                       batch_idx=batch_idx.to(img.device, dtype=torch.long).view(-1),
                       gt_groups=[int((batch_idx == i).sum()) for i in range(img.shape[0])])
        if preds is None:
            preds = self.predict(img, batch=targets)
        raw = preds if self.training else preds[1]
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = raw[:5]
        payload = raw[5] if len(raw) == 6 else None
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
        losses, details = self.criterion((dec_bboxes, dec_scores), targets,
                                         dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta,
                                         peq_payload=payload, return_details=True)
        if details is not None and self.peq_stats is not None:
            self.peq_stats.observe(payload, details)
        return losses

    def loss(self, batch, preds=None):
        if not self.model[-1].peq.config["enabled"]:
            return super().loss(batch, preds)
        losses = self.loss_components(batch, preds)
        # Match the mother implementation's insertion order and total-loss reduction.
        return sum(losses.values()), torch.as_tensor(
            [losses[k].detach() for k in ("loss_giou", "loss_class", "loss_bbox")], device=batch["img"].device)
