"""Stable importable PDS model; inference inherits the original RT-DETR path."""
from __future__ import annotations
from copy import deepcopy
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.modules.pds import CONFIG, PDSHead, dense_loss, ramp, finite


class PDSDetectionModel(RTDETRDetectionModel):
    @classmethod
    def attach(cls, native):
        """Called only after the original native strict topology/value audit."""
        if type(native) is not RTDETRDetectionModel or len(native.model) != 27 or native.model[-1].nc != 1:
            raise ValueError("PDS F requires an audited native nc1 27-node model")
        before = native.state_dict()
        native.__class__ = cls
        native.nc = 1
        native.pds_head = PDSHead()
        native.pds_config = deepcopy(CONFIG)
        native.pds_epoch = 0
        native.pds_stats = {}
        after = native.state_dict()
        assert all(torch.equal(v, after[k]) for k, v in before.items())
        assert len(after) - len(before) == 11
        assert sum(p.numel() for p in native.pds_head.parameters()) == 10821
        assert not list(native.pds_head.buffers())
        return native

    def training_forward(self, image, targets):
        if not self.training:
            raise RuntimeError("PDS capture is training-only")
        x, saved, local = image, [], {}
        for module in self.model[:-1]:
            if module.f != -1:
                x = saved[module.f] if isinstance(module.f, int) else [
                    x if j == -1 else saved[j] for j in module.f]
            x = module(x)
            saved.append(x if module.i in self.save else None)
            if module.i in (4, 19):
                local[module.i] = x
        if set(local) != {4, 19}:
            raise RuntimeError("PDS taps missing")
        head = self.model[-1]
        preds = head([saved[j] for j in head.f], targets)
        return preds, (local[4], local[19])

    def loss(self, batch, preds=None):
        weight = .25 * ramp(self.pds_epoch)
        if not self.training or weight == 0:
            self.pds_stats = dict(active=False, epoch=self.pds_epoch, weight=0.)
            return super().loss(batch, preds)
        if preds is not None:
            raise RuntimeError("PDS v1 rejects precomputed training preds without features; compile=false")
        img, idx = batch["img"], batch["batch_idx"]
        targets = dict(cls=batch["cls"].to(img.device, dtype=torch.long).view(-1),
                       bboxes=batch["bboxes"].to(img.device),
                       batch_idx=idx.to(img.device, dtype=torch.long).view(-1),
                       gt_groups=[int((idx == i).sum()) for i in range(len(img))])
        preds, features = self.training_forward(img, targets)
        original, items = super().loss(batch, preds)
        finite(original, "main_loss")
        logits, raw = self.pds_head(*features)
        auxiliary, stats = dense_loss(logits, raw, batch)
        self.pds_stats = dict(stats, active=True, epoch=self.pds_epoch, weight=weight,
                              weighted=float((weight * auxiliary).detach()), L0=float(original.detach()))
        return original + weight * auxiliary, items
