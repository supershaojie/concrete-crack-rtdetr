"""Importable specialization; all predictions and the native loss sum stay unchanged."""
from ultralytics.nn.tasks import RTDETRDetectionModel
from .dtr_loss import DTRDetectionLoss


class DTRDetectionModel(RTDETRDetectionModel):
    def init_criterion(self):
        return DTRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        if not isinstance(getattr(self, "criterion", None), DTRDetectionLoss):
            self.criterion = self.init_criterion()
        criterion = self.criterion
        criterion.enabled = self.training and getattr(self, "dtr_enabled", True)
        criterion.epoch = getattr(self, "dtr_epoch", 0)
        criterion.sample = self.training and getattr(self, "dtr_sample", False)
        criterion.input_hw = tuple(batch["img"].shape[-2:])
        try:
            return super().loss(batch, preds)
        finally:
            criterion.input_hw = None
