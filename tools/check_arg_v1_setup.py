"""Local reduced-size native setup check. No optimizer update or capacity claim."""
from __future__ import annotations

from pathlib import Path
import time
import traceback

from arg_v1_common import OUT, SOURCE, data_config, recipe, read_json, initialize, write_json, runtime
from arg_v1 import strict_amp_resources
from arg_v1_preflight import close_workers
from ultralytics.models.rtdetr.arg_model import ARGTrainer, ARGDetectionModel
from ultralytics.models.rtdetr.arg_loss import ARGDetectionLoss
import torch


if __name__ == "__main__":
    torch.set_num_threads(4)
    report = dict(status="FAIL", scope="local native setup only; B2/160/workers0 diagnostic overrides, not capacity", runtime=runtime())
    trainer = None
    try:
        checkpoint = OUT / "local_setup_init.pt"
        if not checkpoint.exists():
            write_json(OUT / "local_setup_init.json", initialize(SOURCE, checkpoint))
        report["amp_resources"] = strict_amp_resources()
        args, _ = recipe(data_config())
        folder = OUT / "setup_runs" / time.strftime("%Y%m%d_%H%M%S")
        args.update(model=str(checkpoint), project=str(folder), name="setup", save_dir=str(folder / "setup"),
                    batch=2, imgsz=160, workers=0, plots=False)
        trainer = ARGTrainer(overrides=args)
        trainer._setup_train()
        assert type(trainer.model) is ARGDetectionModel
        assert type(trainer.model.init_criterion()) is ARGDetectionLoss
        trainer.epoch = 6
        trainer._arg_epoch_start(trainer)
        assert trainer.model.arg_epoch == 6
        report.update(status="PASS", actual_args=vars(trainer.args), rebuild=trainer.arg_rebuild,
                      amp=trainer.arg_amp_evidence, scaler=trainer.scaler.state_dict(), accumulate=trainer.accumulate)
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc()); raise
    finally:
        if trainer:
            close_workers(trainer)
        write_json(OUT / "local_setup.json", report)
