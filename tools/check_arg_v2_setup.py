"""Local reduced-size native setup check. No optimizer update or capacity claim."""
from __future__ import annotations

from pathlib import Path
import argparse
import time
import traceback

from arg_v2_common import ROOT, OUT, SOURCE, data_config, recipe, read_json, initialize, write_json, runtime, require
from arg_v2 import strict_amp_resources
from arg_v2_preflight import close_workers
from ultralytics.models.rtdetr.arg_v2_model import ARGv2Trainer, ARGv2DetectionModel
from ultralytics.models.rtdetr.arg_v2_loss import ARGv2DetectionLoss
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-alias", type=Path, help="Optional alias to this same worktree for local path-loader compatibility")
    options = parser.parse_args()
    if options.path_alias:
        require(options.path_alias.resolve() == ROOT.resolve(), "Alias must resolve to this v2 worktree")
    def local_path(path):
        return options.path_alias / path.relative_to(ROOT) if options.path_alias else path
    torch.set_num_threads(4)
    report = dict(status="FAIL", scope="local native setup only; B2/160/workers0 diagnostic overrides, not capacity", runtime=runtime())
    trainer = None
    try:
        checkpoint = local_path(OUT / "local_setup_init.pt")
        if not checkpoint.exists():
            write_json(OUT / "local_setup_init.json", initialize(SOURCE, checkpoint))
        report["amp_resources"] = strict_amp_resources()
        args, _ = recipe(local_path(data_config()))
        folder = local_path(OUT / "setup_runs" / time.strftime("%Y%m%d_%H%M%S"))
        args.update(model=str(checkpoint), project=str(folder), name="setup", save_dir=str(folder / "setup"),
                    batch=2, imgsz=160, workers=0, plots=False)
        trainer = ARGv2Trainer(overrides=args)
        trainer._setup_train()
        assert type(trainer.model) is ARGv2DetectionModel
        assert type(trainer.model.init_criterion()) is ARGv2DetectionLoss
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
