"""Audit native save_model/resume on disposable learned engineering copies only."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from init_bfr_p4 import ROOT, VARIANTS, require, write_json, sha256
from check_bfr_p4 import optimizer
from preflight_bfr_p4 import native_resume
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA


def run(variant, checkpoint_path, output, device):
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    require(checkpoint.get("epoch") == 0 and checkpoint.get("model") is not None,
            "This diagnostic requires a disposable local learned checkpoint")
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.device = torch.device(device)
    trainer.model = checkpoint["model"].float().to(device)
    require(torch.count_nonzero(trainer.model.model[22].bfr.wo.weight) > 0, "Need learned nonzero BFR")
    trainer.data = {"nc": 1, "channels": 3}
    args = YAML.load(ROOT / "docs/bfr_p4/parent_args.yaml")
    args.update(model=str(checkpoint_path.resolve()), project=str(output), name="native_save_diagnostic",
                save_dir=str(output / "native_save_diagnostic"))
    trainer.args = SimpleNamespace(**args)
    trainer.wdir = output / "native_save_diagnostic/weights"
    trainer.last, trainer.best = trainer.wdir / "last.pt", trainer.wdir / "best.pt"
    trainer.csv = output / "intentionally_absent_no_formal_epoch_results.csv"
    trainer.metrics, trainer.save_period = {}, -1
    trainer.optimizer = optimizer(trainer.model)
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    enabled = bool(checkpoint["scaler"])
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    trainer.scaler.load_state_dict(checkpoint["scaler"])
    trainer.ema = ModelEMA(trainer.model)
    trainer.ema.ema.load_state_dict(checkpoint["ema"].float().state_dict())
    trainer.ema.updates = checkpoint["updates"]
    report = {"status": "RUNNING", "variant": variant, "device": device, "native_scaler_enabled": enabled,
              "source_checkpoint_sha256": sha256(checkpoint_path), "formal_training": "NOT_STARTED",
              "final_test": "NOT_RUN", "scope": "Native save/rebuild/resume methods on learned disposable copies; no train setup/loop"}
    try:
        report["native_roundtrip"] = native_resume(trainer, variant, output)
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        write_json(output / "native_save_resume.json", report)
    print(variant, device, report["status"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.variant, args.checkpoint, args.output, args.device)
