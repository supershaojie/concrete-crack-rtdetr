"""Create an nc=1 DCC controlled initialization and audit native reconstruction, without training."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from dcc_common import (ROOT, MODEL_DIR, VARIANTS, BASE_COMMIT, SOURCE_SHA256, require, sha256,
                        write_json, runtime, verify_model, controlled_models, build_training_model)
from ultralytics import RTDETR
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.patches import torch_load


def audit_checkpoint_reconstruction(output, expected, variant):
    """Exercise setup_model and actual Model.train get_model dispatch; no train loop.

    The dispatch-only subclass deliberately omits dataset/optimizer setup and raises
    at train() entry. This is reconstruction evidence, not a training/resume test.
    """
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.model, trainer.data = str(output), dict(nc=1, channels=3)
    trainer.args = SimpleNamespace(pretrained=False)
    checkpoint = trainer.setup_model()
    verify_model(trainer.model, variant, zero=True)
    require(checkpoint["epoch"] == -1, "Unexpected trained state")
    require(all(torch.equal(v, trainer.model.state_dict()[k]) for k, v in expected.state_dict().items()), "setup_model reload differs")
    captured = {}

    class ReconstructionComplete(Exception):
        pass

    class ReconstructionOnlyTrainer(RTDETRTrainer):
        def __init__(self, overrides=None, _callbacks=None):
            self.data = dict(nc=1, channels=3)
            captured["overrides"] = overrides

        def get_model(self, cfg=None, weights=None, verbose=True):
            captured["native_get_model_calls"] = captured.get("native_get_model_calls", 0) + 1
            return RTDETRTrainer.get_model(self, cfg=cfg, weights=weights, verbose=False)

        def train(self):
            captured["model"] = self.model
            raise ReconstructionComplete()

    wrapper = RTDETR(str(output))
    with patch("ultralytics.engine.model.checks.check_pip_update_available", return_value=None):
        try:
            wrapper.train(trainer=ReconstructionOnlyTrainer, data="reconstruction-only-no-data-read.yaml", resume=False)
        except ReconstructionComplete:
            pass
    require(captured.get("native_get_model_calls") == 1 and "model" in captured, "Actual Model.train did not reconstruct through native get_model")
    verify_model(captured["model"], variant, zero=True)
    require(all(torch.equal(v, captured["model"].state_dict()[k]) for k, v in expected.state_dict().items()), "Model.train reconstruction differs")
    return dict(status="PASSED", native_setup_model=True, native_get_model=True,
                actual_model_train_dispatch=True, public_and_added_values_exact=True,
                trainer_constructor="Minimal reconstruction-only shim; native data/optimizer setup not executed",
                stopped_at="Trainer.train entry before training loop", training_updates=0)


def initialize(source, output, variant="cbr_lif_dcc_v1", verify_existing=False):
    output = Path(output)
    require(output.is_file() if verify_existing else not output.exists(),
            "Expected existing initialization for verification" if verify_existing else "Existing initialization preserved: " + str(output))
    _, target, report = controlled_models(source, variant)
    target.eval()
    target.args = {**DEFAULT_CFG_DICT, "model": str(MODEL_DIR / VARIANTS[variant][0]), "task": "detect"}
    target.task, target.pt_path = "detect", str(output.resolve())
    metadata = dict(variant=variant, source_sha256=SOURCE_SHA256, base_commit=BASE_COMMIT,
                    kind="controlled_initialization", source_nc=80, target_nc=1, seed=42,
                    new_parameters=18432, common_exact=True, variant_new_state_exact=True)
    checkpoint = dict(epoch=-1, best_fitness=None, model=deepcopy(target).float(), ema=None,
                      updates=None, optimizer=None, scaler=None, train_args=target.args,
                      train_metrics=None, train_results=None, date=datetime.now(timezone.utc).isoformat(),
                      dcc=metadata, dcc_provenance=report)
    if verify_existing:
        prior = torch_load(output, map_location="cpu")
        require(prior.get("dcc") == metadata and prior.get("epoch") == -1 and
                all(prior.get(k) is None for k in ("optimizer", "scaler", "ema", "updates")),
                "Existing checkpoint is not this experiment's controlled initialization")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            torch.save(checkpoint, stream)
    restored = RTDETR(str(output)).model
    require(set(restored.state_dict()) == set(target.state_dict()) and
            all(torch.equal(v, restored.state_dict()[k]) for k, v in target.state_dict().items()), "Controlled checkpoint reload differs")
    verify_model(restored, variant, zero=True)
    report.update(dcc=metadata, output=str(output.resolve()), output_sha256=sha256(output), reload_exact=True,
                  training_reconstruction=audit_checkpoint_reconstruction(output, target, variant),
                  status="PASSED", existing_checkpoint_verified=verify_existing, runtime=runtime())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true", help="Audit existing init against freshly controlled source without modifying it")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        result = initialize(args.source, args.output, args.variant, args.verify_existing)
    except Exception as error:
        write_json(args.report, dict(status="FAILED", variant=args.variant, error=repr(error)))
        raise
    write_json(args.report, result)
    print("PASSED: controlled initialization + native Trainer reconstruction", args.report)


if __name__ == "__main__":
    main()
