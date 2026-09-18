"""CPU-only toy differential fixture for the DCC optimizer checkpoint policy.

Does not instantiate a detector, read real data, or start any training run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.torch_utils import ModelEMA

from dcc_checkpoint import (DCCCheckpointTrainer, OPTIMIZER_POLICY, native_serializer_identity,
                            optimizer_param_names, require_checkpoint_policy)


def compare_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            compare_exact(a, b)
    else:
        assert left == right


def build_fixture(cls, folder, model, optimizer, ema, scaler):
    trainer = cls.__new__(cls)
    trainer.model, trainer.optimizer, trainer.ema, trainer.scaler = model, optimizer, ema, scaler
    trainer.epoch, trainer.best_fitness, trainer.fitness = 2, .125, .125
    trainer.metrics = {"metrics/toy": .25}
    trainer.args = SimpleNamespace(task="detect", epochs=200, optimizer="AdamW", amp=True)
    trainer.wdir, trainer.last, trainer.best = folder, folder / "last.pt", folder / "best.pt"
    trainer.csv, trainer.save_period = folder / "absent_results.csv", 2
    return trainer


def check():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
        optimizer = torch.optim.AdamW(
            [{"params": [model[0].weight, model[1].weight]},
             {"params": [model[0].bias, model[1].bias], "weight_decay": 0.0}],
            lr=.0005, weight_decay=.0001,
        )
        (model(torch.randn(4, 3)).sum() * 1e-5).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    ema = ModelEMA(model)
    ema.update(model)
    # Disabled scaler exercises the same portable API without initializing CUDA.
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    active_before = deepcopy(optimizer.state_dict())
    model_before = deepcopy(model.state_dict())
    ema_before = deepcopy(ema.ema.state_dict())
    tensors_before = {id(value): (value.data_ptr(), value.dtype)
                      for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)}
    names = optimizer_param_names(model, optimizer)
    assert names == [["0.weight", "1.weight"], ["0.bias", "1.bias"]]
    report = {"status": "RUNNING", "python": sys.version, "torch": str(torch.__version__), "device": "cpu",
              "policy": OPTIMIZER_POLICY,
              "source_sha256": {name: hashlib.sha256((ROOT / "tools" / name).read_bytes()).hexdigest()
                                for name in ("dcc_checkpoint.py", "check_dcc_checkpoint.py")}}
    with tempfile.TemporaryDirectory(prefix="dcc_checkpoint_fixture_") as temporary:
        folder = Path(temporary).resolve()
        assert folder.is_relative_to(Path(tempfile.gettempdir()).resolve())
        native = build_fixture(RTDETRTrainer, folder / "native", model, optimizer, ema, scaler)
        dcc = build_fixture(DCCCheckpointTrainer, folder / "dcc", model, optimizer, ema, scaler)
        native.save_model()
        dcc.save_model()
        original = torch.load(native.last, map_location="cpu", weights_only=False)
        updated = torch.load(dcc.last, map_location="cpu", weights_only=False)
        assert set(updated) == set(original) | {"dcc_checkpoint"}
        assert updated["model"] is original["model"] is None
        for key in original:
            if key not in {"date", "ema", "optimizer"}:
                compare_exact(original[key], updated[key])
        datetime.fromisoformat(original["date"])
        datetime.fromisoformat(updated["date"])
        compare_exact(original["ema"].state_dict(), updated["ema"].state_dict())
        assert all(parameter.dtype == torch.float16 for parameter in updated["ema"].parameters())
        compare_exact(updated["optimizer"], active_before)
        compare_exact(optimizer.state_dict(), active_before)
        compare_exact(model.state_dict(), model_before)
        compare_exact(ema.ema.state_dict(), ema_before)
        tensors_after = {id(value): (value.data_ptr(), value.dtype)
                         for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)}
        assert tensors_after == tensors_before
        assert updated["dcc_checkpoint"]["optimizer_param_names"] == names
        assert updated["dcc_checkpoint"]["policy"] == OPTIMIZER_POLICY
        compare_exact({key: updated["dcc_checkpoint"][key] for key in native_serializer_identity()}, native_serializer_identity())
        source_ema_before = deepcopy(updated["ema"].state_dict())
        policy_audit = require_checkpoint_policy(updated)
        compare_exact(source_ema_before, updated["ema"].state_dict())
        compare_exact(active_before, updated["optimizer"])
        rejected = []

        def must_reject(name, candidate):
            try:
                require_checkpoint_policy(candidate)
            except RuntimeError:
                rejected.append(name)
            else:
                raise AssertionError("Invalid checkpoint policy accepted: " + name)

        must_reject("historical_native_half_without_policy", original)
        half_mislabeled = deepcopy(original)
        half_mislabeled["dcc_checkpoint"] = deepcopy(updated["dcc_checkpoint"])
        must_reject("half_moments_cannot_be_relabelled_fp32", half_mislabeled)
        for case in ("missing_name", "duplicate_name", "duplicate_param_id", "unknown_state_id", "missing_optimizer"):
            broken = deepcopy(updated)
            if case == "missing_name":
                broken["dcc_checkpoint"]["optimizer_param_names"][0].pop()
            elif case == "duplicate_name":
                broken["dcc_checkpoint"]["optimizer_param_names"][0][1] = broken["dcc_checkpoint"]["optimizer_param_names"][0][0]
            elif case == "duplicate_param_id":
                broken["optimizer"]["param_groups"][0]["params"][1] = broken["optimizer"]["param_groups"][0]["params"][0]
            elif case == "unknown_state_id":
                broken["optimizer"]["state"][9999] = deepcopy(next(iter(broken["optimizer"]["state"].values())))
            else:
                broken["optimizer"] = None
            must_reject(case, broken)
        try:
            optimizer_param_names(model, SimpleNamespace(param_groups=[{"params": [model[0].weight]}]))
        except RuntimeError as error:
            assert "every trainable model parameter" in str(error)
            rejected.append("missing_live_trainable_parameter")
        else:
            raise AssertionError("Incomplete live optimizer coverage accepted")
        moments = []
        for key, values in active_before["state"].items():
            before = values["exp_avg_sq"]
            prior = original["optimizer"]["state"][key]["exp_avg_sq"]
            current = updated["optimizer"]["state"][key]["exp_avg_sq"]
            moments.append({"parameter_id": key, "nonzero_live": int(before.count_nonzero()),
                            "nonzero_native_half": int(prior.count_nonzero()),
                            "nonzero_dcc_fp32": int(current.count_nonzero()),
                            "native_dtype": str(prior.dtype), "dcc_dtype": str(current.dtype),
                            "live_min": float(before.min()), "live_max": float(before.max())})
        assert any(row["nonzero_live"] > row["nonzero_native_half"] for row in moments)
        assert all(row["nonzero_live"] == row["nonzero_dcc_fp32"] for row in moments)
        # Last/best/periodic write routing and identical serialized byte reuse.
        saved = dcc.last.read_bytes()
        assert saved == dcc.best.read_bytes() == (dcc.wdir / "epoch2.pt").read_bytes()
        dcc.epoch, dcc.fitness, dcc.save_period = 3, .01, -1
        dcc.save_model()
        assert dcc.best.read_bytes() == saved and not (dcc.wdir / "epoch3.pt").exists()
        assert torch.load(dcc.last, map_location="cpu", weights_only=False)["epoch"] == 3
        # An actual half live moment is not silently promoted and mislabeled.
        # Use a matching independent model/optimizer pair for rejection testing.
        bad_model = deepcopy(model)
        bad_optimizer = torch.optim.AdamW(bad_model.parameters())
        for parameter in bad_model.parameters():
            bad_optimizer.state[parameter] = {"step": torch.tensor(1.), "exp_avg": torch.zeros_like(parameter),
                                              "exp_avg_sq": torch.zeros_like(parameter).half()}
        bad = build_fixture(DCCCheckpointTrainer, folder / "bad", bad_model, bad_optimizer, ModelEMA(bad_model), scaler)
        try:
            bad.save_model()
        except RuntimeError as error:
            assert "active FP32 AdamW moments" in str(error)
        else:
            raise AssertionError("Half active moment was incorrectly accepted")
        assert not bad.wdir.exists()
        report.update(status="PASSED", native_fields=sorted(original), extra_fields=["dcc_checkpoint"],
                      native_semantics_except_optimizer_precision=True, optimizer_copy_exact_fp32=True,
                      active_optimizer_values_and_storage_unchanged=True, active_model_and_ema_unchanged=True,
                      ema_checkpoint_dtype="torch.float16", optimizer_param_names=names, second_moments=moments,
                      last_best_periodic_routing=True, mismatched_half_live_moments_rejected=True,
                      policy_validation=policy_audit, rejected_invalid_policies=rejected,
                      native_serializer=native_serializer_identity(), formal_training="NOT_STARTED", final_test="NOT_RUN")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dcc/checkpoint_policy_fixture.json")
    args = parser.parse_args()
    report = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "second_moments": report["second_moments"]}))


if __name__ == "__main__":
    main()
