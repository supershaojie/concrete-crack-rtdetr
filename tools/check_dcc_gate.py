"""Synthetic negative tests for formal admission; never detector/capacity evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dcc_common import sha256, write_json, require
import train_dcc


def check():
    variant = "cbr_lif_dcc_v1"
    identity = {"fixture": "current-source"}
    environment = {key: "fixture-only" for key in ("python", "torch", "cuda", "gpu", "ultralytics", "worktree")}
    cases = {}
    with TemporaryDirectory(prefix="dcc_gate_") as directory:
        folder = Path(directory)
        init = folder / "synthetic-init.txt"
        init.write_text("not model weights", encoding="utf-8")
        capacity = dict(status="PASSED", variant=variant, init_sha256=sha256(init),
                        code_identity=identity, runtime=environment, dataset_identity={"synthetic": True},
                        capacity=dict(batch=16, imgsz=640, AMP=True, effective_updates=2, observed_batches=5))
        checks = dict(status="PASSED", variant=variant, initialization_sha256=sha256(init),
                      code_identity=identity, runtime=environment, devices={})
        for device in ("cpu_fp32", "cuda_fp32", "cuda_native_amp"):
            checks["devices"][device] = dict(status="PASSED", fusion=dict(status="PASSED", cuda_half=dict(status="PASSED")),
                lifecycle=dict(status="PASSED", checkpoint_policy="optimizer_fp32_v1", raw_next_update_allclose=True,
                               restoration=dict(status="PASSED"),
                               same_gradient_replay=dict(status="PASSED", amp_scaler_path=device == "cuda_native_amp")))

        def attempt(name, cap, engineering, should_pass):
            write_json(folder / "capacity.json", cap)
            write_json(folder / "checks.json", engineering)
            error = None
            try:
                train_dcc.strict_gate(folder / "capacity.json", folder / "checks.json", variant, init, folder / "data.yaml")
                passed = True
            except RuntimeError as exception:
                passed, error = False, str(exception)
            require(passed == should_pass, "Gate fixture result incorrect: " + name)
            cases[name] = dict(expected_accepted=should_pass, observed_accepted=passed, reason=error)

        with patch.multiple(train_dcc, code_identity=lambda: identity, runtime=lambda: environment,
                            dataset_identity=lambda _: {"synthetic": True}):
            attempt("complete_synthetic_contract", capacity, checks, True)
            for name, mutation in (
                ("raw_false_with_exact_replay", lambda x: x.update(raw_next_update_allclose=False, fixed_gradient_replay_exact=True)),
                ("old_half_serializer", lambda x: x.update(checkpoint_policy="native_optimizer_half")),
                ("missing_complete_restoration", lambda x: x.pop("restoration")),
                ("model_only_replay", lambda x: x.pop("same_gradient_replay")),
                ("amp_replay_disabled", lambda x: x["same_gradient_replay"].update(amp_scaler_path=False)),
            ):
                modified = deepcopy(checks)
                mutation(modified["devices"]["cuda_native_amp"]["lifecycle"])
                attempt(name, capacity, modified, False)
            stale = deepcopy(capacity)
            stale["code_identity"] = {"fixture": "old-source"}
            attempt("stale_capacity_after_code_change", stale, checks, False)
            stale = deepcopy(checks)
            stale["code_identity"] = {"fixture": "old-source"}
            attempt("stale_engineering_after_code_change", capacity, stale, False)
    return dict(status="PASSED", scope="synthetic gate regression only; no training/capacity certification", cases=cases,
                formal_training="NOT_STARTED", final_test="NOT_RUN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Existing fixture report protected")
    write_json(args.output, check())
    print("PASSED: synthetic strict-gate rejection checks")
