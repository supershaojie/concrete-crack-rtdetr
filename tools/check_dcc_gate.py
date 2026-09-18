"""Offline full-admission regressions. Historical evidence is used ONLY as fixture data."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dcc_common import ROOT, sha256, write_json, require
from dcc_acceptance import CONTRACT_VERSION, evaluate_mode
import train_dcc

EVIDENCE = ROOT / "docs/dcc/acceptance_v2/evidence"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def case(mode, model, kind):
    return json.loads(gzip.decompress((EVIDENCE / (mode + "__" + model + "__" + kind + ".json.gz")).read_bytes()))


def fixture_dtype_schema(value):
    """Synthetic v2 schema only; never infer or rewrite historical evidence dtypes."""
    if isinstance(value, dict):
        if "shape" in value and "finite" in value:
            value.setdefault("dtype", "torch.float32")
        for item in value.values():
            fixture_dtype_schema(item)
    elif isinstance(value, list):
        for item in value:
            fixture_dtype_schema(item)


def check():
    identity = {"fixture_only": "never a current-source certificate"}
    environment = {k: "fixture-only" for k in ("python", "torch", "cuda", "gpu", "ultralytics", "worktree")}
    variant = "cbr_lif_dcc_v1"
    cases = {}
    with TemporaryDirectory(prefix="dcc_v2_gate_") as directory:
        folder = Path(directory)
        init = folder / "synthetic-init.txt"
        init.write_text("synthetic fixture, not weights", encoding="utf-8")
        reports = {
            "capacity": load(EVIDENCE / "preflight.json"),
            "checks": load(ROOT / "docs/dcc/checks_local.json"),
            "initialization": load(ROOT / "docs/dcc/initialization_cbr_lif_dcc_v1.json"),
            "math": load(ROOT / "docs/dcc/module_checks_final.json"),
        }
        kinds = dict(capacity="native_capacity", checks="full_preflight_engineering",
                     initialization="controlled_initialization_audit", math="dcc_math_audit")
        for key, report in reports.items():
            report.update(contract_version=CONTRACT_VERSION, report_kind=kinds[key],
                          code_identity=identity, runtime=environment)
        reports["capacity"]["init_sha256"] = reports["checks"]["initialization_sha256"] = sha256(init)
        reports["initialization"]["output_sha256"] = sha256(init)
        for mode in ("cpu_fp32", "cuda_fp32", "cuda_native_amp"):
            observed = reports["checks"]["devices"][mode]
            legacy = observed["lifecycle"]
            lifecycle = case(mode, "dcc", "checkpoint")
            # These full-lifecycle fixture fields come from the full historical
            # check, never from the partial resume report being promoted.
            for key in ("state_dict_exact", "full_model_exact", "native_get_model_nonzero_preserved",
                        "learned_nc80_to1", "ema_own_copy"):
                lifecycle[key] = legacy[key]
            controls = dict(parent=case(mode, "parent_cbr_lif", "live"), dcc=case(mode, "dcc", "live"))
            for item in (lifecycle, controls["parent"], controls["dcc"]):
                item["contract_version"] = CONTRACT_VERSION
                fixture_dtype_schema(item)
            acceptance = evaluate_mode(mode, lifecycle, controls["parent"], controls["dcc"])
            require(acceptance["status"] in {"PASSED", "PRECISION_NOTE"}, "Historical fixture classification failed: " + str(acceptance))
            observed.update(lifecycle=lifecycle, live_controls=controls, acceptance=acceptance,
                            status=acceptance["status"])
        reports["checks"]["status"] = "PRECISION_NOTE"
        paths = {name: folder / (name + ".json") for name in reports}

        def attempt(name, mutation=None, expected=False, break_reference=False):
            current = deepcopy(reports)
            if mutation:
                mutation(current)
            for key in ("initialization", "math"):
                write_json(paths[key], current[key])
            current["checks"]["prerequisites"] = {
                k: dict(path=str(paths[k].resolve()), sha256=sha256(paths[k])) for k in ("initialization", "math")}
            if break_reference:
                current["checks"]["prerequisites"].pop("math")
            for key in ("capacity", "checks"):
                write_json(paths[key], current[key])
            error = None
            try:
                admission = train_dcc.strict_gate(paths["capacity"], paths["checks"], variant, init, folder / "data.yaml")
                accepted = True
                require(admission["status"] == "PRECISION_NOTE", "Raw CUDA differences hidden by gate")
            except (RuntimeError, ValueError, KeyError, TypeError) as exception:
                accepted, error = False, str(exception)
            require(accepted == expected, "Incorrect full-gate fixture: " + name + " error=" + str(error))
            cases[name] = dict(expected_accepted=expected, observed_accepted=accepted, reason=error)

        def checkpoint(x):
            return x["checks"]["devices"]["cuda_native_amp"]["lifecycle"]

        def state_row(x, scope, suffix):
            tensors = checkpoint(x)[scope]["tensors"]
            key = next(k for k in tensors if suffix in k)
            tensors[key]["equal"] = False

        with patch.multiple(train_dcc, code_identity=lambda: identity, runtime=lambda: environment,
                            dataset_identity=lambda _: reports["capacity"]["dataset_identity"]):
            attempt("complete_fixture_accepts_precision_note_with_raw_false", expected=True)
            changes = [
                ("partial_report_cannot_authorize", lambda x: x["checks"].update(report_kind="partial_resume_diagnostic")),
                ("old_contract_rejected", lambda x: x["checks"].pop("contract_version")),
                ("stale_capacity", lambda x: x["capacity"].update(code_identity={})),
                ("stale_engineering", lambda x: x["checks"].update(code_identity={})),
                ("stale_initialization", lambda x: x["initialization"].update(code_identity={})),
                ("stale_math", lambda x: x["math"].update(code_identity={})),
                ("math_cuda_pending", lambda x: x["math"]["devices"][1].update(status="PENDING")),
                ("missing_math_fusion", lambda x: x["math"]["devices"][0].pop("nonzero_wrapper_fusion")),
                ("wrong_wiring", lambda x: x["checks"]["wiring_640"].update(concat=[17, 16])),
                ("missing_network_fusion", lambda x: x["checks"]["devices"]["cuda_fp32"].pop("fusion")),
                ("wrong_recipe", lambda x: x["capacity"]["actual_recipe"].update(lr0=.001)),
                ("wrong_engineering_dataset", lambda x: x["checks"]["data_identity"]["train"].update(images=1)),
                ("capacity_missing_updates", lambda x: x["capacity"].update(steps=[])),
                ("missing_parent_control", lambda x: x["checks"]["devices"]["cuda_native_amp"]["live_controls"].pop("parent")),
                ("missing_complete_restore", lambda x: checkpoint(x).pop("restoration")),
                ("missing_current_dtype_evidence", lambda x: next(r for r in checkpoint(x)["restoration"]["tensors"].values() if "finite" in r).pop("dtype")),
                ("missing_saved_byte_oracle", lambda x: checkpoint(x)["restoration"].pop("native_vs_saved_checkpoint")),
                ("wrong_parameter_map", lambda x: checkpoint(x)["checkpoint_source_validation"]["optimizer_param_names"][0].reverse()),
                ("optimizer_state_corruption", lambda x: state_row(x, "restoration", "exp_avg_sq")),
                ("ema_state_corruption", lambda x: state_row(x, "same_gradient_replay", "state.ema.")),
                ("scaler_state_corruption", lambda x: checkpoint(x)["same_gradient_replay"]["metadata_errors"].append("state.scaler.scale differs")),
                ("shared_storage", lambda x: checkpoint(x)["restoration"]["storage_independence"].update(shared_storage_count=1)),
                ("amp_path_removed", lambda x: checkpoint(x)["same_gradient_replay"].update(amp_scaler_path=False)),
                ("overflow_evidence_removed", lambda x: checkpoint(x)["same_gradient_replay"].pop("overflow_fixture")),
                ("raw_state_nonfinite", lambda x: next(iter(checkpoint(x)["state_comparison"]["tensors"].values())).update(finite=False)),
                ("cpu_raw_not_exact", lambda x: x["checks"]["devices"]["cpu_fp32"]["lifecycle"].update(raw_next_update_allclose=False)),
                ("raw_cuda_false_forged_true", lambda x: checkpoint(x).update(raw_next_update_allclose=True)),
                ("weakened_tolerance", lambda x: checkpoint(x).update(atol=.01)),
            ]
            for name, mutation in changes:
                attempt(name, mutation)
            attempt("missing_math_reference", break_reference=True)
    return dict(status="PASSED", contract_version=CONTRACT_VERSION,
                scope="Offline admission regressions using composite historical fixtures; NOT a new detector, capacity, or server acceptance",
                cases=cases, new_server_full_preflight="PENDING", formal_training="NOT_STARTED", final_test="NOT_RUN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Existing fixture report protected")
    result = check()
    write_json(args.output, result)
    print("PASSED:", len(result["cases"]), "offline full-admission regression cases")
