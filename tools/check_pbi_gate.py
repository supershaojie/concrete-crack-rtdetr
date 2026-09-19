"""Exercise admission schema with real audit leaves and explicitly synthetic capacity/environment.

This is an API compatibility and corruption-rejection test only. It cannot create
a start permit, certify server capacity, or rewrite the supplied audit reports.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pbi_common import PBI_KEYS, require, sha256, write_json
from pbi_acceptance import CONTRACT_VERSION, evaluate_mode
import train_pbi as train


def run(args):
    observed = json.loads(args.checks.read_text(encoding="utf-8"))
    initial = json.loads(args.initialization_report.read_text(encoding="utf-8"))
    mathematical = json.loads(args.math_report.read_text(encoding="utf-8"))
    require(observed.get("status") in {"PASSED", "PRECISION_NOTE"}, "Real engineering leaves are incomplete")
    train._controlled_pass(initial, observed["variant"])
    train._math_pass(mathematical)
    init = Path(initial["output"])
    require(init.is_file() and sha256(init) == initial["output_sha256"], "Actual initial checkpoint identity changed")
    data = {"fixture_only": True, "splits": {split: dict(images=row["images"], boxes=row["boxes"],
            paths_sha256=row["split_paths_sha256"], labels_sha256=row["label_inventory_sha256"])
            for split, row in observed["data_identity"].items()}}
    identity = {"TEST_FIXTURE_NOT_PERMISSION": "synthetic identity"}
    environment = {"status": "PASSED", "TEST_FIXTURE_NOT_PERMISSION": True}
    runtime, head = train.runtime(), train.git_head()
    results = []
    with TemporaryDirectory(prefix="pbi_gate_schema_") as temporary:
        root = Path(temporary)
        for report in (observed, initial, mathematical):
            report.update(code_identity=identity, runtime=runtime, git_head=head)
        initial_path, math_path = root / "init.json", root / "math.json"
        write_json(initial_path, initial)
        write_json(math_path, mathematical)
        observed["prerequisites"] = {name: dict(path=str(path), sha256=sha256(path))
                                     for name, path in (("initialization", initial_path), ("math", math_path))}
        # Refresh only the status summaries in the isolated fixture according to
        # the current reducer. Raw real model/optimizer/gradient evidence stays.
        for mode, record in observed["devices"].items():
            acceptance = evaluate_mode(mode, record["lifecycle"], record["live_controls"]["parent"],
                                       record["live_controls"]["pbi"])
            require(acceptance["status"] in {"PASSED", "PRECISION_NOTE"}, "Actual raw A/B leaves rejected")
            record["acceptance"] = acceptance
            half_note = record.get("fusion", {}).get("cuda_half", {}).get("status") == "PRECISION_NOTE"
            record["status"] = "PRECISION_NOTE" if half_note else acceptance["status"]
        observed["status"] = "PRECISION_NOTE" if any(r["status"] == "PRECISION_NOTE" for r in observed["devices"].values()) else "PASSED"
        recipe, _ = train.recipe(observed["variant"], init)
        capacity = dict(status="PASSED", report_kind="native_capacity", contract_version=CONTRACT_VERSION,
                        code_identity=identity, runtime=runtime, git_head=head, server_environment=environment,
                        variant=observed["variant"], init_sha256=sha256(init), dataset_identity=data,
                        checkpoint_policy="optimizer_fp32_v1", recipe=recipe, actual_recipe=recipe,
                        capacity=dict(batch=16, imgsz=640, AMP=True, effective_updates=2, observed_batches=2),
                        optimizer_coverage={"every_parameter_exactly_once": True},
                        steps=[dict(effective=True, gradients={name: {"finite": True, "norm": 1.0}
                                                              for name in PBI_KEYS}) for _ in range(2)])
        capacity_path, checks_path = root / "capacity.json", root / "checks.json"
        write_json(capacity_path, capacity)
        write_json(checks_path, observed)
        with patch.multiple(train, code_identity=lambda: identity, runtime=lambda: runtime,
                            server_environment=lambda: environment, git_head=lambda: head,
                            dataset_identity=lambda _: data):
            admission = train.strict_gate(capacity_path, checks_path, observed["variant"], init, root / "data.yaml")
            results.append(dict(case="full_schema_real_leaves_synthetic_capacity", status="PASSED",
                                fixture_result=admission["status"]))
            cases = [("missing native rebuild", ("controlled_initialization", "native_nc1_rebuild"), {}),
                     ("lost fused branch", ("devices", "cpu_fp32", "fusion", "pbi_calls"), 0),
                     ("missing continuous PBI output", ("devices", "cpu_fp32", "fusion", "strict_fp32", "continuous", "pbi_output"), {}),
                     ("missing same-gradient replay", ("devices", "cuda_native_amp", "lifecycle", "same_gradient_replay"), {}),
                     ("missing half formula", ("devices", "cuda_fp32", "fusion", "cuda_half", "explicit_formula"), {})]
            for label, keys, value in cases:
                damaged = deepcopy(observed)
                cursor = damaged
                for key in keys[:-1]:
                    cursor = cursor[key]
                cursor[keys[-1]] = value
                write_json(checks_path, damaged)
                try:
                    train.strict_gate(capacity_path, checks_path, observed["variant"], init, root / "data.yaml")
                except (RuntimeError, AssertionError) as error:
                    results.append(dict(case=label, status="PASSED", rejection=str(error)))
                else:
                    raise AssertionError("Corrupted audit admitted: " + label)
    result = dict(status="PASSED", report_kind="TEST_FIXTURE_NOT_PERMISSION", cases=results,
                  source_reports={str(path): sha256(path) for path in
                                  (args.checks, args.initialization_report, args.math_report)},
                  synthetic=["capacity", "runtime/environment acceptance", "freshness identities", "derived fixture summaries"],
                  original_reports_unmodified=True, formal_start_authorized=False, server_capacity="PENDING")
    write_json(args.output, result)
    print(json.dumps(dict(status=result["status"], cases=len(results), formal_start_authorized=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checks", "initialization-report", "math-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Existing fixture report protected")
    run(args)
