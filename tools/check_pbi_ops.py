"""Bounded fault injection for fail-closed admission; synthetic fixtures never authorize training."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pbi_common import PBI_KEYS, require, sha256, write_json
import train_pbi as train
from pbi_acceptance import CONTRACT_VERSION


def check(output):
    results = []
    with TemporaryDirectory(prefix="pbi_gate_faults_") as temporary:
        root = Path(temporary)
        init = root / "fixture_init.bin"
        init.write_bytes(b"synthetic admission test; not a checkpoint")
        identity = {"fixture.py": "a" * 64}
        runtime = {key: "fixture" for key in ("python", "torch", "cuda", "gpu", "ultralytics", "worktree")}
        head = "b" * 40
        environment = {"status": "PASSED", "fixture_only": True}
        dataset = {"splits": {split: dict(images=1, boxes=1, paths_sha256="paths", labels_sha256="labels")
                               for split in ("train", "val", "test")}}
        common = dict(status="PASSED", contract_version=CONTRACT_VERSION, code_identity=identity,
                      runtime=runtime, git_head=head, variant="cbr_lif_pbi_v1")
        capacity = dict(common, report_kind="native_capacity", init_sha256=sha256(init),
                        server_environment=environment, dataset_identity=dataset,
                        checkpoint_policy="optimizer_fp32_v1",
                        capacity=dict(batch=16, imgsz=640, AMP=True, effective_updates=2, observed_batches=2),
                        recipe={"epochs": 200}, actual_recipe={"epochs": 200},
                        optimizer_coverage={"every_parameter_exactly_once": True},
                        steps=[dict(effective=True, gradients={name: {"finite": True, "norm": 1.0}
                                                              for name in PBI_KEYS}) for _ in range(2)])
        checks = dict(common, report_kind="full_preflight_engineering", initialization_sha256=sha256(init),
                      data_identity={split: dict(images=1, boxes=1, split_paths_sha256="paths",
                                                 label_inventory_sha256="labels")
                                     for split in ("train", "val", "test")})
        cases = [
            ("missing prerequisite", "checks", (), None, "Missing/changed bound prerequisite"),
            ("partial report", "checks", ("report_kind",), "partial_resume_diagnostic", "Wrong/old report"),
            ("old contract", "checks", ("contract_version",), "old", "Wrong/old report"),
            ("pending report", "capacity", ("status",), "PENDING", "Incomplete report"),
            ("false pass with pending", "checks", ("pending",), ["cuda"], "Incomplete report"),
            ("stale source", "checks", ("code_identity",), {}, "Code/config differs"),
            ("wrong runtime", "capacity", ("runtime", "torch"), "other", "actual training environment"),
            ("wrong HEAD", "checks", ("git_head",), "c" * 40, "full HEAD differs"),
            ("wrong server target", "capacity", ("server_environment",), {}, "server environment not verified"),
            ("wrong variant", "capacity", ("variant",), "pbi_v1", "Capacity init/variant/status changed"),
            ("wrong initialization", "checks", ("initialization_sha256",), "other", "Engineering initialization/variant"),
            ("wrong data", "capacity", ("dataset_identity",), {}, "data path/label identity changed"),
            ("engineering data mismatch", "checks", ("data_identity", "val", "images"), 2, "engineering/capacity dataset differs"),
            ("reduced batch", "capacity", ("capacity", "batch"), 8, "B16/640 native AMP"),
            ("disabled AMP", "capacity", ("capacity", "AMP"), False, "B16/640 native AMP"),
            ("changed recipe", "capacity", ("actual_recipe", "epochs"), 150, "Capacity recipe changed"),
            ("missing optimizer parameter", "capacity", ("optimizer_coverage",), {}, "optimizer coverage missing"),
            ("counted calls not updates", "capacity", ("steps",), [], "Capacity update evidence missing"),
            ("missing PBI gradients", "capacity", ("steps",), [{"effective": True}] * 2, "gradient startup missing"),
        ]
        with patch.multiple(train, code_identity=lambda: identity, runtime=lambda: runtime,
                            git_head=lambda: head, server_environment=lambda: environment,
                            dataset_identity=lambda path: dataset,
                            recipe=lambda *a, **k: ({"epochs": 200}, [])):
            for label, selected, keys, value, expected in cases:
                payloads = {"capacity": deepcopy(capacity), "checks": deepcopy(checks)}
                if keys:
                    target = payloads[selected]
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = value
                for name, payload in payloads.items():
                    write_json(root / (name + ".json"), payload)
                try:
                    train.strict_gate(root / "capacity.json", root / "checks.json", "cbr_lif_pbi_v1", init, root / "data.yaml")
                except (RuntimeError, AssertionError) as error:
                    require(expected in str(error), label + " rejected for unexpected reason: " + str(error))
                    results.append(dict(case=label, status="PASSED", rejected=str(error)))
                else:
                    raise AssertionError("Unsafe admission: " + label)
    report = dict(status="PASSED", report_kind="admission_fault_injection_only", cases=results,
                  engineering_pass_evidence=False, formal_start_authorized=False,
                  statement="Synthetic adversarial fixtures test rejection only; they are not model/GPU evidence.")
    write_json(output, report)
    print(f"PASSED {len(results)} fail-closed admission cases; no training permission generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Existing evidence protected")
    check(args.output)
