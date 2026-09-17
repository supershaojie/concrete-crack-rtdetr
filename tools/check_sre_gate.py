"""Bounded tests of the actual formal-start gate; no model, data, or training execution."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import sre_common as common


def fixture():
    identity = dict(commit="a" * 40, files={"tools/sre_common.py": "b" * 64}, manifest_sha256="c" * 64)
    dataset = dict(config="synthetic/data.yaml", config_sha256="d" * 64,
                   inventory={"val": dict(images=1728, boxes=12840, label_inventory_sha256="e" * 64)})
    locations = {key: Path("synthetic") / name for key, name in
                 (("preflight", "preflight.json"), ("initialization", "initialization.json"),
                  ("init", "controlled_init.pt"))}
    initialization = dict(source="synthetic/source.pt", output_sha256="f" * 64)
    report = dict(status="PASSED", variant="cbr_lif_sre_v1", identity=deepcopy(identity),
                  dataset=deepcopy(dataset), init_sha256="f" * 64, source_sha256=common.SOURCE_SHA256,
                  capacity=dict(status="PASSED", batch=16, imgsz=640, native_amp=True, effective_updates=2,
                                upstream_gradient_after_nonzero_output=True, resume=dict(status="PASSED")),
                  core=dict(status="PASSED"), runtime=dict(cuda_available=True))
    return dict(identity=identity, dataset=dataset, locations=locations, initialization=initialization, report=report,
                hashes={str(locations["init"]): "f" * 64, "synthetic/source.pt": common.SOURCE_SHA256})


def invoke(value, dirty=False):
    """Mock external evidence only. common.gate and common.require remain real."""
    def read_json(path):
        if Path(path) == value["locations"]["preflight"]:
            return deepcopy(value["report"])
        if Path(path) == value["locations"]["initialization"]:
            return deepcopy(value["initialization"])
        raise AssertionError("Unexpected evidence access: " + str(path))
    def digest(path):
        if str(path) not in value["hashes"]:
            raise AssertionError("Unexpected file hashing: " + str(path))
        return value["hashes"][str(path)]
    def clean():
        if dirty:
            raise RuntimeError("Synthetic dirty executable/config state rejected")
    with patch.object(common, "paths", return_value=value["locations"]), \
         patch.object(common, "read_json", side_effect=read_json), \
         patch.object(common, "code_identity", return_value=value["identity"]), \
         patch.object(common, "dataset_identity", return_value=value["dataset"]), \
         patch.object(common, "sha256", side_effect=digest), \
         patch.object(common, "require_clean", side_effect=clean):
        return common.gate("cbr_lif_sre_v1", Path("synthetic/data.yaml"))


def checks():
    results = []
    valid = fixture()
    assert invoke(valid) == valid["report"], "The complete valid fixture must be accepted"
    results.append(dict(case="valid complete evidence", result="ACCEPTED"))

    def rejected(name, change, message):
        value = fixture()
        change(value)
        try:
            invoke(value)
        except RuntimeError as error:
            assert message in str(error), f"{name} rejected for the wrong reason: {error}"
            results.append(dict(case=name, result="REJECTED", reason=str(error)))
        else:
            raise AssertionError(name + " bypassed the actual formal-start gate")

    rejected("PENDING server preflight", lambda v: v["report"].update(status="PENDING"), "not PASSED")
    rejected("stale execution commit", lambda v: v["report"]["identity"].update(commit="0" * 40), "identity changed")
    rejected("modified executable hash", lambda v: v["report"]["identity"]["files"].update({"tools/sre_common.py": "0" * 64}), "identity changed")
    rejected("wrong experiment variant", lambda v: v["report"].update(variant="sre_v1"), "identity changed")
    rejected("controlled init file changed", lambda v: v["hashes"].update({str(v["locations"]["init"]): "0" * 64}), "Controlled init")
    rejected("initialization report hash changed", lambda v: v["initialization"].update(output_sha256="0" * 64), "Controlled init")
    rejected("unified source file changed", lambda v: v["hashes"].update({"synthetic/source.pt": "0" * 64}), "Unified source")
    rejected("source report hash changed", lambda v: v["report"].update(source_sha256="0" * 64), "Unified source")
    rejected("data identity changed", lambda v: v["dataset"]["inventory"]["val"].update(label_inventory_sha256="0" * 64), "Data identity")
    rejected("only one effective update", lambda v: v["report"]["capacity"].update(effective_updates=1), "two effective updates")
    rejected("native AMP disabled", lambda v: v["report"]["capacity"].update(native_amp=False), "native AMP")
    rejected("batch reduced", lambda v: v["report"]["capacity"].update(batch=8), "B16/640")
    rejected("image size reduced", lambda v: v["report"]["capacity"].update(imgsz=320), "B16/640")
    rejected("upstream gradients unverified", lambda v: v["report"]["capacity"].update(upstream_gradient_after_nonzero_output=False), "two effective updates")
    rejected("capacity still PENDING", lambda v: v["report"]["capacity"].update(status="PENDING"), "two effective updates")
    rejected("native resume evidence missing", lambda v: v["report"]["capacity"].pop("resume"), "resume not verified")
    rejected("core lifecycle still PENDING", lambda v: v["report"]["core"].update(status="PENDING"), "lifecycle checks not passed")
    rejected("CUDA evidence absent", lambda v: v["report"]["runtime"].update(cuda_available=False), "CUDA server preflight")
    try:
        invoke(fixture(), dirty=True)
    except RuntimeError as error:
        assert "dirty executable" in str(error)
        results.append(dict(case="dirty source guard propagated", result="REJECTED", reason=str(error)))
    else:
        raise AssertionError("Dirty source guard ignored")
    sources = {name: hashlib.sha256((common.ROOT / "tools" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
               for name in ("sre_common.py", "check_sre_gate.py")}
    return dict(status="PASSED", schema="sre_actual_gate_checks_v1", count=len(results), checks=results,
                source_lf_sha256=sources, execution_commit=common.commit(),
                scope="Actual common.gate; only external paths/JSON/hashes/identity/clean-state probes mocked. No model or training execution.",
                server_preflight="NOT_RUN_BY_THIS_TEST", formal_training="NOT_STARTED", final_test="NOT_RUN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = checks()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, indent=2))
