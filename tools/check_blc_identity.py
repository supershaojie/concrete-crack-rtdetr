"""Real-data JSON identity regression plus isolated gate faults; never trains/evaluates."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import blc_common as common
import blc_server
import c19_lif_v1_data
from ultralytics.data import utils as data_utils


OLD_SHA = "010038a1115cc757d2aba6fad480e73e0a36b820"


def rejected(action, message):
    try:
        action()
    except RuntimeError as error:
        common.require(str(error) == message, "Unexpected rejection: " + str(error))
        return str(error)
    raise AssertionError("Expected rejection: " + message)


def old_dataset_identity():
    """Execute only the historical function, with the same current dependencies."""
    source = subprocess.check_output(
        ["git", "show", OLD_SHA + ":tools/blc_common.py"], cwd=common.ROOT, text=True)
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "dataset_identity")
    namespace = dict(vars(common))
    exec(compile(ast.Module(body=[function], type_ignores=[]), OLD_SHA + ":dataset_identity", "exec"), namespace)
    return namespace["dataset_identity"]


def check():
    report = dict(status="RUNNING", historical_commit=OLD_SHA,
                  scope="Local real inventory/weight hashes and synthetic gate fixtures; no model forward, training or evaluation",
                  server_init_preflight="PENDING", server_main_preflight="PENDING")
    output = common.ROOT / "outputs/blc_v1" / ("json-identity-" + common.stamp() + ".json")
    temporary = common.ROOT / "outputs/blc_v1/temporary"
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        data = common.paths("cbr_lif_blc_v1")["data"]
        legacy = old_dataset_identity()(data)
        contexts = {variant: common.evidence_context(variant) for variant in common.VARIANTS}
        current = contexts["cbr_lif_blc_v1"]
        canonical = current["data"]
        common.require(legacy["data_yaml"]["names"] == {0: "crack"}, "Historical integer class key missing")
        common.require(canonical == json.loads(json.dumps(legacy, ensure_ascii=False, allow_nan=False)),
                       "Fix changed data identity beyond JSON representation")
        report.update(data=canonical, code_sha256=current["code"]["sha256"], runtime=current["runtime"],
                      weight_hashes={v: {k: c[k] for k in ("init_sha256", "source_sha256")}
                                     for v, c in contexts.items()})
        with tempfile.TemporaryDirectory(prefix="json-identity-", dir=temporary) as tmp:
            folder = Path(tmp)
            report["roundtrip"] = {}
            for variant, context in contexts.items():
                path = folder / (variant + ".json")
                common.write_json(path, context)
                restored = json.loads(path.read_text(encoding="utf-8"))
                fields = {key: restored[key] == value for key, value in context.items()}
                common.require(all(fields.values()), "Context JSON mismatch: " + variant)
                common.require(json.loads(json.dumps(context, allow_nan=False)) == context,
                               "Context needs a lossy JSON fallback: " + variant)
                report["roundtrip"][variant] = fields
            legacy_path = folder / "historical.json"
            common.write_json(legacy_path, legacy)
            restored_legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            common.require(restored_legacy != legacy, "Historical JSON bug not reproduced")
            common.require(restored_legacy == canonical, "Historical serialization unexpectedly differs")
            report["historical_roundtrip"] = dict(equal=False, before_key_type="int", after_key_type="str")

            # Write new, exclusive fixtures only. Never touch historical experiment reports.
            fixture = folder / "gate/init-preflight-01/report.json"
            common.write_json(fixture, dict(status="PASSED", context=current))
            gate_cases = {"historical_integer_key": {**current, "data": legacy}}
            data_faults = {
                "image_list": ("inventory", "train", "split_paths_sha256"),
                "label_hash": ("inventory", "train", "label_inventory_sha256"),
                "class_name": ("data_yaml", "names", "0"),
                **{split + "_path": ("data_yaml", split) for split in ("train", "val", "test")},
                "resolved_root": ("resolved_root",), "yaml_hash": ("data_sha256",),
            }
            for name, keys in data_faults.items():
                changed = deepcopy(current)
                target = changed["data"]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] += "-changed"
                gate_cases[name] = changed
            report["evidence_gate"] = {}
            with patch.object(blc_server, "paths", return_value={"evidence": fixture.parent.parent}):
                with patch.object(blc_server, "evidence_context", return_value=current):
                    common.require(blc_server.require_evidence("cbr_lif_blc_v1", "init-preflight") == str(fixture),
                                   "Fixed identity was not accepted")
                    report["evidence_gate"]["unchanged"] = "accepted"
                for name, changed in gate_cases.items():
                    with patch.object(blc_server, "evidence_context", return_value=changed):
                        report["evidence_gate"][name] = rejected(
                            lambda: blc_server.require_evidence("cbr_lif_blc_v1", "init-preflight"),
                            "Stale init-preflight evidence: data")

            # Also exercise the strict validations BEFORE the normalization itself.
            report["strict_validation"] = {}
            spec = common.YAML.load(data)
            for field in ("names", "train", "val", "test"):
                changed = deepcopy(spec)
                changed[field] = {0: "changed"} if field == "names" else "images/changed"
                with patch.object(common.YAML, "load", return_value=changed):
                    report["strict_validation"][field] = rejected(
                        lambda: common.dataset_identity(data), "Data split/classes changed")
            changed = {**spec, "names": {"0": "crack"}}
            with patch.object(common.YAML, "load", return_value=changed):
                report["strict_validation"]["input_class_key_still_strict"] = rejected(
                    lambda: common.dataset_identity(data), "Data split/classes changed")
            inventory = canonical["inventory"]
            for split, row in inventory.items():
                for field in row:
                    changed = deepcopy(inventory)
                    value = changed[split][field]
                    changed[split][field] = value + 1 if isinstance(value, int) else "0" * 64
                    with patch.object(c19_lif_v1_data, "dataset_inventory", return_value=changed):
                        report["strict_validation"][split + "." + field] = rejected(
                            lambda: common.dataset_identity(data),
                            "Image lists/label hashes differ from the successful parent; counts alone are insufficient")
            # An equivalent inventory at a different root must still have a different identity.
            with patch.object(c19_lif_v1_data, "dataset_inventory", return_value=inventory), \
                 patch.object(data_utils, "check_det_dataset", return_value={"path": canonical["resolved_root"] + "-moved"}):
                moved = common.dataset_identity(data)
                common.require(moved["inventory"] == inventory and moved != canonical
                               and moved["resolved_root"] != canonical["resolved_root"], "Resolved root dropped")
            # A comment-only YAML change preserves parsed fields but changes the exact file hash.
            copied_yaml = folder / "comment.yaml"
            copied_yaml.write_bytes(data.read_bytes() + b"\n# JSON identity regression fixture\n")
            with patch.object(c19_lif_v1_data, "dataset_inventory", return_value=inventory), \
                 patch.object(data_utils, "check_det_dataset", return_value={"path": canonical["resolved_root"]}):
                commented = common.dataset_identity(copied_yaml)
                common.require(commented["data_yaml"] == canonical["data_yaml"]
                               and commented["data_sha256"] != canonical["data_sha256"], "YAML SHA256 dropped")
            with patch.object(blc_server, "paths", return_value={"evidence": fixture.parent.parent}):
                for name, identity in (("equivalent_inventory_moved_root", moved), ("comment_only_yaml", commented)):
                    with patch.object(blc_server, "evidence_context", return_value={**current, "data": identity}):
                        report["evidence_gate"][name] = rejected(
                            lambda: blc_server.require_evidence("cbr_lif_blc_v1", "init-preflight"),
                            "Stale init-preflight evidence: data")
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        common.write_json(output, report)
        print(output, report["status"], flush=True)


if __name__ == "__main__":
    check()
