"""Offline acceptance-v2 reassessment and negative fixtures; no model/GPU execution.

Reads the nine immutable gzip JSON reports from the supplied historical server
archive. It tests classification logic, not a new server engineering run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from dcc_acceptance import CONTRACT_VERSION, evaluate_checkpoint, evaluate_mode

ROOT = Path(__file__).resolve().parents[1]
MODES = ("cpu_fp32", "cuda_fp32", "cuda_native_amp")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def load_evidence(folder):
    provenance_bytes = (folder / "provenance.json").read_bytes()
    provenance = json.loads(provenance_bytes)
    reports, identities = {}, {}
    for mode in MODES:
        reports[mode] = {}
        for case, tail in (("parent", "parent_cbr_lif__live"), ("dcc", "dcc__live"), ("checkpoint", "dcc__checkpoint")):
            path = folder / (mode + "__" + tail + ".json.gz")
            compressed = path.read_bytes()
            raw = gzip.decompress(compressed)
            recorded = provenance["cases"][mode + "__" + tail]
            assert recorded["file"] == path.name
            assert recorded["compressed_sha256"] == digest(compressed), "Compressed historical evidence changed: " + path.name
            assert recorded["raw_sha256"] == digest(raw), "Original historical JSON changed: " + path.name
            reports[mode][case] = json.loads(raw)
            identities[path.name] = dict(path=str(path.resolve()), compressed_sha256=digest(compressed),
                                         original_json_sha256=digest(raw), original_json_bytes=len(raw))
    origin = {key: provenance[key] for key in ("archive_sha256", "actual_attachment", "historical_commit", "scope")}
    origin["provenance_sha256"] = digest(provenance_bytes)
    return reports, identities, origin


def fail_row(comparison, predicate):
    name = next(key for key in comparison["tensors"] if predicate(key))
    comparison["tensors"][name].update(equal=False, allclose=False, max_abs=.25)
    # Deliberately leave aggregate PASSED/true fields untouched. A robust checker
    # must inspect leaf evidence instead of trusting a forged summary status.


def reassess(folder):
    reports, identities, origin = load_evidence(folder)
    before = digest(encoded(reports))
    accepted, raw = {}, {}
    expected = {"cpu_fp32": (0, 0, 0), "cuda_fp32": (3, 3, 5), "cuda_native_amp": (6, 6, 7)}
    for mode, cases in reports.items():
        accepted[mode] = evaluate_mode(mode, cases["checkpoint"], cases["parent"], cases["dcc"])
        assert accepted[mode]["restoration"]["status"] == "PASSED", accepted[mode]
        assert accepted[mode]["status"] == ("PASSED" if mode == "cpu_fp32" else "PRECISION_NOTE"), accepted[mode]
        raw[mode] = {}
        for case in ("parent", "dcc", "checkpoint"):
            source = cases[case]
            failed = {name: row for name, row in source["state_comparison"]["tensors"].items()
                      if name.startswith("state.model.") and not row["allclose"]}
            dcc_grad = {name: row for name, row in source["gradient_comparison"]["tensors"].items()
                        if ".dcc." in name and not row["allclose"]}
            raw[mode][case] = dict(original_status=source["status"],
                raw_next_update_allclose=source["raw_next_update_allclose"],
                raw_all_states_allclose=source["raw_all_states_allclose"],
                model_failed_tensor_count=len(failed), model_failed_tensors=failed,
                dcc_gradient_failed_tensors=dcc_grad)
        assert tuple(raw[mode][case]["model_failed_tensor_count"] for case in ("parent", "dcc", "checkpoint")) == expected[mode]
    for case in ("dcc", "checkpoint"):
        row = raw["cuda_native_amp"][case]["dcc_gradient_failed_tensors"]["state.model.17.dcc.W_o.weight"]
        assert row["allclose"] is False and row["shape"] == [256, 32, 1, 1]
        assert row["exceeded_fraction"] == 1 / 8192
        assert row["atol"] == 2e-5 and row["rtol"] == 2e-4
    assert digest(encoded(reports)) == before, "Acceptance evaluator mutated historical reports"

    fixtures = []

    def reject(name, mutation):
        cases = deepcopy(reports["cuda_native_amp"])
        mutation(cases)
        result = evaluate_mode("cuda_native_amp", cases["checkpoint"], cases["parent"], cases["dcc"])
        assert result["status"] == "FAILED", name + " was incorrectly accepted: " + repr(result)
        fixtures.append(dict(name=name, status="PASSED", expected="FAILED", actual=result["status"], failures=result["failures"]))

    reject("mapping_group_order_mismatch", lambda c: c["checkpoint"]["serializer"]["optimizer_param_names"][0].reverse())
    reject("duplicate_parameter_id", lambda c: c["checkpoint"]["checkpoint_source_validation"]["parameter_ids_by_group"][0].__setitem__(1, 0))
    reject("historical_half_policy", lambda c: c["checkpoint"].__setitem__("checkpoint_policy", "historical_half"))
    reject("saved_optimizer_leaf_changed", lambda c: fail_row(c["checkpoint"]["saved_optimizer_live_exact"], lambda k: "exp_avg_sq" in k))
    reject("native_optimizer_step_changed", lambda c: fail_row(c["checkpoint"]["restoration"]["native_vs_saved_checkpoint"], lambda k: k.endswith(".step")))
    reject("native_ema_weight_changed", lambda c: fail_row(c["checkpoint"]["restoration"]["native_vs_saved_checkpoint"], lambda k: k.startswith("state.ema.")))
    reject("native_scaler_mismatch", lambda c: c["checkpoint"]["restoration"]["native_vs_saved_checkpoint"]["metadata_errors"].append("state.scaler.scale: value differs"))
    reject("native_epoch_mismatch", lambda c: c["checkpoint"]["restoration"].__setitem__("restored_start_epoch", 9))
    reject("ema_update_counter_mismatch", lambda c: c["checkpoint"]["restoration"]["native_vs_saved_checkpoint"]["metadata_errors"].append("state.ema_updates: value differs"))
    reject("missing_saved_checkpoint_oracle", lambda c: c["checkpoint"]["restoration"].pop("native_vs_saved_checkpoint"))
    reject("missing_replay_evidence", lambda c: c["checkpoint"].pop("same_gradient_replay"))
    reject("restored_storage_alias", lambda c: c["checkpoint"]["restoration"]["storage_independence"].__setitem__("shared_storage_count", 1))
    reject("replay_storage_alias", lambda c: c["checkpoint"]["same_gradient_replay"]["storage_independence"].__setitem__("shared_storage_count", 1))
    reject("amp_replay_disabled", lambda c: c["checkpoint"]["same_gradient_replay"].__setitem__("amp_scaler_path", False))
    reject("replay_optimizer_leaf_changed", lambda c: fail_row(c["checkpoint"]["same_gradient_replay"], lambda k: "exp_avg_sq" in k))
    reject("missing_parent_live_control", lambda c: c.__setitem__("parent", {}))
    reject("missing_dcc_live_initial_state", lambda c: c["dcc"].pop("initial_state"))
    reject("dcc_live_alias", lambda c: c["dcc"]["storage_independence"].__setitem__("shared_storage_count", 1))

    def nonfinite(cases, case, section):
        comparison = cases[case][section]
        name = next(name for name in comparison["tensors"] if ".dcc." in name)
        comparison["tensors"][name].update(finite=False, allclose=False, equal=False, max_abs=None,
                                            relative_L2=None, exceeded_fraction=None)

    reject("nonfinite_dcc_live_gradient", lambda c: nonfinite(c, "dcc", "gradient_comparison"))
    reject("nonfinite_checkpoint_model", lambda c: nonfinite(c, "checkpoint", "state_comparison"))
    reject("changed_raw_forward_hash", lambda c: c["checkpoint"].__setitem__("captured_forward_hashes_exact", False))
    reject("raw_false_relabelled_true", lambda c: c["checkpoint"].__setitem__("raw_next_update_allclose", True))
    reject("raw_tolerance_changed", lambda c: c["checkpoint"].__setitem__("atol", .01))

    # A is intentionally independent of unrelated top-level historical status.
    copied = deepcopy(reports["cuda_native_amp"]["checkpoint"])
    copied["status"] = "UNRESOLVED"
    assert evaluate_checkpoint(copied, "cuda_native_amp")["status"] == "PASSED"
    assert digest(encoded(reports)) == before, "Negative fixtures mutated source evidence"
    return dict(status="PASSED", contract_version=CONTRACT_VERSION,
                report_kind="historical_evidence_reassessment", created=datetime.now(timezone.utc).isoformat(),
                fresh_server_acceptance=False, model_execution="NOT_RUN", cuda_execution="NOT_RUN",
                formal_training="NOT_STARTED", final_test="NOT_RUN", admission="BLOCKED",
                full_engineering_eligible=False, historical_origin=origin, source_evidence=identities,
                source_reports_canonical_sha256_before=before, source_reports_canonical_sha256_after=digest(encoded(reports)),
                raw_evidence_unchanged=True, acceptance=accepted, preserved_raw_results=raw,
                negative_fixtures=fixtures, negative_fixture_count=len(fixtures),
                statement="Offline reclassification of original server evidence under the authorized A/B contract; not a fresh runtime or full engineering acceptance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/dcc/acceptance_v2/evidence")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/dcc/acceptance_v2/offline_reassessment.json")
    args = parser.parse_args()
    result = reassess(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=result["status"], negative_fixtures=result["negative_fixture_count"],
                         acceptance={mode: value["status"] for mode, value in result["acceptance"].items()},
                         output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
