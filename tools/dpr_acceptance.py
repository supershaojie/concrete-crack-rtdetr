"""Approved R1 policy. Raw measurements are immutable; decisions are recomputed.

This module deliberately does not import a model or execute training. Partial
supplements authorize only the next engineering check, never formal training.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CONTRACT = "dpr_acceptance_v2"
FULL_SCHEMA = "dpr_full_preflight_v3"
SUPPLEMENT_SCHEMA = "dpr_r1_supplement_v1"
PROFILE = "native_amp_native_half_epoch_final_independent_fp32_v1"
ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = "docs/dpr/acceptance_review_bb768787.md"
POLICY_SHA256 = "d8c0b79c54654e0bf263fb21a11bd22ac5d9fde5446f2e1a6fcd6758b8d9ea0e"
UNSUPPORTED = "NOT_SUPPORTED_UNDER_THIS_PROFILE"
MODES = ("cpu_fp32", "cuda_fp32", "cuda_native_amp")
TARGET = "model.5.blocks.1.branch2b."
ADDED = tuple(TARGET + x for x in ("dpr_cd", "dpr_hd", "dpr_vd", "dpr_ad"))


def need(condition, message):
    if not condition:
        raise RuntimeError("R1 BLOCKED: " + message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def policy():
    # Git text normalization makes this identity portable across Windows/Linux.
    raw = (ROOT / POLICY_PATH).read_bytes().replace(b"\r\n", b"\n")
    need(hashlib.sha256(raw).hexdigest() == POLICY_SHA256, "approved policy document changed")
    return dict(contract=CONTRACT, profile=PROFILE, approval="USER_APPROVED_R1",
                document=POLICY_PATH, sha256=POLICY_SHA256)


def scope():
    return dict(profile=PROFILE, native_amp_training="REQUIRES_FULL_ADMISSION",
                native_half_epoch_final_validation="REQUIRES_H0_H1_H2",
                independent_evaluation="FP32_ONLY", fp16_cross_precision_H3=UNSUPPORTED,
                fp16_cross_conversion_H4=UNSUPPORTED, arbitrary_half_deployment="NOT_AUTHORIZED")


def validate_scope(value):
    need(value == scope(), "unknown/missing capability scope")
    return True


def artifact(record, files=True):
    need(isinstance(record, dict) and len(record.get("sha256", "")) == 64 and record.get("bytes", 0) > 0,
         "missing evidence artifact identity")
    if files:
        path = Path(record["path"])
        need(path.is_file() and path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
             "evidence artifact missing/changed: " + str(path))


def raw_metric(row, atol=2e-5, rtol=2e-4, passing=True):
    need(row.get("finite") is True, "nonfinite/missing raw metric")
    need(row.get("atol") == atol and row.get("rtol") == rtol, "changed/missing tolerance")
    for key in ("max_abs", "relative_L2"):
        need(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) and row[key] >= 0, "invalid " + key)
    fraction = row.get("exceeded_fraction", row.get("exceed_fraction"))
    need(isinstance(fraction, (int, float)) and 0 <= fraction <= 1, "missing exceeded fraction")
    need(row.get("raw_allclose") is (fraction == 0), "inconsistent raw result")
    if passing:
        need(row["raw_allclose"], "original raw tolerance failed")


def table(value, exact=False, passing=False):
    rows = value.get("tensors", {})
    need(rows and value.get("metadata_errors") == [] and value.get("finite") is True, "incomplete tensor table")
    failed = []
    for name, row in rows.items():
        need(row.get("finite") is True, "nonfinite " + name)
        if row.get("floating", True):
            raw_metric(row, passing=passing)
        else:
            need(row.get("equal") is True, "integer/step differs: " + name)
        if not row.get("raw_allclose"):
            failed.append(name)
            need(row.get("exceeded_coordinates"), "failed coordinates missing: " + name)
        if exact or name.endswith(".step"):
            need(row.get("equal") is True, "exact state differs: " + name)
    need(set(failed) == set(value.get("failed_tensors", [])), "raw failed-name inventory changed")
    return rows


def validate_a(a, mode):
    need(a.get("status") == "PASSED", "B1 fresh A missing/failed")
    for key in ("saved_live_optimizer_exact", "saved_byte_restoration_exact", "optimizer_names_groups_exact",
                "no_shared_storage", "same_gradient_complete_state_exact", "nonzero_dpr_preserved",
                "state_dict_reload_exact", "full_model_reload_exact", "native_trainer_rebuild_exact", "effective_step_exact"):
        need(a.get(key) is True, "A " + key)
    p = a.get("policy", {})
    need(p.get("policy") == "optimizer_fp32_v1" and p.get("original_tensors_inspected_before_cast") is True
         and p.get("fp32_moment_tensors", 0) > 0, "A original saved optimizer bytes")
    for key in ("saved_live_optimizer_comparison", "restoration", "replay"):
        entry = a.get(key, {})
        need(entry.get("tensors") and entry.get("metadata_errors") == [] and entry.get("exact") is True
             and all(r.get("finite") is True and r.get("equal") is True for r in entry["tensors"].values()), "A " + key)
    need(len(a.get("replay_steps", [])) == 2 and all(r.get("effective_update") is True for r in a["replay_steps"]), "A steps")
    if mode == "cpu_fp32":
        need(a.get("cpu_native_continuation_exact") is True, "CPU native A continuation")
    if mode == "cuda_native_amp":
        need(a.get("overflow_skip_exact") is True and len(a.get("overflow_rows", [])) == 2
             and all(r.get("effective_update") is False and 0 < r.get("scale_after", 0) < r.get("scale_before", 0)
                     for r in a["overflow_rows"]), "A real AMP overflow/skip")


def validate_trace(entry):
    rows = entry.get("comparisons", {})
    for name in ("effective_kernel", "target", "P3", "P4", "P5", "encoder_features", "candidate_scores"):
        try: raw_metric(rows.get(name, {}))
        except RuntimeError as error:
            raise RuntimeError("R1 BLOCKED: B7 " + name + " " + json.dumps(rows.get(name, {}), sort_keys=True)) from error
    raw_metric(rows.get("output", {}), passing=False)
    if not rows["output"]["raw_allclose"]:
        need(entry.get("selection", {}).get("changed_positions", 0) > 0, "B7 actual query indices unchanged")
        raw_metric(entry.get("fixed_candidate_replay_output", {}))
    for selection in entry.get("actual_selections", []):
        need(selection.get("fixed") is False and len(selection.get("calls", [])) == 1
             and selection["calls"][0].get("k") == 300 and selection["calls"][0].get("actual_indices"), "B7 actual indices")
    need(len(entry.get("actual_selections", [])) == 2, "B7 missing actual selection capture")
    indices = [s["calls"][0]["actual_indices"] for s in entry["actual_selections"]]
    need(len(indices[0]) == len(indices[1]) and indices[0], "B7 selection batch mismatch")
    changed = 0
    for a, b in zip(*indices):
        need(len(a) == len(b) == 300 and len(set(a)) == len(set(b)) == 300, "B7 invalid actual selection geometry")
        changed += sum(x != y for x, y in zip(a, b))
    need(changed == entry.get("selection", {}).get("changed_positions"), "B7 reported index change count is inconsistent")
    if not rows["output"]["raw_allclose"]:
        replay = entry.get("replay_selection", {})
        need(replay.get("fixed") is True and len(replay.get("calls", [])) == 1
             and replay["calls"][0].get("actual_indices") == indices[0], "B7 replay did not use original actual indices")
    need(entry.get("tf32_restored") is True, "B7 temporary flags not restored")


def validate_matched_controls(observations):
    parent = observations["parent"]["starts"][0]
    candidate = observations["candidate"]["starts"][0]
    for key in ("python_rng", "numpy_rng", "cpu_rng", "cuda_rng", "batch_sha256", "amp", "autocast_dtype",
                "matmul_tf32", "cudnn_tf32", "cudnn_benchmark", "cudnn_deterministic", "deterministic", "warn_only"):
        need(key in parent and parent[key] == candidate.get(key), "B2 parent/candidate controls differ: " + key)


def validate_b(item, mode, files=True):
    """B8: ignore all assessment labels; derive the decision from B1--B7."""
    need(mode in MODES and item.get("mode") == mode and item.get("policy") == policy(), "B mode/policy binding")
    validate_a(item.get("A", {}), mode)
    artifact(item.get("A_artifact"), files)
    need(item["A_artifact"]["sha256"] == item["A"].get("checkpoint_sha256"), "B1 different saved A checkpoint")
    need(item.get("A_binding") == item.get("binding") and item.get("binding", {}).get("mode") == mode,
         "B1 A not collected in this version/mode/session")
    raw = item.get("B", {})
    need(raw.get("atol") == 2e-5 and raw.get("rtol") == 2e-4, "B original tolerance")
    observations = item.get("B_evidence", {})
    for label in ("parent", "candidate"):
        control = raw.get(label, {})
        extra = observations.get(label, {})
        for key in ("initial_exact", "storage_independent", "forward_exact", "loss_exact", "exact_metadata_buffers"):
            need(control.get(key) is True, "B2/B3 " + label + " " + key)
        table(control.get("forward", {}), exact=True)
        states = table(control.get("state", {}), passing=mode == "cpu_fp32")
        initial = table(extra.get("initial_comparison", {}), exact=True)
        need(set(initial) == set(states), "B2 incomplete initial state comparison")
        grads = table(control.get("gradients", {}), passing=mode == "cpu_fp32")
        inventory = extra.get("inventory", {})
        parameters, buffers = set(inventory.get("parameters", [])), set(inventory.get("buffers", []))
        need(parameters and buffers and len(parameters) == len(inventory["parameters"]), "B3 full names missing")
        for prefix in ("state.model.", "state.ema."):
            need({n[len(prefix):] for n in states if n.startswith(prefix)} == parameters | buffers, "B3 model/EMA coverage")
        need({n[6:] for n in grads} == parameters, "B3 incomplete gradient coverage")
        for prefix in ("state.buffers.", "state.ema_buffers."):
            need({n[len(prefix):] for n in states if n.startswith(prefix)} == buffers, "B3 buffer coverage")
            need(all(states[prefix+n]["equal"] for n in buffers), "B3 buffers not exact")
        for name in parameters:
            for field in ("exp_avg", "exp_avg_sq", "step"):
                need("state.optimizer."+name+"."+field in states, "B3 missing optimizer moment/step")
        group_names = [n for group in inventory.get("groups", []) for n in group]
        need(len(group_names) == len(set(group_names)) and set(group_names) == parameters, "B3 optimizer groups")
        need(len(control.get("steps", [])) == 2 and all(r.get("effective_update") is True and r.get("finite_gradients") is True
             for r in control["steps"]), "B3 ineffective update")
        starts = extra.get("starts", [])
        need(len(starts) == 2 and starts[0] == starts[1] and starts[0].get("batch_sha256") == item["binding"].get("batch_sha256"), "B2 RNG/batch/precision mismatch")
        start = starts[0]
        need(all(len(start.get(k, "")) == 64 for k in ("python_rng", "numpy_rng", "cpu_rng", "batch_sha256")), "B2 missing RNG fingerprint")
        need(start.get("amp") is (mode == "cuda_native_amp") and start.get("scaler_enabled") is (mode == "cuda_native_amp")
             and start.get("model_dtypes") == ["torch.float32"] and start.get("deterministic") is True
             and start.get("autocast_dtype") == ("torch.float16" if mode == "cuda_native_amp" else "torch.float32")
             and all(type(start.get(k)) is bool for k in ("matmul_tf32", "cudnn_tf32", "cudnn_benchmark", "cudnn_deterministic", "warn_only")), "B2 precision flags")
        if mode != "cpu_fp32":
            need(start.get("cuda_rng") and all(len(x) == 64 for x in start["cuda_rng"]), "B2 CUDA RNG")
        for name in control["state"]["failed_tensors"]:
            prefix = next((p for p in ("state.model.", "state.ema.", "state.optimizer.") if name.startswith(p)), None)
            need(prefix and not name.endswith(".step"), "B4 non-float failure")
            parameter = name[len(prefix):].rsplit(".", 1)[0] if prefix == "state.optimizer." else name[len(prefix):]
            gradient = grads.get("state."+parameter, {})
            need(gradient.get("finite") is True and gradient.get("equal") is False, "B4 no same-stage gradient variation")
        replays = extra.get("native_replays", [])
        need(len(replays) == 2, "B6 missing native causal replay")
        for replay in replays:
            artifact(replay.get("artifact"), files)
            need(replay.get("capture_stage") == "after_backward_before_native_optimizer_step_scaled"
                 and replay.get("native_calls") == 1 and replay.get("storage_independent") is True
                 and replay.get("scaled_gradient_names") == sorted(parameters)
                 and replay.get("pre_state_exact") is True and replay.get("scaled_gradients_exact") is True,
                 "B6 wrong capture or incomplete state/gradients")
            post_rows = table(replay.get("post_comparison", {}), exact=True)
            need(set(post_rows) == set(states), "B6 incomplete actual/replay state inventory")
        delta_rows = table(extra.get("replay_delta_comparison", {}), exact=True)
        need(set(delta_rows) == set(states), "B6 incomplete paired delta inventory")
        table(extra.get("metadata_comparison", {}), exact=True)
    validate_matched_controls(observations)
    states = raw["candidate"]["state"]["tensors"]
    state_names = {n[len("state.model."):] for n in states if n.startswith("state.model.")}
    for key in ("restoration", "replay"):
        for prefix in ("state.model.", "state.ema."):
            need({n[len(prefix):] for n in item["A"][key]["tensors"] if n.startswith(prefix)} == state_names, "B1 incomplete A inventory")
    expected = {p+n for p in ("state.model.", "state.ema.") for n in ADDED}
    expected |= {"state.optimizer."+n+"."+f for n in ADDED for f in ("exp_avg", "exp_avg_sq", "step")}
    need({n for n in states if ".dpr_" in n} == expected, "B3 added persistent inventory must contain all 20 rows")
    need(all(states[n]["raw_allclose"] is True for n in expected), "B3 added persistent state fails original tolerance")
    mapping = observations.get("mapping", {})
    need(len(mapping.get("runs", [])) == 2, "B5 both same-backward captures required")
    for index, run in enumerate(mapping["runs"]):
        artifact(run.get("artifact"), files)
        need(run.get("scale", 0) > 0 and run.get("amp") is (mode == "cuda_native_amp"), "B5 real scale")
        need(run["scale"] == raw["candidate"]["steps"][index].get("scale_before")
             == observations["candidate"]["starts"][index].get("scaler_state", {}).get("scale", 1.),
             "B5 scale does not match the actual same backward")
        for collection in (run.get("comparisons", {}), run.get("local_mapping", {}).get("comparisons", {}), run.get("frozen_local_replay", {})):
            need(set(collection) == {n[len(TARGET):] for n in ADDED}, "B5 four adjoints")
            for row in collection.values(): raw_metric(row)
        raw_metric(run.get("kernel_gradient_equals_W", {})); raw_metric(run.get("local_output", {}))
    for key, atol, rtol in (("difference_propagation", 2e-5, 2e-4), ("fp64_reference", 1e-10, 1e-9)):
        rows = mapping.get(key, {}).get("comparisons", {})
        need(set(rows) == {n[len(TARGET):] for n in ADDED}, "B5 delta/FP64 reference missing")
        for row in rows.values(): raw_metric(row, atol, rtol)
    validate_trace(observations.get("updated_function", {}))
    need(raw.get("long_term_trajectory_equivalence") == "NOT_CLAIMED", "trajectory equivalence not established")
    passed = all(raw[c][k]["raw_allclose"] for c in ("parent", "candidate") for k in ("state", "gradients"))
    need(raw.get("raw_allclose") is passed, "raw B summary inconsistent")
    return dict(status="PASSED" if passed else "EXPLAINED_BACKWARD_VARIATION", rules=["B"+str(i) for i in range(1, 9)],
                raw_status=raw.get("status"), raw_allclose=passed, long_term_trajectory_equivalence="NOT_CLAIMED")


def validate_half(entry, files=True):
    from dpr_diagnostics import validate_backend_record
    need(entry.get("policy") == policy() and entry.get("scope") == scope(), "H policy/scope")
    h0 = entry.get("H0", {})
    for key in ("actual_native_call", "unfused", "norm_retained", "all_q_nonzero", "finite_loss", "finite_postprocess",
                "returned_float32", "native_half_roundtrip_exact", "original_trainer_unchanged"):
        need(h0.get(key) is True, "H0 " + key)
    need(h0.get("batches") == 1 and h0.get("input_dtypes") == ["torch.float16"]
         and h0.get("parameter_dtypes") == ["torch.float16"] and h0.get("ground_truth", 0) > 0
         and 0 < h0.get("elapsed_seconds", 0) <= 120, "H0 bounded true half EMA validation")
    artifact(h0.get("checkpoint"), files)
    h1 = entry.get("H1", {})
    artifact(h1.get("checkpoint"), files)
    need(all(h1.get(k) is True for k in ("state_bytes_exact", "deployed", "norm_retained", "checkpoint_unchanged", "nonzero_source")), "H1 identity")
    need(h1.get("dtype") == "torch.float16" and 0 < h1.get("elapsed_seconds", 0) <= 120, "H1 dtype/budget")
    for name in ("effective_kernel", "target", "P3", "P4", "P5", "encoder_features", "candidate_scores", "output"):
        raw_metric(h1.get("comparisons", {}).get(name, {}), 2e-3, 2e-2)
    h2 = entry.get("H2", {})
    need(set(h2) == {"native_ema_half", "final_eval_half", "independent_memory_fp32"}, "H2 missing production entry")
    for name, row in h2.items():
        artifact(row.get("checkpoint"), files)
        need(0 < row.get("elapsed_seconds", 0) <= 120, "H2 budget")
        half = name != "independent_memory_fp32"
        validate_backend_record(row["audit"], 2e-3 if half else 2e-5, 2e-2 if half else 2e-4)
        need(row["audit"]["dtype"] == ("torch.float16" if half else "torch.float32"), "H2 actual precision")
        need(row["audit"]["provenance"]["selected"] == ("model" if name == "final_eval_half" else "ema"), "H2 wrong file role")
        if name == "independent_memory_fp32":
            need(row.get("actual_memory_entry") is True, "H2 missing in-memory backend entry")
        if name != "final_eval_half":
            need(row["checkpoint"] == h0["checkpoint"], "H2 not the same actual H0 EMA file")
        if name == "final_eval_half":
            need(row.get("disposable_strip") is True and row.get("terminal_fields_verified") is True, "H2 final_eval representation")
    for name in ("H3", "H4"):
        need(entry.get(name, {}).get("capability") == UNSUPPORTED and entry[name].get("raw"), "H3/H4 cannot become pass labels")
        expected = {"U32_to_U16", "D32_to_D16", "N32_to_N16"} if name == "H3" else {"U16_to_D16", "D16_to_N16", "U16_to_N16"}
        need(set(entry[name]["raw"]) == expected, "H3/H4 missing original comparison")
        for comparison in entry[name]["raw"].values():
            for tensor in ("effective_kernel", "target", "P3", "P4", "P5", "encoder_features", "candidate_scores", "output"):
                raw_metric(comparison.get("comparisons", {}).get(tensor, {}), 2e-3, 2e-2, passing=False)
        def finite_tree(value):
            if isinstance(value, dict):
                if "finite" in value: need(value["finite"] is True, "H3/H4 nonfinite remains hard")
                for child in value.values(): finite_tree(child)
            elif isinstance(value, list):
                for child in value: finite_tree(child)
        finite_tree(entry[name]["raw"])
    return dict(status="PASSED", H0="PASSED", H1="PASSED", H2="PASSED", H3=UNSUPPORTED, H4=UNSUPPORTED, scope=scope())


def validate_original_half_finiteness(entry):
    """Unsupported cross paths still retain every original finite-value gate."""
    need(entry.get("dtype") == "float16" and entry.get("atol") == 2e-3 and entry.get("rtol") == 2e-2,
         "original half report/threshold missing")
    for key in ("native_half_vs_fp32", "fp32_fold_then_half"):
        rows = entry.get("checks", {}).get(key, {}).get("details", {})
        for name in ("target", "P3", "P4", "P5", "encoder_features", "candidate_scores", "output"):
            raw_metric(rows.get(name, {}), 2e-3, 2e-2, passing=False)
        if "fixed_candidate_replay_output" in rows:
            raw_metric(rows["fixed_candidate_replay_output"], 2e-3, 2e-2, passing=False)
    return True


def validate_supplement(report, identity=None, files=True):
    need(report.get("report_schema") == SUPPLEMENT_SCHEMA and report.get("contract") == CONTRACT
         and report.get("admission_eligible") is False and report.get("policy") == policy(), "not an approved R1 supplement")
    validate_scope(report.get("capability_scope"))
    need(report.get("source_stable") is True and report.get("code_identity") == report.get("code_identity_at_end"), "unstable code")
    if identity is not None: need(report["code_identity"] == identity, "stale supplement source")
    decisions = {}
    for mode in ("cuda_fp32", "cuda_native_amp"):
        need(mode in report.get("modes", {}), "missing targeted mode: " + mode)
        item = report.get("modes", {}).get(mode, {})
        need(item.get("binding", {}).get("code_identity") == report["code_identity"]
             and item["binding"].get("session") == report.get("session")
             and item["binding"].get("variant") == report.get("variant"), "supplement mode identity")
        decisions[mode] = validate_b(item, mode, files)
    decisions["half"] = validate_half(report.get("half", {}), files)
    return dict(status="SUPPLEMENT_PASSED", admission="BLOCKED_REQUIRES_FULL_PREFLIGHT", decisions=decisions)


def validate_full_assessments(report, files=True):
    need(report.get("contract") == CONTRACT and report.get("report_schema") == FULL_SCHEMA
         and report.get("policy") == policy(), "full report policy/schema")
    validate_scope(report.get("capability_scope"))
    for mode in MODES:
        item = report.get("checks", {}).get(mode, {})
        need(item.get("binding", {}).get("code_identity") == report.get("code_identity")
             and item["binding"].get("session") == report.get("output"), "full mode identity")
        validate_b(item, mode, files)
    validate_half(report.get("half_R1", {}), files)
    validate_original_half_finiteness(report.get("checks", {}).get("learned_inference", {}).get("cuda_half", {}))
    return True
