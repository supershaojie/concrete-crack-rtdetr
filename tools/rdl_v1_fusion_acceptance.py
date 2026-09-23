"""Fail-closed acceptance of ordered output or a fully evidenced ID permutation.

Only engineering reports use this predicate. No production query/output is sorted.
"""
import hashlib
import math
import re

from init_c19_lif_v1 import ROOT, SOURCE_SHA256
from c19_lif_v1_diagnostic import LIF_KEYS, PRE_KEYS, schema
from c19_lif_v1_cutoff import selection_from_scores

TOL = dict(atol=3e-5, rtol=3e-5)
KEYS = schema(dict(mode="pair", layers=3))
PRE = LIF_KEYS + PRE_KEYS
POST = [k for k in KEYS if k not in PRE + ["candidate_indices"]]
HEAD = [k for k in KEYS if k not in LIF_KEYS + ["downsample", "scale_0", "scale_1", "scale_2"]]


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values(): finite_tree(item)
    elif isinstance(value, list):
        for item in value: finite_tree(item)
    elif isinstance(value, float):
        need(math.isfinite(value), "Nonfinite report value")


def metric(row, key, exact=False, allow_failure=False, ids=True):
    need(row["key"] == key and row["precision"] == "fp32", "Wrong comparison identity: " + key)
    need(isinstance(row["stage"], str) and bool(row["stage"]) and
         re.fullmatch(r"cpu|cuda(?::\d+)?", row["device"]) is not None, "Missing comparison context: " + key)
    need(row["atol"] == (0 if exact else TOL["atol"]) and row["rtol"] == (0 if exact else TOL["rtol"]),
         "Changed tolerance: " + key)
    shape = row["shape"]
    need(isinstance(shape, list) and shape and all(type(n) is int and n > 0 for n in shape), "Empty/wrong shape: " + key)
    dtype = "torch.int64" if key == "candidate_indices" else ("torch.bool" if key == "valid_mask" else "torch.float32")
    need(row["dtype_a"] == row["dtype_b"] == dtype, "Wrong dtype: " + key)
    need(row["candidate_ids_available"] is ids, "Missing ID proof: " + key)
    count = row["allclose_failed_count"]
    need(type(count) is int and 0 <= count <= math.prod(shape), "Invalid failure count: " + key)
    need(type(row["max_abs_error"]) in (int, float) and row["max_abs_error"] >= 0, "Invalid error: " + key)
    if dtype == "torch.float32":
        need(type(row["max_rel_error"]) in (int, float) and row["max_rel_error"] >= 0, "Invalid relative error: " + key)
        # This fixed engineering fixture has no masked nonfinite anchors either.
        need(type(row["masked_nonfinite_count"]) is int and row["masked_nonfinite_count"] == 0, "Nonfinite tensor: " + key)
    else:
        need(row["max_rel_error"] is None, "Invalid integer/bool comparison: " + key)
    need(row.get("status") == ("FAILED_REAL_NUMERICAL_MISMATCH" if count else None), "Inconsistent comparison status: " + key)
    if not allow_failure:
        need(count == 0, "Comparison exceeded tolerance: " + key)
        if exact or dtype != "torch.float32":
            need(row["max_abs_error"] == 0, "Non-exact identity: " + key)
            if dtype == "torch.float32": need(row["max_rel_error"] == 0, "Non-exact relative error: " + key)


def group(report, name, keys, exact=False, allow_failure=False, ids=True, shapes=None):
    rows = report[name]
    need(isinstance(rows, dict) and bool(rows) and set(rows) == set(keys), "Incomplete comparison group: " + name)
    for key in keys:
        metric(rows[key], key, exact, allow_failure, ids)
        expected_stage = key if not ids else {"unfused":"mother_unfused", "fused":"mother_fused"}.get(name,name)
        need(rows[key]["stage"] == expected_stage, "Wrong comparison stage: " + name)
        if shapes is not None:
            need(rows[key]["shape"] == shapes[key]["shape"], "Inconsistent tensor shape: " + name + "." + key)


def selection(value):
    rows = value["images"]
    actual = selection_from_scores([r["natural_a"] for r in rows], [r["natural_b"] for r in rows],
        value["scores_a"], value["scores_b"], value["shapes"], value["score_dtype_a"], value["score_dtype_b"])
    need(actual == value and actual["score_dtype_a"] == "torch.float32", "Selection evidence inconsistent")
    need(len(rows) == 2 and all(len(r["natural_a"]) == len(r["natural_b"]) == 300 for r in rows), "Wrong query count")
    need(all(r["cutoff_a"]["native_order_valid"] and r["cutoff_b"]["native_order_valid"] for r in rows), "Invalid natural top-k")
    return actual


def invariants(r):
    need(r["schema"] == "rdl_fusion_diagnostic_v2" and r["tolerance"] == TOL, "Unknown protocol/tolerance")
    need(r["controlled_source_sha256"] == SOURCE_SHA256, "Wrong initialization source")
    p = r["precision_scope"]
    settings = {"cudnn_tf32", "matmul_tf32", "float32_matmul_precision", "cuda_autocast", "cpu_autocast",
                "cuda_autocast_dtype", "cpu_autocast_dtype", "autocast_cache_enabled", "nvidia_tf32_override"}
    need(all(set(p[k]) == settings for k in ("before", "inside", "after")), "Incomplete precision evidence")
    need(p["restored"] is True and p["before"] == p["after"], "Precision not restored")
    need(p["before"]["float32_matmul_precision"] in {"highest", "high", "medium"} and
         type(p["before"]["autocast_cache_enabled"]) is bool, "Invalid precision settings")
    for key in ("cuda_autocast_dtype", "cpu_autocast_dtype"):
        need(p["before"][key] in {"torch.float16", "torch.bfloat16", "torch.float32"}, "Invalid autocast dtype")
    for key in ("cudnn_tf32", "matmul_tf32", "cuda_autocast", "cpu_autocast"):
        need(p["inside"][key] is False and type(p["before"][key]) is bool, "Not strict FP32: " + key)
    need(p["inside"]["float32_matmul_precision"] == "highest", "Not full FP32 matmul")
    for key in settings - {"cudnn_tf32", "matmul_tf32", "cuda_autocast", "cpu_autocast", "float32_matmul_precision"}:
        need(p["inside"][key] == p["before"][key], "Unexplained precision change: " + key)
    flags = r["validation_invariants"]
    need(set(flags) == {"prediction_unchanged", "state_unchanged", "input_unchanged", "cache_unchanged"} and
         all(v is True for v in flags.values()), "Validation changed fixture")
    for key in ("unfused_snapshot_state_exact", "input_unchanged_after_fuse", "lif_bn_retained"):
        need(r[key] is True, "Invariant failed: " + key)
    first = r["before_predict"]
    need(re.fullmatch("[0-9a-f]{64}", first["state_sha256"]) is not None, "Missing state identity")
    c = first["cache"]
    need(set(c) == {"shapes", "anchors", "valid_mask"} and c["shapes"] == [[20,24],[10,12],[5,6]], "Incomplete cache evidence")
    for name, dtype, shape in (("anchors","torch.float32",[1,630,4]), ("valid_mask","torch.bool",[1,630,1])):
        need(set(c[name]) == {"sha256", "dtype", "device", "shape"} and c[name]["dtype"] == dtype and
             c[name]["shape"] == shape and c[name]["device"] == first["precision"]["input_device"] and
             re.fullmatch("[0-9a-f]{64}",c[name]["sha256"]) is not None, "Invalid cache identity")
    for stage in ("before_predict", "before_fuse", "after_fuse", "after_predict"):
        s = r[stage]["precision"]
        device = s["input_device"]
        need(s == dict(cuda_autocast=False, cpu_autocast=False, input_dtype="torch.float32", input_device=device,
                       parameter_dtypes=["torch.float32"], parameter_devices=[device], training_modules=[]), "Model/input precision or eval differs")
        need(s == first["precision"] and r[stage]["cache"] == first["cache"], "Device/cache changed")
    for stage in ("unfused", "fused"):
        need(r["inside_score_head"][stage] == dict(cuda_autocast=False, cpu_autocast=False, input_dtype="torch.float32",
             input_device=first["precision"]["input_device"], weight_dtype="torch.float32", training=False), "Actual forward precision differs")


def review_fusion(r):
    """Recompute acceptance from complete evidence, never from finding/status alone."""
    try:
        finite_tree(r)
        invariants(r)
        metric(r["original_assertion"], "ordered_output", allow_failure=True, ids=False)
        need(r["original_assertion"]["shape"] == [2, 300, 5], "Wrong inference shape")
        if "saved_fixture_replay" in r:
            group(r, "saved_fixture_replay", ["before", "after"], exact=True, ids=False)
        if r["original_status"] == "PASS":
            need(r["original_assertion"]["allclose_failed_count"] == 0, "Ordered PASS contradicts measurements")
            return dict(accepted=True, status="PASS_ORDERED", policy="rdl_fusion_id_v1")
        need(r["original_status"] == "FAIL" and r["original_assertion"]["allclose_failed_count"] > 0, "Missing original failure")
        need(r["diagnostic_status"] == "COMPLETE" and not r.get("diagnostic_error"), "Incomplete diagnostic")
        group(r, "reproduce_original", ["before", "after"], exact=True, ids=False)
        s = selection(r["selection"])
        need(s["kind"] == "PERMUTATION", "Only a same-set permutation may bypass ordered failure")
        a, b = ([v[side] for v in s["images"]] for side in ("natural_a", "natural_b"))
        expected = {"natural_a": (a,a), "natural_b": (b,b), "fixed_A_a": (a,a), "fixed_A_b": (a,b),
                    "fixed_B_a": (b,a), "fixed_B_b": (b,b), "common_head_a": (a,a), "common_head_b": (a,a),
                    "mother_a": (a,a), "mother_b": (b,b)}
        need(set(r["traces"]) == set(expected), "Missing gather trace")
        for name, (consumed, native) in expected.items():
            trace = r["traces"][name]
            need(trace["actual_gather_verified"] is True and type(trace["topk_calls"]) is int and trace["topk_calls"] == 1,
                 "Actual gather not verified: " + name)
            need(trace["candidate_indices"] == consumed and trace["native_candidate_indices"] == native, "Candidate identity differs: " + name)
            need(all(type(i) is int for field in ("candidate_indices", "native_candidate_indices") for row in trace[field] for i in row),
                 "Invalid candidate ID type: " + name)
        shapes = r["fixed_ids_A"]
        need(shapes["candidate_indices"]["shape"] == [2,300] and shapes["candidate_scores"]["shape"] == [2,630,1] and
             shapes["boxes"]["shape"] == [1,2,300,4] and shapes["scores"]["shape"] == [1,2,300,1], "Wrong candidate/output dimensions")
        for name, keys, exact, allow in (("pre_selection", PRE, False, False), ("natural_outputs", POST, False, True),
                ("candidate_id_alignment", POST, False, False), ("fixed_ids_A", KEYS, False, False), ("fixed_ids_B", KEYS, False, False),
                ("common_head_inputs", ["scale_0", "scale_1", "scale_2"], True, False), ("common_head_outputs", HEAD, False, False),
                ("common_candidate_inputs", ["candidate_indices", "selected_features", "selected_anchors", "flattened_features"], True, False)):
            group(r, name, keys, exact=exact, allow_failure=allow, shapes=shapes)
        f = r["fusion_invariants"]
        need(f["lif_bn_retained"] is True and f["lif_state_exact"] is True and
             type(f["bn_before"]) is int and type(f["bn_after"]) is int and 0 < f["bn_after"] < f["bn_before"], "LIF/fusion invariant failed")
        mother = r["mother_source"]
        need(mother["base_commit"] == "a0459d6a652cb702699087c88fa39a3e4c4087ec" and mother["fuse_ast_unchanged"] is True, "Wrong mother control")
        files = ("lif_down.py", "cbr.py", "head.py", "transformer.py", "conv.py")
        need(set(mother["protected_modules"]) == set(files), "Incomplete mother source identity")
        for name in files:
            entry = mother["protected_modules"][name]
            raw = (ROOT/"ultralytics-main/ultralytics/nn/modules"/name).read_bytes().replace(b"\r\n", b"\n")
            need(entry["unchanged"] is True and entry["sha256"] == hashlib.sha256(raw).hexdigest(), "Changed mother module: " + name)
        need(set(r["mother_vs_rdl"]) == {"unfused", "fused"}, "Incomplete mother comparison")
        for side in ("unfused", "fused"):
            group(r["mother_vs_rdl"], side, KEYS, exact=True, shapes=shapes)
        need(selection(r["mother_selection"]) == s, "Mother selection differs")
        return dict(accepted=True, status="PASS_CANDIDATE_PERMUTATION", policy="rdl_fusion_id_v1")
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, OSError) as error:
        return dict(accepted=False, status="FAIL", policy="rdl_fusion_id_v1", reason=str(error))
