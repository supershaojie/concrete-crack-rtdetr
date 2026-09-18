"""Pure evidence validation for DCC restoration and independent trajectories.

No torch import, model execution, report mutation, or tolerance adjustment. A
report's prefilled status/acceptance is never an admission decision. Historical
reports may be evaluated read-only; only the caller can establish fresh source,
runtime, sample and contract identities for an actual launch gate.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

CONTRACT_VERSION = "dcc_acceptance_v2"
ATOL, RTOL = 2e-5, 2e-4
LABELS = {"cpu_fp32", "cuda_fp32", "cuda_native_amp"}
DCC_SHAPES = {
    "model.17.dcc.W_d.weight": [32, 256, 1, 1],
    "model.17.dcc.W_q.weight": [32, 32, 1, 1],
    "model.17.dcc.W_k.weight": [32, 32, 1, 1],
    "model.17.dcc.W_o.weight": [256, 32, 1, 1],
}
FORWARD_KEYS = {
    "lif_input", "lif_conv", "lif_residual", "lif_pre_bn", "lif_post_bn", "downsample",
    "scale_0", "scale_1", "scale_2", "projection_0", "projection_1", "projection_2",
    "flattened_features", "encoder_features", "candidate_scores", "native_candidate_indices",
    "candidate_indices", "selected_features", "selected_bbox_delta", "anchors", "valid_mask",
    "selected_anchors", "reference_boxes", "decoder_embeddings", "query_layer_0", "query_layer_1",
    "query_layer_2", "raw_boxes", "raw_scores", "returned_query", "cbr_p3", "cbr_query",
    "cbr_before", "cbr_offsets", "cbr_after", "boxes", "scores", "enc_boxes", "enc_scores", "cbr_residual",
}


def _forward_keys(parameters):
    """Match probe hooks for the two authorized architectures, including C2."""
    keys = set(FORWARD_KEYS)
    if not any(name.startswith("model.26.cbr.") for name in parameters):
        keys = {key for key in keys if not key.startswith("cbr_") and key != "returned_query"}
    if "model.20.B_proj.weight" not in parameters:
        keys = {key for key in keys if not key.startswith("lif_")}
    return keys


class _Evidence:
    def __init__(self):
        self.failures = []

    def need(self, condition, path, reason):
        if not condition:
            message = path + ": " + reason
            if message not in self.failures:
                self.failures.append(message)
        return bool(condition)


def _number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _hash(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _shape(row):
    return isinstance(row.get("shape"), list) and all(type(x) is int and x >= 0 for x in row["shape"])


def _row(e, row, path, sentinel=False):
    if not e.need(isinstance(row, dict), path, "missing tensor comparison"):
        return
    e.need(_shape(row), path, "missing/invalid shape or shape mismatch")
    e.need(not any(key in row for key in ("shape_a", "shape_b", "dtype_a", "dtype_b", "reason")),
           path, "shape/dtype identity is not established")
    e.need(type(row.get("equal")) is bool and type(row.get("allclose")) is bool, path, "missing exact/allclose flags")
    if sentinel and row.get("finite") is False:
        e.need(row.get("equal") is True and row.get("allclose") is True and
               row.get("note") == "Matching native invalid-anchor +inf sentinel" and
               all(row.get(k) is None for k in ("max_abs", "relative_L2", "exceeded_fraction")),
               path, "unverified nonfinite anchor sentinel")
        return
    e.need(_number(row.get("max_abs")), path, "nonfinite/missing absolute error")
    if "finite" in row:
        e.need(row["finite"] is True, path, "nonfinite tensor")
        e.need(_number(row.get("relative_L2")) and _number(row.get("exceeded_fraction")) and
               row.get("exceeded_fraction", 2) <= 1, path, "invalid relative error/exceedance fraction")
        e.need(row.get("atol") == ATOL and row.get("rtol") == RTOL, path, "original declared tolerance changed")
        e.need(row.get("allclose") is (row.get("exceeded_fraction") == 0), path, "allclose/fraction disagree")
        # Legacy tensor_metric checked matching dtypes before emitting float
        # rows, but omitted the dtype label. New rows must not contradict it.
        if "dtype" in row:
            e.need(row["dtype"] in {"torch.float16", "torch.float32", "torch.float64", "torch.bfloat16"},
                   path, "invalid floating dtype")
    else:
        e.need(row.get("dtype") in {"torch.bool", "torch.int8", "torch.uint8", "torch.int16", "torch.int32", "torch.int64"},
               path, "finite floating evidence or exact integer dtype is missing")
        e.need(row.get("equal") is row.get("allclose"), path, "integer identity is not exact")
    if row.get("equal") is True:
        e.need(row.get("allclose") is True and row.get("max_abs") == 0 and
               row.get("relative_L2", 0) == 0 and row.get("exceeded_fraction", 0) == 0,
               path, "claimed exact tensor has nonzero error")
    elif row.get("equal") is False:
        e.need(_number(row.get("max_abs")) and row["max_abs"] > 0, path, "nonexact tensor has no measured difference")


def _comparison(e, value, path, exact=False, forward=False):
    if not e.need(isinstance(value, dict), path, "missing complete comparison"):
        return {}
    e.need(value.get("metadata_errors") == [], path, "missing or unequal names/groups/hyperparameters/dtypes/scaler/epoch metadata")
    rows = value.get("tensors")
    if not e.need(isinstance(rows, dict) and bool(rows), path, "missing tensor inventory"):
        return {}
    for name, row in rows.items():
        _row(e, row, path + "." + name, sentinel=forward and name in {"state.anchors", "state.selected_anchors"})
    equal = all(row.get("equal") is True for row in rows.values())
    close = all(row.get("allclose") is True for row in rows.values())
    e.need(value.get("exact") is equal and value.get("allclose") is close, path, "aggregate exact/allclose contradicts rows")
    requested = value.get("exact_requested")
    e.need(type(requested) is bool, path, "missing comparison mode")
    failed = {name for name, row in rows.items() if row.get("equal" if requested else "allclose") is not True}
    e.need(isinstance(value.get("failed_tensors"), list) and set(value["failed_tensors"]) == failed,
           path, "failed tensor inventory contradicts rows")
    e.need(value.get("status") == ("PASSED" if not failed else "FAILED"), path, "comparison status contradicts rows")
    measured = max((row.get("max_abs") or 0 for row in rows.values()), default=0)
    e.need(value.get("max_abs") == measured, path, "aggregate maximum error contradicts rows")
    if exact:
        e.need(equal and close and not failed, path, "strict tensor identity failed")
    return rows


def _same_inventory(e, left, right, path):
    e.need(set(left) == set(right), path, "tensor inventory differs")
    for name in left.keys() & right.keys():
        e.need(left[name].get("shape") == right[name].get("shape"), path + "." + name, "tensor shape differs")
        if "dtype" in left[name] and "dtype" in right[name]:
            e.need(left[name]["dtype"] == right[name]["dtype"], path + "." + name, "tensor dtype differs")


def _storage(e, item, path):
    e.need(isinstance(item, dict) and item.get("status") == "PASSED" and
           item.get("shared_storage_count") == 0 and type(item.get("first_tensors")) is int and item["first_tensors"] > 0 and
           type(item.get("second_tensors")) is int and item["second_tensors"] > 0,
           path, "missing independence evidence or shared mutable tensor storage")


def _current_dtype_evidence(e, report, path):
    """New reports must name dtypes; legacy originals retain their old schema."""
    if report.get("contract_version") != CONTRACT_VERSION:
        return

    def visit(value, location):
        if not isinstance(value, dict):
            return
        if isinstance(value.get("tensors"), dict):
            for name, row in value["tensors"].items():
                floating = isinstance(row, dict) and "finite" in row
                allowed = {"torch.float16", "torch.float32", "torch.float64", "torch.bfloat16"} if floating else {
                    "torch.bool", "torch.int8", "torch.uint8", "torch.int16", "torch.int32", "torch.int64"}
                e.need(isinstance(row, dict) and row.get("dtype") in allowed, location + "." + name,
                       "current contract requires an explicit matching tensor dtype")
        for key, item in value.items():
            if key != "tensors":
                visit(item, location + "." + key)

    visit(report, path)


def _state_inventory(e, rows, names, path):
    """Require all model/EMA, persistent/nonpersistent buffers and AdamW rows."""
    names = set(names)
    model = {k[len("state.model."):]: v for k, v in rows.items() if k.startswith("state.model.")}
    ema = {k[len("state.ema."):]: v for k, v in rows.items() if k.startswith("state.ema.")}
    buffers = {k[len("state.model_buffers."):]: v for k, v in rows.items() if k.startswith("state.model_buffers.")}
    ema_buffers = {k[len("state.ema_buffers."):]: v for k, v in rows.items() if k.startswith("state.ema_buffers.")}
    e.need(names <= set(model), path, "missing named trainable model state")
    _same_inventory(e, model, ema, path + ".model_ema_inventory")
    _same_inventory(e, buffers, ema_buffers, path + ".buffer_inventory")
    persistent = set(model) - names
    nonpersistent = {"model.17.dcc.identity"} if set(DCC_SHAPES) <= names else set()
    e.need(bool(persistent) and set(buffers) == persistent | nonpersistent, path, "incomplete persistent/nonpersistent buffer inventory")
    e.need(all(name.endswith((".running_mean", ".running_var", ".num_batches_tracked")) for name in persistent),
           path, "unexpected model buffer")
    for name in persistent:
        _same_inventory(e, {name: model[name]}, {name: buffers.get(name, {})}, path + ".persistent_buffer")
    expected = {"state." + prefix + "." + name for prefix, table in
                (("model", model), ("ema", ema), ("model_buffers", buffers), ("ema_buffers", ema_buffers)) for name in table}
    for name in names:
        for field in ("step", "exp_avg", "exp_avg_sq"):
            key = "state.optimizer." + name + "." + field
            expected.add(key)
            if e.need(key in rows, path + "." + key, "missing AdamW state"):
                expected_shape = [] if field == "step" else model.get(name, {}).get("shape")
                e.need(rows[key].get("shape") == expected_shape, path + "." + key, "AdamW state shape differs")
    e.need(set(rows) == expected, path, "unknown or missing complete-state tensor rows")
    if names <= set(model):
        total = sum(math.prod(model[name].get("shape", [])) for name in names)
        has_cbr = any(name.startswith("model.26.cbr.") for name in names)
        expected_total = (20149765 if has_cbr else 20082772) + (18432 if nonpersistent else 0)
        e.need(total == expected_total, path, "parameter inventory does not match the fixed nc=1 architecture")
    for name, shape in DCC_SHAPES.items():
        if name in names:
            e.need(model.get(name, {}).get("shape") == shape, path + "." + name, "DCC projection shape differs")


def _source_names(e, report):
    source, serializer = report.get("checkpoint_source_validation", {}), report.get("serializer", {})
    e.need(report.get("checkpoint_policy") == "optimizer_fp32_v1" and source.get("policy") == "optimizer_fp32_v1" and
           serializer.get("policy") == "optimizer_fp32_v1", "checkpoint_policy", "FP32 source policy is missing")
    e.need(source.get("original_tensors_inspected_before_cast") is True and source.get("checkpoint_unmodified") is True,
           "checkpoint_source_validation", "original source dtype/state inspection missing")
    groups, ids = source.get("optimizer_param_names"), source.get("parameter_ids_by_group")
    if not e.need(isinstance(groups, list) and bool(groups) and isinstance(ids, list) and len(ids) == len(groups),
                  "checkpoint_source_validation", "missing group/name/ID mapping"):
        return [], {}
    e.need(groups == serializer.get("optimizer_param_names"), "serializer", "saved group/name order differs from source validation")
    mapping = {}
    for index, (group, numbers) in enumerate(zip(groups, ids)):
        if not e.need(isinstance(group, list) and isinstance(numbers, list) and len(group) == len(numbers),
                      "optimizer_group." + str(index), "param-ID/name lengths differ"):
            continue
        for name, number in zip(group, numbers):
            e.need(isinstance(name, str) and type(number) is int and number not in mapping,
                   "optimizer_group." + str(index), "invalid/duplicate parameter ID or name")
            mapping[number] = name
    names = list(mapping.values())
    e.need(len(set(names)) == len(names) and set(DCC_SHAPES) <= set(names), "optimizer_param_names", "duplicate/incomplete parameter names")
    e.need(source.get("state_entries") == len(names) and source.get("fp32_moment_tensors") == 2 * len(names),
           "checkpoint_source_validation", "not every parameter has both original FP32 moments")
    e.need(report.get("saved_ema_dtypes") == ["torch.float16", "torch.int64"], "saved_ema_dtypes", "native EMA-half source evidence differs")
    e.need(_hash(report.get("checkpoint_sha256")) and _hash(serializer.get("native_serializer_sha256")) and
           _hash(serializer.get("native_serializer_ast_sha256")), "serializer", "checkpoint/source identities missing")
    return names, mapping


def _replay(e, replay, label, reference):
    rows = _comparison(e, replay, "same_gradient_replay", exact=True)
    _same_inventory(e, reference, rows, "same_gradient_replay.inventory")
    _storage(e, replay.get("storage_independence", {}), "same_gradient_replay.storage")
    amp = label == "cuda_native_amp"
    e.need(replay.get("amp_scaler_path") is amp and replay.get("native_scaler_mode_preserved") is True,
           "same_gradient_replay", "native scaler mode was not preserved")
    first, second = replay.get("first", {}), replay.get("second", {})
    for name, item in (("first", first), ("second", second)):
        e.need(item.get("amp_scaler_path") is amp and item.get("scaler_enabled") is amp and item.get("effective_update") is True,
               "same_gradient_replay." + name, "disabled scaler or skipped native update")
        e.need(_number(item.get("scale_before"), 1e-300) and _number(item.get("scale_after"), 1e-300) and
               item.get("scale_after", 0) >= item.get("scale_before", 1), "same_gradient_replay." + name, "invalid native scaler transition")
        e.need(item.get("path") == "native scale -> assigned scaled gradients -> native optimizer_step/unscale/clip/step/update/EMA",
               "same_gradient_replay." + name, "replay bypassed the native optimizer/scaler path")
    e.need(first == second, "same_gradient_replay", "native replay transitions differ")
    if not amp:
        e.need(first.get("scale_before") == first.get("scale_after") == 1, "same_gradient_replay", "FP32 scaler must remain disabled")
        return
    overflow = replay.get("overflow_fixture", {})
    e.need(overflow.get("status") == "PASSED" and overflow.get("amp_scaler_path") is True and
           overflow.get("model_optimizer_unchanged_on_skip") is True and overflow.get("native_ema_updates_on_skipped_step") is True,
           "overflow_fixture", "native enabled-AMP overflow/skip evidence missing")
    _storage(e, overflow.get("storage_independence", {}), "overflow_fixture.storage")
    tiny = _comparison(e, overflow.get("complete_state_comparison"), "overflow_fixture.complete_state", exact=True)
    e.need(len(tiny) == 10, "overflow_fixture.complete_state", "incomplete two-parameter toy state")
    a, b = overflow.get("first", {}), overflow.get("second", {})
    e.need(a == b, "overflow_fixture", "skip/scaler/EMA transitions differ")
    e.need(a.get("skipped") is True and _number(a.get("scale_before"), 1e-300) and
           _number(a.get("scale_after"), 1e-300) and a.get("scale_after", 1) < a.get("scale_before", 0) and
           type(a.get("ema_updates_before")) is int and a.get("ema_updates_after") == a["ema_updates_before"] + 1,
           "overflow_fixture", "overflow must skip optimizer, back off scaler, and retain native EMA update behavior")


def evaluate_checkpoint(report, label, check_contract_version=False):
    """A: strict original-FP32 source, saved-byte restoration and native replay."""
    e = _Evidence()
    try:
        e.need(label in LABELS, "label", "unknown precision mode")
        if check_contract_version:
            e.need(report.get("contract_version") == CONTRACT_VERSION, "contract_version", "fresh contract identity missing")
        _current_dtype_evidence(e, report, "checkpoint")
        e.need(report.get("native_setup_model") is True and report.get("native_resume_training") is True,
               "native_restore", "native setup_model/resume_training evidence missing")
        e.need(report.get("atol") == ATOL and report.get("rtol") == RTOL and report.get("seed") == 42,
               "contract", "seed/tolerances differ")
        names, mapping = _source_names(e, report)
        saved = _comparison(e, report.get("saved_optimizer_live_exact"), "saved_optimizer_live_exact", exact=True)
        expected_saved = {"state.state.%s.%s" % (identifier, field) for identifier in mapping
                          for field in ("step", "exp_avg", "exp_avg_sq")}
        e.need(set(saved) == expected_saved, "saved_optimizer_live_exact", "missing original optimizer state rows")
        restoration = report.get("restoration", {})
        reference = _comparison(e, restoration, "restoration", exact=True)
        _state_inventory(e, reference, names, "restoration")
        for identifier, parameter in mapping.items():
            for field in ("step", "exp_avg", "exp_avg_sq"):
                key = "state.state.%s.%s" % (identifier, field)
                shape = [] if field == "step" else reference.get("state.model." + parameter, {}).get("shape")
                e.need(saved.get(key, {}).get("shape") == shape, "saved_optimizer_live_exact." + key,
                       "saved parameter-ID state shape differs from its named parameter")
                if "dtype" in saved.get(key, {}):
                    e.need(saved[key]["dtype"] == "torch.float32", "saved_optimizer_live_exact." + key,
                           "saved AdamW state must retain original FP32 precision")
        _storage(e, restoration.get("storage_independence", {}), "restoration.storage")
        e.need(type(restoration.get("checkpoint_epoch")) is int and
               restoration.get("restored_start_epoch") == restoration["checkpoint_epoch"] + 1,
               "restoration", "epoch restoration mismatch")
        for name in ("direct_vs_saved_checkpoint", "native_vs_saved_checkpoint"):
            rows = _comparison(e, restoration.get(name), "restoration." + name, exact=True)
            _same_inventory(e, reference, rows, "restoration." + name + ".inventory")
        _replay(e, report.get("same_gradient_replay", {}), label, reference)
        if label == "cpu_fp32":
            _case(e, report, label, "checkpoint", names)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        e.need(False, "schema", "malformed/missing evidence: " + str(error))
    return dict(contract_version=CONTRACT_VERSION, status="FAILED" if e.failures else "PASSED", failures=e.failures,
                scope="A: checkpoint correctness; independent CUDA backward trajectories are evaluated separately",
                historical_read_only_allowed=not check_contract_version)


def _case(e, report, label, name, parameters, reference=None):
    path = "trajectory." + name
    e.need(isinstance(report, dict) and bool(report), path, "missing independent control")
    _current_dtype_evidence(e, report, path)
    e.need(report.get("atol") == ATOL and report.get("rtol") == RTOL and report.get("seed") == 42,
           path, "seed or original tolerance changed")
    initial = report.get("restoration") if name == "checkpoint" else report.get("initial_state")
    initial_rows = _comparison(e, initial, path + ".initial", exact=True)
    _state_inventory(e, initial_rows, parameters, path + ".initial")
    if name != "checkpoint":
        e.need(report.get("actual_updates_required") is True and report.get("quantization") == "NONE: copied live FP32 model/optimizer/EMA",
               path, "control is not genuinely updated live FP32 state")
        _storage(e, report.get("storage_independence", {}), path + ".storage")
        _storage(e, report.get("original_first_independence", {}), path + ".original_storage")
    rows = _comparison(e, report.get("state_comparison"), path + ".state", exact=label == "cpu_fp32")
    _same_inventory(e, initial_rows, rows, path + ".raw_inventory")
    if reference is not None:
        _same_inventory(e, reference, initial_rows, path + ".cross_control_inventory")
    gradients = _comparison(e, report.get("gradient_comparison"), path + ".gradients", exact=label == "cpu_fp32")
    e.need(set(gradients) == {"state." + key for key in parameters}, path + ".gradients", "missing/unknown parameter gradients")
    for parameter in parameters:
        if "state." + parameter in gradients and "state.model." + parameter in rows:
            e.need(gradients["state." + parameter].get("shape") == rows["state.model." + parameter].get("shape"),
                   path + ".gradients." + parameter, "gradient shape differs from parameter")
    forward = _comparison(e, report.get("captured_forward_comparison"), path + ".forward", exact=True, forward=True)
    expected_forward = _forward_keys(parameters)
    e.need(set(forward) == {"state." + key for key in expected_forward}, path + ".forward", "continuous/candidate forward inventory incomplete")
    first, second = report.get("direct_step", {}), report.get("resumed_step", {})
    e.need(report.get("captured_forward_hashes_exact") is True and first.get("forward_hashes") == second.get("forward_hashes") and
           isinstance(first.get("forward_hashes"), dict) and set(first["forward_hashes"]) == expected_forward and
           all(_hash(value) for value in first["forward_hashes"].values()), path, "forward/candidate hashes differ or are missing")
    e.need(report.get("loss_exact") is True and _number(first.get("loss")) and first.get("loss") == second.get("loss"),
           path, "original forward loss is not exactly equal and finite")
    for item in (first, second):
        e.need(item.get("effective_update") is True and item.get("skipped") is False, path, "raw comparison lacks two effective updates")
        e.need(_number(item.get("scale_before"), 1e-300) and _number(item.get("scale_after"), 1e-300) and
               item.get("scale_after", 0) >= item.get("scale_before", 1), path, "raw native scaler transition invalid")
        e.need(type(item.get("gt")) is int and item["gt"] > 0 and isinstance(item.get("dn_split"), list), path, "GT/DN evidence missing")
        norms = item.get("dcc_grad_norms", {})
        required_dcc = set(parameters) & set(DCC_SHAPES)
        e.need(set(norms) == required_dcc and all(_number(v, 1e-300) for v in norms.values()), path, "DCC gradient missing/nonfinite/not activated")
    for key in ("scale_before", "scale_after", "gt", "dn_split"):
        e.need(first.get(key) == second.get(key), path, "raw update/scaler/GT transition differs: " + key)
    if label != "cuda_native_amp":
        e.need(first.get("scale_before") == first.get("scale_after") == 1, path, "FP32 run unexpectedly used a scaler")
    parameters = set(parameters)
    mutable_keys = {"state." + prefix + "." + parameter
                    for prefix in ("model", "ema") for parameter in parameters}
    mutable_keys.update("state.optimizer." + parameter + "." + moment
                        for parameter in parameters for moment in ("exp_avg", "exp_avg_sq"))
    for key, row in rows.items():
        if key not in mutable_keys:
            e.need(row.get("equal") is True, path + "." + key, "buffer/counter or unrecognized state changed")
    for parameter in parameters & set(DCC_SHAPES):
        for prefix in ("model", "ema"):
            e.need(rows.get("state." + prefix + "." + parameter, {}).get("allclose") is True,
                   path + "." + parameter, "DCC parameter/EMA update exceeds the original tolerance")
    model_close = all(row.get("allclose") is True for key, row in rows.items() if key.startswith("state.model."))
    e.need(report.get("raw_next_update_allclose") is model_close and
           report.get("raw_all_states_allclose") is all(row.get("allclose") is True for row in rows.values()),
           path, "raw aggregate flags contradict tensor evidence")
    bad_model = [key for key, row in rows.items() if key.startswith("state.model.") and row.get("allclose") is False]
    dcc_gradient_rows = {key: deepcopy(row) for key, row in gradients.items() if key[len("state."):] in DCC_SHAPES}
    return dict(rows=rows, gradients=gradients, initial=initial_rows,
                summary=dict(raw_status=report.get("raw_status", report.get("status")),
                             raw_next_update_allclose=report.get("raw_next_update_allclose"),
                             raw_all_states_allclose=report.get("raw_all_states_allclose"),
                             model_failed_tensors=bad_model, model_failed_count=len(bad_model),
                             gradient_failed_tensors=[k for k, v in gradients.items() if v.get("allclose") is False],
                             dcc_gradient_comparison=dcc_gradient_rows,
                             forward_hashes_exact=report.get("captured_forward_hashes_exact"), loss_exact=report.get("loss_exact")))


def evaluate_mode(label, checkpoint, parent_live, dcc_live, check_contract_version=False):
    """Validate A, then classify B without changing any original raw evidence.

    CUDA notes require both same-mode live controls to reproduce finite backward
    differences under exact initial state, forward tensors, candidates and loss.
    Only known trainable parameters, their EMA copies and AdamW moments may vary.
    Beyond-tolerance public updates require a changed gradient at the same named
    parameter in the case and in both live controls. No numerical envelope is
    fitted to a particular package; DCC updates retain the original tolerance.
    """
    a = evaluate_checkpoint(checkpoint, label, check_contract_version)
    e, cases = _Evidence(), {}
    try:
        e.need(label in LABELS, "label", "unknown precision mode")
        names, _ = _source_names(e, checkpoint)
        parent_names = [name for name in names if name not in DCC_SHAPES]
        cases["parent_live"] = _case(e, parent_live, label, "parent_live", parent_names)
        cases["dcc_live"] = _case(e, dcc_live, label, "dcc_live", names)
        cases["checkpoint"] = _case(e, checkpoint, label, "checkpoint", names, cases["dcc_live"]["initial"])
        if check_contract_version:
            for name, report in (("parent_live", parent_live), ("dcc_live", dcc_live)):
                e.need(report.get("contract_version") == CONTRACT_VERSION, name, "fresh contract identity missing")
        if label != "cpu_fp32":
            changed = {name: any(row.get("equal") is False for row in case["gradients"].values()) for name, case in cases.items()}
            any_difference = any(any(row.get("equal") is False for row in case["rows"].values()) or changed[name]
                                 for name, case in cases.items())
            if any_difference:
                e.need(changed["parent_live"] and changed["dcc_live"], "trajectory.mechanism", "both same-mode live controls must reproduce finite backward variation")
                for case_name, case in cases.items():
                    for key, row in case["rows"].items():
                        if row.get("allclose") is not False:
                            continue
                        if key.startswith(("state.model.", "state.ema.")):
                            parameter = key.split(".", 2)[2]
                        elif key.startswith("state.optimizer.") and key.endswith((".exp_avg", ".exp_avg_sq")):
                            parameter = key[len("state.optimizer."):].rsplit(".", 1)[0]
                        else:
                            e.need(False, "trajectory." + case_name + "." + key, "difference is outside the identified backward/AdamW/EMA mechanism")
                            continue
                        if parameter in DCC_SHAPES:
                            # Gradient differences are retained and disclosed;
                            # DCC state updates were already checked above.
                            e.need(False, "trajectory." + case_name + "." + key, "unexpected DCC optimizer/state divergence")
                            continue
                        gradient_key = "state." + parameter
                        e.need(all(control["gradients"].get(gradient_key, {}).get("equal") is False
                                   for control in (case, cases["parent_live"], cases["dcc_live"])),
                               "trajectory." + case_name + "." + key,
                               "public update is not corroborated by that parameter's backward variation in both live controls")
        else:
            any_difference = False
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        e.need(False, "trajectory.schema", "malformed/missing evidence: " + str(error))
        any_difference = False
    trajectory_status = "FAILED" if e.failures else "PRECISION_NOTE" if any_difference else "PASSED"
    failures = ["restoration." + value for value in a["failures"]] + e.failures
    return dict(contract_version=CONTRACT_VERSION, status="FAILED" if failures else trajectory_status,
                restoration=a,
                trajectory=dict(status=trajectory_status, failures=e.failures,
                    mechanism="Exact forward/loss and state restoration; same-mode independent live CUDA backward variation propagates through native AdamW/EMA",
                    cases={name: case["summary"] for name, case in cases.items()},
                    original_tolerances=dict(atol=ATOL, rtol=RTOL), raw_evidence_unchanged=True,
                    note="A precision note is not raw trajectory equality or a detector accuracy claim; DCC gradient exceptions remain listed"),
                failures=failures)
