"""Versioned metadata-only training admission; fusion observations stay diagnostic."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re

CONTRACT = "blc_training_admission_v3_amp_backoff"
VARIANTS = ("cbr_lif_blc_v1", "blc_v1")
KEYS = {"model.5.blc." + name for name in ("Wd.weight", "Wg.weight", "Wg.bias", "Wo.weight")}
OBSERVED = ("PASSED", "PRECISION_NOTE", "PENDING")
# Fixed producer 0d233512: engine/trainer.py constructs
# torch.cuda.amp.GradScaler(enabled=self.amp), with no configuration overrides.
# Verified upstream: pytorch/pytorch v2.1.2 torch/cuda/amp/grad_scaler.py
# __init__ and aten/src/ATen/native/cuda/AmpKernels.cu amp_update_scale_cuda_kernel.
# These are source-derived defaults, not values inferred from a server screenshot.
NATIVE_SCALER = dict(torch="2.1.2+cu121", init_scale=65536.0, backoff_factor=0.5,
                     growth_factor=2.0, growth_interval=2000)


def read_report(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class Checks:
    def __init__(self, errors=()):
        self.errors = list(errors)

    def need(self, condition, path):
        if not condition:
            self.errors.append(path)

    def scan(self, value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("error", "traceback", "probe_error", "exception_stage", "resource_cleanup_errors"):
                    self.need(not item, path + "." + key + ": conflicting runtime error")
                self.scan(item, path + "." + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                self.scan(item, f"{path}[{i}]")
        elif isinstance(value, float) or isinstance(value, str) and value.lower() in (
                "nan", "inf", "-inf", "infinity", "-infinity", "+inf"):
            self.need(finite(value), path + ": nonfinite value")

    def difference(self, row, path, exact=False, passed=False):
        self.need(isinstance(row, dict) and bool(row), path + ": missing difference")
        row = row if isinstance(row, dict) else {}
        self.need(row.get("finite") is True, path + ".finite")
        for key in ("max_abs", "relative_L2", "atol", "rtol"):
            self.need(finite(row.get(key)) and row[key] >= 0, path + "." + key)
        self.need(row.get("atol") == 2e-5 and row.get("rtol") == 2e-4, path + ": original tolerance")
        self.need(type(row.get("allclose")) is bool, path + ".allclose missing")
        if exact:
            self.need(row.get("allclose") is True and row.get("max_abs") == 0
                      and row.get("relative_L2") == 0, path + ": same-dtype exact restoration")
        if passed:
            self.need(row.get("status") == "PASSED" and row.get("allclose") is True,
                      path + ": same-file AutoBackend check")

    def branch(self, row, path):
        self.need(row.get("calls") == 1 and row.get("completed") is True, path + ": branch call/completion")
        values = row.get("increment_norms", [])
        self.need(isinstance(values, list) and len(values) == 1 and finite(values[0]) and values[0] > 0,
                  path + ": finite nonzero BLC increment")

    def audit(self, row, path, adaptation=False):
        for key in ("NEW_BUFFER", "MISSING", "UNEXPECTED", "SHAPE_MISMATCH"):
            self.need(row.get(key) == [], path + "." + key)
        self.need(isinstance(row.get("NEW_TRAINABLE"), list) and set(row["NEW_TRAINABLE"]) == KEYS,
                  path + ".NEW_TRAINABLE")
        common = row.get("COMMON")
        if isinstance(common, list):
            self.need(bool(common) and all(v.get("equal") is True and digest(v.get("sha256")) for v in common),
                      path + ".COMMON equality/inventory")
        elif isinstance(common, dict):
            self.need(finite(common.get("count")) and common["count"] > 0 and common.get("unequal_count") == 0
                      and common.get("unequal_samples") == [] and digest(common.get("inventory_sha256")),
                      path + ".COMMON compact equality/inventory")
        else:
            self.need(False, path + ".COMMON missing")
        if adaptation:
            for key in ("native_get_model", "classification_parent_candidate_exact", "new_state_exact"):
                self.need(row.get(key) is True, path + "." + key)
            self.need(row.get("nc") == 1, path + ".nc")

    def initialization(self, row, variant):
        p = "init." + variant
        self.scan(row, p)
        self.need(row.get("status") == "PASSED" and row.get("phase") == "init-preflight"
                  and row.get("variant") == variant, p + ": completed matching initialization")
        source = row.get("current_source_audit", {})
        self.need(source.get("status") == "PASSED" and source.get("reload_exact") is True
                  and source.get("variant") == variant, p + ": source initialization/reload")
        self.need(source.get("source_sha256") == row.get("context", {}).get("source_sha256")
                  and digest(source.get("source_sha256")), p + ": source SHA256")
        for name in ("constructor", "transfer", "nc1_rebuild"):
            self.audit(source.get(name, {}), p + ".source." + name, name == "nc1_rebuild")
        structure = row.get("structure", {})
        self.need(structure.get("status") == "PASSED", p + ".structure.status")
        if "exact_saved_state" in structure:
            self.need(structure.get("exact_saved_state") is True
                      and structure.get("topology_parameters_and_zero_initialization") is True
                      and finite(structure.get("tensors")) and structure["tensors"] > 0, p + ": light structure audit")
        else:  # Verified older producer asserts exact saved state before topology_checks.
            for key in ("exact_initial_features", "exact_native_output", "public_rng_unchanged"):
                self.need(structure.get(key) is True, p + ".structure." + key)
            self.need(structure.get("blc_calls") == 1 and structure.get("consumers") == [6, 17]
                      and structure.get("added_parameters") == 8561, p + ": original topology/routing")
            self.audit(structure.get("native_get_model", {}), p + ".structure.rebuild", True)
        api = row.get("actual_train_api", {})
        self.need(api.get("status") == "PASSED" and api.get("actual_RTDERT_train_api") is True
                  and api.get("all_states_exact") is True and api.get("batch_count") == 0, p + ": actual Trainer reconstruction")
        self.audit(api.get("audit", {}), p + ".actual_train_api.audit", True)
        self.need(api.get("audit", {}).get("ALLOWED_CLASS_ADAPTATION") == [], p + ": nc1 reconstruction changed classes")

    def updates(self, row, path, minimum, initial_state):
        """Interpret recorded attempts; never run/replay an optimizer or scaler."""
        phase_errors = len(self.errors)
        self.need(row.get("status") == "PASSED", path + ".status")
        batches, steps = row.get("batches", []), row.get("steps", [])
        self.need(isinstance(batches, list) and isinstance(steps, list), path + ": batch/step lists missing")
        batches = batches if isinstance(batches, list) else []
        steps = steps if isinstance(steps, list) else []
        self.need(bool(batches) and row.get("attempted_batches") == len(batches), path + ": attempted/completed batches")
        valid_batches = []
        for i, batch in enumerate(batches):
            valid = (isinstance(batch, dict) and finite(batch.get("loss"))
                     and type(batch.get("gt")) is int and batch["gt"] > 0)
            self.need(valid,
                      f"{path}.batches[{i}]: finite loss/real GT")
            if path == "capacity":
                numbered = isinstance(batch, dict) and batch.get("batch") == i + 1
                self.need(numbered, f"{path}.batches[{i}]: original batch numbering")
                valid = valid and numbered
            valid_batches.append(valid)
        self.need(initial_state is not None, path + ": initial optimizer/scaler state is unproven")
        state = deepcopy(initial_state)
        effective, upstream, previous_batch = 0, False, 0
        attempts = []
        for i, step in enumerate(steps):
            p = f"{path}.steps[{i}]"
            start_errors = len(self.errors)
            if not isinstance(step, dict):
                self.need(False, p + ": missing attempt object")
                attempts.append(dict(classification="INVALID", errors=self.errors[start_errors:]))
                state = None
                continue
            norms, delta = step.get("new_gradient_norms", {}), step.get("parameter_delta", {})
            self.need(isinstance(norms, dict) and isinstance(delta, dict), p + ": BLC norm/delta mappings missing")
            norms = norms if isinstance(norms, dict) else {}
            delta = delta if isinstance(delta, dict) else {}
            self.need(set(norms) == KEYS and set(delta) == KEYS, p + ": BLC parameter inventory")
            self.need(all(finite(v) and v >= 0 for v in norms.values())
                      and all(finite(v) and v >= 0 for v in delta.values()), p + ": finite gradients/deltas")
            for key in ("scale_before", "scale_after"):
                self.need(finite(step.get(key)) and step[key] > 0, p + "." + key)
            optimizer_step = step.get("optimizer_state_step")
            self.need(finite(optimizer_step) and optimizer_step >= 0 and int(optimizer_step) == optimizer_step,
                      p + ": optimizer_state_step must be finite nonnegative integer")
            index = step.get("batch")
            valid_index = type(index) is int and previous_batch < index <= len(batches)
            self.need(valid_index, p + ": batch index must increase and be within completed batches")
            self.need(valid_index and valid_batches[index - 1], p + ": corresponding finite-loss/valid-GT batch missing")
            for key in ("all_gradients_finite", "scaler_skipped", "effective_update"):
                self.need(type(step.get(key)) is bool, p + "." + key + ": explicit boolean required")
            self.need(state is not None, p + ": preceding optimizer/scaler state unproven")
            entry = dict(batch=index, classification="INVALID", scale_before=step.get("scale_before"),
                         scale_after=step.get("scale_after"), optimizer_state_step=optimizer_step,
                         recorded_effective_update=step.get("effective_update"),
                         expected_previous_optimizer_state_step=state["optimizer_state_step"] if state else None,
                         expected_scale_before=state["scale"] if state else None)
            kind, next_state = "INVALID", None
            if len(self.errors) == start_errors:
                self.need(step["scale_before"] == state["scale"], p + ": scale_before breaks previous/restored state")
                if step["all_gradients_finite"] is False and step["scaler_skipped"] is True and step["effective_update"] is False:
                    self.need(step["scale_after"] == state["scale"] * NATIVE_SCALER["backoff_factor"],
                              p + ": AMP skip must use the verified native backoff ratio")
                    self.need(optimizer_step == state["optimizer_state_step"], p + ": AMP skip advanced optimizer step")
                    self.need(all(v == 0 for v in delta.values()), p + ": AMP skip changed a BLC parameter")
                    kind = "AMP_BACKOFF"
                    next_state = dict(optimizer_state_step=optimizer_step, scale=step["scale_after"], growth_tracker=0)
                elif step["all_gradients_finite"] is True and step["scaler_skipped"] is False:
                    self.need(optimizer_step == state["optimizer_state_step"] + 1,
                              p + ": native optimizer step must advance once from previous/restored state")
                    tracker = state["growth_tracker"] + 1
                    grow = tracker == NATIVE_SCALER["growth_interval"]
                    self.need(step["scale_after"] == state["scale"] * (NATIVE_SCALER["growth_factor"] if grow else 1),
                              p + ": applied step has inconsistent native scale")
                    wo_changed = delta["model.5.blc.Wo.weight"] > 0
                    self.need(step["effective_update"] is wo_changed, p + ": effective_update contradicts actual Wo delta")
                    kind = "EFFECTIVE_UPDATE" if wo_changed else "APPLIED_NO_WO_CHANGE"
                    next_state = dict(optimizer_state_step=optimizer_step, scale=step["scale_after"],
                                      growth_tracker=0 if grow else tracker)
                else:
                    self.need(False, p + ": nonfinite/skip/effective flags contradict native execution")
            if len(self.errors) == start_errors:
                entry["classification"] = kind
                state = next_state
                if kind == "EFFECTIVE_UPDATE":
                    effective += 1
                    upstream |= all(v > 0 for v in norms.values())
            else:
                state = None  # Do not invent a resume origin after an invalid/missing observation.
            entry["errors"] = self.errors[start_errors:]
            attempts.append(entry)
            if type(index) is int:
                previous_batch = index
        self.need(effective >= minimum and row.get("effective_updates") == effective, path + ": insufficient/inconsistent effective updates")
        if minimum == 2:
            self.need(upstream, path + ": original finite nonzero upstream gradient rule")
        summary = dict(initial_state=deepcopy(initial_state), attempts=attempts, effective_updates=effective,
                       reported_effective_updates=row.get("effective_updates"),
                       amp_backoff_count=sum(r["classification"] == "AMP_BACKOFF" for r in attempts),
                       applied_no_wo_change_count=sum(r["classification"] == "APPLIED_NO_WO_CHANGE" for r in attempts),
                       final_state=state if len(self.errors) == phase_errors else None,
                       scope="Derived classifications/state from original attempts and source defaults; not new observations")
        return len(batches), summary


def evaluate(report, init_reports, identity_errors=()):
    """The one decision function used by preflight, reassessment and start/resume."""
    checks = Checks(identity_errors)
    raw = {k: v for k, v in report.items() if k not in ("training_admission", "fusion_diagnostic", "contract")}
    checks.scan(raw, "preflight")
    checks.need(report.get("status") in OBSERVED, "preflight.status: incomplete/failed execution")
    checks.need(report.get("phase") == "preflight" and report.get("variant") in VARIANTS, "preflight.phase/variant")
    checks.need(report.get("stage") == "complete", "preflight.stage incomplete")
    checks.need(report.get("formal_init_untouched") is True, "preflight.formal_init_untouched")
    for variant in VARIANTS:
        checks.initialization(init_reports.get(variant, {}), variant)
    capacity = report.get("capacity", {})
    checks.need(capacity.get("batch") == 16 and capacity.get("imgsz") == 640 and capacity.get("AMP") is True,
                "capacity: original B16/640/native AMP")
    checks.need(report.get("context", {}).get("runtime", {}).get("torch") == NATIVE_SCALER["torch"],
                "AMP evidence must use the source-verified torch 2.1.2+cu121 scaler configuration")
    # Fresh AdamW starts without state; the protected producer takes max(..., default=0).
    count, capacity_attempts = checks.updates(capacity, "capacity", 2,
        dict(optimizer_state_step=0, scale=NATIVE_SCALER["init_scale"], growth_tracker=0))
    for batch in capacity.get("batches", []) if isinstance(capacity.get("batches"), list) else []:
        checks.need(isinstance(batch, dict) and batch.get("amp") is True, "capacity: batch AMP evidence")
    resume = report.get("native_resume", {})
    restored = resume.get("restored_state", {})
    resume_origin = capacity_attempts["final_state"] if all(
        restored.get(key) is True for key in ("epoch", "optimizer_moments_steps", "scaler", "ema", "updates")
    ) and restored.get("shared_storage") is False else None
    resumed_count, resume_attempts = checks.updates(resume, "native_resume", 1, resume_origin)
    count += resumed_count
    checks.need(0 < count <= 16 and report.get("total_attempted_training_batches") == count
                and report.get("batch_budget") == 16 and report.get("validation_batch_budget") == 1,
                "preflight: combined 16 training/1 validation batch budget")
    for key in ("epoch", "optimizer_moments_steps", "scaler", "ema", "updates"):
        checks.need(resume.get("restored_state", {}).get(key) is True, "native_resume.restored_state." + key)
    checks.need(resume.get("restored_state", {}).get("shared_storage") is False, "native_resume: independent storage")
    checkpoint = report.get("bounded_checkpoint", {})
    checks.need(checkpoint.get("epoch") == 0 and checkpoint.get("incomplete_epoch") is True
                and checkpoint.get("native_resume_next_epoch") == resume.get("epoch") == 1
                and digest(checkpoint.get("sha256")), "native checkpoint/resume epoch identity")
    val = report.get("native_half_ema_epoch_val", {})
    checks.need(val.get("status") == "PASSED" and val.get("batches") == 1 and val.get("half") is True,
                "native_half_ema_epoch_val: completed real half EMA batch")
    checks.need(isinstance(val.get("metrics"), dict) and bool(val["metrics"])
                and all(finite(v) for v in val["metrics"].values()), "native_half_ema_epoch_val.metrics finite/missing")
    life = report.get("lifecycle", {})
    checks.need(life.get("status") in OBSERVED and life.get("stage") == "complete", "lifecycle: incomplete/error")
    checks.need(life.get("state_dict_and_full_model_exact") is True, "lifecycle.same_dtype exact")
    for name in ("state_dict", "full_model"):
        checks.difference(life.get("serialization", {}).get(name, {}), "serialization." + name, exact=True)
    checks.branch(life.get("live_blc", {}), "lifecycle.live_blc")
    checks.need(finite(life.get("nonzero_increment_norm")) and life["nonzero_increment_norm"] > 0, "lifecycle.nonzero_increment_norm")
    checks.difference(life.get("live_vs_file_quantization", {}), "lifecycle.live_vs_file_quantization")
    storage = life.get("native_storage", {})
    checks.need("model" in storage and storage["model"] is None and storage.get("ema_dtype") == "torch.float16"
                and storage.get("optimizer") == "native convert_optimizer_state_dict_to_fp16"
                and finite(storage.get("updates")) and storage["updates"] >= 2, "lifecycle.native_storage")
    diagnostic = {}
    for mode in ("FP32", "AMP", "half"):
        row = life.get("paths", {}).get(mode, {})
        p = "lifecycle.paths." + mode
        checks.need(row.get("status") in OBSERVED and row.get("stage") == "complete", p + ": incomplete/error")
        checks.need(row.get("dtype") == ("torch.float16" if mode == "half" else "torch.float32")
                    and row.get("autocast") is (mode == "AMP") and str(row.get("device", "")).startswith("cuda")
                    and row.get("input_shape") == [1, 3, 160, 160] and digest(row.get("input_sha256")), p + ": precision/input identity")
        state = row.get("independent_state", {})
        checks.need(state.get("exact") is True and state.get("shared_storage") is False
                    and finite(state.get("tensors")) and state["tensors"] > 0, p + ": independent exact state")
        for name in ("direct_blc", "backend_blc"):
            checks.branch(row.get(name, {}), p + "." + name)
        checks.difference(row.get("autobackend", {}), p + ".autobackend", passed=True)
        fusion = row.get("fusion", {})
        checks.need(fusion.get("status") in OBSERVED, p + ".fusion: missing/error/NOT_RUN")
        checks.difference(fusion.get("natural", {}), p + ".fusion.natural")
        for name in ("5", "6", "7", "scores"):
            checks.difference(fusion.get("features", {}).get(name, {}), p + ".fusion.features." + name)
        replay = fusion.get("fixed_indices_diagnostic_only")
        needed = fusion.get("natural", {}).get("allclose") is False
        checks.need(not needed or isinstance(replay, dict), p + ".fusion: missing fixed-index diagnostic")
        if replay is not None:
            checks.difference(replay, p + ".fusion.fixed_indices_diagnostic_only")
        checks.need(fusion.get("forward_count") == (3 if replay is not None else 2), p + ".fusion.forward_count")
        positions = fusion.get("differing_candidate_positions")
        checks.need(type(positions) is int and 0 <= positions <= 300
                    and type(fusion.get("same_candidate_set")) is bool, p + ".fusion.candidates")
        indices = fusion.get("indices_sha256", [])
        checks.need(isinstance(indices, list) and len(indices) == 2 and all(digest(v) for v in indices), p + ".fusion.indices_sha256")
        diagnostic[mode] = dict(original_path_status=row.get("status", "NOT_RUN"), **deepcopy(fusion))
    statuses = [row.get("status", "NOT_RUN") for row in diagnostic.values()]
    diagnostic_status = ("FAILED" if any(s not in OBSERVED for s in statuses) else
                         "PENDING" if "PENDING" in statuses else
                         "PRECISION_NOTE" if "PRECISION_NOTE" in statuses else "PASSED")
    return dict(contract=CONTRACT,
                training_admission=dict(status="FAILED" if checks.errors else "PASSED", errors=checks.errors,
                    native_scaler_contract=deepcopy(NATIVE_SCALER),
                    optimizer_attempts=dict(capacity=capacity_attempts, native_resume=resume_attempts),
                    scope="Evidence permits a native training experiment only; no accuracy/convergence claim"),
                fusion_diagnostic=dict(status=diagnostic_status, modes=diagnostic,
                    fixed_indices_diagnostic_only=True, scope="Original fusion observations; not fusion equivalence approval"))


def print_result(path, result):
    print("Admission report:", path, flush=True)
    for phase, summary in result["training_admission"]["optimizer_attempts"].items():
        for i, row in enumerate(summary["attempts"]):
            print(f"{phase}.steps[{i}] batch={row.get('batch')} {row['classification']} "
                  f"scale={row.get('scale_before')} -> {row.get('scale_after')} "
                  f"optimizer_state_step={row.get('optimizer_state_step')} "
                  f"previous_step={row.get('expected_previous_optimizer_state_step')}", flush=True)
        print(f"{phase}: AMP_BACKOFF={summary['amp_backoff_count']} "
              f"APPLIED_NO_WO_CHANGE={summary['applied_no_wo_change_count']} "
              f"effective_updates={summary['effective_updates']} "
              f"reported_effective_updates={summary['reported_effective_updates']}", flush=True)
    print("training_admission=" + result["training_admission"]["status"] +
          "; fusion_diagnostic=" + result["fusion_diagnostic"]["status"], flush=True)
    print(result["training_admission"]["scope"], flush=True)
    print(result["fusion_diagnostic"]["scope"], flush=True)
    for error in result["training_admission"]["errors"]:
        print("BLOCKED:", error, flush=True)


def reassess(variant, from_report):
    from blc_common import paths, stamp, write_json
    from blc_admission_identity import collect
    raw, inits, bindings, contexts, proof, errors = collect(variant, from_report)
    result = evaluate(raw, inits, errors)
    result.update(phase="training-admission", variant=variant, original_status=raw.get("status"),
                  contexts=contexts, sources=bindings, migration=proof,
                  execution=dict(new_training_batches=0, new_validation_batches=0, new_checkpoints=0))
    destination = paths(variant)["evidence"] / ("admission-" + stamp()) / "report.json"
    write_json(destination, result)
    print_result(destination, result)
    if result["training_admission"]["status"] != "PASSED":
        raise RuntimeError("Training admission blocked; inspect the specific missing/conflicting evidence above")
    return str(destination)


def require_admission(variant):
    from blc_common import paths
    from blc_admission_identity import collect, verify_bindings
    reports = sorted(paths(variant)["evidence"].glob("admission-*/report.json"))
    if not reports:
        raise RuntimeError("PENDING: reassess --from-report <original preflight report>; no training is run by reassess")
    path = reports[-1]
    saved = read_report(path)
    if saved.get("contract") != CONTRACT or saved.get("variant") != variant or saved.get("phase") != "training-admission":
        raise RuntimeError("Admission contract/variant/phase mismatch")
    verify_bindings(saved["sources"])
    raw, inits, bindings, contexts, proof, errors = collect(variant, saved["sources"]["preflight"]["path"], saved["sources"])
    if contexts != saved.get("contexts"):
        errors.append("Admission complete current context changed")
    if bindings != saved.get("sources") or proof != saved.get("migration"):
        errors.append("Admission raw evidence/provenance binding changed")
    result = evaluate(raw, inits, errors)
    for key in ("training_admission", "fusion_diagnostic"):
        if saved.get(key) != result[key]:
            raise RuntimeError("Admission summary differs from recomputed " + key)
    print_result(path, result)
    if result["training_admission"]["status"] != "PASSED":
        raise RuntimeError("Training admission failed on recheck")
    return str(path)
