"""Bounded evidence for the RDL EMA fusion check; preserves its original assertion.

The original ordered FP32 assertion is preserved as evidence. Only a fully
verified candidate-ID permutation can pass independent fusion acceptance.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import MethodType

import torch

from init_c19_lif_v1 import ROOT, require
from c19_lif_v1_diagnostic import (DiagnosticError, LIF_KEYS, PRE_KEYS, align_to_ids,
    atomic_json, compare_records, restore_rng, rng_state, schema, selection_report, tensor_compare)
from c19_lif_v1_probe import capture

BASE = "a0459d6a652cb702699087c88fa39a3e4c4087ec"
TOL = dict(atol=3e-5, rtol=3e-5)


def precision_settings():
    """Settings touched by the strict scope, plus dtype/cache/environment evidence."""
    return dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_autocast=torch.is_autocast_enabled(), cpu_autocast=torch.is_autocast_cpu_enabled(),
                cuda_autocast_dtype=str(torch.get_autocast_gpu_dtype()),
                cpu_autocast_dtype=str(torch.get_autocast_cpu_dtype()),
                autocast_cache_enabled=torch.is_autocast_cache_enabled(),
                nvidia_tf32_override=os.environ.get("NVIDIA_TF32_OVERRIDE"))


@contextmanager
def strict_fusion_precision(evidence):
    """Only the engineering comparison uses full FP32; always restore the caller."""
    before = evidence["before"] = precision_settings()
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            evidence["inside"] = precision_settings()
            yield
    finally:
        torch.backends.cudnn.allow_tf32 = before["cudnn_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = before["matmul_tf32"]
        # allow_tf32=True alone maps a caller's "medium" precision to "high".
        # Restore the exact original precision as well as the Boolean flags.
        torch.set_float32_matmul_precision(before["float32_matmul_precision"])
        evidence["after"] = precision_settings()
        evidence["restored"] = evidence["after"] == before
        require(evidence["restored"], "Strict fusion precision settings were not restored")


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((value.shape, value.dtype)).encode() +
                          value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(tensor_hash(value).encode())
    return h.hexdigest()


def cache(model):
    head = model.model[-1]
    return dict(shapes=deepcopy(getattr(head, "shapes", None)),
                **{key: dict(sha256=tensor_hash(getattr(head, key)),
                             dtype=str(getattr(head, key).dtype), device=str(getattr(head, key).device),
                             shape=list(getattr(head, key).shape)) for key in ("anchors", "valid_mask")})


def precision(model, image):
    return dict(cuda_autocast=torch.is_autocast_enabled(), cpu_autocast=torch.is_autocast_cpu_enabled(),
                input_dtype=str(image.dtype), input_device=str(image.device),
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                parameter_devices=sorted({str(p.device) for p in model.parameters()}),
                training_modules=[name for name, module in model.named_modules() if module.training])


def metric(a, b, name, atol=3e-5, rtol=3e-5):
    try:
        return tensor_compare(a.detach().cpu(), b.detach().cpu(), name, name, atol, rtol, device=str(a.device))
    except DiagnosticError as error:
        return error.detail


def passed(rows):
    return bool(rows) and all(not row.get("status") and row["allclose_failed_count"] == 0 for row in rows.values())


def historical_methods():
    """Execute ONLY the pinned predict/fuse methods, with the verified module code.

    This controls inference/fusion on the exact current fixture, not a claim of
    equivalence to a historically trained checkpoint or the mother's full recipe.
    """
    from ultralytics.nn import tasks
    path = "ultralytics-main/ultralytics/nn/tasks.py"
    raw = git("show", BASE + ":" + path)
    old, current = ast.parse(raw), ast.parse((ROOT/path).read_text(encoding="utf-8"))
    def method(tree, cls, name):
        parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
    methods = {}
    for cls, name in (("BaseModel", "fuse"), ("RTDETRDetectionModel", "predict")):
        node = method(old, cls, name)
        namespace = dict(vars(tasks))
        exec(compile(ast.Module(body=[node], type_ignores=[]), BASE + ":" + name, "exec"), namespace)
        methods[name] = namespace[name]
    protected = {}
    for filename in ("lif_down.py", "cbr.py", "head.py", "transformer.py", "conv.py"):
        path = "ultralytics-main/ultralytics/nn/modules/" + filename
        old_bytes = git("show", BASE + ":" + path).replace(b"\r\n", b"\n")
        new_bytes = (ROOT/path).read_bytes().replace(b"\r\n", b"\n")
        protected[filename] = dict(unchanged=old_bytes == new_bytes, sha256=hashlib.sha256(new_bytes).hexdigest())
    evidence = dict(base_commit=BASE, protected_modules=protected,
                    fuse_ast_unchanged=ast.dump(method(old, "BaseModel", "fuse")) ==
                                       ast.dump(method(current, "BaseModel", "fuse")),
                    scope="Pinned predict/fuse on identical post-lifecycle weights/input/cache; not a historical training run")
    require(evidence["fuse_ast_unchanged"] and all(x["unchanged"] for x in protected.values()),
            "Mother control needs unchanged native fusion/modules")
    return methods, evidence


def diagnose(unfused_cpu, image_cpu, before_cpu, after_cpu, rng, report, folder, device):
    """At most ten traced forwards; same runtime flags and precision as the failure."""
    records = {}
    def persist():
        atomic_json(folder/"fusion.json", report)
    def compare(a, b, stage, keys=None, exact=False):
        return compare_records(a, b, atol=0 if exact else TOL["atol"], rtol=0 if exact else TOL["rtol"],
                               stage=stage, device=device, precision="fp32", keys=keys, collect=True)
    image = image_cpu.to(device)
    unfused = unfused_cpu.to(device)
    fused = deepcopy(unfused).fuse(verbose=False).eval()
    def run(model, name, ids=None, common=None):
        restore_rng(rng)
        handle = None
        if common is not None:
            # Equal inputs to ALL head paths: memory, candidate features/anchors,
            # decoder and CBR P3. No replacement in production model code.
            def replace_input(module, args):
                return ([v.to(device).clone() for v in common], *args[1:])
            handle = model.model[-1].register_forward_pre_hook(replace_input)
        try:
            with torch.no_grad():
                value, record = capture(model, image, fixed_ids=ids)
            records[name] = record
            report.setdefault("traces", {})[name] = {k: record[k].tolist() if isinstance(record[k], torch.Tensor) else record[k]
                for k in ("actual_gather_verified", "topk_calls", "candidate_indices", "native_candidate_indices")}
            return value[0].detach().cpu(), record
        finally:
            if handle is not None:
                handle.remove()
    try:
        lif_a, lif_b = unfused.model[20], fused.model[20]
        report["fusion_invariants"] = dict(lif_bn_retained=hasattr(lif_b, "bn"),
            lif_state_exact=lif_a.state_dict().keys() == lif_b.state_dict().keys() and
                all(torch.equal(v, lif_b.state_dict()[k]) for k, v in lif_a.state_dict().items()),
            bn_before=sum(isinstance(m, torch.nn.BatchNorm2d) for m in unfused.modules()),
            bn_after=sum(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules()))
        pa, a = run(unfused, "natural_a")
        pb, b = run(fused, "natural_b")
        report["reproduce_original"] = dict(before=metric(before_cpu, pa, "before", 0, 0),
                                             after=metric(after_cpu, pb, "after", 0, 0))
        report["selection"] = selection_report(a, b)
        pre = LIF_KEYS + PRE_KEYS
        post = [k for k in schema(a) if k not in pre + ["candidate_indices"]]
        report["pre_selection"] = compare(a, b, "pre_selection", pre)
        report["natural_outputs"] = compare(a, b, "natural_outputs", post)
        persist()
        if report["selection"]["kind"] != "SET_DRIFT":
            report["candidate_id_alignment"] = compare(a, align_to_ids(a, b), "candidate_id_alignment", post)
        # Replay both native candidate lists; own encoder features remain live.
        for side, ids in (("A", a["candidate_indices"]), ("B", b["candidate_indices"])):
            _, ra = run(unfused, "fixed_" + side + "_a", ids)
            _, rb = run(fused, "fixed_" + side + "_b", ids)
            report["fixed_ids_" + side] = compare(ra, rb, "fixed_ids_" + side)
        # Isolate head numerics with bit-identical multiscale inputs AND query IDs.
        common = [a[f"scale_{i}"] for i in range(3)]
        _, ca = run(unfused, "common_head_a", a["candidate_indices"], common)
        _, cb = run(fused, "common_head_b", a["candidate_indices"], common)
        report["common_head_inputs"] = compare(ca, cb, "common_head_inputs", [f"scale_{i}" for i in range(3)], exact=True)
        head_keys = [k for k in schema(a) if k not in LIF_KEYS + ["downsample", "scale_0", "scale_1", "scale_2"]]
        report["common_head_outputs"] = compare(ca, cb, "common_head_outputs", head_keys)
        report["common_candidate_inputs"] = compare(ca, cb, "common_candidate_inputs",
            ["candidate_indices", "selected_features", "selected_anchors", "flattened_features"], exact=True)
        persist()
        # This copy has only served eval forwards; no optimizer/RDL is invoked.
        del fused
        parent = deepcopy(unfused)
        methods, report["mother_source"] = historical_methods()
        for name, method in methods.items():
            setattr(parent, name, MethodType(method, parent))
        for name in ("rdl_config", "rdl_epoch", "criterion"):
            if hasattr(parent, name):
                delattr(parent, name)
        _, ma = run(parent, "mother_a")
        parent.fuse(verbose=False).eval()
        _, mb = run(parent, "mother_b")
        report["mother_vs_rdl"] = dict(unfused=compare(a, ma, "mother_unfused", exact=True),
                                       fused=compare(b, mb, "mother_fused", exact=True))
        report["mother_selection"] = selection_report(ma, mb)
        report["diagnostic_status"] = "COMPLETE"
        faithful = passed(report["reproduce_original"])
        mother_exact = all(passed(v) for v in report["mother_vs_rdl"].values())
        continuous = all(passed(report[k]) for k in ("pre_selection", "fixed_ids_A", "fixed_ids_B", "common_head_inputs", "common_head_outputs", "common_candidate_inputs"))
        if report["original_assertion"]["allclose_failed_count"] == 0:
            report["finding"] = "STRICT_FP32_CHECK_PASSED"
        elif not faithful:
            report["finding"] = "REPLAY_DIFFERS_FROM_ORIGINAL_REQUIRES_REVIEW"
        elif mother_exact and continuous and report["selection"]["kind"] == "PERMUTATION" and passed(report["candidate_id_alignment"]):
            report["finding"] = "MOTHER_CANDIDATE_PERMUTATION_ON_SAME_FIXTURE"
        elif mother_exact and continuous and report["selection"]["kind"] == "SET_DRIFT":
            report["finding"] = "MOTHER_CANDIDATE_SET_DRIFT_ON_SAME_FIXTURE_REQUIRES_REVIEW"
        else:
            report["finding"] = "CONTINUOUS_OR_STATE_DIFFERENCE_REQUIRES_REVIEW"
    except Exception as error:
        report["diagnostic_status"] = "INCOMPLETE"
        report["diagnostic_error"] = repr(error)
        raise
    finally:
        # Never discard a failing trace, including incomplete probes.
        torch.save(records, folder/"records.pt")
        persist()
        restore_rng(rng)


def check_ema_fusion(ema, batch, folder=None, source_sha256=None, reference=None):
    """Strict FP32 comparison on a disposable copy; preserve caller precision."""
    report = dict(precision_scope={})
    try:
        with strict_fusion_precision(report["precision_scope"]):
            _, ordered_error = _check_ema_fusion(deepcopy(ema).float().eval(), batch, report, folder, source_sha256, reference)
        # Acceptance is evaluated only AFTER actual precision restoration.
        from rdl_v1_fusion_acceptance import review_fusion
        report["fusion_acceptance"] = review_fusion(report)
        if not report["fusion_acceptance"]["accepted"]:
            if ordered_error is not None:
                raise ordered_error
            raise RuntimeError(report["fusion_acceptance"]["reason"])
        return report
    except BaseException as error:
        report.setdefault("fusion_acceptance", dict(accepted=False, status="FAIL", reason=repr(error)))
        raise
    finally:
        # The inner trace is written before context exit; append actual restoration
        # evidence even when the original assertion or a probe raised an exception.
        if report.get("evidence_directory"):
            atomic_json(Path(report["evidence_directory"])/"fusion.json", report)
        print("RDL fusion precision: " + json.dumps(report["precision_scope"]), flush=True)


def _check_ema_fusion(ema, batch, report, folder, source_sha256, reference):
    """Original assertion and candidate diagnostics, inside the strict scope."""
    report.update(schema="rdl_fusion_diagnostic_v2", tolerance=TOL, acceptance="ORDERED_OR_VERIFIED_CANDIDATE_ID_PERMUTATION",
                  controlled_source_sha256=source_sha256,
                  runtime=dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                    executable=sys.executable, gpu=torch.cuda.get_device_name(batch["img"].device) if batch["img"].is_cuda else None,
                    commit=git("rev-parse", "HEAD").decode().strip(), threads=torch.get_num_threads(),
                    deterministic=torch.are_deterministic_algorithms_enabled(),
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    cudnn_benchmark=torch.backends.cudnn.benchmark), formal_training="NOT_RUN")
    image = batch["img"]
    require(image.dtype == torch.float32 and all(p.dtype == torch.float32 and p.device == image.device
            for p in ema.parameters()), "Strict fusion requires FP32 model/input on the same device")
    require(not any(m.training for m in ema.modules()), "Strict fusion requires eval")
    state_before = state_hash(ema)
    input_before = tensor_hash(image)
    report["before_predict"] = dict(precision=precision(ema, image), cache=cache(ema), state_sha256=state_before)
    def predict_observed(name):
        def observe(module, args):
            report.setdefault("inside_score_head", {})[name] = dict(
                cuda_autocast=torch.is_autocast_enabled(), cpu_autocast=torch.is_autocast_cpu_enabled(),
                input_dtype=str(args[0].dtype), input_device=str(args[0].device),
                weight_dtype=str(module.weight.dtype), training=module.training)
        handle = ema.model[-1].enc_score_head.register_forward_pre_hook(observe)
        try:
            return ema.predict(image)
        finally:
            handle.remove()
    with torch.no_grad():
        pred = predict_observed("unfused")
        require(pred[0].shape == (2, 300, 5), "Inference contract")
        before_validation = pred[0].clone()
        cache_before_validation = cache(ema)
        val_loss, display = ema.loss(batch, preds=pred)
        require(torch.isfinite(val_loss) and len(display) == 3, "Validation loss")
        if hasattr(ema, "criterion"):
            require(ema.criterion.last_stats["validation_L0_only"], "Validation requested RDL")
        report["validation_invariants"] = dict(prediction_unchanged=torch.equal(before_validation, pred[0]),
            state_unchanged=state_before == state_hash(ema), input_unchanged=input_before == tensor_hash(image),
            cache_unchanged=cache_before_validation == cache(ema))
        require(all(report["validation_invariants"].values()), "Validation mutated fusion fixture")
        before = pred[0].clone()
        # Only a diagnostic snapshot; the real fusion still acts on this same EMA.
        unfused = deepcopy(ema).cpu()
        report["unfused_snapshot_state_exact"] = state_before == state_hash(unfused)
        require(report["unfused_snapshot_state_exact"], "Diagnostic snapshot changed model weights")
        rng = rng_state()
        report["before_fuse"] = dict(precision=precision(ema, image), cache=cache(ema))
        ema.fuse(verbose=False).eval()
        require(not any(m.training for m in ema.modules()), "Fused check copy must remain eval")
        report["after_fuse"] = dict(precision=precision(ema, image), cache=cache(ema))
        fused = predict_observed("fused")[0]
        report["after_predict"] = dict(precision=precision(ema, image), cache=cache(ema))
        report["original_assertion"] = metric(before, fused, "ordered_output")
        report["input_unchanged_after_fuse"] = input_before == tensor_hash(image)
        report["lif_bn_retained"] = hasattr(ema.model[20], "bn")
        error = None
        try:
            torch.testing.assert_close(before, fused, **TOL, check_dtype=False)
            require(report["lif_bn_retained"], "LIF fusion guard")
            require(report["input_unchanged_after_fuse"], "Fusion mutated input")
        except Exception as caught:
            error = caught
        report["original_status"] = "FAIL" if error else "PASS"
        if error is not None:
            report["original_error"] = repr(error)
        if reference is not None:
            report["saved_fixture_replay"] = dict(before=metric(reference["before"], before.cpu(), "before", 0, 0),
                                                  after=metric(reference["after"], fused.cpu(), "after", 0, 0))
        if error is not None or folder is not None:
            folder = Path(folder) if folder is not None else ROOT/"outputs/rdl_v1"/(
                "fusion_failure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
            folder.mkdir(parents=True, exist_ok=False)
            report["evidence_directory"] = str(folder.resolve())
            atomic_json(folder/"fusion.json", report)
            torch.save(dict(model=unfused, image=image.cpu().clone(), before=before.cpu(), after=fused.cpu(), rng=rng), folder/"fixture.pt")
            try:
                diagnose(unfused, image.cpu(), before.cpu(), fused.cpu(), rng, report, folder, str(image.device))
            except Exception:
                if error is None:
                    raise
            finally:
                print("RDL fusion evidence: " + str(folder/"fusion.json"), flush=True)
    return report, error
