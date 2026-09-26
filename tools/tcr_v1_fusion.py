"""Scoped TCR fusion diagnostics; never changes the training model or recipe.

The original mother tensor tolerances and candidate-aware predicate are reused.
Runtime comparisons retain their failures, independently of the strict FP32 gate.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import pickle
import traceback

from tcr_v1_core import Path, torch, require, write_json, read_json, verify_model
from c19_lif_v1_diagnostic import (
    DiagnosticError, fusion_protocol, tensor_compare, rng_state, restore_rng,
)
from c19_lif_v1_cutoff import fusion_accepted

ATOL, RTOL = 2e-5, 2e-4  # Unchanged mother FP32 tolerances.
VERSION = "tcr_fusion_precision_v1"


def _autocast_enabled(device):
    try:
        return torch.is_autocast_enabled(device)
    except TypeError:  # torch 2.1 API
        return torch.is_autocast_enabled() if device == "cuda" else torch.is_autocast_cpu_enabled()


def precision_state():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                autocast_cuda=_autocast_enabled("cuda"), autocast_cpu=_autocast_enabled("cpu"),
                cudnn_enabled=torch.backends.cudnn.enabled,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


@contextmanager
def precision_scope(strict=False, recorded=None):
    """Restore TF32/matmul settings and nested autocast on success or exception.

Only strict diagnostics disable autocast. A runtime check inherits its caller's
actual autocast. Replay can restore the *recorded* TF32 flags, without claiming
that missing legacy settings such as cuDNN benchmark/autocast were recorded.
"""
    before = precision_state()
    try:
        if strict:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        elif recorded is not None:
            torch.backends.cuda.matmul.allow_tf32 = recorded["tf32"]
            torch.backends.cudnn.allow_tf32 = recorded["cudnn_tf32"]
        with ExitStack() as stack:
            if strict:
                for device in ("cuda", "cpu"):
                    stack.enter_context(torch.autocast(device_type=device, enabled=False))
            yield precision_state()
    finally:
        torch.set_float32_matmul_precision(before["float32_matmul_precision"])
        # Setting allow_tf32=True unconditionally collapses "medium" to "high"
        # on supported torch versions. Restore the precision enum first.
        if torch.backends.cuda.matmul.allow_tf32 != before["matmul_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = before["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_tf32"]


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def rng_hash(state):
    parts = {"cpu": tensor_hash(state["cpu"]), "cuda": [tensor_hash(s) for s in state["cuda"]]}
    for key in ("python", "numpy"):
        if key in state:
            parts[key] = hashlib.sha256(pickle.dumps(state[key], protocol=4)).hexdigest()
    return parts


def model_identity(model):
    state = model.state_dict()
    bn = {k: v for k, v in state.items() if any(s in k for s in ("running_mean", "running_var", "num_batches_tracked"))}
    return dict(state_sha256=state_hash(state), bn_buffers_sha256=state_hash(bn),
                states=len(state), bn_buffers=len(bn),
                all_eval=all(not m.training for m in model.modules()),
                floating_dtypes=sorted({str(v.dtype) for v in state.values() if v.is_floating_point()}),
                P_sha256=tensor_hash(model.model[17].tcr.P.weight),
                O_sha256=tensor_hash(model.model[17].tcr.O.weight),
                O_nonzero=int(torch.count_nonzero(model.model[17].tcr.O.weight)))


def pair_invariants(unfused, fused):
    from ultralytics.nn.modules import ConvTCR
    a, b = unfused.model[17], fused.model[17]
    lif_a, lif_b = unfused.model[20], fused.model[20]
    result = dict(
        convtcr_forward_fuse=getattr(b.forward, "__func__", None) is ConvTCR.forward_fuse,
        convtcr_bn_fused=not hasattr(b, "bn"),
        tcr_enabled=a.tcr.enabled and b.tcr.enabled,
        P_exact=torch.equal(a.tcr.P.weight, b.tcr.P.weight),
        O_exact=torch.equal(a.tcr.O.weight, b.tcr.O.weight),
        O_nonzero=bool(torch.count_nonzero(b.tcr.O.weight)),
        lif_bn_retained=hasattr(lif_b, "bn"),
        lif_state_exact=state_hash(lif_a.state_dict()) == state_hash(lif_b.state_dict()),
        lif_forward_retained=getattr(lif_a.forward, "__func__", None) is getattr(lif_b.forward, "__func__", None),
        other_bn_fused=sum(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules()) <
                       sum(isinstance(m, torch.nn.BatchNorm2d) for m in unfused.modules()))
    require(all(result.values()), f"Fusion structure/state changed: {result}")
    verify_model(fused)
    return result


class _TraceComplete(Exception):
    pass


def early_trace(model, image, state):
    """Observe execution order through node 25, before any decoder selection.

Node 17's original Conv means conv/bn/act as a block: its semantic boundary is
the input to TCR. Raw conv alone has different semantics after BN folding.
"""
    records, handles, counts = {}, [], {"TCR": 0, "P": 0, "O": 0}

    def save(name, value):
        require(name not in records and isinstance(value, torch.Tensor), f"Ambiguous trace: {name}")
        records[name] = value.detach().cpu().clone()

    def output(name):
        return lambda mod, args, value: save(name, value)

    def tcr_input(mod, args):
        save("node17.original_conv_output", args[0])
        save("tcr.input", args[0])

    def tcr_output(mod, args, value):
        counts["TCR"] += 1
        save("tcr.output", value)
        save("tcr.residual", value - args[0])

    def projection(name):
        def hook(mod, args, value):
            counts[name] += 1
            save("tcr." + name, value)
        return hook

    def stop(mod, args, value):
        raise _TraceComplete()

    outer_rng = rng_state()
    try:
        # Includes each original backbone block, then every top-level neck node.
        for name, module in model.named_modules():
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit() and int(parts[1]) <= 7:
                if type(module).__name__ in {"ConvNormLayer", "BasicBlock"}:
                    handles.append(module.register_forward_hook(output("backbone." + name)))
        for i, module in enumerate(model.model[:26]):
            handles.append(module.register_forward_hook(output(f"node{i}.output")))
        tcr = model.model[17].tcr
        handles.extend([tcr.register_forward_pre_hook(tcr_input), tcr.register_forward_hook(tcr_output),
                        tcr.P.register_forward_hook(projection("P")), tcr.O.register_forward_hook(projection("O")),
                        model.model[20].register_forward_pre_hook(lambda mod, args: save("lif_input", args[0])),
                        model.model[25].register_forward_hook(stop)])
        restore_rng(state)
        with torch.no_grad():
            try:
                model(image)
            except _TraceComplete:
                pass
        require("node25.output" in records, "Early trace incomplete")
        require(counts == dict(TCR=1, P=1, O=1), f"TCR/P/O execution missing: {counts}")
        require(torch.count_nonzero(records["tcr.residual"]) > 0, "Nonzero O did not produce a visible residual")
        return records, counts
    finally:
        for handle in handles:
            handle.remove()
        restore_rng(outer_rng)


def compare_early(unfused, fused, image, folder, state):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    a, counts_a = early_trace(unfused, image, state)
    b, counts_b = early_trace(fused, image, state)
    require(list(a) == list(b), "Fusion changed early trace order")
    result = dict(status="PASSED", order=list(a), first_nonidentical=None, first_tolerance_failure=None,
                  tolerances=dict(atol=ATOL, rtol=RTOL), comparisons={}, calls=dict(unfused=counts_a, fused=counts_b),
                  scope="execution order before decoder topk; failures here cannot be candidate permutations")
    for key in a:
        if not torch.equal(a[key], b[key]) and result["first_nonidentical"] is None:
            result["first_nonidentical"] = key
        try:
            row = tensor_compare(a[key], b[key], key, "early_nodes", ATOL, RTOL, image.device.type, "fp32")
        except DiagnosticError as error:
            row = error.detail
            result["status"] = error.detail["status"]
            if result["first_tolerance_failure"] is None:
                result["first_tolerance_failure"] = key
                path = folder / "first_mismatch.pt"
                torch.save(dict(key=key, unfused=a[key], fused=b[key]), path)
                result["first_failure_tensors"] = str(path)
        result["comparisons"][key] = row
    write_json(folder / "early_nodes.json", result)
    return result


def _one_comparison(unfused, fused, image, folder, state):
    folder = Path(folder)
    before_a, before_b = model_identity(unfused), model_identity(fused)
    result = dict(settings=precision_state(), unfused=before_a, fused=before_b,
                  input_sha256=tensor_hash(image), rng=rng_hash(state),
                  invariants=pair_invariants(unfused, fused))
    try:
        result["early"] = compare_early(unfused, fused, image, folder / "early", state)
        restore_rng(state)
        # The mother protocol still runs on an early mismatch, retaining its
        # original lif_input failure, full records and reproducible fixture.
        try:
            full = fusion_protocol(unfused, fused, image, folder / "full", image.device.type, "fp32")
        except DiagnosticError:
            full = read_json(folder / "full/fuse_diagnostic.json")
            require(full is not None, "Missing original fusion failure evidence")
        result["full"] = dict(status=full["status"], accepted=fusion_accepted(full, image.device.type, "fp32"),
                              failure=full.get("failure"), evidence=str(folder / "full/fuse_diagnostic.json"))
        require(result["early"]["status"] != "FAILED_INCOMPLETE_DIAGNOSTIC" and full["status"] != "FAILED_INCOMPLETE_DIAGNOSTIC",
                "Incomplete diagnostic is an error, not a runtime precision comparison result")
        result["accepted"] = result["early"]["status"] == "PASSED" and result["full"]["accepted"]
        result["status"] = full["status"] if result["early"]["status"] == "PASSED" else result["early"]["status"]
        return result
    finally:
        result["models_unchanged"] = before_a == model_identity(unfused) and before_b == model_identity(fused)
        write_json(folder / "comparison.json", result)
        require(result["models_unchanged"], "Diagnostic forward mutated model/BN state")


def diagnose(unfused, image, folder, state=None, historical_fused=None):
    """Strict gate + independent runtime comparisons on frozen nonzero-O state.

The same strict-fused weights are replayed at runtime precision to isolate the
forward environment. A separately runtime-fused copy covers precision during BN
folding too. No failed runtime record is converted into a passing record.
"""
    from ultralytics.nn.autobackend import AutoBackend
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    entry_rng, entry_precision = rng_state(), precision_state()
    state = state or entry_rng
    source = model_identity(unfused)
    image_sha = tensor_hash(image)
    report = dict(version=VERSION, status="FAILED", source=source, input_sha256=image_sha,
                  input_shape=list(image.shape), input_dtype=str(image.dtype), rng=rng_hash(state),
                  entry_precision=entry_precision, strict_fp32={}, runtime={},
                  acceptance_scope="strict FP32 only; runtime results are independent, never inferred from strict PASS")
    try:
        require(source["all_eval"] and source["floating_dtypes"] == ["torch.float32"], "Diagnostic source must be entirely eval/float")
        require(image.dtype == torch.float32 and source["O_nonzero"] > 0, "FP32 input and nonzero O required")
        with precision_scope(strict=True) as settings:
            # Precision control includes torch.mm used inside native BN folding.
            copy_for_fuse = deepcopy(unfused)
            copied = model_identity(copy_for_fuse)
            require(copied == source, "Fused copy did not start with identical weights/BN")
            fused = copy_for_fuse.fuse(verbose=False).eval().float()
            report["strict_fp32"]["construction"] = dict(settings=settings, before_fuse=copied,
                source_unchanged=source == model_identity(unfused), after_fuse=model_identity(fused))
            require(report["strict_fp32"]["construction"]["source_unchanged"], "Fusion modified source model")
            report["strict_fp32"]["fuse"] = _one_comparison(unfused, fused, image, folder / "strict_fp32/fuse", state)
            copy_for_backend = deepcopy(unfused)
            require(model_identity(copy_for_backend) == source, "AutoBackend copy differs before fusion")
            backend = AutoBackend(copy_for_backend, device=image.device, fp16=False, fuse=True, verbose=False)
            backend.model.eval().float()
            report["strict_fp32"]["backend"] = _one_comparison(unfused, backend.model, image, folder / "strict_fp32/backend", state)
            with torch.no_grad():
                restore_rng(state)
                direct = backend.model(image)[0]
                restore_rng(state)
                actual = backend(image)[0]
                torch.testing.assert_close(direct, actual, atol=0, rtol=0)
            report["strict_fp32"]["backend_wrapper_exact"] = True
            del backend, copy_for_backend
        require(precision_state() == entry_precision, "Strict context leaked precision settings")
        write_json(folder / "precision_diagnostic.json", report)

        # Both runs below use the same input/RNG/unfused model as strict FP32.
        report["runtime"]["same_strict_weights"] = _one_comparison(unfused, fused, image, folder / "runtime/same_strict_weights", state)
        runtime_copy = deepcopy(unfused)
        require(model_identity(runtime_copy) == source, "Runtime copy differs before fusion")
        runtime_fused = runtime_copy.fuse(verbose=False).eval().float()
        report["runtime"]["construction"] = dict(settings=precision_state(), before_fuse=source,
            after_fuse=model_identity(runtime_fused),
            matches_strict_fused_weights=state_hash(runtime_fused.state_dict()) == state_hash(fused.state_dict()))
        report["runtime"]["native_fuse"] = _one_comparison(unfused, runtime_fused, image, folder / "runtime/native_fuse", state)
        if historical_fused is not None:
            historical = deepcopy(fused)
            historical.load_state_dict(historical_fused, strict=True)
            with precision_scope(strict=True):
                report["historical_weights_strict"] = _one_comparison(unfused, historical, image, folder / "historical_weights_strict", state)
            report["historical_weights_runtime"] = _one_comparison(unfused, historical, image, folder / "historical_weights_runtime", state)
        report["strict_accepted"] = all(report["strict_fp32"][k]["accepted"] for k in ("fuse", "backend"))
        report["runtime_accepted"] = all(report["runtime"][k]["accepted"] for k in ("same_strict_weights", "native_fuse"))
        report["status"] = ("PASSED_BOTH_PRECISIONS" if report["runtime_accepted"] else "STRICT_FP32_ONLY_RUNTIME_FAILED") if report["strict_accepted"] else "FAILED_STRICT_FP32"
        report["same_weight_precision_control"] = dict(
            unfused_exact=report["strict_fp32"]["fuse"]["unfused"] == report["runtime"]["same_strict_weights"]["unfused"],
            fused_exact=report["strict_fp32"]["fuse"]["fused"] == report["runtime"]["same_strict_weights"]["fused"],
            input_exact=report["strict_fp32"]["fuse"]["input_sha256"] == report["runtime"]["same_strict_weights"]["input_sha256"],
            rng_exact=report["strict_fp32"]["fuse"]["rng"] == report["runtime"]["same_strict_weights"]["rng"])
        require(all(report["same_weight_precision_control"].values()), "Precision control changed input/weights/RNG")
        strict_settings = report["strict_fp32"]["fuse"]["settings"]
        runtime_settings = report["runtime"]["same_strict_weights"]["settings"]
        report["same_weight_precision_control"]["setting_differences"] = {
            key: dict(strict=strict_settings[key], runtime=runtime_settings[key])
            for key in strict_settings if strict_settings[key] != runtime_settings[key]}
    except BaseException as error:
        report.update(status="FAILED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        restore_rng(entry_rng)
        report["precision_restored"] = precision_state() == entry_precision
        report["source_unchanged"] = source == model_identity(unfused)
        report["input_unchanged"] = image_sha == tensor_hash(image)
        write_json(folder / "precision_diagnostic.json", report)
        require(report["precision_restored"] and report["source_unchanged"] and report["input_unchanged"],
                "Diagnostic altered precision/input/source state")
    return report


def replay(fixture, folder, device):
    """Read the old failure fixture; write a new directory without training."""
    from tcr_v1_core import RTDETRDetectionModel, torch_load, file_identity
    fixture = Path(fixture)
    payload = torch_load(fixture, map_location="cpu")
    old_report = read_json(fixture.parent / "fuse_diagnostic.json", {})
    recorded = old_report.get("runtime")
    if recorded is not None:
        require(all(type(recorded.get(k)) is bool for k in ("tf32", "cudnn_tf32")), "Invalid recorded TF32 settings")
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        model = RTDETRDetectionModel(payload["yaml"], nc=payload.get("nc", 1), verbose=False)
        model.load_state_dict(payload["unfused"], strict=True)
        model.to(device).eval().float()
        image = payload["image"].to(device).float()
        with precision_scope(recorded=recorded):
            result = diagnose(model, image, folder, state=payload["rng"], historical_fused=payload["fused"])
    result["replay"] = dict(fixture=file_identity(fixture), recorded_runtime=recorded,
        old_failure=old_report.get("failure"),
        replayed_settings="TF32 flags and saved RNG/input/weights; other effective settings are explicitly reported, not inferred",
        limitation="Legacy reports recorded TF32, threads and deterministic only; actual ambient autocast/cuDNN benchmark were not captured. Current effective settings are recorded separately.")
    write_json(Path(folder) / "precision_diagnostic.json", result)
    return result
