"""Observation-only DPR diagnostics. No training recipe or admission policy lives here."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from types import FunctionType, MethodType

import torch
from torch import nn

TARGET = "model.5.blocks.1.branch2b"
SCHEMA = "dpr_observation_v1"
STATISTICS = "allclose_right_reference_v2"
CONTINUOUS = ("target", "P3", "P4", "P5", "encoder_features", "candidate_scores")
LOCAL = ("dpr_input", "effective_kernel", "conv_output", "norm_output")


def digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_record(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
                sha256=digest(tensor), finite=bool(torch.isfinite(tensor).all()))


def state_fingerprint(model):
    """Bind checkpoint content to the original learned state, independent of device."""
    records = [(name, str(value.dtype), tuple(value.shape), digest(value))
               for name, value in sorted(model.state_dict().items())]
    return hashlib.sha256(repr(records).encode()).hexdigest()


def metric(a, b, atol=2e-5, rtol=2e-4):
    from check_dpr import comparison
    return comparison(a, b, atol, rtol)


def model_identity(model, image):
    target = model.get_submodule(TARGET)
    tensors = {name: tensor_record(value) for name, value in model.state_dict().items()}
    modes = {name: dict(training=m.training, inplace=getattr(m, "inplace", None))
             for name, m in model.named_modules()}
    buffers = {name: tensor_record(value) for name, value in model.named_buffers()}
    head = model.model[-1]
    caches = {name: tensor_record(getattr(head, name)) for name in ("anchors", "valid_mask")
              if isinstance(getattr(head, name, None), torch.Tensor)}
    return dict(input=tensor_record(image), state=tensors, buffers=buffers, decoder_caches=caches,
                modes=modes, requires_grad={name: p.requires_grad for name, p in model.named_parameters()},
                deployed=getattr(target, "deployed", None), norm_retained=isinstance(target.norm, nn.BatchNorm2d))


@contextmanager
def decoder_selection(decoder, fixed_indices=None):
    """Observe the real selection in this instance's original method only.

    Bind the unchanged code object to a private globals dictionary with a torch
    facade. No process-global torch.topk replacement and no copied decoder math.
    """
    original = decoder._get_decoder_input
    function = original.__func__
    source = inspect.getsource(function)
    if source.count("torch.topk(") != 1:
        raise RuntimeError("Unrecognized decoder query-selection callsite")
    old = decoder.__dict__.get("_get_decoder_input")
    existed = "_get_decoder_input" in decoder.__dict__
    record = dict(schema=SCHEMA, calls=[], method=function.__qualname__,
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  scope="one decoder instance/original _get_decoder_input code object", fixed=fixed_indices is not None)

    def topk(value, k, *args, **kwargs):
        dim = kwargs.get("dim", args[0] if args else -1)
        frame = inspect.currentframe().f_back
        location = dict(function=frame.f_code.co_name, filename=frame.f_code.co_filename, line=frame.f_lineno)
        del frame
        if value.ndim != 2 or dim != 1 or k != decoder.num_queries:
            raise RuntimeError("Unexpected selection shape/dimension/count")
        raw = torch.topk(value, k, *args, **kwargs)
        selected = raw.indices
        if fixed_indices is not None:
            selected = fixed_indices.to(device=value.device, dtype=torch.long)
            if selected.shape != raw.indices.shape or bool((selected < 0).any()) or bool((selected >= value.shape[1]).any()):
                raise RuntimeError("Invalid fixed query indices")
            if any(row.unique().numel() != k for row in selected):
                raise RuntimeError("Fixed query indices are not unique")
        record["calls"].append(dict(input_shape=list(value.shape), k=k, dim=dim, location=location,
                                    actual_indices=selected.detach().cpu().clone(),
                                    free_indices=raw.indices.detach().cpu().clone()))
        return raw if fixed_indices is None else torch.return_types.topk((value.gather(1, selected), selected))

    class TorchFacade:
        def __getattr__(self, name):
            return topk if name == "topk" else getattr(torch, name)

    namespace = dict(function.__globals__, torch=TorchFacade())
    instrumented = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    instrumented.__kwdefaults__ = function.__kwdefaults__
    decoder._get_decoder_input = MethodType(instrumented, decoder)
    try:
        yield record
        if len(record["calls"]) != 1:
            raise RuntimeError(f"Expected one actual decoder selection, observed {len(record['calls'])}")
    finally:
        if existed:
            decoder._get_decoder_input = old
        else:
            del decoder._get_decoder_input


def capture_trace(model, image, invoke=None, fixed_indices=None, detailed=True):
    """Observe model internals through invoke(image), including a real file backend."""
    invoke = model if invoke is None else invoke
    values, handles = {}, []
    target = model.get_submodule(TARGET)
    kernel_patched = False
    kernel_existed = "get_equivalent_kernel" in target.__dict__
    old_kernel = target.__dict__.get("get_equivalent_kernel")
    layers = dict(target=target, P3=model.model[5], P4=model.model[6], P5=model.model[7],
                  encoder_features=model.model[-1].enc_output, candidate_scores=model.model[-1].enc_score_head)

    def save(name, value):
        values[name] = value.detach().cpu().clone()

    def local_input(module, args):
        save("dpr_input", args[0])
        if not kernel_patched:
            save("effective_kernel", module.conv.weight)

    try:
        for name, layer in layers.items():
            handles.append(layer.register_forward_hook(lambda m, a, o, key=name: save(key, o)))
        if detailed:
            if hasattr(target, "get_equivalent_kernel") and not target.deployed:
                native_kernel = target.get_equivalent_kernel
                def observed_kernel(instance):
                    value = native_kernel()
                    save("effective_kernel", value.to(dtype=instance.conv.weight.dtype))
                    return value
                target.get_equivalent_kernel = MethodType(observed_kernel, target)
                kernel_patched = True
            handles.append(target.register_forward_pre_hook(local_input))
            # Norm's real input is the executed conv output, including F.conv2d.
            handles.append(target.norm.register_forward_pre_hook(lambda m, a: save("conv_output", a[0])))
            handles.append(target.norm.register_forward_hook(lambda m, a, o: save("norm_output", o)))
        with decoder_selection(model.model[-1], fixed_indices) as selection, torch.no_grad():
            output = invoke(image)
            save("output", output[0] if isinstance(output, (tuple, list)) else output)
        values["candidate_indices"] = selection["calls"][0]["actual_indices"]
        serial = deepcopy(selection)
        for call in serial["calls"]:
            call["actual_indices"] = call["actual_indices"].tolist()
            call["free_indices"] = call["free_indices"].tolist()
        return values, serial
    finally:
        for handle in handles:
            handle.remove()
        if kernel_patched:
            if kernel_existed:
                target.get_equivalent_kernel = old_kernel
            else:
                del target.get_equivalent_kernel


def selection_difference(before, after):
    rows = []
    for left, right in zip(before.tolist(), after.tolist()):
        a, b = set(left), set(right)
        rows.append(dict(same_set=a == b, removed=sorted(a-b), added=sorted(b-a),
                         changed_positions=sum(x != y for x, y in zip(left, right))))
    return dict(changed_positions=int((before != after).sum()), batches=rows,
                interpretation="index-array positions; not a count of added/lost detections")


def compare_traces(before, after, model, image, invoke=None, atol=2e-5, rtol=2e-4):
    rows = {key: metric(before[key], after[key], atol, rtol) for key in before if key != "candidate_indices"}
    indices = selection_difference(before["candidate_indices"], after["candidate_indices"])
    replay, selection = capture_trace(model, image, invoke, before["candidate_indices"], detailed="dpr_input" in before)
    replay_metric = metric(before["output"], replay["output"], atol, rtol)
    continuous = all(rows[key]["raw_allclose"] for key in (*LOCAL, *CONTINUOUS) if key in rows)
    complete = continuous and indices["changed_positions"] > 0 and replay_metric["raw_allclose"]
    first = next((key for key in (*LOCAL, *CONTINUOUS, "output") if key in rows and not rows[key]["raw_allclose"]), None)
    return dict(schema=SCHEMA, admission_eligible=False, comparisons=rows, selection=indices,
                first_exceeded=first, fixed_candidate_replay_output=replay_metric, replay_selection=selection,
                candidate_sensitivity_complete=complete,
                conclusion="CANDIDATE_SENSITIVITY_COMPLETE" if complete else "RAW_PASS" if first is None else "UNRESOLVED",
                note="Diagnostic replay never overrides continuous failures or the backend raw gate")


def file_reference(checkpoint, device, half):
    """Independently mirror the inspected file-load sequence, not AutoBackend.forward."""
    from ultralytics.utils import DEFAULT_CFG_DICT
    from ultralytics.nn.tasks import guess_model_task
    checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selected = "ema" if ckpt.get("ema") is not None else "model"
    model = ckpt[selected]
    checkpoint_state_sha256 = state_fingerprint(model)
    model = model.float()
    model.args = {**DEFAULT_CFG_DICT, **ckpt.get("train_args", {})}
    model.pt_path = str(checkpoint)
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])
    model.fuse(verbose=False)
    model.eval().to(device)
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = True
        elif isinstance(module, nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    model.half() if half else model.float()
    model.requires_grad_(False)
    return model, dict(checkpoint_sha256=checkpoint_sha256, checkpoint_state_sha256=checkpoint_state_sha256,
                       selected=selected, order="load CPU -> float -> fuse CPU -> eval -> to(device) -> inplace=True -> dtype -> requires_grad=False")


def backend_audit(checkpoint, model, image, legacy_capture, half=False, legacy_model=None):
    """Keep the raw backend gate; separately retain the former mismatched comparison."""
    from ultralytics.nn.autobackend import AutoBackend
    device = next(model.parameters()).device
    reference, provenance = file_reference(checkpoint, device, half)
    source_state_sha256 = state_fingerprint(model)
    source_matches_checkpoint = source_state_sha256 == provenance["checkpoint_state_sha256"]
    backend = AutoBackend(model=str(checkpoint), device=device, fp16=half, fuse=True, verbose=False).eval()
    reference_identity = model_identity(reference, image)
    backend_identity = model_identity(backend.model, image)
    legacy_identity = model_identity(legacy_model, image) if legacy_model is not None else None
    identity_changes = {}
    if legacy_identity is not None:
        for group in ("state", "buffers", "modes", "requires_grad", "decoder_caches"):
            a, b = legacy_identity[group], reference_identity[group]
            identity_changes[group] = [name for name in sorted(set(a)|set(b)) if a.get(name) != b.get(name)]
    identities_equal = reference_identity == backend_identity
    storages = lambda m: {(str(v.device), v.untyped_storage().data_ptr()) for v in m.state_dict().values() if v.numel()}
    storage_independent = not (storages(reference) & storages(backend.model))
    with torch.no_grad():
        raw = backend(image)
        raw = (raw[0] if isinstance(raw, (tuple, list)) else raw).detach().cpu().clone()
    ref_capture, ref_selection = capture_trace(reference, image)
    actual, actual_selection = capture_trace(backend.model, image, invoke=backend)
    observer_equal = torch.equal(raw, actual["output"])
    atol, rtol = (2e-3, 2e-2) if half else (2e-5, 2e-4)
    comparison = metric(ref_capture["output"], actual["output"], atol, rtol)
    aligned = compare_traces(ref_capture, actual, backend.model, image, backend, atol, rtol)
    legacy = {key: value for key, value in actual.items() if key in legacy_capture}
    old_comparison = metric(legacy_capture["output"], actual["output"], atol, rtol)
    old_trace = compare_traces(legacy_capture, legacy, backend.model, image, backend, atol, rtol)
    target = backend.model.get_submodule(TARGET)
    checkpoint_sha256_after = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    checkpoint_unchanged = provenance["checkpoint_sha256"] == checkpoint_sha256_after
    passed = (source_matches_checkpoint and checkpoint_unchanged and identities_equal and storage_independent and observer_equal
              and comparison["raw_allclose"] and aligned["first_exceeded"] is None
              and target.deployed and isinstance(target.norm, nn.BatchNorm2d))
    return dict(status="PASSED" if passed else "FAILED", comparison=comparison, deployed=target.deployed,
                norm_retained=isinstance(target.norm, nn.BatchNorm2d), dtype=str(next(backend.model.parameters()).dtype),
                backend_schema=SCHEMA, identities_equal=identities_equal, observer_output_exact=observer_equal,
                storage_independent=storage_independent,
                source_matches_checkpoint=source_matches_checkpoint, source_state_sha256=source_state_sha256,
                checkpoint_unchanged=checkpoint_unchanged, checkpoint_sha256_after=checkpoint_sha256_after,
                provenance=provenance, reference_identity=reference_identity, backend_identity=backend_identity,
                legacy_identity=legacy_identity, legacy_to_file_reference_identity_changes=identity_changes,
                actual_selection=actual_selection, reference_selection=ref_selection,
                diagnostic=aligned, legacy_comparison=old_comparison, legacy_diagnostic=old_trace,
                gate="aligned actual backend output raw_allclose; no precision-note exception")


def validate_backend_record(entry, atol, rtol):
    """Shared math/full-preflight gate: diagnostic fields cannot replace raw proof."""
    def need(condition, message):
        if not condition:
            raise RuntimeError("Backend evidence rejected: " + message)
    need(entry.get("backend_schema") == SCHEMA and entry.get("status") == "PASSED", "schema/status")
    for key in ("deployed", "norm_retained", "identities_equal", "storage_independent", "observer_output_exact",
                "source_matches_checkpoint", "checkpoint_unchanged"):
        need(entry.get(key) is True, key)
    need(entry.get("reference_identity") and entry["reference_identity"] == entry.get("backend_identity"), "model identities")
    need(len(entry.get("provenance", {}).get("checkpoint_sha256", "")) == 64, "checkpoint identity")
    need(entry["provenance"]["checkpoint_sha256"] == entry.get("checkpoint_sha256_after"), "checkpoint changed during capture")
    need(len(entry.get("source_state_sha256", "")) == 64 and
         entry["source_state_sha256"] == entry["provenance"].get("checkpoint_state_sha256"), "wrong learned checkpoint state")
    for key in ("actual_selection", "reference_selection"):
        record = entry.get(key, {})
        calls = record.get("calls", [])
        need(record.get("schema") == SCHEMA and record.get("fixed") is False and len(calls) == 1, "actual selection")
        need(calls[0].get("k") == 300 and calls[0].get("dim") == 1 and calls[0].get("actual_indices"), "selection geometry")
    rows = entry.get("diagnostic", {}).get("comparisons", {})
    for name in (*LOCAL, *CONTINUOUS, "output", "comparison"):
        value = entry.get("comparison", {}) if name == "comparison" else rows.get(name, {})
        need(value.get("raw_allclose") is True and value.get("finite") is True and
             value.get("atol") == atol and value.get("rtol") == rtol and
             value.get("statistics_version") == STATISTICS, "raw metric " + name)
    return True


def half_decomposition(model, image, parent=None):
    """Factor the original whole-network half checks without relaxing them."""
    from ultralytics.nn.modules.block import ConvNormLayer
    model = deepcopy(model).float().eval()
    device = next(model.parameters()).device
    image = image.to(device=device, dtype=torch.float32)
    u32 = model
    d32 = deepcopy(u32)
    d32.get_submodule(TARGET).switch_to_deploy()
    n32 = deepcopy(d32).fuse(verbose=False)
    u16, d16, n16 = deepcopy(u32).half(), deepcopy(d32).half(), deepcopy(n32).half()
    models = dict(U32=u32, D32=d32, N32=n32, U16=u16, D16=d16, N16=n16)
    captures, selections, identities = {}, {}, {}
    for name, current in models.items():
        inp = image.half() if name.endswith("16") else image
        identities[name] = model_identity(current, inp)
        captures[name], selections[name] = capture_trace(current, inp)
    comparisons = {}
    for before, after in (("U32", "D32"), ("D32", "N32"), ("U32", "U16"), ("D32", "D16"),
                          ("N32", "N16"), ("U16", "D16"), ("D16", "N16")):
        half = after.endswith("16")
        comparisons[before+"_to_"+after] = compare_traces(captures[before], captures[after], models[after],
            image.half() if half else image, atol=2e-3 if half else 2e-5, rtol=2e-2 if half else 2e-4)
    # A real ordinary ConvNormLayer with the learned effective kernel; no DPR forward.
    ordinary = deepcopy(u32)
    target = ordinary.get_submodule(TARGET)
    plain = ConvNormLayer.__new__(ConvNormLayer)
    nn.Module.__init__(plain)
    plain.conv, plain.norm, plain.act = deepcopy(target.conv), deepcopy(target.norm), deepcopy(target.act)
    with torch.no_grad():
        plain.conv.weight.copy_(target.get_equivalent_kernel())
    plain.eval()
    ordinary.model[5].blocks[1].branch2b = plain
    ordinary32, ordinary_selection = capture_trace(ordinary, image)
    ordinary16_model = deepcopy(ordinary).half()
    ordinary16, _ = capture_trace(ordinary16_model, image.half())
    comparisons["U32_to_same_effective_plain32"] = compare_traces(captures["U32"], ordinary32, ordinary, image)
    comparisons["same_effective_plain32_to_half"] = compare_traces(ordinary32, ordinary16, ordinary16_model,
                                                                 image.half(), atol=2e-3, rtol=2e-2)
    local_input = captures["U32"]["dpr_input"].to(device).half()
    with torch.no_grad():
        # Same input/norm state; only parameter-rounding order differs.
        first, second = d16.get_submodule(TARGET), u16.get_submodule(TARGET)
        local = dict(kernel=metric(first.conv.weight, second.get_equivalent_kernel().half(), 2e-3, 2e-2),
                     output=metric(first(local_input), second(local_input), 2e-3, 2e-2),
                     input=tensor_record(local_input),
                     paths=["FP32 parameters -> effective kernel -> half", "parameters -> half -> FP32 effective kernel -> half"],
                     production_unfolded_half="parameters -> half -> FP32 effective kernel -> half")
    background = dict(status="PENDING", reason="Parent learned model not supplied")
    if parent is not None:
        p32 = deepcopy(parent).float().eval().to(device)
        p16 = deepcopy(p32).half()
        pc, _ = capture_trace(p32, image)
        ph, _ = capture_trace(p16, image.half())
        background = compare_traces(pc, ph, p16, image.half(), atol=2e-3, rtol=2e-2)
        background["scope"] = "same public source and batch, independently learned original parent; background only"
    return dict(schema=SCHEMA, admission_eligible=False, comparisons=comparisons, actual_selections=selections,
                path_identities=identities,
                local_rounding_order=local, parent_precision_background=background,
                interpretation="Whole-network precision/conversion failures remain failures regardless of parent similarity")


def transpose_maps(kernel_gradient):
    """Independent adjoints of the four fixed linear maps (no DPR autograd reuse)."""
    diagonal = kernel_gradient.diagonal(dim1=0, dim2=1).permute(2, 0, 1)
    flat = diagonal.flatten(1)
    inverse = (1, 2, 5, 0, 4, 8, 3, 6, 7)
    return dict(dpr_cd=flat-flat[:, 4:5], dpr_hd=diagonal[:, :, 0]-diagonal[:, :, 2],
                dpr_vd=diagonal[:, 0, :]-diagonal[:, 2, :], dpr_ad=flat-flat[:, inverse])


def gradient_mapping_check(kernel, actual, atol=2e-5, rtol=2e-4):
    expected = transpose_maps(kernel)
    rows = {key: metric(actual[key], expected[key], atol, rtol) for key in expected}
    return dict(status="PASSED" if all(row["raw_allclose"] for row in rows.values()) else "FAILED",
                comparisons=rows, stage="pre-clip gradients; scaled values divided once by the same recorded scale")


class GradientObserver:
    """Capture same-backward tensors without changing native unscale/clip/step/EMA."""
    def __init__(self, model):
        self.target = model.get_submodule(TARGET)
        self.handles, self.values = [], {}

    def __enter__(self):
        self.initial = deepcopy(self.target)
        target = self.target
        self.existed = "get_equivalent_kernel" in target.__dict__
        self.old = target.__dict__.get("get_equivalent_kernel")
        native = target.get_equivalent_kernel

        def kernel(instance):
            value = native()
            self.values["kernel"] = value.detach().cpu().clone()
            if value.requires_grad:
                self.handles.append(value.register_hook(lambda grad: self.save("kernel_gradient_scaled", grad)))
            return value

        target.get_equivalent_kernel = MethodType(kernel, target)
        self.handles.append(target.register_forward_pre_hook(lambda m, a: self.save("input", a[0])))

        def output_hook(module, args, output):
            self.save("output", output)
            self.handles.append(output.register_hook(lambda grad: self.save("upstream_scaled", grad)))

        self.handles.append(target.register_forward_hook(output_hook))
        return self

    def save(self, key, value):
        self.values[key] = value.detach().cpu().clone()

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        if self.existed:
            self.target.get_equivalent_kernel = self.old
        else:
            del self.target.get_equivalent_kernel

    def evidence(self, gradients, scale, amp, device):
        from ultralytics.utils.torch_utils import autocast
        names = self.target.parameter_names
        actual = {name: gradients[TARGET+"."+name] for name in names}
        kernel = self.values["kernel_gradient_scaled"].float()/scale
        own = gradient_mapping_check(kernel, actual)
        own["kernel_gradient_equals_W"] = metric(kernel, gradients[TARGET+".conv.weight"])
        local = self.initial.to(device)
        local.zero_grad(set_to_none=True)
        x = self.values["input"].to(device).requires_grad_(True)
        with autocast(amp):
            y = local(x)
        y.backward(self.values["upstream_scaled"].to(device))
        replay = {name: getattr(local, name).grad.detach().cpu().float()/scale for name in names}
        replay_w = local.conv.weight.grad.detach().cpu().float()/scale
        own["local_mapping"] = gradient_mapping_check(replay_w, replay)
        own["frozen_local_replay"] = {name: metric(actual[name], replay[name]) for name in names}
        own["local_output"] = metric(self.values["output"], y)
        own["scale"] = scale
        own["amp"] = amp
        own["kernel_gradient"] = tensor_record(kernel)
        own["capture_stage"] = "same backward, before native unscale and clip; arithmetic division only, no second unscale_"
        return own, kernel, actual
