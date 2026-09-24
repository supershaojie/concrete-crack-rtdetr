"""Strip only the auxiliary head; deployment always derives from selected trained EMA/model."""
from __future__ import annotations
from copy import deepcopy
import torch
from pds_v1_common import (CONFIG, require, build, verify_model, state_audit, sha256,
                           write_json, git, native_view)
from ultralytics.models.rtdetr.pds import PDSDetectionModel
from ultralytics.utils.patches import torch_load


def compare_outputs(a, b, path="output"):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype,
                f"Deployment output shape/dtype mismatch: {path}")
        require(torch.equal(a, b), f"Deployment raw outputs differ: {path}, max={float((a-b).abs().max())}")
    elif isinstance(a, (tuple, list)):
        require(type(a) is type(b) and len(a) == len(b), "Output structure mismatch")
        for i, (x, y) in enumerate(zip(a, b)):
            compare_outputs(x, y, f"{path}.{i}")
    elif isinstance(a, dict):
        require(a.keys() == b.keys(), "Output dict keys differ")
        for k in a:
            compare_outputs(a[k], b[k], f"{path}.{k}")
    else:
        require(a == b, f"Output metadata differs at {path}")


def strip_model(selected):
    require(type(selected) is PDSDetectionModel and selected.pds_config == CONFIG,
            "Selected checkpoint lacks the complete PDS model")
    source = deepcopy(selected).cpu().float().eval()
    native = build(nc=1).cpu().float().eval()
    verify_model(native)
    public = {k: v for k, v in source.state_dict().items() if not k.startswith("pds_head.")}
    require(set(public) == set(native.state_dict()), "Deployment public keys mismatch")
    native.load_state_dict(public, strict=True)
    # EMA freezes parameters. Preserve that execution state as well as values:
    # CPU attention/linear kernels can choose another path when the flags differ.
    source_parameters = dict(source.named_parameters())
    for name, parameter in native.named_parameters():
        parameter.requires_grad_(source_parameters[name].requires_grad)
    for name in ("names", "nc", "yaml", "args", "task", "stride"):
        if hasattr(source, name):
            setattr(native, name, deepcopy(getattr(source, name)))
    caches = {}
    for attr in ("shapes", "anchors", "valid_mask"):
        value = deepcopy(getattr(source.model[-1], attr))
        setattr(native.model[-1], attr, value)
        caches[attr] = dict(shape=list(value.shape), dtype=str(value.dtype)) if isinstance(value, torch.Tensor) else value
    audit = state_audit(native, source)
    calls = source.pds_head.calls
    # Isolated CPU FP32, finite input. No TF32/autocast global changes are required.
    x = torch.linspace(0, 1, 3 * 640 * 640).reshape(1, 3, 640, 640)
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        compare_outputs(source(x), native(x))
    require(source.pds_head.calls == calls, "Deployment comparison executed PDS")
    require(all(torch.equal(v, source.state_dict()[k]) for k, v in native.state_dict().items()),
            "Deploy conversion changed public tensors")
    import thop
    with torch.no_grad():
        macs = thop.profile(deepcopy(native), inputs=(torch.zeros_like(x),), verbose=False)[0]
    return native, dict(public_audit=audit, pds_calls=0, raw_output_exact=True,
                        parameter_requires_grad_preserved=True, gflops_640=2 * macs / 1e9,
                        flops_method="THOP multiply-add=2, standard unfused model, finite B1/640 input; custom-op coverage limited",
                        precision="same-source CPU FP32 unfused", source_caches=caches,
                        parameters_unfused=sum(p.numel() for p in native.parameters()))


def deploy(checkpoint, output, report):
    require(not output.exists(), "Existing deploy preserved")
    ckpt = torch_load(checkpoint, map_location="cpu")
    require(ckpt.get("epoch", -1) >= 0, "Deploy must derive from a trained checkpoint")
    key = "ema" if ckpt.get("ema") is not None else "model"
    model, audit = strip_model(ckpt[key])
    provenance = dict(source=str(checkpoint), source_sha256=sha256(checkpoint), selected=key,
                      source_epoch=ckpt["epoch"], commit=git("rev-parse", "HEAD"), **audit)
    model.args = ckpt["train_args"].copy()
    model.task = "detect"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(dict(model=model, ema=None, optimizer=None, epoch=-1, train_args=model.args,
                        pds_deployment=provenance), stream)
    provenance["deploy_sha256"] = sha256(output)
    write_json(report, provenance)
    return provenance
