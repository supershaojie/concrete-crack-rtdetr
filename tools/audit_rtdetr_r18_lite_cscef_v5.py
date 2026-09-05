"""Audit CSCEF-v5 using synthetic inputs and the real RTDETRTrainer.get_model entry point; no datasets."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import inspect
from pathlib import Path

import torch

from init_rtdetr_r18_lite_cscef_v5_controlled import (
    BASE_CFG, CLASSIFICATION_KEYS, DEFAULT_OUTPUT, DEFAULT_SOURCE, NEW_SUFFIXES, ROOT, SOURCE_SHA256,
    V5_CFG, clean_checkpoint, read_source, remap_baseline_key, require,
    runtime_info, sha256, tensor_rows, verify_module, verify_reloads, write_json,
)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import CSCEFv4, CSCEFv5
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import init_seeds


def verify_topology() -> dict:
    base = YAML.load(BASE_CFG)
    v4 = YAML.load(V5_CFG.with_name("rtdetr-resnet18-lite-cscef-v4.yaml"))
    v5 = YAML.load(V5_CFG)
    expected = deepcopy(v4)
    expected["head"][10][2] = "CSCEFv5"
    require(v5 == expected, "V5 YAML changed beyond the single layer-18 class replacement.")
    require(v5["backbone"] == base["backbone"], "Backbone changed.")
    layers = v5["backbone"] + v5["head"]
    require(layers[18] == [[17, 16], 1, "CSCEFv5", []], "V5 insertion differs from the specified graph.")
    require(layers[19][0] == [16, 18] and layers[27][0] == [20, 23, 26], "Concat/decoder order changed.")
    return {"insertion": layers[18], "concat": layers[19], "decoder": layers[27], "v4_only_class_changed": True}


def audit_structure() -> dict:
    """Compare V4 and V5 using the same gradients, including degenerate boundaries."""
    old, new = CSCEFv4(256, 256), CSCEFv5(256, 256)
    max_error = 0.0
    for shape in ((1, 32, 1, 1), (1, 32, 1, 7), (1, 32, 9, 1), (2, 32, 9, 13)):
        for x in (torch.zeros(shape), torch.ones(shape), torch.randn(shape, requires_grad=True)):
            a = old._compute_scharr_components(x)
            b = new._compute_scharr_components(x)
            for expected, actual in zip(a, b):
                require(torch.equal(expected, actual), "V4/V5 Scharr or padding differs.")
                require(not actual.requires_grad and actual.dtype == torch.float32, "Scharr must be detached FP32.")
            confidence = new._compute_structure_confidence(*b)
            expected_confidence = old._compute_structure_confidence(*a)
            require(torch.equal(confidence, expected_confidence), "V4/V5 structure confidence differs.")
            require(not confidence.requires_grad and torch.isfinite(confidence).all().item(), "Invalid confidence.")
            max_error = max(max_error, float((confidence - expected_confidence).abs().max()))
        gx, gy = torch.randn(shape, requires_grad=True), torch.randn(shape, requires_grad=True)
        for first, second in ((gx, gy), (gx, torch.zeros_like(gx)), (torch.zeros_like(gx), gy)):
            require(torch.equal(old._compute_structure_confidence(first, second),
                                new._compute_structure_confidence(first, second)), "Fixed-gradient formula differs.")
    return {"v4_formula_exact": True, "scharr_and_boundaries_exact": True, "max_abs": max_error, "detached": True}


def audit_module(mode: str = "cpu_fp32") -> dict:
    """Check identity, two-stage gradient opening and semantic dependence in one numerical pass."""
    device = "cpu" if mode == "cpu_fp32" else "cuda"
    use_amp, explicit_half = mode == "cuda_amp_fp16", mode == "cuda_half"
    module_rng = torch.get_rng_state().clone()
    module = CSCEFv5(256, 256)
    require(torch.equal(module_rng, torch.get_rng_state()), "Module construction advanced external CPU RNG.")
    module = module.to(device)
    if explicit_half:
        module.half()
    dtype = torch.float16 if explicit_half or use_amp else torch.float32
    lateral = torch.randn(2, 256, 12, 14, device=device, dtype=dtype, requires_grad=True)
    semantic = torch.randn(2, 256, 6, 7, device=device, dtype=dtype, requires_grad=True)
    probe = torch.randn(lateral.shape, device=device)

    def forward():
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            return module([lateral, semantic])

    forward_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    output = forward()
    require(torch.equal(forward_rng, torch.get_rng_state()), "Forward consumed CPU RNG.")
    if cuda_rng is not None:
        require(torch.equal(cuda_rng, torch.cuda.get_rng_state()), "Forward consumed CUDA RNG.")
    require(torch.equal(output, lateral), f"{mode}: initialization is not exact identity.")
    require(output.dtype == lateral.dtype and output.shape == lateral.shape, "Output contract failed.")
    ((output.float() * probe).sum() / lateral.shape[0]).backward()
    first_gradients = {}
    for name, parameter in module.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(), f"Invalid gradient: {name}")
        magnitude = float(parameter.grad.float().abs().sum())
        first_gradients[name] = magnitude
        require((magnitude > 0) == (name == "output_projection.weight"), f"Unexpected first-step gradient: {name}")
    with torch.no_grad():
        module.output_projection.weight.add_(module.output_projection.weight.grad, alpha=-0.001)
    module.zero_grad(set_to_none=True)
    lateral.grad = semantic.grad = None
    output = forward()
    require(torch.isfinite(output).all().item(), f"{mode}: non-finite output after projection update.")
    ((output.float() * probe).sum() / lateral.shape[0]).backward()
    second_gradients = {}
    for name, parameter in module.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(), f"Invalid second gradient: {name}")
        magnitude = float(parameter.grad.float().abs().sum())
        require(magnitude > 0, f"{mode}: upstream gradient did not open: {name}")
        second_gradients[name] = magnitude
    require(semantic.grad is not None and torch.isfinite(semantic.grad).all().item()
            and torch.count_nonzero(semantic.grad).item() > 0, "Semantic input receives no usable gradient.")
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        altered = module([lateral, -semantic])
    semantic_difference = float((output.float() - altered.float()).abs().max())
    require(semantic_difference > 0, "Activated residual does not depend on semantic content.")
    return {"mode": mode, "identity_exact": True, "construction_rng_preserved": True, "forward_rng_preserved": True,
            "first_backward_abs_sum": first_gradients, "after_projection_update_abs_sum": second_gradients,
            "semantic_change_max_abs": semantic_difference, "output_dtype": str(output.dtype), "status": "passed"}


def trainer_build(cfg: dict, weights: torch.nn.Module, nc: int) -> tuple[torch.nn.Module, torch.Tensor]:
    """Call the unchanged real method; only skip __init__'s dataset/device/directory setup.

    The complete attributes read by get_model are data[nc] and data[channels]. Seeding is the actual
    BaseTrainer single-process formula seed + 1 + RANK = 42 + 1 - 1, using the real init_seeds function.
    No load_state_dict or manual head copying is performed after the real method returns.
    """
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = {"nc": nc, "channels": 3}
    init_seeds(42, deterministic=True)
    model = trainer.get_model(cfg=deepcopy(cfg), weights=weights, verbose=False)
    return model.eval(), torch.get_rng_state().clone()


def compare_output(a, b, prefix: str = "output") -> list[dict]:
    """Compare all tensors in the actual prediction tree, not just the final boxes."""
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.shape == b.shape, f"{prefix}: output shape differs.")
        require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), f"{prefix}: non-finite output.")
        error = float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
        require(torch.equal(a, b), f"{prefix}: exact initial equivalence failed, max_abs={error}; no tolerance widening.")
        return [{"tensor": prefix, "shape": list(a.shape), "max_abs": error, "equal": True}]
    if isinstance(a, (tuple, list)):
        require(type(a) is type(b) and len(a) == len(b), f"{prefix}: output container differs.")
        return [row for i, (x, y) in enumerate(zip(a, b)) for row in compare_output(x, y, f"{prefix}.{i}")]
    require(a == b, f"{prefix}: non-tensor output differs.")
    return []


def audit_initialization(source: Path, initialized: Path) -> tuple[dict, torch.nn.Module, torch.nn.Module]:
    _, baseline_weights = read_source(source)
    require(initialized.is_file(), f"Missing V5 initialization: {initialized}")
    checkpoint = torch_load(initialized, map_location="cpu")
    require(clean_checkpoint(checkpoint), "V5 checkpoint contains training state.")
    require(checkpoint.get("cscef_v5_provenance", {}).get("source_sha256") == SOURCE_SHA256, "Missing verified provenance.")
    initialized_weights = RTDETR(str(initialized)).model
    verify_module(initialized_weights)
    mapping = {k: remap_baseline_key(k) for k in baseline_weights.state_dict()}
    nc80_rows = tensor_rows(baseline_weights.state_dict(), initialized_weights.state_dict(), mapping)
    require(len(nc80_rows) == 533 and all(row["equal"] for row in nc80_rows), "Inexact nc=80 checkpoint mapping.")
    report = {"source_sha256": sha256(source), "initialized_sha256": sha256(initialized),
              "nc80_checkpoint_tensors": nc80_rows, "reloads": verify_reloads(initialized, initialized_weights.state_dict()),
              "entry_point": "RTDETRTrainer.get_model (unchanged implementation, __new__ fixture avoids dataset I/O)",
              "entry_point_source_sha256": hashlib.sha256(inspect.getsource(RTDETRTrainer.get_model).encode()).hexdigest(),
              "seed": 42, "deterministic": True, "ranks": "single process RANK=-1", "classes": {}}
    baseline_nc1 = target_nc1 = None
    for nc in (1, 80):
        baseline, baseline_rng = trainer_build(baseline_weights.yaml, baseline_weights, nc)
        target, target_rng = trainer_build(initialized_weights.yaml, initialized_weights, nc)
        rows = tensor_rows(baseline.state_dict(), target.state_dict(), mapping)
        require(len(rows) == 533 and all(row["equal"] for row in rows), f"nc={nc} common tensors differ: {[r for r in rows if not r['equal']]}")
        require(set(target.state_dict()) - set(mapping.values()) == {f"model.18.{s}" for s in NEW_SUFFIXES},
                "Unexpected extra target states.")
        require(torch.equal(baseline_rng, target_rng), f"nc={nc}: CPU RNG differs after real trainer construction/load.")
        verify_module(target)
        image = torch.rand(1, 3, 640, 640, generator=torch.Generator().manual_seed(123))
        with torch.inference_mode():
            a, b = baseline(image), target(image)
        require(a[0].shape == (1, 300, nc + 4), f"nc={nc} prediction shape is unexpected.")
        outputs = compare_output(a, b)
        report["classes"][str(nc)] = {
            "compared": len(rows), "differences": [], "all_common_tensors": rows,
            "five_classification_weights": [r for r in rows if r["baseline"] in CLASSIFICATION_KEYS],
            "cpu_rng_equal_after_build_load": True,
            "cpu_rng_sha256": hashlib.sha256(baseline_rng.numpy().tobytes()).hexdigest(),
            "cpu_fp32_640_outputs": outputs,
        }
        print(f"nc={nc}: 533/533 common tensors exact, five classification weights exact, RNG exact, all 640 outputs exact")
        if nc == 1:
            baseline_nc1, target_nc1 = baseline, target
    return report, baseline_nc1, target_nc1


def complexity(baseline: torch.nn.Module, target: torch.nn.Module) -> dict:
    report = {"scope": "model.info/THOP; functional Scharr convolutions, avg_pool2d and pointwise math are NOT fully counted",
              "module_parameters": 26912}
    for name, model in (("C2", baseline), ("V5", target)):
        unfused = model.info(imgsz=640, verbose=True)
        require(unfused is not None and unfused[3] > 0, f"Missing THOP complexity for {name}.")
        fused = deepcopy(model).fuse(verbose=False)
        fused_info = fused.info(imgsz=640, verbose=True)
        require(fused_info is not None and fused_info[3] > 0, f"Missing fused THOP complexity for {name}.")
        report[name] = {"unfused_parameters": unfused[1], "unfused_gflops": unfused[3],
                        "fused_parameters": fused_info[1], "fused_gflops": fused_info[3]}
    require(report["C2"]["unfused_parameters"] == 20082772, "Unexpected C2 parameter count.")
    require(report["V5"]["unfused_parameters"] == 20109684, "Unexpected V5 parameter count.")
    return report


def audit_cuda(model: torch.nn.Module) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA unavailable", "pending": ["cuda_fp32", "cuda_amp_fp16", "cuda_half"]}
    report = {"status": "passed", "device": torch.cuda.get_device_name(), "modes": {}}
    for mode in ("cuda_fp32", "cuda_amp_fp16", "cuda_half"):
        module_report = audit_module(mode)
        actual = deepcopy(model).eval().cuda()
        if mode == "cuda_half":
            actual.half()
        image = torch.rand(1, 3, 640, 640, device="cuda", dtype=torch.float16 if mode == "cuda_half" else torch.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=mode == "cuda_amp_fp16"):
            output = actual(image)[0]
        require(output.shape == (1, 300, 5) and torch.isfinite(output).all().item(), f"{mode}: complete 640 forward failed.")
        report["modes"][mode] = {"module": module_report, "full_shape": list(output.shape),
                                  "full_dtype": str(output.dtype), "full_finite": True}
        print(f"{mode}: module forward/backward passed; full model 640 -> {tuple(output.shape)} finite")
        del actual, image, output
        gc.collect()
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--initialized", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/cscef_v5/audit.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {"runtime": runtime_info(), "status": "running", "datasets_used": False,
              "rng_scope": "CLI runs in its own process; do not import and run this audit inside training"}
    try:
        init_seeds(42, deterministic=True)
        report["topology"] = verify_topology()
        report["structure"] = audit_structure()
        report["module_cpu"] = audit_module()
        report["initialization"], base, target = audit_initialization(args.source, args.initialized)
        report["complexity"] = complexity(base, target)
        report["cuda"] = audit_cuda(target)
        require(sha256(args.source) == SOURCE_SHA256, "Original C2 source changed.")
        require(sha256(args.initialized) == report["initialization"]["initialized_sha256"], "V5 checkpoint changed.")
        report["status"] = "passed" if report["cuda"]["status"] == "passed" else "passed_cpu_cuda_pending"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.report, report)
        print(f"Audit status: {report['status']}; report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
