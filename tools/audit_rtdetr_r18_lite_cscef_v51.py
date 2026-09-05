"""Audit CSCEF-v5.1 using synthetic inputs and the real RTDETRTrainer.get_model entry point; no datasets."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import hashlib
import inspect
from pathlib import Path
from types import MethodType

import torch

from init_rtdetr_r18_lite_cscef_v51_controlled import (
    BASE_CFG, CLASSIFICATION_KEYS, DEFAULT_OUTPUT, DEFAULT_SOURCE, NEW_SUFFIXES, ROOT, SOURCE_SHA256,
    V51_CFG, clean_checkpoint, read_source, remap_baseline_key, require,
    runtime_info, sha256, tensor_rows, verify_module, verify_reloads, write_json,
)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import CSCEFv5, CSCEFv51
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import init_seeds


def audit_autocast_context(device: str | torch.device, use_amp: bool):
    """Create a fresh context; non-AMP audits must not construct FP16 autocast on PyTorch 2.1."""
    if not use_amp:
        return nullcontext()
    if torch.device(device).type != "cuda":
        raise ValueError("FP16 AMP audit requires CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def verify_topology() -> dict:
    base = YAML.load(BASE_CFG)
    v5 = YAML.load(V51_CFG.with_name("rtdetr-resnet18-lite-cscef-v5.yaml"))
    v51 = YAML.load(V51_CFG)
    expected = deepcopy(v5)
    expected["head"][10][2] = "CSCEFv51"
    require(v51 == expected, "V51 YAML changed beyond the single layer-18 class replacement.")
    require(v51["backbone"] == base["backbone"], "Backbone changed.")
    layers = v51["backbone"] + v51["head"]
    require(layers[18] == [[17, 16], 1, "CSCEFv51", []], "V51 insertion differs from the specified graph.")
    require(layers[19][0] == [16, 18] and layers[27][0] == [20, 23, 26], "Concat/decoder order changed.")
    return {"insertion": layers[18], "concat": layers[19], "decoder": layers[27], "v5_only_class_changed": True}


def audit_structure() -> dict:
    """Check the complete V5 nonlinear confidence BEFORE per-image H/W averaging."""
    old, new = CSCEFv5(256, 256), CSCEFv51(256, 256)
    cases = 0
    for shape in ((1, 32, 1, 1), (1, 32, 1, 7), (1, 32, 9, 1), (2, 32, 9, 13)):
        for x in (torch.zeros(shape), torch.ones(shape), torch.randn(shape, requires_grad=True)):
            a, b = old._compute_scharr_components(x), new._compute_scharr_components(x)
            require(all(torch.equal(u, v) for u, v in zip(a, b)), "V5 Scharr/padding changed.")
            require(all(not v.requires_grad and v.dtype == torch.float32 for v in b), "Scharr must be detached FP32.")
            raw = old._compute_structure_confidence(*a)
            expected = raw.mean(dim=(-2, -1), keepdim=True).expand_as(raw)
            actual = new._compute_structure_confidence(*b)
            require(torch.equal(actual, expected), "V51 is not the per-image spatial mean of complete V5 confidence.")
            require(actual.dtype == torch.float32 and not actual.requires_grad
                    and actual.grad_fn is None and torch.isfinite(actual).all().item(), "Invalid confidence contract.")
            cases += 1
        gx, gy = torch.randn(shape, requires_grad=True), torch.randn(shape, requires_grad=True)
        for first, second in ((gx, gy), (gx, torch.zeros_like(gx)), (torch.zeros_like(gx), gy)):
            raw = old._compute_structure_confidence(first, second)
            actual = new._compute_structure_confidence(first, second)
            require(torch.equal(actual, raw.mean((-2, -1), keepdim=True).expand_as(raw)), "Nonlinear order changed.")
            require(not actual.requires_grad and actual.dtype == torch.float32, "Fixed-gradient confidence must detach.")
    gx, gy = torch.randn(1, 32, 9, 13), torch.randn(1, 32, 9, 13)
    single = new._compute_structure_confidence(gx, gy)
    for others in (torch.zeros_like(gx), 100 * torch.randn(3, 32, 9, 13)):
        batch = new._compute_structure_confidence(torch.cat((others, gx)), torch.cat((others, gy)))
        require(torch.equal(single, batch[-1:]), "Another image changed this image's structure confidence.")
    return {"v5_then_spatial_mean_exact": True, "scharr_and_boundaries_exact": True,
            "max_abs": 0.0, "detached_fp32": True, "sample_independent": True, "boundary_cases": cases}


@torch.no_grad()
def diagnostic_mean_structure(self, gx, gy):
    """Independent V5 inference intervention used as a reference, with no residual scaling."""
    raw = CSCEFv5._compute_structure_confidence(self, gx, gy)
    return raw.mean(dim=(-2, -1), keepdim=True).expand_as(raw)


def audit_intervention(mode: str = "cpu_fp32") -> dict:
    """Compare full NONZERO-residual forwards to V5 + mean_structure, including edge shapes."""
    device = "cpu" if mode == "cpu_fp32" else "cuda"
    amp, half = mode == "cuda_amp_fp16", mode == "cuda_half"
    reference, actual = CSCEFv5(256, 256), CSCEFv51(256, 256)
    with torch.no_grad():
        reference.output_projection.weight.normal_(0, 0.01)
    actual.load_state_dict(reference.state_dict(), strict=True)
    reference._compute_structure_confidence = MethodType(diagnostic_mean_structure, reference)
    reference, actual = reference.to(device), actual.to(device)
    if half:
        reference.half()
        actual.half()
    dtype = torch.float16 if half or amp else torch.float32
    cases, nonzero = 0, 0
    for height, width in ((1, 1), (1, 9), (7, 1), (9, 13)):
        for factory in (torch.zeros, torch.ones, torch.randn):
            lateral = factory(2, 256, height, width, device=device, dtype=dtype)
            semantic = factory(2, 256, max(1, height // 2), max(1, width // 2), device=device,
                               dtype=torch.float32 if amp else dtype)
            with torch.no_grad(), audit_autocast_context(device, amp):
                expected, output = reference([lateral, semantic]), actual([lateral, semantic])
            require(torch.equal(expected, output), f"{mode}: full mean_structure forward differs.")
            require(output.shape == lateral.shape and output.dtype == lateral.dtype
                    and torch.isfinite(output).all().item(), f"{mode}: invalid boundary forward.")
            nonzero += int(not torch.equal(output, lateral))
            cases += 1
    require(nonzero > 0, "Intervention check did not exercise an active residual.")
    return {"mode": mode, "full_forward_exact": True, "max_abs": 0.0,
            "nonzero_output_projection": True, "nonidentity_cases": nonzero, "boundary_cases": cases}


def audit_module(mode: str = "cpu_fp32") -> dict:
    """Check identity, two-stage gradient opening and semantic dependence in one numerical pass."""
    device = "cpu" if mode == "cpu_fp32" else "cuda"
    use_amp, explicit_half = mode == "cuda_amp_fp16", mode == "cuda_half"
    module_rng = torch.get_rng_state().clone()
    module = CSCEFv51(256, 256)
    require(torch.equal(module_rng, torch.get_rng_state()), "Module construction advanced external CPU RNG.")
    module = module.to(device)
    if explicit_half:
        module.half()
    dtype = torch.float16 if explicit_half or use_amp else torch.float32
    lateral = torch.randn(2, 256, 12, 14, device=device, dtype=dtype, requires_grad=True)
    semantic = torch.randn(2, 256, 6, 7, device=device, dtype=dtype, requires_grad=True)
    probe = torch.randn(lateral.shape, device=device)

    def forward():
        with audit_autocast_context(device, use_amp):
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
    with torch.no_grad(), audit_autocast_context(device, use_amp):
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
    require(initialized.is_file(), f"Missing V51 initialization: {initialized}")
    checkpoint = torch_load(initialized, map_location="cpu")
    require(clean_checkpoint(checkpoint), "V51 checkpoint contains training state.")
    require(checkpoint.get("cscef_v51_provenance", {}).get("source_sha256") == SOURCE_SHA256, "Missing verified provenance.")
    require(checkpoint["cscef_v51_provenance"]["runtime"]["code_sha256"] == runtime_info()["code_sha256"],
            "Initialization was generated with different source; regenerate it.")
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
        require(sum(row["baseline"] in CLASSIFICATION_KEYS for row in rows) == 5, "Missing classification checks.")
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
              "module_parameters": sum(p.numel() for p in target.model[18].parameters())}
    for name, model in (("C2", baseline), ("V51", target)):
        unfused = model.info(imgsz=640, verbose=True)
        require(unfused is not None and unfused[3] > 0, f"Missing THOP complexity for {name}.")
        fused = deepcopy(model).fuse(verbose=False)
        fused_info = fused.info(imgsz=640, verbose=True)
        require(fused_info is not None and fused_info[3] > 0, f"Missing fused THOP complexity for {name}.")
        report[name] = {"unfused_parameters": unfused[1], "unfused_gflops": unfused[3],
                        "fused_parameters": fused_info[1], "fused_gflops": fused_info[3]}
    require(report["C2"]["unfused_parameters"] == 20082772, "Unexpected C2 parameter count.")
    require(report["V51"]["unfused_parameters"] == 20109684, "Unexpected V51 parameter count.")
    return report


def audit_optimizer(model: torch.nn.Module) -> dict:
    """Build actual C2 AdamW groups through the unchanged Trainer method; no optimizer step."""
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    optimizer = trainer.build_optimizer(model, name="AdamW", lr=0.0005, momentum=0.937, decay=0.0001)
    names = {id(p): name for name, p in model.named_parameters()}
    rows = [{"group": group["param_group"], "count": len(group["params"]),
             "weight_decay": group["weight_decay"], "lr": group["lr"], "betas": list(group["betas"]),
             "parameters": [names[id(p)] for p in group["params"]]} for group in optimizer.param_groups]
    require({r["group"]: (r["count"], r["weight_decay"]) for r in rows}
            == {"weight": (122, 0.0001), "bn": (81, 0.0), "bias": (128, 0.0)}, "Optimizer groups changed.")
    all_names = [name for row in rows for name in row["parameters"]]
    require(len(all_names) == len(set(all_names)) == len(names), "Optimizer parameter coverage is not exact.")
    return {"optimizer": type(optimizer).__name__, "groups": rows,
            "meaning": {"weight": "weight decay weights", "bn": "norm weights without decay", "bias": "bias without decay"},
            "status": "passed"}


def audit_cuda(model: torch.nn.Module) -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA unavailable", "pending": ["cuda_fp32", "cuda_amp_fp16", "cuda_half"]}
    report = {"status": "passed", "device": torch.cuda.get_device_name(), "modes": {}}
    for mode in ("cuda_fp32", "cuda_amp_fp16", "cuda_half"):
        module_report = audit_module(mode)
        intervention = audit_intervention(mode)
        actual = deepcopy(model).eval().cuda()
        if mode == "cuda_half":
            actual.half()
        image = torch.rand(1, 3, 640, 640, device="cuda", dtype=torch.float16 if mode == "cuda_half" else torch.float32)
        with torch.inference_mode(), audit_autocast_context("cuda", mode == "cuda_amp_fp16"):
            output = actual(image)[0]
        require(output.shape == (1, 300, 5) and torch.isfinite(output).all().item(), f"{mode}: complete 640 forward failed.")
        report["modes"][mode] = {"module": module_report, "intervention": intervention, "full_shape": list(output.shape),
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
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/cscef_v51/audit.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {"runtime": runtime_info(), "status": "running", "datasets_used": False,
              "rng_scope": "CLI runs in its own process; do not import and run this audit inside training"}
    try:
        init_seeds(42, deterministic=True)
        report["topology"] = verify_topology()
        report["structure"] = audit_structure()
        report["module_cpu"] = audit_module()
        report["intervention_cpu"] = audit_intervention()
        report["initialization"], base, target = audit_initialization(args.source, args.initialized)
        report["complexity"] = complexity(base, target)
        report["optimizer"] = audit_optimizer(target)
        report["cuda"] = audit_cuda(target)
        require(sha256(args.source) == SOURCE_SHA256, "Original C2 source changed.")
        require(sha256(args.initialized) == report["initialization"]["initialized_sha256"], "V51 checkpoint changed.")
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
