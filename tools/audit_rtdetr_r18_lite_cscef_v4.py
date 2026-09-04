"""Audit CSCEF-v4 structure, numerics, complexity, and controlled checkpoint initialization."""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
MODEL_DIR = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
MODULE_DIR = ULTRALYTICS_ROOT / "ultralytics" / "nn" / "modules"
BASE_CFG = MODEL_DIR / "rtdetr-resnet18-lite.yaml"
V1_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef.yaml"
V2_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v2.yaml"
V3_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v3.yaml"
DEFAULT_V4_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v4.yaml"
DEFAULT_SOURCE = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_V3_INITIALIZED = ROOT / "weights" / "rtdetr_r18_lite_cscef_v3_imagenet_backbone_init.pt"
DEFAULT_INITIALIZED = ROOT / "weights" / "rtdetr_r18_lite_cscef_v4_imagenet_backbone_init.pt"
PROTECTED_SHA256 = {
    BASE_CFG: "3493d693cbef958a38a2a135822a6c5379b65804905bc71566e395069e8f4f16",
    V1_CFG: "27545ed78ed8308f8c0e5f18abfecdcfc5a0ef69cdf1c605467d564e432bf895",
    V2_CFG: "8275efea9212131799508c57da8c411baf89625cac0be05f9a5c42c3410b906e",
    V3_CFG: "cf679dc1cde990419ca549299be0f991b002cc31d64fdfb9b44e50a3ea10cdbc",
    MODULE_DIR / "cscef.py": "6826911701ce14b08bb99945fa7790897d9389838a78cd3227c5514c0d945cda",
    MODULE_DIR / "cscef_v2.py": "050aa1c9e2a2fc5ffd83bd217cf859fb9607c9f00b7ba405af4e78faf625b9aa",
    MODULE_DIR / "cscef_v3.py": "40f5fae030d21b9eeee67385ebdd058f2c1d1e283396c76d2915136ee8f62143",
    ULTRALYTICS_ROOT / "tests" / "test_cscef_v2.py": (
        "834ab9fa70415a358ae0036c764f5ee50d6abaa6f9a35d12556a5dce95e2ae12"
    ),
    ULTRALYTICS_ROOT / "tests" / "test_cscef_v3.py": (
        "995659103686f97f2603c7a1c08443a339fd6885bb97539cd246a52cd5c1c211"
    ),
    ROOT / "tools" / "audit_rtdetr_r18_lite_cscef_v2.py": (
        "fe1e86d2312c50375c293784d11852db4faa21ab1e2c6538e76facf8b8006310"
    ),
    ROOT / "tools" / "audit_rtdetr_r18_lite_cscef_v3.py": (
        "254500dffdde0b93c7a4bde92e2fac7f1f428ebc20958266d9572f3872d316d3"
    ),
}
BACKBONE_PREFIXES = tuple(f"model.{index}." for index in range(8))

# Prefer this repository's Ultralytics package over any pip-installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics import RTDETR  # noqa: E402
from ultralytics.nn.modules import CSCEFv2, CSCEFv3, CSCEFv4  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402

from init_rtdetr_r18_lite_cscef_v4_controlled import (  # noqa: E402
    build_strict_mapping,
    checkpoint_model,
    clean_initialization_state,
    exact_value_mismatches,
    layer_state_keys,
    nc_shape_audit,
    sha256,
    verify_mapped_values,
    verify_v3_module_source,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_V4_CFG, help="CSCEF-v4 model YAML.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Baseline initialization checkpoint.")
    parser.add_argument(
        "--v3-initialized",
        type=Path,
        default=DEFAULT_V3_INITIALIZED,
        help="Training-free CSCEF-v3 initialization checkpoint.",
    )
    parser.add_argument(
        "--initialized", type=Path, default=DEFAULT_INITIALIZED, help="Generated CSCEF-v4 initialization checkpoint."
    )
    parser.add_argument(
        "--skip-checkpoint", action="store_true", help="Run structure/numerics/complexity checks without weight files."
    )
    return parser.parse_args()


def canonical_lf_sha256(data: bytes) -> str:
    """Hash text bytes after normalizing CRLF and CR line endings to canonical LF."""
    canonical = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def file_sha256(path: Path) -> str:
    """Return a digest, canonicalizing line endings for source and YAML files."""
    data = path.read_bytes()
    if path.suffix.lower() in {".py", ".yaml", ".yml"}:
        return canonical_lf_sha256(data)
    return hashlib.sha256(data).hexdigest()


def verify_protected_files() -> None:
    """Fail if any protected baseline, V1, V2, or V3 artifact changed."""
    mismatches = []
    for path, expected in PROTECTED_SHA256.items():
        actual = file_sha256(path)
        print(f"protected_sha256[{path.relative_to(ROOT)}]={actual}")
        if actual != expected:
            mismatches.append(f"{path}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError(f"Protected files changed: {mismatches}")
    print("protected_baseline_v1_v2_v3_files_unchanged=True")


def verify_yaml_topology(v4_cfg: Path) -> None:
    """Verify that parsed V4 topology differs from V3 only at layer 18."""
    v3 = YAML.load(V3_CFG)
    v4 = YAML.load(v4_cfg)
    v3_layers = v3["backbone"] + v3["head"]
    v4_layers = v4["backbone"] + v4["head"]
    expected = [[*layer] for layer in v3_layers]
    expected[18] = [[17, 16], 1, "CSCEFv4", []]
    if v3["backbone"] != v4["backbone"] or v4_layers != expected:
        raise RuntimeError("CSCEF-v4 parsed YAML differs from CSCEF-v3 beyond the layer-18 module name.")
    if v4_layers[19][0] != [16, 18] or v4_layers[-1][0] != [20, 23, 26]:
        raise RuntimeError("CSCEF-v4 CCFM or decoder inputs changed unexpectedly.")
    print("yaml_topology_matches_v3_except_layer18_module=True")
    print("cscef_v4_inputs=[17, 16]")
    print("concat_inputs=[16, 18]")
    print("decoder_inputs=[20, 23, 26]")


def local_state_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    """Return state shapes without a global layer prefix."""
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def verify_model_structure(
    baseline: RTDETRDetectionModel,
    v2: RTDETRDetectionModel,
    v3: RTDETRDetectionModel,
    v4: RTDETRDetectionModel,
) -> CSCEFv4:
    """Check unchanged layers, topology, parameterization, normalization, and initialization."""
    errors = []
    for baseline_index in range(27):
        inserted_index = baseline_index if baseline_index <= 17 else baseline_index + 1
        baseline_layer = baseline.model[baseline_index]
        v4_layer = v4.model[inserted_index]
        if type(baseline_layer) is not type(v4_layer):
            errors.append(
                f"type mismatch {baseline_index}->{inserted_index}: "
                f"{type(baseline_layer).__name__} != {type(v4_layer).__name__}"
            )
        if local_state_shapes(baseline_layer) != local_state_shapes(v4_layer):
            errors.append(f"state-shape mismatch for unchanged layer {baseline_index}->{inserted_index}")
    if errors:
        raise RuntimeError(f"Baseline and CSCEF-v4 unchanged layer structures differ: {errors}")

    instances = [module for module in v4.modules() if isinstance(module, CSCEFv4)]
    if len(instances) != 1 or v4.model[18] is not instances[0]:
        raise RuntimeError(f"Expected one CSCEFv4 instance at layer 18, found {len(instances)}.")
    module = instances[0]
    if module.f != [17, 16] or v4.model[19].f != [16, 18] or v4.model[-1].f != [20, 23, 26]:
        raise RuntimeError("Constructed CSCEF-v4 topology is incorrect.")
    if sum(isinstance(item, CSCEFv2) for item in v2.modules()) != 1:
        raise RuntimeError("CSCEF-v2 comparison model does not contain exactly one CSCEFv2.")
    if sum(isinstance(item, CSCEFv3) for item in v3.modules()) != 1:
        raise RuntimeError("CSCEF-v3 comparison model does not contain exactly one CSCEFv3.")

    v3_module = v3.model[18]
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    parameter_names = set(dict(module.named_parameters()))
    expected_names = {"shared_projection.weight", "depthwise_conv.weight", "output_projection.weight", "raw_alpha"}
    v3_shapes = {name: tuple(value.shape) for name, value in v3_module.named_parameters()}
    v4_shapes = {name: tuple(value.shape) for name, value in module.named_parameters()}
    if trainable != 16673 or parameter_names != expected_names or v3_shapes != v4_shapes:
        raise RuntimeError(
            f"Unexpected CSCEF-v4 parameters: count={trainable}, names={sorted(parameter_names)}, "
            f"v3_shapes_equal={v3_shapes == v4_shapes}"
        )
    if module.shared_norm.affine or module.edge_norm.affine:
        raise RuntimeError("CSCEF-v4 GroupNorm layers must use affine=False.")
    convolutions = (module.shared_projection, module.depthwise_conv, module.output_projection)
    if any(conv.bias is not None for conv in convolutions):
        raise RuntimeError("Every CSCEF-v4 convolution must use bias=False.")
    if not torch.equal(module.scharr_x, v3_module.scharr_x) or not torch.equal(
        module.scharr_y, v3_module.scharr_y
    ):
        raise RuntimeError("CSCEF-v4 Scharr buffers differ from CSCEF-v3.")
    raw_initial = module.raw_alpha.item()
    alpha_initial = module._effective_alpha().item()
    if abs(raw_initial - (-1.38629436112)) > 1e-6 or abs(alpha_initial - 0.01) > 1e-7:
        raise RuntimeError(f"Unexpected alpha initialization: raw={raw_initial}, effective={alpha_initial}")

    print("cscef_v4_instance_count=1")
    print("layer18_inputs=[17, 16]")
    print(f"cscef_v4_trainable_parameters={trainable}")
    print(f"cscef_v4_parameter_names={sorted(parameter_names)}")
    print(f"cscef_v4_parameter_shapes={v4_shapes}")
    print("parameter_names_and_shapes_equal_v3=True")
    print("groupnorm_affine=False")
    print("all_cscef_v4_conv_bias=False")
    print("scharr_buffers_equal_v3=True")
    print(f"raw_alpha_initial={raw_initial:.12f}")
    print(f"effective_alpha_initial={alpha_initial:.12f}")
    print("unchanged_layer_types_and_state_shapes_identical=True")
    return module


def channel_direction(values: list[float], height: int = 3, width: int = 4) -> torch.Tensor:
    """Create a non-degenerate channel direction repeated spatially."""
    return torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1).expand(1, -1, height, width).clone()


def audit_module_numerics(module: CSCEFv4) -> None:
    """Exercise semantic/structure gates, CPU forward, RMS ordering, alpha bounds, and backward gradients."""
    direction = channel_direction([1.0, -1.0, 0.0, 0.0, 0.5, -0.5, 0.25, -0.25])
    orthogonal = channel_direction([0.0, 0.0, 1.0, -1.0, -0.25, 0.25, 0.5, -0.5])
    bandpass = {
        "identical": module._compute_semantic_bandpass(direction, direction).mean().item(),
        "orthogonal": module._compute_semantic_bandpass(direction, orthogonal).mean().item(),
        "opposite": module._compute_semantic_bandpass(direction, -direction).mean().item(),
    }
    bandpass_boundaries_valid = (
        abs(bandpass["identical"]) <= 1e-6
        and abs(bandpass["orthogonal"] - 1.0) <= 1e-6
        and abs(bandpass["opposite"]) <= 1e-6
    )
    if not bandpass_boundaries_valid:
        raise RuntimeError(f"CSCEF-v4 semantic bandpass boundaries failed: {bandpass}")

    zero = torch.zeros(1, 8, 5, 7)
    zero_confidence = module._compute_structure_confidence(zero, zero)
    directional = torch.ones_like(zero)
    coherence_x, _ = module._compute_structure_terms(directional, zero)
    coherence_y, _ = module._compute_structure_terms(zero, directional)
    isotropic_x = torch.zeros_like(zero)
    isotropic_y = torch.zeros_like(zero)
    isotropic_x[:, :4] = 1.0
    isotropic_y[:, 4:] = 1.0
    coherence_isotropic, _ = module._compute_structure_terms(isotropic_x, isotropic_y)
    diagonal_confidence = module._compute_structure_confidence(directional, directional)
    ramp = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7).expand(1, 8, 5, 7)
    horizontal_confidence = module._compute_structure_confidence(ramp, torch.zeros_like(ramp))
    vertical_transposed_confidence = module._compute_structure_confidence(
        torch.zeros_like(ramp.transpose(-2, -1)), ramp.transpose(-2, -1)
    )
    transpose_consistent = torch.allclose(
        horizontal_confidence.transpose(-2, -1), vertical_transposed_confidence
    )
    if zero_confidence.abs().max().item() != 0.0:
        raise RuntimeError("Zero-gradient structure confidence is not zero.")
    if min(coherence_x.min().item(), coherence_y.min().item()) < 1.0 - 3e-6:
        raise RuntimeError("Single-direction structure coherence is not approximately one.")
    if coherence_isotropic.abs().max().item() > 1e-7:
        raise RuntimeError("Constructed isotropic structure coherence is not approximately zero.")
    if not torch.isfinite(diagonal_confidence).all() or diagonal_confidence.requires_grad:
        raise RuntimeError("Diagonal structure confidence is non-finite or not detached.")
    if not transpose_consistent:
        raise RuntimeError("Horizontal/vertical transposed structure confidences are inconsistent.")

    mixed_direction = channel_direction([1.0, -1.0, 0.0, 0.0, 0.5, -0.5, 0.25, -0.25], 4, 4)
    mixed_orthogonal = channel_direction([0.0, 0.0, 1.0, -1.0, -0.25, 0.25, 0.5, -0.5], 4, 4)
    mixed_semantic = mixed_direction.clone()
    mixed_semantic[:, :, :, -1] = mixed_orthogonal[:, :, :, -1]
    unstandardized_mean = module._compute_semantic_bandpass(mixed_direction, mixed_semantic).mean().item()
    if abs(unstandardized_mean - 0.25) > 1e-6 or abs(unstandardized_mean - 0.5) < 1e-3:
        raise RuntimeError(f"Semantic bandpass appears forcibly standardized: mean={unstandardized_mean}")

    torch.manual_seed(12)
    lateral = torch.randn(2, 256, 12, 14, requires_grad=True)
    semantic = torch.randn(2, 256, 6, 7, requires_grad=True)
    output = module([lateral, semantic])
    if output.shape != lateral.shape or output.dtype != lateral.dtype or not torch.isfinite(output).all():
        raise RuntimeError("CSCEF-v4 CPU FP32 output shape, dtype, or finiteness audit failed.")
    delta = output - lateral
    lateral_rms = torch.sqrt(lateral.float().square().mean(dim=(1, 2, 3)))
    residual_ratios = torch.sqrt(delta.float().square().mean(dim=(1, 2, 3))) / lateral_rms
    if residual_ratios.max().item() > module.alpha_max + 1e-6:
        raise RuntimeError(f"CSCEF-v4 residual RMS bound failed: {residual_ratios.tolist()}")
    (output * torch.randn_like(output)).mean().backward()
    for name, parameter in module.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum().item() == 0:
            raise RuntimeError(f"Missing, non-finite, or zero CSCEF-v4 gradient: {name}")

    small_results = {}
    for shape in ((1, 1), (1, 5), (5, 1)):
        gx, gy = module._compute_scharr_components(torch.ones(2, 32, *shape))
        confidence = module._compute_structure_confidence(gx, gy)
        if confidence.shape != (2, 1, *shape) or not torch.isfinite(confidence).all():
            raise RuntimeError(f"CSCEF-v4 small-spatial audit failed for {shape}.")
        small_results[str(shape)] = tuple(confidence.shape)

    with torch.no_grad():
        saved_raw_alpha = module.raw_alpha.detach().clone()
        module.raw_alpha.fill_(-100.0)
        alpha_low = module._effective_alpha().item()
        module.raw_alpha.fill_(100.0)
        alpha_high = module._effective_alpha().item()
        module.raw_alpha.copy_(saved_raw_alpha)
    if not 0.0 < alpha_low < module.alpha_max or not 0.0 < alpha_high < module.alpha_max:
        raise RuntimeError(f"Effective alpha escaped its open interval: low={alpha_low}, high={alpha_high}")

    print(f"semantic_bandpass_identical={bandpass['identical']:.9f}")
    print(f"semantic_bandpass_orthogonal={bandpass['orthogonal']:.9f}")
    print(f"semantic_bandpass_opposite={bandpass['opposite']:.9f}")
    print(f"structure_zero_confidence_max={zero_confidence.max().item():.9f}")
    print(f"structure_directional_coherence_min={min(coherence_x.min().item(), coherence_y.min().item()):.9f}")
    print(f"structure_isotropic_coherence_max={coherence_isotropic.max().item():.9f}")
    print(f"structure_diagonal_finite={torch.isfinite(diagonal_confidence).all().item()}")
    print(f"structure_horizontal_vertical_transpose_consistent={transpose_consistent}")
    print("structure_confidence_detached=True")
    print(f"semantic_bandpass_unstandardized_example_mean={unstandardized_mean:.9f}")
    print(f"small_spatial_confidence_shapes={small_results}")
    print(f"cpu_fp32_output_shape={tuple(output.shape)}")
    print(f"cpu_fp32_output_dtype={output.dtype}")
    print("cpu_fp32_output_finite=True")
    print(f"residual_rms_ratio_per_sample={residual_ratios.detach().tolist()}")
    print(f"residual_rms_ratio_max={residual_ratios.max().item():.9f}")
    print(f"effective_alpha_extreme_raw_bounds=({alpha_low:.12g}, {alpha_high:.12g})")
    print("all_cscef_v4_parameter_gradients_finite_nonzero=True")


def audit_cuda(v4_cfg: Path) -> None:
    """Run CUDA FP32, AMP FP16, explicit-half module, and complete 640x640 half-model checks."""
    if not torch.cuda.is_available():
        print("cuda_fp32=SKIPPED (CUDA unavailable)")
        print("cuda_amp_fp16=SKIPPED (CUDA unavailable)")
        print("cuda_explicit_half_module=SKIPPED (CUDA unavailable)")
        print("cuda_complete_640_explicit_half=SKIPPED (CUDA unavailable)")
        return
    for use_amp in (False, True):
        module = CSCEFv4(256, 256).cuda().eval()
        dtype = torch.float16 if use_amp else torch.float32
        lateral = torch.randn(2, 256, 12, 14, device="cuda", dtype=dtype)
        semantic = torch.randn(2, 256, 6, 7, device="cuda", dtype=dtype)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = module([lateral, semantic])
        if output.dtype != dtype or output.shape != lateral.shape or not torch.isfinite(output).all():
            raise RuntimeError(f"CSCEF-v4 CUDA audit failed with amp={use_amp}.")
        print(f"cuda_{'amp_fp16' if use_amp else 'fp32'}=PASSED shape={tuple(output.shape)} dtype={output.dtype}")
        del output, semantic, lateral, module
        gc.collect()
        torch.cuda.empty_cache()

    module = CSCEFv4(256, 256).cuda().half().eval()
    lateral = torch.randn(2, 256, 12, 14, device="cuda", dtype=torch.float16)
    semantic = torch.randn(2, 256, 6, 7, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        output = module([lateral, semantic])
    if output.dtype != torch.float16 or output.shape != lateral.shape or not torch.isfinite(output).all():
        raise RuntimeError("CSCEF-v4 explicit-half module audit failed.")
    print(f"cuda_explicit_half_module=PASSED shape={tuple(output.shape)} dtype={output.dtype}")
    del output, semantic, lateral, module
    gc.collect()
    torch.cuda.empty_cache()

    model = RTDETRDetectionModel(str(v4_cfg), ch=3, nc=1, verbose=False).eval().cuda().half()
    image = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        output = model(image)
    if output[0].shape != (1, 300, 5) or not torch.isfinite(output[0]).all():
        raise RuntimeError("CSCEF-v4 complete 640x640 explicit-half model audit failed.")
    print(f"cuda_complete_640_explicit_half=PASSED shape={tuple(output[0].shape)} dtype={output[0].dtype}")
    del output, image, model
    gc.collect()
    torch.cuda.empty_cache()


def complexity_table(v4_cfg: Path) -> tuple[dict[str, RTDETRDetectionModel], dict[str, tuple]]:
    """Build baseline/V2/V3/V4 nc=1 models and report model.info parameter and GFLOPs comparisons."""
    models = {
        "R18-Lite baseline": RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=1, verbose=False).eval(),
        "R18-Lite + CSCEF-v2": RTDETRDetectionModel(str(V2_CFG), ch=3, nc=1, verbose=False).eval(),
        "R18-Lite + CSCEF-v3": RTDETRDetectionModel(str(V3_CFG), ch=3, nc=1, verbose=False).eval(),
        "R18-Lite + CSCEF-v4": RTDETRDetectionModel(str(v4_cfg), ch=3, nc=1, verbose=False).eval(),
    }
    rows = {}
    for name, model in models.items():
        info = model.info(imgsz=640)
        if info is None or not info[3]:
            raise RuntimeError(f"model.info(imgsz=640) did not produce FLOPs for {name}.")
        rows[name] = info
    _, base_params, _, base_flops = rows["R18-Lite baseline"]
    _, _, _, v3_flops = rows["R18-Lite + CSCEF-v3"]
    _, v4_params, _, v4_flops = rows["R18-Lite + CSCEF-v4"]
    if v4_params - base_params != 16673:
        raise RuntimeError(f"CSCEF-v4 total parameter delta is {v4_params - base_params}, expected 16673.")
    if v4_flops != v3_flops:
        raise RuntimeError(
            "model.info should report equal traced GFLOPs for parameter-identical V3/V4 functional paths."
        )

    print("complexity_table:")
    print("model | layers | parameters | parameter_delta | parameter_delta_ratio | model.info_GFLOPs | GFLOPs_delta")
    for name, (layers, parameters, gradients, flops) in rows.items():
        del gradients
        print(
            f"{name} | {layers} | {parameters} | {parameters - base_params} | "
            f"{(parameters - base_params) / base_params:.6%} | {flops:.6f} | {flops - base_flops:.6f}"
        )
    print(f"cscef_v4_parameter_delta={v4_params - base_params}")
    print(f"cscef_v4_model_info_gflops={v4_flops:.6f}")
    print(
        "gflops_scope_note=model.info/THOP does not count functional Scharr, pointwise structure-tensor math, "
        "or avg_pool2d; reported GFLOPs are not complete theoretical FLOPs"
    )
    return models, rows


def audit_complete_models(v4_cfg: Path, nc1_model: RTDETRDetectionModel) -> None:
    """Run nc=1 and nc=80 complete-model CPU FP32 forwards and verify decoder output shapes."""
    for nc, model in ((1, nc1_model), (80, RTDETRDetectionModel(str(v4_cfg), ch=3, nc=80, verbose=False).eval())):
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 128, 128))
        expected_shape = (1, 300, nc + 4)
        if not isinstance(output, tuple) or output[0].shape != expected_shape or not torch.isfinite(output[0]).all():
            raise RuntimeError(f"CSCEF-v4 complete-model nc={nc} CPU forward failed.")
        print(f"nc={nc}_cpu_output_shape={tuple(output[0].shape)} finite=True")
        del output


def audit_checkpoint_mapping(
    source_path: Path,
    v3_initialized_path: Path,
    initialized_path: Path,
    baseline_model: RTDETRDetectionModel,
    v4_model: RTDETRDetectionModel,
    v4_nc1_model: RTDETRDetectionModel,
    v4_cfg: Path,
) -> None:
    """Verify baseline coverage, V3 module equality, clean state, reload paths, and nc shape changes."""
    for path, label in (
        (source_path, "baseline initialization checkpoint"),
        (v3_initialized_path, "V3 initialization checkpoint"),
        (initialized_path, "V4 initialization checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    source_checkpoint = torch_load(source_path, map_location="cpu")
    source_state = checkpoint_model(source_checkpoint).state_dict()
    v3_checkpoint = torch_load(v3_initialized_path, map_location="cpu")
    v3_reference_state = checkpoint_model(v3_checkpoint).state_dict()
    v3_module_keys = verify_v3_module_source(v3_checkpoint, v3_reference_state)
    target_state = v4_model.state_dict()
    plan = build_strict_mapping(source_state, baseline_model.state_dict(), target_state)

    initialized_checkpoint = torch_load(initialized_path, map_location="cpu")
    initialized_model = checkpoint_model(initialized_checkpoint)
    initialized_state = initialized_model.state_dict()
    missing = sorted(set(target_state) - set(initialized_state))
    unexpected = sorted(set(initialized_state) - set(target_state))
    shape_mismatches = sorted(
        key
        for key in target_state
        if key in initialized_state and target_state[key].shape != initialized_state[key].shape
    )
    mapped_mismatches = verify_mapped_values(source_state, initialized_state, plan.source_to_target)
    v3_module_mismatches = exact_value_mismatches(v3_reference_state, initialized_state, v3_module_keys)
    target_parameters = dict(v4_model.named_parameters())
    unchanged_parameter_numel = sum(
        parameter.numel() for key, parameter in target_parameters.items() if not key.startswith("model.18.")
    )
    mapped_parameter_numel = sum(target_parameters[key].numel() for key in plan.state if key in target_parameters)
    backbone_keys = [key for key in target_parameters if key.startswith(BACKBONE_PREFIXES)]
    mapped_backbone_keys = [key for key in backbone_keys if key in plan.state]
    backbone_numel = sum(target_parameters[key].numel() for key in backbone_keys)
    mapped_backbone_numel = sum(target_parameters[key].numel() for key in mapped_backbone_keys)
    clean_state = clean_initialization_state(initialized_checkpoint)

    if not isinstance(initialized_model.model[18], CSCEFv4):
        raise RuntimeError("Generated initialization checkpoint does not contain CSCEFv4 at layer 18.")
    direct = RTDETR(str(initialized_path))
    if not isinstance(direct.model.model[18], CSCEFv4):
        raise RuntimeError("Direct RTDETR(checkpoint) load did not preserve CSCEFv4.")
    yaml_loaded = RTDETR(str(v4_cfg))
    yaml_loaded.load(str(initialized_path))
    yaml_mismatches = exact_value_mismatches(
        initialized_state, yaml_loaded.model.state_dict(), sorted(initialized_state)
    )

    print(f"source_checkpoint_sha256={sha256(source_path)}")
    print(f"v3_initialized_checkpoint_sha256={sha256(v3_initialized_path)}")
    print(f"v4_initialized_checkpoint_sha256={sha256(initialized_path)}")
    print(f"mapped_unchanged_state_keys={len(plan.state)}/{len(baseline_model.state_dict())}")
    print(f"mapped_unchanged_parameter_numel={mapped_parameter_numel}/{unchanged_parameter_numel}")
    print(f"mapping_shape_mismatches={plan.shape_mismatches}")
    print(f"unchanged_baseline_values_exact={not mapped_mismatches}")
    print(f"backbone_parameter_key_coverage={len(mapped_backbone_keys)}/{len(backbone_keys)}")
    print(f"backbone_parameter_numel_coverage={mapped_backbone_numel}/{backbone_numel}")
    print(f"initialized_missing_keys={missing}")
    print(f"initialized_unexpected_keys={unexpected}")
    print(f"initialized_shape_mismatches={shape_mismatches}")
    print(f"v4_module_equals_v3_init_exact={not v3_module_mismatches}")
    print(f"v4_module_v3_init_mismatches={v3_module_mismatches}")
    print(f"clean_initialization_checkpoint_state={clean_state}")
    print(f"checkpoint_direct_load=True")
    print(f"checkpoint_yaml_load_exact={not yaml_mismatches}")
    nc_shape_audit(initialized_state, v4_nc1_model.state_dict())

    errors = []
    if len(plan.state) != len(baseline_model.state_dict()) or plan.shape_mismatches or mapped_mismatches:
        errors.append("baseline mapping is incomplete or inexact")
    if mapped_parameter_numel != unchanged_parameter_numel:
        errors.append("unchanged parameter numel coverage is incomplete")
    if len(mapped_backbone_keys) != len(backbone_keys) or mapped_backbone_numel != backbone_numel:
        errors.append("backbone mapping coverage is incomplete")
    if missing or unexpected or shape_mismatches:
        errors.append("initialized V4 state structure differs from the YAML model")
    if v3_module_mismatches:
        errors.append("V4 layer 18 does not exactly equal the V3 training-free initialization")
    if not clean_state:
        errors.append("generated checkpoint retained training state")
    if yaml_mismatches:
        errors.append(f"YAML load values differ: {yaml_mismatches}")
    if errors:
        raise RuntimeError(f"CSCEF-v4 checkpoint audit failed: {errors}")
    print("strict_controlled_checkpoint_audit_passed=True")


def main() -> None:
    """Run all requested CSCEF-v4 audits and fail nonzero on any discrepancy."""
    args = parse_args()
    v4_cfg = args.model.resolve()
    source_path = args.source.resolve()
    v3_initialized_path = args.v3_initialized.resolve()
    initialized_path = args.initialized.resolve()
    imported_path = Path(ultralytics.__file__).resolve()
    print(f"ultralytics_version={ultralytics.__version__}")
    print(f"ultralytics_file={imported_path}")
    if not imported_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")

    verify_protected_files()
    verify_yaml_topology(v4_cfg)
    baseline_nc80 = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False).eval()
    v2_nc80 = RTDETRDetectionModel(str(V2_CFG), ch=3, nc=80, verbose=False).eval()
    v3_nc80 = RTDETRDetectionModel(str(V3_CFG), ch=3, nc=80, verbose=False).eval()
    v4_nc80 = RTDETRDetectionModel(str(v4_cfg), ch=3, nc=80, verbose=False).eval()
    module = verify_model_structure(baseline_nc80, v2_nc80, v3_nc80, v4_nc80)
    audit_module_numerics(module)
    audit_cuda(v4_cfg)
    nc1_models, _ = complexity_table(v4_cfg)
    audit_complete_models(v4_cfg, nc1_models["R18-Lite + CSCEF-v4"])
    if args.skip_checkpoint:
        print("checkpoint_mapping_audit=SKIPPED (--skip-checkpoint)")
    else:
        audit_checkpoint_mapping(
            source_path,
            v3_initialized_path,
            initialized_path,
            baseline_nc80,
            v4_nc80,
            nc1_models["R18-Lite + CSCEF-v4"],
            v4_cfg,
        )
    print("CSCEF-v4 audit passed.")


if __name__ == "__main__":
    main()
