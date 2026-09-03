"""Audit CSCEF-v3 structure, numerics, complexity, and strict baseline-checkpoint initialization."""

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
BASE_CFG = MODEL_DIR / "rtdetr-resnet18-lite.yaml"
V1_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef.yaml"
V2_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v2.yaml"
DEFAULT_V3_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v3.yaml"
DEFAULT_SOURCE = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_INITIALIZED = ROOT / "weights" / "rtdetr_r18_lite_cscef_v3_imagenet_backbone_init.pt"
PROTECTED_SHA256 = {
    BASE_CFG: "3493d693cbef958a38a2a135822a6c5379b65804905bc71566e395069e8f4f16",
    V1_CFG: "863ed9f2786737bbc0f96879b5f79c65b8a72fba2f65094cd64624ff54f70820",
    V2_CFG: "8275efea9212131799508c57da8c411baf89625cac0be05f9a5c42c3410b906e",
    ULTRALYTICS_ROOT / "ultralytics" / "nn" / "modules" / "cscef.py": (
        "9c6ade426fa468915e8325fb88b40e31facc3bff41472866e96aefdc11ae3cbf"
    ),
    ULTRALYTICS_ROOT / "ultralytics" / "nn" / "modules" / "cscef_v2.py": (
        "050aa1c9e2a2fc5ffd83bd217cf859fb9607c9f00b7ba405af4e78faf625b9aa"
    ),
}
BACKBONE_PREFIXES = tuple(f"model.{index}." for index in range(8))

# Prefer this repository's Ultralytics package over any pip-installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics.nn.modules import CSCEFv2, CSCEFv3  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402

from init_rtdetr_r18_lite_cscef_v3_from_baseline import (  # noqa: E402
    build_strict_mapping,
    checkpoint_model,
    nc_shape_audit,
    sha256,
    verify_mapped_values,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Baseline initialization checkpoint.")
    parser.add_argument("--model", type=Path, default=DEFAULT_V3_CFG, help="CSCEF-v3 model YAML.")
    parser.add_argument(
        "--initialized", type=Path, default=DEFAULT_INITIALIZED, help="Generated CSCEF-v3 initialization checkpoint."
    )
    parser.add_argument(
        "--skip-checkpoint", action="store_true", help="Run structure/numerics/complexity checks without weight files."
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Return a file digest without depending on checkpoint utilities."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protected_files() -> None:
    """Fail if the baseline, CSCEF-v1, or CSCEF-v2 source/configuration changed."""
    mismatches = []
    for path, expected in PROTECTED_SHA256.items():
        actual = file_sha256(path)
        print(f"protected_sha256[{path.relative_to(ROOT)}]={actual}")
        if actual != expected:
            mismatches.append(f"{path}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError(f"Protected baseline/CSCEF-v1/v2 files changed: {mismatches}")
    print("protected_baseline_cscef_v1_cscef_v2_files_unchanged=True")


def verify_yaml_topology(v3_cfg: Path) -> None:
    """Verify that the v3 YAML differs from v2 only by the inserted module name."""
    v2 = YAML.load(V2_CFG)
    v3 = YAML.load(v3_cfg)
    v2_layers = v2["backbone"] + v2["head"]
    v3_layers = v3["backbone"] + v3["head"]
    if v2["backbone"] != v3["backbone"]:
        raise RuntimeError("CSCEF-v3 changed the R18-Lite backbone YAML.")
    if len(v3_layers) != len(v2_layers):
        raise RuntimeError("CSCEF-v3 and CSCEF-v2 must have the same layer count.")
    expected_v3 = [[*layer] for layer in v2_layers]
    expected_v3[18] = [[17, 16], 1, "CSCEFv3", []]
    if v3_layers != expected_v3:
        raise RuntimeError("CSCEF-v3 YAML topology differs from v2 beyond replacing layer 18.")
    if v3_layers[19][0] != [16, 18]:
        raise RuntimeError("CSCEF-v3 must preserve Concat [upsampled P4, enhanced S3].")
    if v3_layers[-1][0] != [20, 23, 26]:
        raise RuntimeError("CSCEF-v3 decoder inputs must remain [20, 23, 26].")
    print("yaml_topology_matches_cscef_v2_except_module=True")
    print("cscef_v3_inputs=[17, 16]")
    print("decoder_inputs=[20, 23, 26]")


def local_state_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    """Return state shapes without a global layer prefix."""
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def verify_model_structure(
    baseline: RTDETRDetectionModel,
    v2: RTDETRDetectionModel,
    v3: RTDETRDetectionModel,
) -> CSCEFv3:
    """Check unchanged layers, topology, parameterization, normalization, and initialization."""
    errors = []
    for baseline_index in range(27):
        inserted_index = baseline_index if baseline_index <= 17 else baseline_index + 1
        baseline_layer = baseline.model[baseline_index]
        v3_layer = v3.model[inserted_index]
        if type(baseline_layer) is not type(v3_layer):
            errors.append(
                f"type mismatch {baseline_index}->{inserted_index}: "
                f"{type(baseline_layer).__name__} != {type(v3_layer).__name__}"
            )
        if local_state_shapes(baseline_layer) != local_state_shapes(v3_layer):
            errors.append(f"state-shape mismatch for unchanged layer {baseline_index}->{inserted_index}")
    if errors:
        raise RuntimeError(f"Baseline and CSCEF-v3 unchanged layer structures differ: {errors}")

    instances = [module for module in v3.modules() if isinstance(module, CSCEFv3)]
    if len(instances) != 1:
        raise RuntimeError(f"Expected one CSCEFv3 instance, found {len(instances)}.")
    module = instances[0]
    if v3.model[18] is not module or module.f != [17, 16]:
        raise RuntimeError("Constructed model does not contain CSCEFv3([17, 16]) at layer 18.")
    if v3.model[19].f != [16, 18] or v3.model[-1].f != [20, 23, 26]:
        raise RuntimeError("Constructed CSCEF-v3 CCFM or decoder topology is incorrect.")
    if sum(isinstance(item, CSCEFv2) for item in v2.modules()) != 1:
        raise RuntimeError("CSCEF-v2 comparison model does not contain exactly one CSCEFv2.")

    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    expected_parameter_names = {
        "shared_projection.weight",
        "depthwise_conv.weight",
        "output_projection.weight",
        "raw_alpha",
    }
    parameter_names = set(dict(module.named_parameters()))
    if trainable != 16673 or parameter_names != expected_parameter_names:
        raise RuntimeError(f"Unexpected CSCEF-v3 parameters: count={trainable}, names={sorted(parameter_names)}")
    if module.raw_alpha.numel() != 1:
        raise RuntimeError("CSCEF-v3 raw_alpha is not a single global scalar.")
    if module.shared_norm.affine or module.edge_norm.affine:
        raise RuntimeError("CSCEF-v3 GroupNorm layers must use affine=False.")
    convolutions = (module.shared_projection, module.depthwise_conv, module.output_projection)
    if any(conv.bias is not None for conv in convolutions):
        raise RuntimeError("Every CSCEF-v3 convolution must use bias=False.")
    forbidden = ("temperature", "similarity_bias", "layer_scale")
    if any(token in name for name in parameter_names for token in forbidden):
        raise RuntimeError(f"CSCEF-v3 retained a forbidden v2 parameter: {sorted(parameter_names)}")
    raw_initial = module.raw_alpha.item()
    alpha_initial = module._effective_alpha().item()
    if abs(raw_initial - (-1.38629436112)) > 1e-6 or abs(alpha_initial - 0.01) > 1e-7:
        raise RuntimeError(f"Unexpected alpha initialization: raw={raw_initial}, effective={alpha_initial}")

    print("cscef_v3_instance_count=1")
    print(f"cscef_v3_trainable_parameters={trainable}")
    print(f"cscef_v3_parameter_names={sorted(parameter_names)}")
    print("shared_groupnorm_affine=False")
    print("edge_groupnorm_affine=False")
    print("all_cscef_v3_conv_bias=False")
    print(f"raw_alpha_initial={raw_initial:.12f}")
    print(f"effective_alpha_initial={alpha_initial:.12f}")
    print("unchanged_layer_types_and_state_shapes_identical=True")
    return module


def channel_direction(values: list[float], height: int = 3, width: int = 4) -> torch.Tensor:
    """Create a non-degenerate channel direction repeated spatially."""
    return torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1).expand(1, -1, height, width).clone()


def audit_module_numerics(module: CSCEFv3) -> None:
    """Exercise gate boundaries, forward shape, RMS bound, small padding, and backward gradients on CPU."""
    direction = channel_direction([1.0, -1.0, 0.0, 0.0, 0.5, -0.5, 0.25, -0.25])
    orthogonal = channel_direction([0.0, 0.0, 1.0, -1.0, -0.25, 0.25, 0.5, -0.5])
    gates = {
        "identical": module._compute_discrepancy_gate(direction, direction).mean().item(),
        "orthogonal": module._compute_discrepancy_gate(direction, orthogonal).mean().item(),
        "opposite": module._compute_discrepancy_gate(direction, -direction).mean().item(),
    }
    if abs(gates["identical"]) > 1e-6 or abs(gates["orthogonal"] - 0.5) > 1e-6:
        raise RuntimeError(f"CSCEF-v3 gate boundary audit failed: {gates}")
    if abs(gates["opposite"] - 1.0) > 1e-6:
        raise RuntimeError(f"CSCEF-v3 opposite-feature gate audit failed: {gates}")

    torch.manual_seed(0)
    lateral = torch.randn(2, 256, 12, 14, requires_grad=True)
    semantic = torch.randn(2, 256, 6, 7, requires_grad=True)
    output = module([lateral, semantic])
    if output.shape != lateral.shape or output.dtype != lateral.dtype or not torch.isfinite(output).all():
        raise RuntimeError("CSCEF-v3 CPU FP32 output shape, dtype, or finiteness audit failed.")
    delta = output - lateral
    reduce_dims = (1, 2, 3)
    ratios = torch.sqrt(delta.float().square().mean(dim=reduce_dims)) / torch.sqrt(
        lateral.float().square().mean(dim=reduce_dims)
    )
    if ratios.max().item() > module.alpha_max + 1e-6:
        raise RuntimeError(f"CSCEF-v3 residual RMS bound failed: {ratios.tolist()}")
    weights = torch.randn_like(output)
    (output * weights).mean().backward()
    required_gradients = {
        "shared_projection.weight",
        "depthwise_conv.weight",
        "output_projection.weight",
        "raw_alpha",
    }
    for name, parameter in module.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing or non-finite CSCEF-v3 gradient: {name}")
        if name in required_gradients and parameter.grad.abs().sum().item() == 0:
            raise RuntimeError(f"Zero CSCEF-v3 gradient: {name}")

    for shape in ((1, 1), (1, 5), (5, 1)):
        magnitude = module._compute_scharr_magnitude(torch.ones(1, 32, *shape))
        if magnitude.shape != (1, 32, *shape) or not torch.isfinite(magnitude).all():
            raise RuntimeError(f"CSCEF-v3 small-spatial Scharr audit failed for {shape}.")

    with torch.no_grad():
        saved_raw_alpha = module.raw_alpha.detach().clone()
        module.raw_alpha.fill_(-100.0)
        alpha_low = module._effective_alpha().item()
        module.raw_alpha.fill_(100.0)
        alpha_high = module._effective_alpha().item()
        module.raw_alpha.copy_(saved_raw_alpha)
    if not 0.0 < alpha_low < module.alpha_max or not 0.0 < alpha_high < module.alpha_max:
        raise RuntimeError(f"Effective alpha escaped its open interval: low={alpha_low}, high={alpha_high}")

    print(f"gate_identical={gates['identical']:.9f}")
    print(f"gate_orthogonal={gates['orthogonal']:.9f}")
    print(f"gate_opposite={gates['opposite']:.9f}")
    print(f"cpu_fp32_output_shape={tuple(output.shape)}")
    print("cpu_fp32_output_finite=True")
    print(f"residual_rms_ratio_per_sample={ratios.detach().tolist()}")
    print(f"residual_rms_ratio_max={ratios.max().item():.9f}")
    print(f"effective_alpha_extreme_raw_bounds=({alpha_low:.12g}, {alpha_high:.12g})")
    print("required_new_parameter_gradients_finite_nonzero=True")


def audit_cuda() -> None:
    """Run real CUDA FP32 and AMP FP16 module forwards when CUDA is available."""
    if not torch.cuda.is_available():
        print("cuda_fp32=SKIPPED (CUDA unavailable)")
        print("cuda_amp_fp16=SKIPPED (CUDA unavailable)")
        return
    for use_amp in (False, True):
        module = CSCEFv3(256, 256).cuda().eval()
        dtype = torch.float16 if use_amp else torch.float32
        lateral = torch.randn(2, 256, 12, 14, device="cuda", dtype=dtype)
        semantic = torch.randn(2, 256, 6, 7, device="cuda", dtype=dtype)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = module([lateral, semantic])
        if output.dtype != lateral.dtype or output.shape != lateral.shape or not torch.isfinite(output).all():
            raise RuntimeError(f"CSCEF-v3 CUDA audit failed with amp={use_amp}.")
        print(f"cuda_{'amp_fp16' if use_amp else 'fp32'}=PASSED shape={tuple(output.shape)} dtype={output.dtype}")
        del output, semantic, lateral, module
        gc.collect()
        torch.cuda.empty_cache()


def complexity_table(v3_cfg: Path) -> tuple[dict[str, RTDETRDetectionModel], dict[str, tuple]]:
    """Build baseline/v2/v3 nc=1 models and report measured parameter and GFLOPs comparisons."""
    models = {
        "R18-Lite baseline": RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=1, verbose=False).eval(),
        "R18-Lite + CSCEF-v2": RTDETRDetectionModel(str(V2_CFG), ch=3, nc=1, verbose=False).eval(),
        "R18-Lite + CSCEF-v3": RTDETRDetectionModel(str(v3_cfg), ch=3, nc=1, verbose=False).eval(),
    }
    rows = {}
    for name, model in models.items():
        info = model.info(imgsz=640)
        if info is None or not info[3]:
            raise RuntimeError(f"model.info(imgsz=640) did not produce FLOPs for {name}.")
        rows[name] = info
    _, base_params, _, base_flops = rows["R18-Lite baseline"]
    _, _, _, _ = rows["R18-Lite + CSCEF-v2"]
    _, v3_params, _, v3_flops = rows["R18-Lite + CSCEF-v3"]
    v3_delta = v3_params - base_params
    if v3_delta != 16673:
        raise RuntimeError(f"CSCEF-v3 total parameter delta is {v3_delta}, expected 16673.")

    print("complexity_table:")
    print("model | layers | parameters | parameter_delta | parameter_delta_ratio | GFLOPs | GFLOPs_delta")
    for name, (layers, parameters, gradients, flops) in rows.items():
        del gradients
        print(
            f"{name} | {layers} | {parameters} | {parameters - base_params} | "
            f"{(parameters - base_params) / base_params:.6%} | {flops:.6f} | {flops - base_flops:.6f}"
        )
    print(f"cscef_v3_parameter_delta={v3_delta}")
    print(f"cscef_v3_parameter_delta_ratio={v3_delta / base_params:.9%}")
    print(f"cscef_v3_measured_gflops={v3_flops:.6f}")
    return models, rows


def audit_complete_models(v3_cfg: Path, nc1_model: RTDETRDetectionModel) -> None:
    """Run nc=1 and nc=80 complete-model CPU FP32 forwards and verify decoder output shapes."""
    for nc, model in ((1, nc1_model), (80, RTDETRDetectionModel(str(v3_cfg), ch=3, nc=80, verbose=False).eval())):
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 128, 128))
        expected_shape = (1, 300, nc + 4)
        if not isinstance(output, tuple) or output[0].shape != expected_shape or not torch.isfinite(output[0]).all():
            raise RuntimeError(f"CSCEF-v3 complete-model nc={nc} forward failed.")
        print(f"nc={nc}_cpu_output_shape={tuple(output[0].shape)} finite=True")
        del output


def audit_checkpoint_mapping(
    source_path: Path,
    initialized_path: Path,
    baseline_model: RTDETRDetectionModel,
    v3_model: RTDETRDetectionModel,
    v3_nc1_model: RTDETRDetectionModel,
) -> None:
    """Verify source coverage, initialized values, clean state, backbone mapping, and nc shape changes."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Baseline initialization checkpoint not found: {source_path}")
    if not initialized_path.is_file():
        raise FileNotFoundError(f"Generated CSCEF-v3 initialization checkpoint not found: {initialized_path}")

    source_checkpoint = torch_load(source_path, map_location="cpu")
    source_state = checkpoint_model(source_checkpoint).state_dict()
    baseline_state = baseline_model.state_dict()
    target_state = v3_model.state_dict()
    plan = build_strict_mapping(source_state, baseline_state, target_state)

    initialized_checkpoint = torch_load(initialized_path, map_location="cpu")
    initialized_model = checkpoint_model(initialized_checkpoint)
    initialized_state = initialized_model.state_dict()
    missing_initialized = sorted(set(target_state) - set(initialized_state))
    unexpected_initialized = sorted(set(initialized_state) - set(target_state))
    initialized_shape_mismatches = sorted(
        key
        for key in target_state
        if key in initialized_state and target_state[key].shape != initialized_state[key].shape
    )
    mapped_value_mismatches = verify_mapped_values(source_state, initialized_state, plan.source_to_target)

    target_parameters = dict(v3_model.named_parameters())
    mapped_parameter_keys = [key for key in plan.state if key in target_parameters]
    unchanged_parameter_numel = sum(
        parameter.numel() for key, parameter in target_parameters.items() if not key.startswith("model.18.")
    )
    mapped_parameter_numel = sum(target_parameters[key].numel() for key in mapped_parameter_keys)
    backbone_parameter_keys = [key for key in target_parameters if key.startswith(BACKBONE_PREFIXES)]
    mapped_backbone_keys = [key for key in backbone_parameter_keys if key in plan.state]
    backbone_numel = sum(target_parameters[key].numel() for key in backbone_parameter_keys)
    mapped_backbone_numel = sum(target_parameters[key].numel() for key in mapped_backbone_keys)
    clean_training_state = (
        initialized_checkpoint.get("epoch") == -1
        and initialized_checkpoint.get("ema") is None
        and initialized_checkpoint.get("optimizer") is None
        and initialized_checkpoint.get("scaler") is None
    )

    print(f"source_checkpoint_sha256={sha256(source_path)}")
    print(f"initialized_checkpoint_sha256={sha256(initialized_path)}")
    print(f"mapped_unchanged_state_keys={len(plan.state)}/{len(baseline_state)}")
    print(f"mapped_unchanged_parameter_numel={mapped_parameter_numel}/{unchanged_parameter_numel}")
    print(f"backbone_parameter_key_coverage={len(mapped_backbone_keys)}/{len(backbone_parameter_keys)}")
    print(f"backbone_parameter_numel_coverage={mapped_backbone_numel}/{backbone_numel}")
    print(f"mapping_shape_mismatches={plan.shape_mismatches}")
    print(f"all_mapped_values_exact={not mapped_value_mismatches}")
    print(f"initialized_missing_keys={missing_initialized}")
    print(f"initialized_unexpected_keys={unexpected_initialized}")
    print(f"initialized_shape_mismatches={initialized_shape_mismatches}")
    print(f"clean_initialization_checkpoint_state={clean_training_state}")
    print("new_cscef_v3_state_keys:")
    for key in plan.new_target_keys:
        print(f"  {key}")
    nc_shape_audit(initialized_state, v3_nc1_model.state_dict())

    errors = []
    if len(plan.state) != len(baseline_state):
        errors.append("not every unchanged baseline state key was mapped")
    if mapped_parameter_numel != unchanged_parameter_numel:
        errors.append("not every unchanged parameter value was mapped")
    if len(mapped_backbone_keys) != len(backbone_parameter_keys) or mapped_backbone_numel != backbone_numel:
        errors.append("backbone parameter mapping coverage is incomplete")
    if missing_initialized or unexpected_initialized or initialized_shape_mismatches:
        errors.append("initialized checkpoint structure differs from the nc=80 CSCEF-v3 model")
    if mapped_value_mismatches:
        errors.append(f"mapped values differ: {mapped_value_mismatches}")
    if not clean_training_state:
        errors.append("generated checkpoint retained training state")
    if not isinstance(initialized_model.model[18], CSCEFv3):
        errors.append("generated checkpoint does not contain CSCEFv3 at layer 18")
    if errors:
        raise RuntimeError(f"CSCEF-v3 checkpoint audit failed: {errors}")
    print("strict_checkpoint_mapping_audit_passed=True")


def main() -> None:
    """Run all requested CSCEF-v3 audits and fail nonzero on any discrepancy."""
    args = parse_args()
    v3_cfg = args.model.resolve()
    source_path = args.source.resolve()
    initialized_path = args.initialized.resolve()
    imported_path = Path(ultralytics.__file__).resolve()
    print(f"ultralytics_version={ultralytics.__version__}")
    print(f"ultralytics_file={imported_path}")
    if not imported_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")

    verify_protected_files()
    verify_yaml_topology(v3_cfg)
    baseline_nc80 = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False).eval()
    v2_nc80 = RTDETRDetectionModel(str(V2_CFG), ch=3, nc=80, verbose=False).eval()
    v3_nc80 = RTDETRDetectionModel(str(v3_cfg), ch=3, nc=80, verbose=False).eval()
    module = verify_model_structure(baseline_nc80, v2_nc80, v3_nc80)
    audit_module_numerics(module)
    audit_cuda()
    nc1_models, _ = complexity_table(v3_cfg)
    audit_complete_models(v3_cfg, nc1_models["R18-Lite + CSCEF-v3"])
    if args.skip_checkpoint:
        print("checkpoint_mapping_audit=SKIPPED (--skip-checkpoint)")
    else:
        audit_checkpoint_mapping(
            source_path,
            initialized_path,
            baseline_nc80,
            v3_nc80,
            nc1_models["R18-Lite + CSCEF-v3"],
        )
    print("CSCEF-v3 audit passed.")


if __name__ == "__main__":
    main()
