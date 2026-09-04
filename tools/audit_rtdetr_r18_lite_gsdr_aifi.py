"""Audit GSDR-AIFI structure, numerics, complexity, and controlled checkpoint initialization."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
MODEL_DIR = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
MODULE_DIR = ULTRALYTICS_ROOT / "ultralytics" / "nn" / "modules"
BASE_CFG = MODEL_DIR / "rtdetr-resnet18-lite.yaml"
DEFAULT_MODEL = MODEL_DIR / "rtdetr-resnet18-lite-gsdr-aifi.yaml"
DEFAULT_SOURCE = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_INITIALIZED = ROOT / "weights" / "rtdetr_r18_lite_gsdr_aifi_imagenet_backbone_init.pt"
PROTECTED_SHA256 = {
    BASE_CFG: "3493d693cbef958a38a2a135822a6c5379b65804905bc71566e395069e8f4f16",
    MODULE_DIR / "transformer.py": "8986f3f58a5e30a42545f544f1a35f5e5861b13ee5d4507d8b2afab1c41df782",
    ROOT / "tools" / "init_rtdetr_r18_lite_imagenet_backbone.py": (
        "1ee77782f64b7e310a14c436a15807ffdca57725a4f1cd9b5a684e9d054df87a"
    ),
}
EXPECTED_SPARSE_PARAMETERS = 117_644
AIFI_LAYER_INDEX = 9
BACKBONE_PREFIXES = tuple(f"model.{index}." for index in range(8))
SPARSE_PREFIX = f"model.{AIFI_LAYER_INDEX}.sparse_relation."

# Prefer this repository's Ultralytics package over any pip-installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics import RTDETR  # noqa: E402
from ultralytics.nn.modules import AIFI, GSDRAIFI, SparseDeformableRelation  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402

from init_rtdetr_r18_lite_gsdr_aifi_controlled import (  # noqa: E402
    build_strict_mapping,
    checkpoint_model,
    clean_initialization_state,
    exact_value_mismatches,
    nc_shape_audit,
    original_aifi_keys,
    sha256,
    verify_mapped_values,
    verify_zero_initialized_boundaries,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="GSDR-AIFI model YAML.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Baseline initialization checkpoint.")
    parser.add_argument("--initialized", type=Path, default=DEFAULT_INITIALIZED, help="Generated checkpoint.")
    parser.add_argument(
        "--skip-checkpoint", action="store_true", help="Run structure/numerics/complexity without weight files."
    )
    return parser.parse_args()


def canonical_lf_sha256(data: bytes) -> str:
    """Hash text after normalizing all line endings to LF."""
    canonical = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def verify_protected_files() -> None:
    """Fail if the baseline YAML, AIFI implementation, or baseline initializer changed."""
    mismatches = []
    for path, expected in PROTECTED_SHA256.items():
        actual = canonical_lf_sha256(path.read_bytes())
        print(f"protected_sha256[{path.relative_to(ROOT)}]={actual}")
        if actual != expected:
            mismatches.append(f"{path}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError(f"Protected baseline files changed: {mismatches}")
    print("protected_baseline_files_unchanged=True")


def verify_yaml_topology(model_cfg: Path) -> None:
    """Require the new YAML to differ from baseline only at AIFI layer 9."""
    baseline = YAML.load(BASE_CFG)
    gsdr = YAML.load(model_cfg)
    baseline_layers = baseline["backbone"] + baseline["head"]
    gsdr_layers = gsdr["backbone"] + gsdr["head"]
    expected = deepcopy(baseline_layers)
    expected[AIFI_LAYER_INDEX] = [-1, 1, "GSDRAIFI", [1024, 8, 128, 4, 4, 2, 2.0, 3]]
    if baseline["backbone"] != gsdr["backbone"] or gsdr_layers != expected:
        raise RuntimeError("GSDR-AIFI YAML differs from baseline beyond the layer-9 module and sparse arguments.")
    if baseline["nc"] != gsdr["nc"] or baseline["scales"] != gsdr["scales"]:
        raise RuntimeError("GSDR-AIFI YAML changed baseline nc or scaling configuration.")
    print("yaml_topology_matches_baseline_except_layer9_module=True")
    print(f"gsdr_aifi_yaml_args={expected[AIFI_LAYER_INDEX][3]}")
    print(f"decoder_inputs={gsdr_layers[-1][0]}")


def local_state_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    """Return local module state shapes."""
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def verify_model_structure(
    baseline: RTDETRDetectionModel, gsdr: RTDETRDetectionModel
) -> tuple[GSDRAIFI, SparseDeformableRelation]:
    """Check topology, unchanged layers, original AIFI parameters, and sparse parameterization."""
    if len(baseline.model) != len(gsdr.model):
        raise RuntimeError("Baseline and GSDR-AIFI parsed layer counts differ.")
    errors = []
    for index, (baseline_layer, gsdr_layer) in enumerate(zip(baseline.model, gsdr.model)):
        if index == AIFI_LAYER_INDEX:
            continue
        if type(baseline_layer) is not type(gsdr_layer):
            errors.append(f"layer {index} type: {type(baseline_layer).__name__} != {type(gsdr_layer).__name__}")
        if local_state_shapes(baseline_layer) != local_state_shapes(gsdr_layer):
            errors.append(f"layer {index} state shapes differ")
        if baseline_layer.f != gsdr_layer.f:
            errors.append(f"layer {index} input routing differs")
    if errors:
        raise RuntimeError(f"Unchanged baseline layers differ: {errors}")

    baseline_aifi = baseline.model[AIFI_LAYER_INDEX]
    module = gsdr.model[AIFI_LAYER_INDEX]
    if type(baseline_aifi) is not AIFI or not isinstance(module, GSDRAIFI):
        raise RuntimeError("Expected original AIFI and GSDRAIFI at layer 9.")
    instances = [item for item in gsdr.modules() if isinstance(item, GSDRAIFI)]
    if len(instances) != 1 or instances[0] is not module:
        raise RuntimeError(f"Expected exactly one GSDRAIFI instance at layer 9, found {len(instances)}.")

    baseline_shapes = local_state_shapes(baseline_aifi)
    gsdr_original_shapes = {
        key: shape for key, shape in local_state_shapes(module).items() if not key.startswith("sparse_relation.")
    }
    if baseline_shapes != gsdr_original_shapes:
        raise RuntimeError("GSDRAIFI did not preserve every original AIFI state key and shape.")
    required_attributes = ("ma", "fc1", "fc2", "norm1", "norm2", "dropout", "dropout1", "dropout2")
    missing_attributes = [name for name in required_attributes if not hasattr(module, name)]
    if missing_attributes:
        raise RuntimeError(f"GSDRAIFI lost baseline AIFI attributes: {missing_attributes}")
    if module.ma.embed_dim != 256 or module.ma.num_heads != 8:
        raise RuntimeError("Dense baseline MultiheadAttention is not 256 channels with 8 heads.")
    if module.fc1.in_features != 256 or module.fc1.out_features != 1024 or module.fc2.out_features != 256:
        raise RuntimeError("Baseline fc1/fc2 dimensions changed.")

    relation = module.sparse_relation
    sparse_parameters = sum(parameter.numel() for parameter in relation.parameters() if parameter.requires_grad)
    if sparse_parameters != EXPECTED_SPARSE_PARAMETERS:
        raise RuntimeError(f"Sparse parameter count is {sparse_parameters}, expected {EXPECTED_SPARSE_PARAMETERS}.")
    if (relation.aux_dim, relation.aux_heads, relation.offset_groups, relation.sample_stride) != (128, 4, 4, 2):
        raise RuntimeError("Parsed sparse relation hyperparameters differ from the controlled defaults.")
    verify_zero_initialized_boundaries(module)
    forbidden_attributes = ("gate", "alpha", "scharr", "sobel", "fft", "wavelet", "strip")
    forbidden = [
        name for name, _ in module.named_modules() if any(token in name.lower() for token in forbidden_attributes)
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden mechanism attributes found: {forbidden}")

    print("gsdr_aifi_instance_count=1")
    print("gsdr_aifi_layer_index=9")
    print("original_aifi_state_keys_and_shapes_preserved=True")
    print(f"original_aifi_attributes_preserved={list(required_attributes)}")
    print("dense_mha_embed_dim=256")
    print("dense_mha_heads=8")
    print("baseline_ffn_dimensions=256->1024->256")
    print(f"sparse_relation_trainable_parameters={sparse_parameters}")
    print("sparse_relation_dimensions=256->128->256")
    print("offset_groups=4")
    print("sparse_heads=4")
    print("sample_stride=2")
    print("zero_initialized_sparse_boundaries=True")
    print("forbidden_mechanisms_absent=True")
    return module, relation


def finite_nonzero_gradient(parameter: torch.Tensor) -> bool:
    """Return whether a parameter has a finite, nonzero gradient."""
    return (
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        and parameter.grad.abs().sum().item() > 0
    )


def audit_baseline_equivalence() -> float:
    """Require exact post-norm and pre-norm equality after loading original AIFI state."""
    torch.manual_seed(20)
    input_tensor = torch.randn(2, 256, 16, 24)
    maximum = 0.0
    for normalize_before in (False, True):
        baseline = AIFI(256, 1024, 8, dropout=0.0, normalize_before=normalize_before).eval()
        gsdr = GSDRAIFI(256, 1024, 8, dropout=0.0, normalize_before=normalize_before).eval()
        result = gsdr.load_state_dict(baseline.state_dict(), strict=False)
        if result.unexpected_keys or not result.missing_keys or any(
            not key.startswith("sparse_relation.") for key in result.missing_keys
        ):
            raise RuntimeError(f"Unexpected module mapping result: {result}")
        with torch.no_grad():
            expected = baseline(input_tensor)
            actual = gsdr(input_tensor)
        difference = (expected - actual).abs().max().item()
        maximum = max(maximum, difference)
        if difference > 1e-6 or not torch.equal(expected, actual):
            raise RuntimeError(
                f"Baseline equivalence failed for normalize_before={normalize_before}: max_abs_diff={difference}"
            )
        print(f"baseline_equivalence_normalize_before={normalize_before}_max_abs_diff={difference:.9g}")
    print(f"baseline_equivalence_max_abs_diff={maximum:.9g}")
    print("baseline_equivalence_elementwise_exact=True")
    return maximum


def audit_relation_numerics() -> None:
    """Exercise dynamic shapes, grids, offsets, finiteness, and staged gradient unblocking."""
    torch.manual_seed(21)
    relation = SparseDeformableRelation(256, 128, 4, 4, 2, 2.0, 3).eval()
    for height, width in ((20, 20), (16, 24), (1, 1), (1, 5), (5, 1)):
        tensor = torch.randn(1, 256, height, width)
        with torch.no_grad():
            projected = relation.input_proj(tensor)
            reference, offsets, deformed = relation._sampling_grid(projected)
            output = relation(tensor)
        expected_h, expected_w = math.ceil(height / 2), math.ceil(width / 2)
        if reference.shape != (1, expected_h, expected_w, 2):
            raise RuntimeError(f"Unexpected reference shape for {(height, width)}: {tuple(reference.shape)}")
        if offsets.shape != (1, 4, expected_h, expected_w, 2):
            raise RuntimeError(f"Unexpected offset shape for {(height, width)}: {tuple(offsets.shape)}")
        if deformed.shape != offsets.shape or output.shape != tensor.shape:
            raise RuntimeError(f"Unexpected deformed/output shape for {(height, width)}.")
        if not all(torch.isfinite(item).all().item() for item in (reference, offsets, deformed, output)):
            raise RuntimeError(f"Non-finite relation output for {(height, width)}.")
        if reference.min().item() < -1 or reference.max().item() > 1:
            raise RuntimeError("Reference grid escaped [-1, 1].")
        x_bound = 2.0 * relation.offset_range_factor / expected_w if expected_w > 1 else 0.0
        y_bound = 2.0 * relation.offset_range_factor / expected_h if expected_h > 1 else 0.0
        if offsets[..., 0].abs().max().item() > x_bound + 1e-6:
            raise RuntimeError("x offset escaped its runtime sparse-grid bound.")
        if offsets[..., 1].abs().max().item() > y_bound + 1e-6:
            raise RuntimeError("y offset escaped its runtime sparse-grid bound.")
        if output.abs().max().item() != 0.0:
            raise RuntimeError("Zero-initialized sparse relation did not output exact zeros.")
        print(
            f"relation_shape_{height}x{width}=output{tuple(output.shape)},offsets{tuple(offsets.shape)},"
            f"reference_range=({reference.min().item():.6f},{reference.max().item():.6f}),finite=True"
        )

    # At initialization, only the zero output boundary should receive a learning signal.
    relation = SparseDeformableRelation(32, 32, 4, 4, 2, 2.0, 3)
    tensor = torch.randn(2, 32, 8, 10, requires_grad=True)
    (relation(tensor) * torch.randn_like(tensor)).mean().backward()
    if not finite_nonzero_gradient(relation.output_proj.weight) or not finite_nonzero_gradient(
        relation.output_proj.bias
    ):
        raise RuntimeError("Zero sparse output projection did not receive a nonzero first-backward gradient.")
    print("first_backward_sparse_output_projection_gradient_finite_nonzero=True")

    # Once the output boundary is nonzero, Q/K/V, offset output, and displacement-bias output layers learn.
    relation.zero_grad(set_to_none=True)
    tensor.grad = None
    with torch.no_grad():
        torch.nn.init.normal_(relation.output_proj.weight, std=0.02)
    (relation(tensor) * torch.randn_like(tensor)).mean().backward()
    representative = {
        "q": relation.q_proj.weight,
        "k": relation.k_proj.weight,
        "v": relation.v_proj.weight,
        "offset": relation.offset_out.weight,
        "relative_bias": relation.relative_bias[0].fc2.weight,
    }
    failed = [name for name, parameter in representative.items() if not finite_nonzero_gradient(parameter)]
    if failed:
        raise RuntimeError(f"Sparse components remained gradient-blocked after enabling output projection: {failed}")
    print("output_enabled_qkv_offset_and_relative_bias_gradients_finite_nonzero=True")

    # Simulate subsequent optimizer updates of both zero inner boundaries and prove every parameter is reachable.
    relation.zero_grad(set_to_none=True)
    tensor.grad = None
    with torch.no_grad():
        torch.nn.init.normal_(relation.offset_out.weight, std=0.02)
        for mlp in relation.relative_bias:
            torch.nn.init.normal_(mlp.fc2.weight, std=0.02)
    (relation(tensor) * torch.randn_like(tensor)).mean().backward()
    failed = [name for name, parameter in relation.named_parameters() if not finite_nonzero_gradient(parameter)]
    if failed:
        raise RuntimeError(f"Sparse parameters with missing, zero, or non-finite eventual gradients: {failed}")
    if tensor.grad is None or not torch.isfinite(tensor.grad).all() or tensor.grad.abs().sum().item() == 0:
        raise RuntimeError("Sparse relation input did not receive a finite nonzero gradient.")
    print("all_sparse_parameters_eventually_receive_finite_nonzero_gradients=True")


def audit_complete_models(model_cfg: Path) -> None:
    """Run nc=1/nc=80 CPU forwards, a 640 forward, and a complete-model backward."""
    for nc in (1, 80):
        model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=nc, verbose=False).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 128, 128))
        expected = (1, 300, nc + 4)
        if not isinstance(output, tuple) or output[0].shape != expected or not torch.isfinite(output[0]).all():
            raise RuntimeError(f"GSDR-AIFI nc={nc} CPU FP32 complete-model forward failed.")
        print(f"nc={nc}_cpu_fp32_output_shape={tuple(output[0].shape)}_finite=True")
        del output, model
        gc.collect()

    model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=1, verbose=False).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 640, 640))
    if output[0].shape != (1, 300, 5) or not torch.isfinite(output[0]).all():
        raise RuntimeError("GSDR-AIFI complete 640x640 CPU FP32 forward failed.")
    print(f"cpu_complete_640_fp32=PASSED shape={tuple(output[0].shape)} dtype={output[0].dtype}")
    del output, model
    gc.collect()

    model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=1, verbose=False).eval()
    image = torch.randn(1, 3, 128, 128, requires_grad=True)
    output = model(image)
    loss = (output[0].float() * torch.randn_like(output[0].float())).mean()
    loss.backward()
    boundary = model.model[AIFI_LAYER_INDEX].sparse_relation.output_proj.weight
    if (
        image.grad is None
        or not torch.isfinite(image.grad).all()
        or image.grad.abs().sum().item() == 0
        or not finite_nonzero_gradient(boundary)
    ):
        raise RuntimeError("Complete RT-DETR CPU FP32 backward failed or did not reach the sparse output boundary.")
    print("cpu_complete_model_forward_backward_finite=True")
    print("cpu_complete_model_sparse_output_projection_gradient_nonzero=True")
    del loss, output, image, model
    gc.collect()


def audit_cuda(model_cfg: Path) -> None:
    """Run module and complete-model CUDA FP32, AMP FP16, and explicit-half checks."""
    if not torch.cuda.is_available():
        print("cuda_fp32=SKIPPED (CUDA unavailable)")
        print("cuda_amp_fp16=SKIPPED (CUDA unavailable)")
        print("cuda_explicit_half=SKIPPED (CUDA unavailable)")
        print("cuda_complete_640_explicit_half=SKIPPED (CUDA unavailable)")
        return

    for use_amp in (False, True):
        module = GSDRAIFI(256, 1024, 8).cuda()
        image = torch.randn(1, 256, 16, 24, device="cuda", requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = module(image)
            loss = (output.float() * torch.randn_like(output.float())).mean()
        if output.shape != image.shape or not torch.isfinite(output).all():
            raise RuntimeError(f"GSDR-AIFI CUDA module failed with amp={use_amp}.")
        loss.backward()
        boundary = module.sparse_relation.output_proj.weight
        if not torch.isfinite(image.grad).all() or not finite_nonzero_gradient(boundary):
            raise RuntimeError(f"GSDR-AIFI CUDA backward failed with amp={use_amp}.")
        label = "amp_fp16" if use_amp else "fp32"
        print(f"cuda_{label}_module_forward_backward=PASSED shape={tuple(output.shape)} dtype={output.dtype}")
        del loss, output, image, module
        gc.collect()
        torch.cuda.empty_cache()

    module = GSDRAIFI(256, 1024, 8).cuda().half()
    image = torch.randn(1, 256, 16, 24, device="cuda", dtype=torch.float16, requires_grad=True)
    output = module(image)
    loss = (output.float() * torch.randn_like(output.float())).mean()
    if output.dtype != torch.float16 or not torch.isfinite(output).all():
        raise RuntimeError("GSDR-AIFI explicit-half module forward failed.")
    loss.backward()
    if not torch.isfinite(image.grad).all() or not finite_nonzero_gradient(module.sparse_relation.output_proj.weight):
        raise RuntimeError("GSDR-AIFI explicit-half module backward failed.")
    print(f"cuda_explicit_half_module_forward_backward=PASSED shape={tuple(output.shape)} dtype={output.dtype}")
    del loss, output, image, module
    gc.collect()
    torch.cuda.empty_cache()

    for use_amp in (False, True):
        model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=1, verbose=False).eval().cuda()
        image = torch.zeros(1, 3, 128, 128, device="cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(image)
        if output[0].shape != (1, 300, 5) or not torch.isfinite(output[0]).all():
            raise RuntimeError(f"GSDR-AIFI complete CUDA model failed with amp={use_amp}.")
        label = "amp_fp16" if use_amp else "fp32"
        print(f"cuda_complete_{label}=PASSED shape={tuple(output[0].shape)} dtype={output[0].dtype}")
        del output, image, model
        gc.collect()
        torch.cuda.empty_cache()

    model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=1, verbose=False).eval().cuda().half()
    image = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        output = model(image)
    if output[0].shape != (1, 300, 5) or output[0].dtype != torch.float16 or not torch.isfinite(output[0]).all():
        raise RuntimeError("GSDR-AIFI complete 640x640 explicit-half CUDA model failed.")
    print(f"cuda_complete_640_explicit_half=PASSED shape={tuple(output[0].shape)} dtype={output[0].dtype}")
    del output, image, model
    gc.collect()
    torch.cuda.empty_cache()


def estimate_sparse_relation_gflops(relation: SparseDeformableRelation, height: int = 20, width: int = 20) -> float:
    """Estimate major sparse-branch operations, including functionals omitted by module-hook profilers."""
    sample_h = math.ceil(height / relation.sample_stride)
    sample_w = math.ceil(width / relation.sample_stride)
    queries = height * width
    samples = sample_h * sample_w
    c1, aux = relation.c1, relation.aux_dim
    groups = relation.offset_groups
    kernel = relation.offset_kernel

    macs = 0
    macs += queries * c1 * aux  # input projection
    macs += 3 * queries * aux * aux  # Q/K/V projections
    macs += sample_h * sample_w * aux * kernel * kernel  # depthwise offset convolution
    macs += sample_h * sample_w * 2 * aux  # grouped offset output convolution
    macs += queries * aux * c1  # output projection
    macs += 2 * queries * samples * aux  # QK and attention-value products
    macs += groups * queries * samples * (2 * 32 + 32)  # continuous-bias two-layer MLPs
    flops = 2 * macs
    flops += 2 * samples * aux * 8  # two bilinear K/V grid samples, approximate 8 operations/value
    flops += relation.aux_heads * queries * samples * 5  # approximate stable softmax operations
    return flops / 1e9


def complexity_table(model_cfg: Path) -> tuple[RTDETRDetectionModel, RTDETRDetectionModel, dict[str, tuple]]:
    """Report baseline/GSDR parameters and model.info GFLOPs plus a functional-aware branch estimate."""
    models = {
        "R18-Lite baseline": RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=1, verbose=False).eval(),
        "R18-Lite + GSDR-AIFI": RTDETRDetectionModel(str(model_cfg), ch=3, nc=1, verbose=False).eval(),
    }
    rows = {name: model.info(imgsz=640) for name, model in models.items()}
    if any(info is None or not info[3] for info in rows.values()):
        raise RuntimeError("model.info(imgsz=640) did not produce parameter/FLOP statistics.")
    _, base_params, _, base_flops = rows["R18-Lite baseline"]
    _, gsdr_params, _, gsdr_flops = rows["R18-Lite + GSDR-AIFI"]
    delta = gsdr_params - base_params
    if delta != EXPECTED_SPARSE_PARAMETERS or delta > 200_000 or delta / base_params > 0.01:
        raise RuntimeError(f"GSDR-AIFI parameter budget failed: delta={delta}, ratio={delta / base_params:.6%}")
    relation = models["R18-Lite + GSDR-AIFI"].model[AIFI_LAYER_INDEX].sparse_relation
    theoretical_delta = estimate_sparse_relation_gflops(relation)
    if theoretical_delta > 0.30:
        raise RuntimeError(f"Estimated sparse relation cost {theoretical_delta:.6f} GFLOPs exceeds 0.30.")

    print("complexity_table:")
    print("model | layers | parameters | parameter_delta | parameter_delta_ratio | model.info_GFLOPs | GFLOPs_delta")
    for name, (layers, parameters, gradients, flops) in rows.items():
        del gradients
        print(
            f"{name} | {layers} | {parameters} | {parameters - base_params} | "
            f"{(parameters - base_params) / base_params:.6%} | {flops:.6f} | {flops - base_flops:.6f}"
        )
    print(f"gsdr_aifi_parameter_delta={delta}")
    print(f"gsdr_aifi_parameter_delta_ratio={delta / base_params:.6%}")
    print(f"model_info_gflops_delta={gsdr_flops - base_flops:.6f}")
    print(f"functional_aware_sparse_branch_estimate_gflops={theoretical_delta:.6f}")
    print(
        "gflops_scope_note=model.info/THOP module hooks do not fully count functional torch.matmul, softmax, "
        "or grid_sample; model.info GFLOPs are not complete theoretical FLOPs"
    )
    return models["R18-Lite baseline"], models["R18-Lite + GSDR-AIFI"], rows


def audit_checkpoint_mapping(
    source_path: Path,
    initialized_path: Path,
    baseline_model: RTDETRDetectionModel,
    gsdr_model: RTDETRDetectionModel,
    model_cfg: Path,
) -> None:
    """Verify exact baseline mapping, sparse-only additions, clean state, reloads, and nc changes."""
    for path, label in ((source_path, "baseline initialization checkpoint"), (initialized_path, "initialized")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    source_checkpoint = torch_load(source_path, map_location="cpu")
    initialized_checkpoint = torch_load(initialized_path, map_location="cpu")
    if not clean_initialization_state(source_checkpoint):
        raise RuntimeError("Baseline source checkpoint is not a clean initialization checkpoint.")
    source_state = checkpoint_model(source_checkpoint).state_dict()
    target_state = gsdr_model.state_dict()
    plan = build_strict_mapping(source_state, baseline_model.state_dict(), target_state)
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
    aifi_keys = original_aifi_keys(source_state)
    aifi_mismatches = exact_value_mismatches(source_state, initialized_state, aifi_keys)
    clean_state = clean_initialization_state(initialized_checkpoint)
    verify_zero_initialized_boundaries(initialized_model.model[AIFI_LAYER_INDEX])

    target_parameters = dict(gsdr_model.named_parameters())
    mapped_parameter_numel = sum(
        target_parameters[key].numel() for key in plan.state if key in target_parameters
    )
    unchanged_parameter_numel = sum(
        parameter.numel() for key, parameter in target_parameters.items() if not key.startswith(SPARSE_PREFIX)
    )
    backbone_keys = [key for key in target_parameters if key.startswith(BACKBONE_PREFIXES)]
    mapped_backbone_keys = [key for key in backbone_keys if key in plan.state]
    backbone_numel = sum(target_parameters[key].numel() for key in backbone_keys)
    mapped_backbone_numel = sum(target_parameters[key].numel() for key in mapped_backbone_keys)

    if not isinstance(initialized_model.model[AIFI_LAYER_INDEX], GSDRAIFI):
        raise RuntimeError("Initialized checkpoint does not contain GSDRAIFI at layer 9.")
    direct = RTDETR(str(initialized_path))
    if not isinstance(direct.model.model[AIFI_LAYER_INDEX], GSDRAIFI):
        raise RuntimeError("Direct RTDETR(checkpoint) load did not preserve GSDRAIFI.")
    yaml_loaded = RTDETR(str(model_cfg))
    yaml_loaded.load(str(initialized_path))
    yaml_mismatches = exact_value_mismatches(
        initialized_state, yaml_loaded.model.state_dict(), sorted(initialized_state)
    )

    print(f"source_checkpoint_sha256={sha256(source_path)}")
    print(f"initialized_checkpoint_sha256={sha256(initialized_path)}")
    print(f"mapped_unchanged_state_keys={len(plan.state)}/{len(baseline_model.state_dict())}")
    print(f"mapped_unchanged_parameter_numel={mapped_parameter_numel}/{unchanged_parameter_numel}")
    print(f"mapping_shape_mismatches={plan.shape_mismatches}")
    print(f"unexpected_baseline_keys=[]")
    print(f"missing_keys_are_sparse_only={all(key.startswith(SPARSE_PREFIX) for key in plan.new_target_keys)}")
    print(f"mapped_baseline_values_exact={not mapped_mismatches}")
    print(f"original_aifi_mapped_keys={len(aifi_keys)}/{len(aifi_keys)}")
    print(f"original_aifi_value_mismatches={aifi_mismatches}")
    print(f"backbone_parameter_key_coverage={len(mapped_backbone_keys)}/{len(backbone_keys)}")
    print(f"backbone_parameter_numel_coverage={mapped_backbone_numel}/{backbone_numel}")
    print(f"initialized_missing_keys={missing}")
    print(f"initialized_unexpected_keys={unexpected}")
    print(f"initialized_shape_mismatches={shape_mismatches}")
    print(f"clean_initialization_checkpoint_state={clean_state}")
    print("sparse_output_projection_zero=True")
    print("offset_output_projection_zero=True")
    print("relative_bias_output_layers_zero=True")
    print("checkpoint_direct_load=True")
    print(f"checkpoint_yaml_load_exact={not yaml_mismatches}")
    nc1_model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=1, verbose=False)
    nc_shape_audit(initialized_state, nc1_model.state_dict())

    errors = []
    if len(plan.state) != len(baseline_model.state_dict()) or plan.shape_mismatches or mapped_mismatches:
        errors.append("baseline mapping is incomplete, shape-incompatible, or inexact")
    if not plan.new_target_keys or any(not key.startswith(SPARSE_PREFIX) for key in plan.new_target_keys):
        errors.append("missing target keys are not exclusively the new sparse relation")
    if aifi_mismatches:
        errors.append("original AIFI ma/fc1/fc2/norm1/norm2 values differ")
    if mapped_parameter_numel != unchanged_parameter_numel:
        errors.append("unchanged parameter coverage is incomplete")
    if len(mapped_backbone_keys) != len(backbone_keys) or mapped_backbone_numel != backbone_numel:
        errors.append("backbone parameter coverage is incomplete")
    if missing or unexpected or shape_mismatches:
        errors.append("initialized checkpoint state differs from target YAML")
    if not clean_state:
        errors.append("initialized checkpoint retained active training state")
    if yaml_mismatches:
        errors.append("YAML loading did not exactly preserve initialized tensors")
    if errors:
        raise RuntimeError(f"GSDR-AIFI checkpoint audit failed: {errors}")
    print("strict_controlled_checkpoint_audit_passed=True")


def main() -> None:
    """Run every requested audit and fail nonzero on any discrepancy."""
    args = parse_args()
    model_cfg = args.model.resolve()
    source_path = args.source.resolve()
    initialized_path = args.initialized.resolve()
    imported_path = Path(ultralytics.__file__).resolve()
    print(f"ultralytics_version={ultralytics.__version__}")
    print(f"ultralytics_file={imported_path}")
    if not imported_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")

    verify_protected_files()
    verify_yaml_topology(model_cfg)
    baseline_nc80 = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False).eval()
    gsdr_nc80 = RTDETRDetectionModel(str(model_cfg), ch=3, nc=80, verbose=False).eval()
    verify_model_structure(baseline_nc80, gsdr_nc80)
    audit_baseline_equivalence()
    audit_relation_numerics()
    audit_complete_models(model_cfg)
    audit_cuda(model_cfg)
    baseline_nc1, gsdr_nc1, _ = complexity_table(model_cfg)
    if args.skip_checkpoint:
        print("checkpoint_mapping_audit=SKIPPED (--skip-checkpoint)")
    else:
        audit_checkpoint_mapping(source_path, initialized_path, baseline_nc80, gsdr_nc80, model_cfg)
    del baseline_nc1, gsdr_nc1
    print("GSDR-AIFI audit passed.")


if __name__ == "__main__":
    main()
