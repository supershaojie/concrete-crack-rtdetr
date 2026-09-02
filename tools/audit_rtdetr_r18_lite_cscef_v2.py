"""Audit CSCEF-v2 structure, complexity, and strict baseline-checkpoint initialization."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
MODEL_DIR = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
BASE_CFG = MODEL_DIR / "rtdetr-resnet18-lite.yaml"
CSCEF_V1_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef.yaml"
DEFAULT_V2_CFG = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v2.yaml"
DEFAULT_SOURCE = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_INITIALIZED = ROOT / "weights" / "rtdetr_r18_lite_cscef_v2_imagenet_backbone_init.pt"
CSCEF_V1_MODULE = ULTRALYTICS_ROOT / "ultralytics" / "nn" / "modules" / "cscef.py"
PROTECTED_SHA256 = {
    BASE_CFG: "3493d693cbef958a38a2a135822a6c5379b65804905bc71566e395069e8f4f16",
    CSCEF_V1_CFG: "863ed9f2786737bbc0f96879b5f79c65b8a72fba2f65094cd64624ff54f70820",
    CSCEF_V1_MODULE: "9c6ade426fa468915e8325fb88b40e31facc3bff41472866e96aefdc11ae3cbf",
}
BACKBONE_PREFIXES = tuple(f"model.{index}." for index in range(8))

# Prefer this repository's Ultralytics package over any pip-installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics.nn.modules import CSCEF, CSCEFv2  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402
from ultralytics.utils.torch_utils import intersect_dicts  # noqa: E402

from init_rtdetr_r18_lite_cscef_v2_from_baseline import (  # noqa: E402
    build_strict_mapping,
    checkpoint_model,
    verify_mapped_values,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Baseline initialization checkpoint.")
    parser.add_argument("--model", type=Path, default=DEFAULT_V2_CFG, help="CSCEF-v2 model YAML.")
    parser.add_argument(
        "--initialized", type=Path, default=DEFAULT_INITIALIZED, help="Generated CSCEF-v2 initialization checkpoint."
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protected_files() -> None:
    """Fail if the baseline or CSCEF-v1 implementation/configuration changed from the v1 branch head."""
    mismatches = []
    for path, expected in PROTECTED_SHA256.items():
        actual = sha256(path)
        print(f"protected_sha256[{path.relative_to(ROOT)}]={actual}")
        if actual != expected:
            mismatches.append(f"{path}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError(f"Protected baseline/CSCEF-v1 files changed: {mismatches}")
    print("protected_baseline_and_cscef_v1_files_unchanged=True")


def absolute_references(reference: int | list[int], layer_index: int) -> list[int]:
    """Resolve negative YAML layer references into absolute layer indices."""
    references = reference if isinstance(reference, list) else [reference]
    return [layer_index + item if item < 0 else item for item in references]


def v2_reference_to_baseline(reference: int) -> int:
    """Map a CSCEF-v2 graph reference back to its baseline semantic counterpart."""
    if reference <= 17:
        return reference
    if reference == 18:
        return 17  # CSCEF-v2 replaces the direct use of projected S3 with its enhanced form.
    return reference - 1


def verify_yaml_topology(v2_cfg: Path) -> None:
    """Verify that CSCEF-v2 is the only semantic topology change from the baseline."""
    baseline = YAML.load(BASE_CFG)
    v2 = YAML.load(v2_cfg)
    baseline_layers = baseline["backbone"] + baseline["head"]
    v2_layers = v2["backbone"] + v2["head"]
    if baseline["backbone"] != v2["backbone"]:
        raise RuntimeError("CSCEF-v2 changed the R18-Lite backbone YAML.")
    if len(v2_layers) != len(baseline_layers) + 1:
        raise RuntimeError("CSCEF-v2 YAML must add exactly one model layer.")
    if v2_layers[18] != [[17, 16], 1, "CSCEFv2", []]:
        raise RuntimeError(f"Unexpected CSCEF-v2 insertion: {v2_layers[18]}")

    errors = []
    for baseline_index, baseline_layer in enumerate(baseline_layers):
        v2_index = baseline_index if baseline_index <= 17 else baseline_index + 1
        v2_layer = v2_layers[v2_index]
        if baseline_layer[1:] != v2_layer[1:]:
            errors.append(f"layer body differs: baseline {baseline_index} vs v2 {v2_index}")
        baseline_refs = absolute_references(baseline_layer[0], baseline_index)
        v2_refs = [v2_reference_to_baseline(ref) for ref in absolute_references(v2_layer[0], v2_index)]
        if baseline_refs != v2_refs:
            errors.append(
                f"layer references differ: baseline {baseline_index} {baseline_refs} vs v2 {v2_index} {v2_refs}"
            )
    if errors:
        raise RuntimeError(f"CSCEF-v2 topology differs beyond its insertion: {errors}")
    if absolute_references(v2_layers[19][0], 19) != [16, 18]:
        raise RuntimeError("CSCEF-v2 must preserve Concat [original upsampled P4, enhanced S3].")
    if v2_layers[-1][0] != [20, 23, 26]:
        raise RuntimeError("CSCEF-v2 decoder inputs must remain [20, 23, 26].")
    print("yaml_topology_only_adds_cscef_v2=True")
    print("decoder_inputs=[20, 23, 26]")


def local_state_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    """Return a module state-shape dictionary independent of its global layer prefix."""
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def verify_model_structure(baseline: RTDETRDetectionModel, v2: RTDETRDetectionModel) -> None:
    """Verify mapped layer types and state shapes for every unchanged model layer."""
    errors = []
    for baseline_index in range(27):
        v2_index = baseline_index if baseline_index <= 17 else baseline_index + 1
        baseline_layer = baseline.model[baseline_index]
        v2_layer = v2.model[v2_index]
        if type(baseline_layer) is not type(v2_layer):
            errors.append(
                f"type mismatch {baseline_index}->{v2_index}: "
                f"{type(baseline_layer).__name__} != {type(v2_layer).__name__}"
            )
        if local_state_shapes(baseline_layer) != local_state_shapes(v2_layer):
            errors.append(f"state-shape mismatch for layer {baseline_index}->{v2_index}")
    if errors:
        raise RuntimeError(f"Baseline and CSCEF-v2 unchanged layer structures differ: {errors}")
    if not isinstance(v2.model[18], CSCEFv2) or v2.model[18].f != [17, 16]:
        raise RuntimeError("Constructed model does not contain CSCEFv2([17, 16]) at layer 18.")
    if v2.model[19].f != [16, 18] or v2.model[-1].f != [20, 23, 26]:
        raise RuntimeError("Constructed CSCEF-v2 CCFM or decoder topology is incorrect.")
    print("unchanged_layer_types_and_state_shapes_identical=True")


def complexity_table(v2_cfg: Path) -> dict[str, RTDETRDetectionModel]:
    """Build nc=1 eval models, report model.info(imgsz=640), and enforce complexity limits."""
    models = {
        "R18-Lite baseline": RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=1, verbose=False).eval(),
        "R18-Lite + CSCEF-v1": RTDETRDetectionModel(str(CSCEF_V1_CFG), ch=3, nc=1, verbose=False).eval(),
        "R18-Lite + CSCEF-v2": RTDETRDetectionModel(str(v2_cfg), ch=3, nc=1, verbose=False).eval(),
    }
    if not isinstance(models["R18-Lite + CSCEF-v1"].model[18], CSCEF):
        raise RuntimeError("CSCEF-v1 model did not build with CSCEF at layer 18.")

    rows = {}
    for name, model in models.items():
        info = model.info(imgsz=640)
        if info is None or not info[3]:
            raise RuntimeError(f"model.info(imgsz=640) did not produce FLOPs for {name}.")
        rows[name] = info
    base_layers, base_params, _, base_flops = rows["R18-Lite baseline"]
    _, v1_params, _, _ = rows["R18-Lite + CSCEF-v1"]
    _, v2_params, _, v2_flops = rows["R18-Lite + CSCEF-v2"]
    v1_delta = v1_params - base_params
    v2_delta = v2_params - base_params
    v2_param_ratio = v2_delta / base_params
    v2_flops_ratio = (v2_flops - base_flops) / base_flops
    print("complexity_table:")
    print("model | layers | parameters | parameter_delta | parameter_delta_ratio | GFLOPs | GFLOPs_delta")
    for name, (layers, parameters, _, flops) in rows.items():
        print(
            f"{name} | {layers} | {parameters} | {parameters - base_params} | "
            f"{(parameters - base_params) / base_params:.6%} | {flops:.6f} | {flops - base_flops:.6f}"
        )
    if v2_delta >= v1_delta:
        raise RuntimeError(f"CSCEF-v2 parameter delta {v2_delta} is not below CSCEF-v1 delta {v1_delta}.")
    if v2_param_ratio >= 0.002:
        raise RuntimeError(f"CSCEF-v2 parameter increase {v2_param_ratio:.6%} is not below 0.2%.")
    if v2_flops_ratio >= 0.01:
        raise RuntimeError(f"CSCEF-v2 FLOPs increase {v2_flops_ratio:.6%} is not below 1%.")
    if base_layers <= 0:
        raise RuntimeError("Invalid baseline layer count from model.info().")
    print("complexity_constraints_passed=True")
    return models


def audit_checkpoint_mapping(
    source_path: Path,
    initialized_path: Path,
    baseline_model: RTDETRDetectionModel,
    v2_model: RTDETRDetectionModel,
    v2_nc1_model: RTDETRDetectionModel,
) -> None:
    """Verify source coverage, initialized values, clean checkpoint state, and nc=80-to-nc=1 load behavior."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Baseline initialization checkpoint not found: {source_path}")
    if not initialized_path.is_file():
        raise FileNotFoundError(f"Generated CSCEF-v2 initialization checkpoint not found: {initialized_path}")

    source_checkpoint = torch_load(source_path, map_location="cpu")
    source_state = checkpoint_model(source_checkpoint).state_dict()
    baseline_state = baseline_model.state_dict()
    target_state = v2_model.state_dict()
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

    target_parameters = dict(v2_model.named_parameters())
    mapped_parameter_keys = [key for key in plan.state if key in target_parameters]
    mapped_parameter_numel = sum(target_parameters[key].numel() for key in mapped_parameter_keys)
    unchanged_parameter_numel = sum(
        parameter.numel() for key, parameter in target_parameters.items() if not key.startswith("model.18.")
    )
    backbone_parameter_keys = [key for key in target_parameters if key.startswith(BACKBONE_PREFIXES)]
    mapped_backbone_parameter_keys = [key for key in backbone_parameter_keys if key in plan.state]
    backbone_parameter_numel = sum(target_parameters[key].numel() for key in backbone_parameter_keys)
    mapped_backbone_parameter_numel = sum(target_parameters[key].numel() for key in mapped_backbone_parameter_keys)

    nc1_state = v2_nc1_model.state_dict()
    transferred_nc1 = intersect_dicts(initialized_state, nc1_state)
    nc1_missing = sorted(set(nc1_state) - set(transferred_nc1))
    expected_decoder_missing = sorted(
        key
        for key in nc1_missing
        if key in initialized_state
        and key.startswith("model.27.")
        and initialized_state[key].shape != nc1_state[key].shape
    )
    unexpected_missing = sorted(set(nc1_missing) - set(expected_decoder_missing))
    unexpected_nc1_keys = sorted(set(initialized_state) - set(nc1_state))
    non_decoder_shape_mismatches = sorted(
        key
        for key in set(initialized_state) & set(nc1_state)
        if initialized_state[key].shape != nc1_state[key].shape and not key.startswith("model.27.")
    )

    clean_training_state = (
        initialized_checkpoint.get("epoch") == -1
        and initialized_checkpoint.get("ema") is None
        and initialized_checkpoint.get("optimizer") is None
        and initialized_checkpoint.get("scaler") is None
    )
    print(f"source_checkpoint_sha256={sha256(source_path)}")
    print(f"initialized_checkpoint_sha256={sha256(initialized_path)}")
    print(f"baseline_yaml_sha256={sha256(BASE_CFG)}")
    print(f"mapped_unchanged_state_keys={len(plan.state)}/{len(baseline_state)}")
    print(f"mapped_unchanged_state_numel={sum(value.numel() for value in plan.state.values())}")
    print(f"mapped_unchanged_parameter_keys={len(mapped_parameter_keys)}")
    print(f"mapped_unchanged_parameter_numel={mapped_parameter_numel}/{unchanged_parameter_numel}")
    print(
        f"backbone_parameter_key_coverage={len(mapped_backbone_parameter_keys)}/{len(backbone_parameter_keys)}"
    )
    print(f"backbone_parameter_numel_coverage={mapped_backbone_parameter_numel}/{backbone_parameter_numel}")
    print(f"all_mapped_values_exact={not mapped_value_mismatches}")
    print(f"mapping_shape_mismatches=[]")
    print(f"initialized_missing_keys={missing_initialized}")
    print(f"initialized_unexpected_keys={unexpected_initialized}")
    print(f"initialized_shape_mismatches={initialized_shape_mismatches}")
    print("new_cscef_v2_state_keys:")
    for key in plan.new_target_keys:
        print(f"  {key}")
    print("expected_nc80_to_nc1_decoder_missing_keys:")
    for key in expected_decoder_missing:
        print(f"  {key}: {tuple(initialized_state[key].shape)} -> {tuple(nc1_state[key].shape)}")
    print(f"unexpected_nc1_missing_keys={unexpected_missing}")
    print(f"unexpected_nc1_source_keys={unexpected_nc1_keys}")
    print(f"non_decoder_nc1_shape_mismatches={non_decoder_shape_mismatches}")
    print(f"clean_initialization_checkpoint_state={clean_training_state}")

    errors = []
    if len(plan.state) != len(baseline_state):
        errors.append("not every unchanged baseline state key was mapped")
    if mapped_parameter_numel != unchanged_parameter_numel:
        errors.append("not every unchanged parameter value was mapped")
    if len(mapped_backbone_parameter_keys) != len(backbone_parameter_keys):
        errors.append("backbone parameter key coverage is incomplete")
    if mapped_backbone_parameter_numel != backbone_parameter_numel:
        errors.append("backbone parameter-count coverage is incomplete")
    if missing_initialized or unexpected_initialized or initialized_shape_mismatches:
        errors.append("initialized checkpoint structure differs from the nc=80 CSCEF-v2 model")
    if mapped_value_mismatches:
        errors.append(f"mapped values differ: {mapped_value_mismatches}")
    if unexpected_missing or unexpected_nc1_keys or non_decoder_shape_mismatches:
        errors.append("nc=80 to nc=1 loading has unexpected key or shape differences")
    if not expected_decoder_missing:
        errors.append("nc=80 to nc=1 loading produced no expected decoder classification mismatch")
    if not clean_training_state:
        errors.append("generated checkpoint retained training state")
    if not isinstance(initialized_model.model[18], CSCEFv2):
        errors.append("generated checkpoint does not contain CSCEFv2 at layer 18")
    if errors:
        raise RuntimeError(f"CSCEF-v2 checkpoint audit failed: {errors}")
    print("strict_checkpoint_mapping_audit_passed=True")


def main() -> None:
    """Run all CSCEF-v2 audits and fail nonzero on any discrepancy."""
    args = parse_args()
    v2_cfg = args.model.resolve()
    source_path = args.source.resolve()
    initialized_path = args.initialized.resolve()
    imported_path = Path(ultralytics.__file__).resolve()
    print(f"ultralytics_version={ultralytics.__version__}")
    print(f"ultralytics_file={imported_path}")
    if not imported_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")

    verify_protected_files()
    verify_yaml_topology(v2_cfg)
    baseline_nc80 = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False).eval()
    v2_nc80 = RTDETRDetectionModel(str(v2_cfg), ch=3, nc=80, verbose=False).eval()
    verify_model_structure(baseline_nc80, v2_nc80)
    nc1_models = complexity_table(v2_cfg)
    audit_checkpoint_mapping(
        source_path,
        initialized_path,
        baseline_nc80,
        v2_nc80,
        nc1_models["R18-Lite + CSCEF-v2"],
    )
    print("CSCEF-v2 audit passed.")


if __name__ == "__main__":
    main()
