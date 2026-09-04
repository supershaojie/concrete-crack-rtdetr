"""Create a controlled CSCEF-v4 checkpoint from baseline and the training-free CSCEF-v3 initialization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
MODEL_DIR = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
BASE_CFG = MODEL_DIR / "rtdetr-resnet18-lite.yaml"
DEFAULT_SOURCE = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_V3_INITIALIZED = ROOT / "weights" / "rtdetr_r18_lite_cscef_v3_imagenet_backbone_init.pt"
DEFAULT_MODEL = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v4.yaml"
DEFAULT_OUTPUT = ROOT / "weights" / "rtdetr_r18_lite_cscef_v4_imagenet_backbone_init.pt"
INSERTED_LAYER_INDEX = 18
BACKBONE_PREFIXES = tuple(f"model.{index}." for index in range(8))
MODULE_STATE_SUFFIXES = {
    "shared_projection.weight",
    "depthwise_conv.weight",
    "output_projection.weight",
    "raw_alpha",
    "scharr_x",
    "scharr_y",
}

# Prefer this repository's Ultralytics package over any pip-installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics import RTDETR  # noqa: E402
from ultralytics.nn.modules import CSCEFv3, CSCEFv4  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402

from init_rtdetr_r18_lite_cscef_v3_from_baseline import (  # noqa: E402
    build_strict_mapping,
    checkpoint_model,
    nc_shape_audit,
    save_ultralytics_checkpoint,
    set_initialization_seed,
    sha256,
    verify_mapped_values,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Baseline initialization checkpoint.")
    parser.add_argument(
        "--v3-initialized",
        type=Path,
        default=DEFAULT_V3_INITIALIZED,
        help="Training-free CSCEF-v3 initialization checkpoint used only for layer 18.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="CSCEF-v4 model YAML.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSCEF-v4 initialization checkpoint."
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic model-construction seed.")
    return parser.parse_args()


def clean_initialization_state(checkpoint: dict) -> bool:
    """Return whether a checkpoint contains no active training state."""
    return (
        checkpoint.get("epoch") == -1
        and checkpoint.get("ema") is None
        and checkpoint.get("optimizer") is None
        and checkpoint.get("scaler") is None
    )


def layer_state_keys(state: dict[str, torch.Tensor], layer_index: int = INSERTED_LAYER_INDEX) -> list[str]:
    """Return sorted state keys belonging directly to one parsed model layer."""
    prefix = f"model.{layer_index}."
    return sorted(key for key in state if key.startswith(prefix))


def verify_v3_module_source(checkpoint: dict, state: dict[str, torch.Tensor]) -> list[str]:
    """Validate the clean V3 initialization source and return its exact layer-18 keys."""
    model = checkpoint_model(checkpoint)
    if not isinstance(model.model[INSERTED_LAYER_INDEX], CSCEFv3):
        raise RuntimeError("The V3 initialization checkpoint does not contain CSCEFv3 at layer 18.")
    if not clean_initialization_state(checkpoint):
        raise RuntimeError("The V3 initialization reference contains active training state.")
    keys = layer_state_keys(state)
    suffixes = {key.split(f"model.{INSERTED_LAYER_INDEX}.", 1)[1] for key in keys}
    if suffixes != MODULE_STATE_SUFFIXES:
        raise RuntimeError(f"Unexpected V3 layer-18 state keys: {sorted(suffixes)}")
    return keys


def exact_value_mismatches(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor], keys: list[str]
) -> list[str]:
    """Return keys whose tensors are absent, shape-incompatible, or not exactly equal."""
    return [
        key
        for key in keys
        if key not in expected
        or key not in actual
        or expected[key].shape != actual[key].shape
        or not torch.equal(expected[key], actual[key])
    ]


def verify_saved_checkpoint(
    output: Path,
    model_cfg: Path,
    expected_model: RTDETRDetectionModel,
    v3_reference_state: dict[str, torch.Tensor],
    v3_module_keys: list[str],
) -> None:
    """Verify direct/YAML reload, state structure, V3 module equality, and clean checkpoint state."""
    expected_state = {key: value.detach().half().float() for key, value in expected_model.state_dict().items()}
    checkpoint = torch_load(output, map_location="cpu")
    clean_state = clean_initialization_state(checkpoint)
    loaded = RTDETR(str(output))
    if not isinstance(loaded.model.model[INSERTED_LAYER_INDEX], CSCEFv4):
        raise RuntimeError("Direct RTDETR(checkpoint) reload did not preserve CSCEFv4 at layer 18.")
    loaded_state = loaded.model.state_dict()
    missing = sorted(set(expected_state) - set(loaded_state))
    unexpected = sorted(set(loaded_state) - set(expected_state))
    shape_mismatches = sorted(
        key for key in expected_state if key in loaded_state and expected_state[key].shape != loaded_state[key].shape
    )
    value_mismatches = exact_value_mismatches(expected_state, loaded_state, sorted(expected_state))
    v3_module_mismatches = exact_value_mismatches(v3_reference_state, loaded_state, v3_module_keys)
    print(f"initialized_missing_keys={missing}")
    print(f"initialized_unexpected_keys={unexpected}")
    print(f"initialized_shape_mismatches={shape_mismatches}")
    print(f"initialized_value_mismatches={value_mismatches}")
    print(f"v4_module_equals_v3_init_exact={not v3_module_mismatches}")
    print(f"v4_module_v3_init_mismatches={v3_module_mismatches}")
    print(f"clean_initialization_checkpoint_state={clean_state}")
    if missing or unexpected or shape_mismatches or value_mismatches or v3_module_mismatches or not clean_state:
        raise RuntimeError("Saved CSCEF-v4 checkpoint failed exact round-trip verification.")

    yaml_loaded = RTDETR(str(model_cfg))
    yaml_loaded.load(str(output))
    yaml_loaded_state = yaml_loaded.model.state_dict()
    yaml_load_mismatches = exact_value_mismatches(loaded_state, yaml_loaded_state, sorted(loaded_state))
    if yaml_load_mismatches:
        raise RuntimeError(f"RTDETR(YAML).load(checkpoint) value mismatches: {yaml_load_mismatches}")
    nc1_model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=1, verbose=False)
    nc_shape_audit(loaded_state, nc1_model.state_dict())


def initialize_controlled(
    source: Path,
    v3_initialized: Path,
    model_cfg: Path,
    output: Path,
    seed: int,
) -> dict[str, int]:
    """Map unchanged baseline tensors and copy the complete V3 layer-18 initialization into V4."""
    imported_path = Path(ultralytics.__file__).resolve()
    if not imported_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")
    source = source.resolve()
    v3_initialized = v3_initialized.resolve()
    model_cfg = model_cfg.resolve()
    output = output.resolve()
    for path, label in ((source, "baseline checkpoint"), (v3_initialized, "V3 initialization checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} not found: {path}")
    if not model_cfg.is_file():
        raise FileNotFoundError(f"CSCEF-v4 YAML not found: {model_cfg}")
    if output in {source, v3_initialized}:
        raise ValueError("The CSCEF-v4 output must not overwrite either initialization source.")

    source_checkpoint = torch_load(source, map_location="cpu")
    source_state = checkpoint_model(source_checkpoint).state_dict()
    v3_checkpoint = torch_load(v3_initialized, map_location="cpu")
    v3_reference_state = checkpoint_model(v3_checkpoint).state_dict()
    v3_module_keys = verify_v3_module_source(v3_checkpoint, v3_reference_state)

    baseline_model = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False)
    baseline_state = baseline_model.state_dict()
    set_initialization_seed(seed)
    target_model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=80, verbose=False)
    target_state = target_model.state_dict()
    plan = build_strict_mapping(source_state, baseline_state, target_state)
    if plan.new_target_keys != v3_module_keys:
        raise RuntimeError(
            f"V4 new state does not exactly match V3 layer 18: V4={plan.new_target_keys}, V3={v3_module_keys}"
        )

    module_shape_mismatches = sorted(
        key
        for key in v3_module_keys
        if key not in target_state or v3_reference_state[key].shape != target_state[key].shape
    )
    if module_shape_mismatches:
        raise RuntimeError(f"V3/V4 module shape mismatches: {module_shape_mismatches}")
    complete_state = {**plan.state, **{key: v3_reference_state[key] for key in v3_module_keys}}
    pre_missing = sorted(set(target_state) - set(complete_state))
    pre_unexpected = sorted(set(complete_state) - set(target_state))
    pre_shape_mismatches = sorted(
        key for key in target_state if key in complete_state and target_state[key].shape != complete_state[key].shape
    )
    print(f"preload_missing_keys={pre_missing}")
    print(f"preload_unexpected_keys={pre_unexpected}")
    print(f"preload_shape_mismatches={pre_shape_mismatches}")
    if pre_missing or pre_unexpected or pre_shape_mismatches:
        raise RuntimeError("Controlled source state does not strictly cover the CSCEF-v4 target.")
    target_model.load_state_dict(complete_state, strict=True)
    loaded_target_state = target_model.state_dict()

    baseline_mismatches = verify_mapped_values(source_state, loaded_target_state, plan.source_to_target)
    module_mismatches = exact_value_mismatches(v3_reference_state, loaded_target_state, v3_module_keys)
    target_parameters = dict(target_model.named_parameters())
    unchanged_parameter_numel = sum(
        parameter.numel() for key, parameter in target_parameters.items() if not key.startswith("model.18.")
    )
    mapped_parameter_numel = sum(
        target_parameters[key].numel() for key in plan.state if key in target_parameters
    )
    backbone_parameter_keys = [key for key in target_parameters if key.startswith(BACKBONE_PREFIXES)]
    mapped_backbone_keys = [key for key in backbone_parameter_keys if key in plan.state]
    backbone_numel = sum(target_parameters[key].numel() for key in backbone_parameter_keys)
    mapped_backbone_numel = sum(target_parameters[key].numel() for key in mapped_backbone_keys)
    module = target_model.model[INSERTED_LAYER_INDEX]
    if not isinstance(module, CSCEFv4):
        raise RuntimeError("Target model does not contain CSCEFv4 at layer 18.")
    module_parameters = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

    print(f"ultralytics_file={imported_path}")
    print(f"baseline_source={source}")
    print(f"baseline_source_sha256={sha256(source)}")
    print(f"v3_initialization_source={v3_initialized}")
    print(f"v3_initialization_source_sha256={sha256(v3_initialized)}")
    print(f"cscef_v4_model={model_cfg}")
    print(f"output_checkpoint={output}")
    print(f"mapped_unchanged_state_keys={len(plan.state)}/{len(baseline_state)}")
    print(f"mapped_unchanged_parameter_numel={mapped_parameter_numel}/{unchanged_parameter_numel}")
    print(f"mapping_shape_mismatches={plan.shape_mismatches}")
    print(f"unchanged_baseline_values_exact={not baseline_mismatches}")
    print(f"backbone_parameter_key_coverage={len(mapped_backbone_keys)}/{len(backbone_parameter_keys)}")
    print(f"backbone_parameter_numel_coverage={mapped_backbone_numel}/{backbone_numel}")
    print(f"v4_module_trainable_parameters={module_parameters}")
    print(f"v4_module_equals_v3_init_exact={not module_mismatches}")
    print(f"v4_module_v3_init_mismatches={module_mismatches}")
    if (
        len(plan.state) != len(baseline_state)
        or mapped_parameter_numel != unchanged_parameter_numel
        or plan.shape_mismatches
        or baseline_mismatches
        or len(mapped_backbone_keys) != len(backbone_parameter_keys)
        or mapped_backbone_numel != backbone_numel
        or module_parameters != 16673
        or module_mismatches
    ):
        raise RuntimeError("Controlled CSCEF-v4 initialization failed a pre-save invariant.")

    save_ultralytics_checkpoint(target_model, output, model_cfg)
    verify_saved_checkpoint(output, model_cfg, target_model, v3_reference_state, v3_module_keys)
    print(f"initialized_checkpoint_sha256={sha256(output)}")
    print("CSCEF-v4 controlled initialization and exact reload verification passed.")
    return {
        "mapped_state_keys": len(plan.state),
        "mapped_parameter_numel": mapped_parameter_numel,
        "backbone_parameter_numel": backbone_numel,
        "new_state_keys": len(v3_module_keys),
        "new_trainable_parameters": module_parameters,
    }


def main() -> None:
    """Run controlled CSCEF-v4 initialization from CLI arguments."""
    args = parse_args()
    initialize_controlled(args.source, args.v3_initialized, args.model, args.output, args.seed)


if __name__ == "__main__":
    main()
