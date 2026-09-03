"""Create a strictly mapped CSCEF-v3 initialization checkpoint from the R18-Lite baseline checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import random
import re
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
MODEL_DIR = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
BASE_CFG = MODEL_DIR / "rtdetr-resnet18-lite.yaml"
DEFAULT_SOURCE = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_MODEL = MODEL_DIR / "rtdetr-resnet18-lite-cscef-v3.yaml"
DEFAULT_OUTPUT = ROOT / "weights" / "rtdetr_r18_lite_cscef_v3_imagenet_backbone_init.pt"
INSERTED_LAYER_INDEX = 18
BASE_LAST_LAYER_INDEX = 26
TARGET_LAST_LAYER_INDEX = 27
BACKBONE_PREFIXES = tuple(f"model.{index}." for index in range(8))

# Prefer this repository's Ultralytics package over any pip-installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics import RTDETR, __version__  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG_DICT  # noqa: E402
from ultralytics.nn.modules import CSCEFv3  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402


@dataclass(frozen=True)
class MappingPlan:
    """A validated baseline-to-CSCEF-v3 state mapping."""

    state: dict[str, torch.Tensor]
    source_to_target: dict[str, str]
    new_target_keys: list[str]
    shape_mismatches: list[str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Baseline initialization checkpoint.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="CSCEF-v3 model YAML.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSCEF-v3 initialization checkpoint."
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic CSCEF-v3 initialization seed.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_initialization_seed(seed: int) -> None:
    """Seed Python, NumPy when available, and PyTorch before constructing the target model."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_model(checkpoint: dict) -> torch.nn.Module:
    """Extract and validate the model stored in an Ultralytics checkpoint."""
    if not isinstance(checkpoint, dict):
        raise TypeError("The source checkpoint must be an Ultralytics checkpoint dictionary.")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The source checkpoint does not contain a torch.nn.Module under 'ema' or 'model'.")
    return model.float()


def remap_baseline_key(source_key: str) -> str:
    """Map a baseline state key around the CSCEF-v3 insertion at model layer 18."""
    match = re.fullmatch(r"model\.(\d+)(\..+)", source_key)
    if match is None:
        raise ValueError(f"Cannot semantically map non-layer baseline state key: {source_key}")
    layer_index = int(match.group(1))
    if layer_index <= 17:
        target_index = layer_index
    elif layer_index <= BASE_LAST_LAYER_INDEX:
        target_index = layer_index + 1
    else:
        raise ValueError(f"Unexpected baseline layer index {layer_index} in state key: {source_key}")
    return f"model.{target_index}{match.group(2)}"


def build_strict_mapping(
    source_state: dict[str, torch.Tensor],
    baseline_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> MappingPlan:
    """Validate and map every unchanged baseline state tensor into the CSCEF-v3 graph."""
    errors: list[str] = []
    shape_mismatches: list[str] = []
    missing_source = sorted(set(baseline_state) - set(source_state))
    unexpected_source = sorted(set(source_state) - set(baseline_state))
    errors.extend(f"missing source key: {key}" for key in missing_source)
    errors.extend(f"unexpected source key: {key}" for key in unexpected_source)

    mapped_state: dict[str, torch.Tensor] = {}
    source_to_target: dict[str, str] = {}
    for source_key, baseline_value in baseline_state.items():
        if source_key not in source_state:
            continue
        source_value = source_state[source_key]
        if source_value.shape != baseline_value.shape:
            mismatch = (
                f"source/baseline: {source_key}: {tuple(source_value.shape)} != {tuple(baseline_value.shape)}"
            )
            shape_mismatches.append(mismatch)
            errors.append(f"shape mismatch: {mismatch}")
            continue
        try:
            target_key = remap_baseline_key(source_key)
        except ValueError as error:
            errors.append(str(error))
            continue
        if target_key not in target_state:
            errors.append(f"missing target key: {source_key} -> {target_key}")
            continue
        if source_value.shape != target_state[target_key].shape:
            mismatch = (
                f"source/target: {source_key} -> {target_key}: "
                f"{tuple(source_value.shape)} != {tuple(target_state[target_key].shape)}"
            )
            shape_mismatches.append(mismatch)
            errors.append(f"shape mismatch: {mismatch}")
            continue
        source_to_target[source_key] = target_key
        mapped_state[target_key] = source_value

    new_target_keys = sorted(set(target_state) - set(mapped_state))
    unexpected_new_keys = [key for key in new_target_keys if not key.startswith(f"model.{INSERTED_LAYER_INDEX}.")]
    errors.extend(f"unexpected unmapped target key: {key}" for key in unexpected_new_keys)
    if not new_target_keys:
        errors.append("CSCEF-v3 contributed no new state keys.")
    if errors:
        details = "\n  ".join(errors)
        raise RuntimeError(f"Strict baseline-to-CSCEF-v3 mapping failed:\n  {details}")
    return MappingPlan(mapped_state, source_to_target, new_target_keys, shape_mismatches)


def verify_mapped_values(
    source_state: dict[str, torch.Tensor],
    loaded_state: dict[str, torch.Tensor],
    source_to_target: dict[str, str],
) -> list[str]:
    """Return mappings whose loaded target values are not exactly equal to their sources."""
    return [
        f"{source_key} -> {target_key}"
        for source_key, target_key in source_to_target.items()
        if not torch.equal(source_state[source_key], loaded_state[target_key])
    ]


def save_ultralytics_checkpoint(model: RTDETRDetectionModel, output: Path, model_cfg: Path) -> None:
    """Save a clean, directly trainable Ultralytics model-initialization checkpoint."""
    model.eval()
    model.args = {**DEFAULT_CFG_DICT, "model": str(model_cfg), "task": "detect"}
    model.task = "detect"
    model.pt_path = str(output)
    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(model).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "scaler": None,
        "train_args": model.args,
        "train_metrics": None,
        "train_results": None,
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


def nc_shape_audit(initialized_state: dict[str, torch.Tensor], nc1_state: dict[str, torch.Tensor]) -> list[str]:
    """Validate that nc=80 to nc=1 differences are confined to decoder classification tensors."""
    missing = sorted(set(nc1_state) - set(initialized_state))
    unexpected = sorted(set(initialized_state) - set(nc1_state))
    shape_mismatches = sorted(
        key
        for key in set(initialized_state) & set(nc1_state)
        if initialized_state[key].shape != nc1_state[key].shape
    )
    expected = [key for key in shape_mismatches if key.startswith(f"model.{TARGET_LAST_LAYER_INDEX}.")]
    unexpected_mismatches = sorted(set(shape_mismatches) - set(expected))
    print(f"nc1_missing_state_keys={missing}")
    print(f"nc1_unexpected_state_keys={unexpected}")
    print("expected_nc80_to_nc1_decoder_shape_changes:")
    for key in expected:
        print(f"  {key}: {tuple(initialized_state[key].shape)} -> {tuple(nc1_state[key].shape)}")
    print(f"unexpected_nc80_to_nc1_shape_mismatches={unexpected_mismatches}")
    if missing or unexpected or unexpected_mismatches or not expected:
        raise RuntimeError("nc=80 to nc=1 state audit found an unexpected key or shape difference.")
    return expected


def verify_saved_checkpoint(
    output: Path,
    model_cfg: Path,
    expected_model: RTDETRDetectionModel,
) -> None:
    """Verify direct checkpoint reload, YAML ``load()``, structure, values, and clean training state."""
    expected_state = {key: value.detach().half().float() for key, value in expected_model.state_dict().items()}
    checkpoint = torch_load(output, map_location="cpu")
    clean_training_state = (
        checkpoint.get("epoch") == -1
        and checkpoint.get("ema") is None
        and checkpoint.get("optimizer") is None
        and checkpoint.get("scaler") is None
    )
    if not clean_training_state:
        raise RuntimeError("Generated checkpoint retained optimizer, EMA, scaler, or active epoch state.")

    loaded = RTDETR(str(output))
    if not isinstance(loaded.model.model[INSERTED_LAYER_INDEX], CSCEFv3):
        raise RuntimeError("Direct RTDETR(checkpoint) reload did not preserve CSCEFv3 at layer 18.")
    loaded_state = loaded.model.state_dict()
    missing = sorted(set(expected_state) - set(loaded_state))
    unexpected = sorted(set(loaded_state) - set(expected_state))
    shape_mismatches = sorted(
        key for key in expected_state if key in loaded_state and expected_state[key].shape != loaded_state[key].shape
    )
    value_mismatches = [
        key for key in expected_state if key in loaded_state and not torch.equal(expected_state[key], loaded_state[key])
    ]
    print(f"initialized_missing_keys={missing}")
    print(f"initialized_unexpected_keys={unexpected}")
    print(f"initialized_shape_mismatches={shape_mismatches}")
    print(f"initialized_value_mismatches={value_mismatches}")
    print(f"clean_initialization_checkpoint_state={clean_training_state}")
    if missing or unexpected or shape_mismatches or value_mismatches:
        raise RuntimeError("Saved checkpoint round-trip did not exactly preserve the nc=80 target state.")

    yaml_loaded = RTDETR(str(model_cfg))
    yaml_loaded.load(str(output))
    yaml_loaded_state = yaml_loaded.model.state_dict()
    load_mismatches = [key for key in loaded_state if not torch.equal(loaded_state[key], yaml_loaded_state[key])]
    if load_mismatches:
        raise RuntimeError(f"RTDETR(YAML).load(checkpoint) value mismatches: {load_mismatches}")
    nc1_model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=1, verbose=False)
    nc_shape_audit(loaded_state, nc1_model.state_dict())


def initialize_from_baseline(source: Path, model_cfg: Path, output: Path, seed: int) -> dict[str, int]:
    """Build, strictly map, save, and reload a deterministic CSCEF-v3 initialization checkpoint."""
    imported_path = Path(ultralytics.__file__).resolve()
    if not imported_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")
    source = source.resolve()
    model_cfg = model_cfg.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Baseline initialization checkpoint not found: {source}")
    if not model_cfg.is_file():
        raise FileNotFoundError(f"CSCEF-v3 YAML not found: {model_cfg}")
    if source == output:
        raise ValueError("The CSCEF-v3 output path must not overwrite the baseline source checkpoint.")

    checkpoint = torch_load(source, map_location="cpu")
    source_model = checkpoint_model(checkpoint)
    source_state = source_model.state_dict()
    baseline_model = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False)
    baseline_state = baseline_model.state_dict()

    set_initialization_seed(seed)
    target_model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=80, verbose=False)
    target_state = target_model.state_dict()
    plan = build_strict_mapping(source_state, baseline_state, target_state)
    load_result = target_model.load_state_dict(plan.state, strict=False)
    if sorted(load_result.missing_keys) != plan.new_target_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Strict mapped load produced an unexpected result: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    value_mismatches = verify_mapped_values(source_state, target_model.state_dict(), plan.source_to_target)
    if value_mismatches:
        raise RuntimeError(f"Mapped values differ after load: {value_mismatches}")

    module = target_model.model[INSERTED_LAYER_INDEX]
    if not isinstance(module, CSCEFv3):
        raise RuntimeError("Target model does not contain CSCEFv3 at layer 18.")
    module_parameters = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    if module_parameters != 16673:
        raise RuntimeError(f"CSCEF-v3 trainable parameter count is {module_parameters}, expected 16673.")
    if abs(module.raw_alpha.item() - (-1.38629436112)) > 1e-6:
        raise RuntimeError(f"Unexpected raw_alpha initialization: {module.raw_alpha.item()}")

    target_parameters = dict(target_model.named_parameters())
    mapped_parameter_keys = [key for key in plan.state if key in target_parameters]
    mapped_state_numel = sum(value.numel() for value in plan.state.values())
    mapped_parameter_numel = sum(target_parameters[key].numel() for key in mapped_parameter_keys)
    backbone_target_keys = [key for key in target_state if key.startswith(BACKBONE_PREFIXES)]
    mapped_backbone_keys = [key for key in backbone_target_keys if key in plan.state]
    backbone_parameter_keys = [key for key in target_parameters if key.startswith(BACKBONE_PREFIXES)]
    mapped_backbone_parameter_keys = [key for key in backbone_parameter_keys if key in plan.state]
    backbone_parameter_numel = sum(target_parameters[key].numel() for key in backbone_parameter_keys)
    mapped_backbone_parameter_numel = sum(target_parameters[key].numel() for key in mapped_backbone_parameter_keys)

    print(f"ultralytics_file={imported_path}")
    print(f"source_checkpoint={source}")
    print(f"source_checkpoint_sha256={sha256(source)}")
    print(f"baseline_model={BASE_CFG}")
    print(f"cscef_v3_model={model_cfg}")
    print(f"output_checkpoint={output}")
    print(f"seed={seed}")
    print(f"mapped_state_keys={len(plan.state)}/{len(baseline_state)}")
    print(f"mapped_state_numel={mapped_state_numel}")
    print(f"mapped_parameter_keys={len(mapped_parameter_keys)}")
    print(f"mapped_parameter_numel={mapped_parameter_numel}")
    print(f"mapping_shape_mismatches={plan.shape_mismatches}")
    print(f"backbone_state_key_coverage={len(mapped_backbone_keys)}/{len(backbone_target_keys)}")
    print(
        f"backbone_parameter_key_coverage="
        f"{len(mapped_backbone_parameter_keys)}/{len(backbone_parameter_keys)}"
    )
    print(f"backbone_parameter_numel_coverage={mapped_backbone_parameter_numel}/{backbone_parameter_numel}")
    print(f"new_cscef_v3_trainable_parameters={module_parameters}")
    print(f"raw_alpha_initial={module.raw_alpha.item():.12f}")
    print(f"effective_alpha_initial={module._effective_alpha().item():.12f}")
    print(f"new_cscef_v3_state_key_count={len(plan.new_target_keys)}")
    print("new_cscef_v3_state_keys:")
    for key in plan.new_target_keys:
        print(f"  {key}")

    if len(mapped_backbone_keys) != len(backbone_target_keys):
        raise RuntimeError("Backbone state-key mapping coverage is incomplete.")
    if len(mapped_backbone_parameter_keys) != len(backbone_parameter_keys):
        raise RuntimeError("Backbone parameter-key mapping coverage is incomplete.")
    if mapped_backbone_parameter_numel != backbone_parameter_numel:
        raise RuntimeError("Backbone parameter-count mapping coverage is incomplete.")

    save_ultralytics_checkpoint(target_model, output, model_cfg)
    verify_saved_checkpoint(output, model_cfg, target_model)
    print(f"initialized_checkpoint_sha256={sha256(output)}")
    print("CSCEF-v3 strict initialization mapping and checkpoint reload verification passed.")
    return {
        "mapped_state_keys": len(plan.state),
        "mapped_state_numel": mapped_state_numel,
        "mapped_parameter_keys": len(mapped_parameter_keys),
        "mapped_parameter_numel": mapped_parameter_numel,
        "new_state_keys": len(plan.new_target_keys),
        "new_trainable_parameters": module_parameters,
    }


def main() -> None:
    """Run strict CSCEF-v3 checkpoint initialization from CLI arguments."""
    args = parse_args()
    initialize_from_baseline(args.source, args.model, args.output, args.seed)


if __name__ == "__main__":
    main()
