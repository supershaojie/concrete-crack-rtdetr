"""Create a strictly mapped CSCEF-v2 initialization checkpoint from the R18-Lite baseline checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
import re
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
BASE_CFG = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18-lite.yaml"
DEFAULT_SOURCE = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_MODEL = (
    ULTRALYTICS_ROOT
    / "ultralytics"
    / "cfg"
    / "models"
    / "rt-detr"
    / "rtdetr-resnet18-lite-cscef-v2.yaml"
)
DEFAULT_OUTPUT = ROOT / "weights" / "rtdetr_r18_lite_cscef_v2_imagenet_backbone_init.pt"
INSERTED_LAYER_INDEX = 18
BASE_LAST_LAYER_INDEX = 26

# Prefer this repository's Ultralytics package over any pip-installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics import RTDETR, __version__  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG_DICT  # noqa: E402
from ultralytics.nn.modules import CSCEFv2  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402


@dataclass(frozen=True)
class MappingPlan:
    """A validated baseline-to-CSCEF-v2 state mapping."""

    state: dict[str, torch.Tensor]
    source_to_target: dict[str, str]
    new_target_keys: list[str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Baseline initialization checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="CSCEF-v2 model YAML.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSCEF-v2 initialization checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic CSCEF-v2 initialization seed.")
    return parser.parse_args()


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
    """Map a baseline state key around the CSCEF-v2 insertion at model layer 18."""
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
    """Validate and construct the complete state mapping for every unchanged baseline layer."""
    errors: list[str] = []
    missing_source = sorted(set(baseline_state) - set(source_state))
    unexpected_source = sorted(set(source_state) - set(baseline_state))
    if missing_source:
        errors.extend(f"missing source key: {key}" for key in missing_source)
    if unexpected_source:
        errors.extend(f"unexpected source key: {key}" for key in unexpected_source)

    mapped_state: dict[str, torch.Tensor] = {}
    source_to_target: dict[str, str] = {}
    for source_key, baseline_value in baseline_state.items():
        if source_key not in source_state:
            continue
        source_value = source_state[source_key]
        if source_value.shape != baseline_value.shape:
            errors.append(
                f"source/baseline shape mismatch: {source_key}: "
                f"{tuple(source_value.shape)} != {tuple(baseline_value.shape)}"
            )
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
            errors.append(
                f"source/target shape mismatch: {source_key} -> {target_key}: "
                f"{tuple(source_value.shape)} != {tuple(target_state[target_key].shape)}"
            )
            continue
        source_to_target[source_key] = target_key
        mapped_state[target_key] = source_value

    new_target_keys = sorted(set(target_state) - set(mapped_state))
    unexpected_new_keys = [key for key in new_target_keys if not key.startswith(f"model.{INSERTED_LAYER_INDEX}.")]
    if unexpected_new_keys:
        errors.extend(f"unexpected unmapped target key: {key}" for key in unexpected_new_keys)
    if not new_target_keys:
        errors.append("CSCEF-v2 contributed no new state keys.")

    if errors:
        details = "\n  ".join(errors)
        raise RuntimeError(f"Strict baseline-to-CSCEF-v2 mapping failed:\n  {details}")
    return MappingPlan(mapped_state, source_to_target, new_target_keys)


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
    """Save a clean, directly trainable Ultralytics initialization checkpoint."""
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


def verify_saved_checkpoint(
    output: Path,
    model_cfg: Path,
    expected_model: RTDETRDetectionModel,
) -> None:
    """Verify direct checkpoint construction, YAML ``load()``, topology, and half-precision round-trip values."""
    expected_state = {key: value.detach().half().float() for key, value in expected_model.state_dict().items()}

    loaded = RTDETR(str(output))
    if not isinstance(loaded.model.model[INSERTED_LAYER_INDEX], CSCEFv2):
        raise RuntimeError("Direct RTDETR(checkpoint) reload did not preserve CSCEFv2 at layer 18.")
    loaded_state = loaded.model.state_dict()
    mismatches = [key for key in expected_state if not torch.equal(expected_state[key], loaded_state[key])]
    if mismatches:
        raise RuntimeError(f"Saved checkpoint round-trip value mismatches: {mismatches}")

    yaml_loaded = RTDETR(str(model_cfg))
    yaml_loaded.load(str(output))
    yaml_loaded_state = yaml_loaded.model.state_dict()
    load_mismatches = [key for key in loaded_state if not torch.equal(loaded_state[key], yaml_loaded_state[key])]
    if load_mismatches:
        raise RuntimeError(f"RTDETR(YAML).load(checkpoint) value mismatches: {load_mismatches}")


def initialize_from_baseline(source: Path, model_cfg: Path, output: Path, seed: int) -> dict[str, int]:
    """Build, strictly map, save, and reload a CSCEF-v2 initialization checkpoint."""
    imported_path = Path(ultralytics.__file__).resolve()
    if not imported_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")
    source = source.resolve()
    model_cfg = model_cfg.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Baseline initialization checkpoint not found: {source}")
    if not model_cfg.is_file():
        raise FileNotFoundError(f"CSCEF-v2 YAML not found: {model_cfg}")
    if source == output:
        raise ValueError("The CSCEF-v2 output path must not overwrite the baseline source checkpoint.")

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

    target_parameters = dict(target_model.named_parameters())
    mapped_parameter_keys = [key for key in plan.state if key in target_parameters]
    mapped_state_numel = sum(value.numel() for value in plan.state.values())
    mapped_parameter_numel = sum(target_parameters[key].numel() for key in mapped_parameter_keys)
    backbone_target_keys = [key for key in target_state if key.startswith(tuple(f"model.{i}." for i in range(8)))]
    mapped_backbone_keys = [key for key in backbone_target_keys if key in plan.state]

    print(f"ultralytics_file={imported_path}")
    print(f"source_checkpoint={source}")
    print(f"baseline_model={BASE_CFG}")
    print(f"cscef_v2_model={model_cfg}")
    print(f"output_checkpoint={output}")
    print(f"seed={seed}")
    print(f"mapped_state_keys={len(plan.state)}")
    print(f"mapped_state_numel={mapped_state_numel}")
    print(f"mapped_parameter_keys={len(mapped_parameter_keys)}")
    print(f"mapped_parameter_numel={mapped_parameter_numel}")
    print(f"backbone_state_key_coverage={len(mapped_backbone_keys)}/{len(backbone_target_keys)}")
    print(f"new_cscef_v2_state_key_count={len(plan.new_target_keys)}")
    print("new_cscef_v2_state_keys:")
    for key in plan.new_target_keys:
        print(f"  {key}")

    save_ultralytics_checkpoint(target_model, output, model_cfg)
    verify_saved_checkpoint(output, model_cfg, target_model)
    print("CSCEF-v2 strict initialization mapping and checkpoint reload verification passed.")
    return {
        "mapped_state_keys": len(plan.state),
        "mapped_state_numel": mapped_state_numel,
        "mapped_parameter_keys": len(mapped_parameter_keys),
        "mapped_parameter_numel": mapped_parameter_numel,
        "new_state_keys": len(plan.new_target_keys),
    }


def main() -> None:
    """Run strict CSCEF-v2 checkpoint initialization from CLI arguments."""
    args = parse_args()
    initialize_from_baseline(args.source, args.model, args.output, args.seed)


if __name__ == "__main__":
    main()
