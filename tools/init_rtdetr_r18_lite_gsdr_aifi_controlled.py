"""Create a strictly mapped GSDR-AIFI initialization checkpoint from the R18-Lite baseline checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import random
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
MODEL_DIR = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
BASE_CFG = MODEL_DIR / "rtdetr-resnet18-lite.yaml"
DEFAULT_SOURCE = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
DEFAULT_MODEL = MODEL_DIR / "rtdetr-resnet18-lite-gsdr-aifi.yaml"
DEFAULT_OUTPUT = ROOT / "weights" / "rtdetr_r18_lite_gsdr_aifi_imagenet_backbone_init.pt"
AIFI_LAYER_INDEX = 9
DECODER_LAYER_INDEX = 26
BACKBONE_PREFIXES = tuple(f"model.{index}." for index in range(8))
SPARSE_PREFIX = f"model.{AIFI_LAYER_INDEX}.sparse_relation."
AIFI_COMPONENTS = ("ma.", "fc1.", "fc2.", "norm1.", "norm2.")
CLASSIFICATION_COMPONENTS = ("denoising_class_embed.", "enc_score_head.", "dec_score_head.")

# Prefer this repository's Ultralytics package over any pip-installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics import RTDETR, __version__  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG_DICT  # noqa: E402
from ultralytics.nn.modules import AIFI, GSDRAIFI  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402


@dataclass(frozen=True)
class MappingPlan:
    """A validated identity mapping from baseline state to unchanged GSDR-AIFI state."""

    state: dict[str, torch.Tensor]
    source_to_target: dict[str, str]
    new_target_keys: list[str]
    shape_mismatches: list[str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Baseline initialization checkpoint.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="GSDR-AIFI model YAML.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output initialization checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic target model-construction seed.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_initialization_seed(seed: int) -> None:
    """Seed Python, NumPy when available, and PyTorch before model construction."""
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
        raise TypeError("The checkpoint must be an Ultralytics checkpoint dictionary.")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The checkpoint does not contain a torch.nn.Module under 'ema' or 'model'.")
    return model.float()


def clean_initialization_state(checkpoint: dict) -> bool:
    """Return whether a checkpoint contains no active training state."""
    return (
        checkpoint.get("epoch") == -1
        and checkpoint.get("ema") is None
        and checkpoint.get("updates") is None
        and checkpoint.get("optimizer") is None
        and checkpoint.get("scaler") is None
    )


def build_strict_mapping(
    source_state: dict[str, torch.Tensor],
    baseline_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> MappingPlan:
    """Map every baseline tensor by identical key and shape, leaving only the new sparse branch."""
    errors: list[str] = []
    shape_mismatches: list[str] = []
    missing_source = sorted(set(baseline_state) - set(source_state))
    unexpected_source = sorted(set(source_state) - set(baseline_state))
    errors.extend(f"missing source key: {key}" for key in missing_source)
    errors.extend(f"unexpected source key: {key}" for key in unexpected_source)

    mapped_state: dict[str, torch.Tensor] = {}
    source_to_target: dict[str, str] = {}
    for key, baseline_value in baseline_state.items():
        if key not in source_state:
            continue
        source_value = source_state[key]
        if source_value.shape != baseline_value.shape:
            mismatch = f"source/baseline {key}: {tuple(source_value.shape)} != {tuple(baseline_value.shape)}"
            shape_mismatches.append(mismatch)
            errors.append(f"shape mismatch: {mismatch}")
            continue
        if key not in target_state:
            errors.append(f"unchanged target key missing: {key}")
            continue
        if source_value.shape != target_state[key].shape:
            mismatch = f"source/target {key}: {tuple(source_value.shape)} != {tuple(target_state[key].shape)}"
            shape_mismatches.append(mismatch)
            errors.append(f"shape mismatch: {mismatch}")
            continue
        source_to_target[key] = key
        mapped_state[key] = source_value

    new_target_keys = sorted(set(target_state) - set(mapped_state))
    unexpected_new = [key for key in new_target_keys if not key.startswith(SPARSE_PREFIX)]
    errors.extend(f"unexpected unmapped target key: {key}" for key in unexpected_new)
    if not new_target_keys:
        errors.append("GSDR-AIFI contributed no new state keys.")
    if errors:
        raise RuntimeError("Strict baseline-to-GSDR mapping failed:\n  " + "\n  ".join(errors))
    return MappingPlan(mapped_state, source_to_target, new_target_keys, shape_mismatches)


def verify_mapped_values(
    source_state: dict[str, torch.Tensor],
    loaded_state: dict[str, torch.Tensor],
    source_to_target: dict[str, str],
) -> list[str]:
    """Return mappings whose loaded target tensors are not exactly equal to their source tensors."""
    return [
        f"{source_key} -> {target_key}"
        for source_key, target_key in source_to_target.items()
        if not torch.equal(source_state[source_key], loaded_state[target_key])
    ]


def original_aifi_keys(state: dict[str, torch.Tensor]) -> list[str]:
    """Return all original dense-attention, FFN, and norm state keys at model layer 9."""
    prefix = f"model.{AIFI_LAYER_INDEX}."
    return sorted(key for key in state if key.startswith(prefix) and key[len(prefix) :].startswith(AIFI_COMPONENTS))


def verify_zero_initialized_boundaries(module: GSDRAIFI) -> None:
    """Verify the three zero-initialized boundaries required by the design."""
    relation = module.sparse_relation
    tensors = {
        "output_proj.weight": relation.output_proj.weight,
        "output_proj.bias": relation.output_proj.bias,
        "offset_out.weight": relation.offset_out.weight,
        "offset_out.bias": relation.offset_out.bias,
    }
    for group, mlp in enumerate(relation.relative_bias):
        tensors[f"relative_bias.{group}.fc2.weight"] = mlp.fc2.weight
        tensors[f"relative_bias.{group}.fc2.bias"] = mlp.fc2.bias
    nonzero = [name for name, tensor in tensors.items() if torch.count_nonzero(tensor).item()]
    if nonzero:
        raise RuntimeError(f"Required zero-initialized sparse boundaries are nonzero: {nonzero}")


def nc_shape_audit(nc80_state: dict[str, torch.Tensor], nc1_state: dict[str, torch.Tensor]) -> list[str]:
    """Require nc=80 to nc=1 differences to be decoder classification tensors only."""
    missing = sorted(set(nc1_state) - set(nc80_state))
    unexpected = sorted(set(nc80_state) - set(nc1_state))
    mismatches = sorted(
        key for key in set(nc80_state) & set(nc1_state) if nc80_state[key].shape != nc1_state[key].shape
    )
    decoder_prefix = f"model.{DECODER_LAYER_INDEX}."
    unexpected_mismatches = [
        key
        for key in mismatches
        if not key.startswith(decoder_prefix)
        or not key[len(decoder_prefix) :].startswith(CLASSIFICATION_COMPONENTS)
    ]
    print(f"nc1_missing_state_keys={missing}")
    print(f"nc1_unexpected_state_keys={unexpected}")
    print("expected_nc80_to_nc1_classification_shape_changes:")
    for key in mismatches:
        print(f"  {key}: {tuple(nc80_state[key].shape)} -> {tuple(nc1_state[key].shape)}")
    print(f"unexpected_nc80_to_nc1_shape_mismatches={unexpected_mismatches}")
    if missing or unexpected or unexpected_mismatches or not mismatches:
        raise RuntimeError("nc=80 to nc=1 state audit found an unexpected key or shape difference.")
    return mismatches


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


def exact_value_mismatches(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor], keys: list[str]
) -> list[str]:
    """Return keys that are absent, shape-incompatible, or not exactly equal."""
    return [
        key
        for key in keys
        if key not in expected
        or key not in actual
        or expected[key].shape != actual[key].shape
        or not torch.equal(expected[key], actual[key])
    ]


def verify_saved_checkpoint(output: Path, model_cfg: Path, expected_model: RTDETRDetectionModel) -> None:
    """Verify direct/YAML reloads, exact values, clean state, zero boundaries, and nc changes."""
    expected_state = {key: value.detach().half().float() for key, value in expected_model.state_dict().items()}
    checkpoint = torch_load(output, map_location="cpu")
    clean_state = clean_initialization_state(checkpoint)
    loaded = RTDETR(str(output))
    if not isinstance(loaded.model.model[AIFI_LAYER_INDEX], GSDRAIFI):
        raise RuntimeError("Direct RTDETR(checkpoint) reload did not preserve GSDRAIFI at layer 9.")
    loaded_state = loaded.model.state_dict()
    missing = sorted(set(expected_state) - set(loaded_state))
    unexpected = sorted(set(loaded_state) - set(expected_state))
    shape_mismatches = sorted(
        key for key in expected_state if key in loaded_state and expected_state[key].shape != loaded_state[key].shape
    )
    value_mismatches = exact_value_mismatches(expected_state, loaded_state, sorted(expected_state))
    verify_zero_initialized_boundaries(loaded.model.model[AIFI_LAYER_INDEX])
    print(f"initialized_missing_keys={missing}")
    print(f"initialized_unexpected_keys={unexpected}")
    print(f"initialized_shape_mismatches={shape_mismatches}")
    print(f"initialized_value_mismatches={value_mismatches}")
    print(f"clean_initialization_checkpoint_state={clean_state}")
    if missing or unexpected or shape_mismatches or value_mismatches or not clean_state:
        raise RuntimeError("Saved GSDR-AIFI checkpoint failed exact round-trip verification.")

    yaml_loaded = RTDETR(str(model_cfg))
    yaml_loaded.load(str(output))
    yaml_state = yaml_loaded.model.state_dict()
    yaml_mismatches = exact_value_mismatches(loaded_state, yaml_state, sorted(loaded_state))
    print(f"checkpoint_yaml_load_exact={not yaml_mismatches}")
    if yaml_mismatches:
        raise RuntimeError(f"RTDETR(YAML).load(checkpoint) value mismatches: {yaml_mismatches}")
    nc1_model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=1, verbose=False)
    nc_shape_audit(loaded_state, nc1_model.state_dict())


def initialize_from_baseline(source: Path, model_cfg: Path, output: Path, seed: int) -> dict[str, int]:
    """Build, strictly map, save, and reload a deterministic GSDR-AIFI initialization checkpoint."""
    imported_path = Path(ultralytics.__file__).resolve()
    if not imported_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")
    source = source.resolve()
    model_cfg = model_cfg.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Baseline initialization checkpoint not found: {source}")
    if not model_cfg.is_file():
        raise FileNotFoundError(f"GSDR-AIFI YAML not found: {model_cfg}")
    if source == output:
        raise ValueError("The GSDR-AIFI output path must not overwrite the baseline source checkpoint.")

    checkpoint = torch_load(source, map_location="cpu")
    if not clean_initialization_state(checkpoint):
        raise RuntimeError("Baseline source checkpoint contains active training state.")
    source_model = checkpoint_model(checkpoint)
    if not isinstance(source_model.model[AIFI_LAYER_INDEX], AIFI) or isinstance(
        source_model.model[AIFI_LAYER_INDEX], GSDRAIFI
    ):
        raise RuntimeError("Baseline checkpoint does not contain the original AIFI at model layer 9.")
    source_state = source_model.state_dict()
    baseline_model = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False)
    baseline_state = baseline_model.state_dict()

    set_initialization_seed(seed)
    target_model = RTDETRDetectionModel(str(model_cfg), ch=3, nc=80, verbose=False)
    target_state = target_model.state_dict()
    if not isinstance(target_model.model[AIFI_LAYER_INDEX], GSDRAIFI):
        raise RuntimeError("Target model does not contain GSDRAIFI at model layer 9.")
    plan = build_strict_mapping(source_state, baseline_state, target_state)
    load_result = target_model.load_state_dict(plan.state, strict=False)
    if sorted(load_result.missing_keys) != plan.new_target_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Strict mapped load produced an unexpected result: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    loaded_state = target_model.state_dict()
    mapped_mismatches = verify_mapped_values(source_state, loaded_state, plan.source_to_target)
    aifi_keys = original_aifi_keys(baseline_state)
    missing_aifi_mappings = sorted(set(aifi_keys) - set(plan.source_to_target))
    aifi_value_mismatches = exact_value_mismatches(source_state, loaded_state, aifi_keys)
    verify_zero_initialized_boundaries(target_model.model[AIFI_LAYER_INDEX])

    target_parameters = dict(target_model.named_parameters())
    mapped_parameter_keys = [key for key in plan.state if key in target_parameters]
    mapped_parameter_numel = sum(target_parameters[key].numel() for key in mapped_parameter_keys)
    unchanged_parameter_numel = sum(
        parameter.numel() for key, parameter in target_parameters.items() if not key.startswith(SPARSE_PREFIX)
    )
    new_parameter_numel = sum(
        parameter.numel() for key, parameter in target_parameters.items() if key.startswith(SPARSE_PREFIX)
    )
    backbone_keys = [key for key in target_parameters if key.startswith(BACKBONE_PREFIXES)]
    mapped_backbone_keys = [key for key in backbone_keys if key in plan.state]
    backbone_numel = sum(target_parameters[key].numel() for key in backbone_keys)
    mapped_backbone_numel = sum(target_parameters[key].numel() for key in mapped_backbone_keys)

    print(f"ultralytics_file={imported_path}")
    print(f"source_checkpoint={source}")
    print(f"source_checkpoint_sha256={sha256(source)}")
    print(f"baseline_model={BASE_CFG}")
    print(f"gsdr_aifi_model={model_cfg}")
    print(f"output_checkpoint={output}")
    print(f"seed={seed}")
    print(f"mapped_unchanged_state_keys={len(plan.state)}/{len(baseline_state)}")
    print(f"mapped_unchanged_parameter_numel={mapped_parameter_numel}/{unchanged_parameter_numel}")
    print(f"mapping_shape_mismatches={plan.shape_mismatches}")
    print(f"mapping_unexpected_baseline_keys=[]")
    print(f"missing_keys_are_sparse_only={all(key.startswith(SPARSE_PREFIX) for key in plan.new_target_keys)}")
    print(f"mapped_values_exact={not mapped_mismatches}")
    print(f"original_aifi_mapped_keys={len(aifi_keys)}/{len(aifi_keys)}")
    print(f"original_aifi_missing_mappings={missing_aifi_mappings}")
    print(f"original_aifi_value_mismatches={aifi_value_mismatches}")
    print(f"backbone_parameter_key_coverage={len(mapped_backbone_keys)}/{len(backbone_keys)}")
    print(f"backbone_parameter_numel_coverage={mapped_backbone_numel}/{backbone_numel}")
    print(f"new_sparse_state_key_count={len(plan.new_target_keys)}")
    print(f"new_sparse_trainable_parameters={new_parameter_numel}")
    print("sparse_output_projection_zero=True")
    print("offset_output_projection_zero=True")
    print("relative_bias_output_layers_zero=True")
    if (
        len(plan.state) != len(baseline_state)
        or mapped_parameter_numel != unchanged_parameter_numel
        or plan.shape_mismatches
        or mapped_mismatches
        or missing_aifi_mappings
        or aifi_value_mismatches
        or len(mapped_backbone_keys) != len(backbone_keys)
        or mapped_backbone_numel != backbone_numel
        or new_parameter_numel > 200_000
    ):
        raise RuntimeError("Controlled GSDR-AIFI initialization failed a pre-save invariant.")

    save_ultralytics_checkpoint(target_model, output, model_cfg)
    verify_saved_checkpoint(output, model_cfg, target_model)
    print(f"initialized_checkpoint_sha256={sha256(output)}")
    print("GSDR-AIFI controlled initialization and exact reload verification passed.")
    return {
        "mapped_state_keys": len(plan.state),
        "mapped_parameter_numel": mapped_parameter_numel,
        "backbone_parameter_numel": backbone_numel,
        "new_state_keys": len(plan.new_target_keys),
        "new_trainable_parameters": new_parameter_numel,
    }


def main() -> None:
    """Run controlled initialization from CLI arguments."""
    args = parse_args()
    initialize_from_baseline(args.source, args.model, args.output, args.seed)


if __name__ == "__main__":
    main()
