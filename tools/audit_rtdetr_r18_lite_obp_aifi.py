"""Audit RT-DETR-R18-Lite OBP-AIFI compatibility with the ImageNet backbone checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
BASE_CFG = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18-lite.yaml"
OBP_CFG = (
    ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18-lite-obp-aifi.yaml"
)
WEIGHTS = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
BACKBONE_PREFIXES = tuple(f"model.{index}." for index in range(8))

# Prefer this repository's Ultralytics package over any installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics.nn.modules import AIFI, OBPAIFI  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402


def is_backbone_key(key: str) -> bool:
    """Return whether a state-dict key belongs to backbone layers 0 through 7."""
    return key.startswith(BACKBONE_PREFIXES)


def backbone_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    """Return all named backbone parameters."""
    return {key: value for key, value in model.named_parameters() if is_backbone_key(key)}


def backbone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return all backbone parameters and persistent buffers."""
    return {key: value for key, value in model.state_dict().items() if is_backbone_key(key)}


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    """Load a full Ultralytics checkpoint and return its FP32 model state dict."""
    checkpoint = torch_load(str(path), map_location="cpu")
    module = checkpoint.get("ema") if isinstance(checkpoint, dict) else None
    if module is None and isinstance(checkpoint, dict):
        module = checkpoint.get("model")
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Checkpoint {path} does not contain a torch.nn.Module under 'ema' or 'model'.")
    return module.float().state_dict()


def compare_backbone_architectures(
    baseline: RTDETRDetectionModel, obp: RTDETRDetectionModel
) -> None:
    """Verify that replacing AIFI does not alter any backbone key, shape, or parameter count."""
    baseline_params = backbone_parameters(baseline)
    obp_params = backbone_parameters(obp)
    if baseline_params.keys() != obp_params.keys():
        raise AssertionError("Baseline and OBP-AIFI backbone parameter keys differ.")

    shape_mismatches = {
        key: (tuple(baseline_params[key].shape), tuple(obp_params[key].shape))
        for key in baseline_params
        if baseline_params[key].shape != obp_params[key].shape
    }
    if shape_mismatches:
        raise AssertionError(f"Backbone parameter shape mismatches: {shape_mismatches}")

    baseline_count = len(baseline_params)
    obp_count = len(obp_params)
    baseline_numel = sum(value.numel() for value in baseline_params.values())
    obp_numel = sum(value.numel() for value in obp_params.values())
    if (baseline_count, baseline_numel) != (obp_count, obp_numel):
        raise AssertionError(
            "Backbone parameter count/numel changed: "
            f"baseline=({baseline_count}, {baseline_numel}), OBP=({obp_count}, {obp_numel})"
        )

    baseline_all = backbone_state(baseline)
    obp_all = backbone_state(obp)
    if baseline_all.keys() != obp_all.keys():
        raise AssertionError("Baseline and OBP-AIFI backbone state keys differ.")
    state_shape_mismatches = {
        key: (tuple(baseline_all[key].shape), tuple(obp_all[key].shape))
        for key in baseline_all
        if baseline_all[key].shape != obp_all[key].shape
    }
    if state_shape_mismatches:
        raise AssertionError(f"Backbone state shape mismatches: {state_shape_mismatches}")

    print("Backbone architecture comparison:")
    print(f"  parameter keys: {baseline_count} (identical)")
    print(f"  parameter numel: {baseline_numel} (identical)")
    print(f"  state tensors including buffers: {len(baseline_all)} (identical)")


def audit_checkpoint_loading(model: RTDETRDetectionModel, source: dict[str, torch.Tensor]) -> None:
    """Load every compatible backbone tensor, print a per-key report, and verify exact values."""
    target_backbone = backbone_state(model)
    source_backbone_keys = {key for key in source if is_backbone_key(key)}
    target_backbone_keys = set(target_backbone)
    unexpected = sorted(source_backbone_keys - target_backbone_keys)
    missing = []
    shape_mismatches = []
    loadable = {}

    print("Checkpoint-to-OBP backbone per-key report:")
    for key, target_tensor in target_backbone.items():
        source_tensor = source.get(key)
        if source_tensor is None:
            missing.append(key)
            print(f"  {key}: target={tuple(target_tensor.shape)}, status=MISSING")
        elif source_tensor.shape != target_tensor.shape:
            shape_mismatches.append((key, tuple(source_tensor.shape), tuple(target_tensor.shape)))
            print(
                f"  {key}: checkpoint={tuple(source_tensor.shape)}, target={tuple(target_tensor.shape)}, "
                "status=SHAPE_MISMATCH"
            )
        else:
            loadable[key] = source_tensor
            print(f"  {key}: shape={tuple(target_tensor.shape)}, status=LOADABLE")

    if missing or unexpected or shape_mismatches:
        raise AssertionError(
            f"Backbone checkpoint incompatibility: missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )

    incompatible = model.load_state_dict(loadable, strict=False)
    if incompatible.unexpected_keys:
        raise AssertionError(f"Unexpected keys while loading backbone subset: {incompatible.unexpected_keys}")

    reloaded = model.state_dict()
    numerical_mismatches = []
    for key, source_tensor in loadable.items():
        expected = source_tensor.to(device=reloaded[key].device, dtype=reloaded[key].dtype)
        if not torch.equal(reloaded[key], expected):
            numerical_mismatches.append(key)
    if numerical_mismatches:
        raise AssertionError(f"Numerical mismatch after backbone load: {numerical_mismatches}")

    loaded_tensors = len(loadable)
    loaded_numel = sum(target_backbone[key].numel() for key in loadable)
    total_numel = sum(tensor.numel() for tensor in target_backbone.values())
    tensor_ratio = 100.0 * loaded_tensors / len(target_backbone)
    numel_ratio = 100.0 * loaded_numel / total_numel
    print("Backbone checkpoint summary:")
    print(f"  loaded state tensors: {loaded_tensors}/{len(target_backbone)} ({tensor_ratio:.2f}%)")
    print(f"  loaded state numel: {loaded_numel}/{total_numel} ({numel_ratio:.2f}%)")
    print("  unexpected backbone keys: 0")
    print("  backbone shape mismatches: 0")
    print("  numerical consistency: exact for every loaded backbone tensor")
    if tensor_ratio != 100.0 or numel_ratio != 100.0:
        raise AssertionError("ImageNet checkpoint did not provide 100% backbone coverage.")


def report_full_model_intersection(model: RTDETRDetectionModel, source: dict[str, torch.Tensor]) -> None:
    """Report expected OBP-only missing keys without calling non-backbone tensors ImageNet-pretrained."""
    target = model.state_dict()
    compatible = {
        key for key, tensor in target.items() if key in source and source[key].shape == tensor.shape
    }
    missing = sorted(set(target) - compatible)
    unexpected = sorted(set(source) - set(target))
    shape_mismatches = sorted(
        key for key, tensor in target.items() if key in source and source[key].shape != tensor.shape
    )
    expected_missing = [key for key in missing if key == "model.9.gamma" or key.startswith("model.9.mixer.")]
    other_missing = sorted(set(missing) - set(expected_missing))
    expected_unexpected = [key for key in unexpected if key.startswith(("model.9.fc1.", "model.9.fc2."))]
    other_unexpected = sorted(set(unexpected) - set(expected_unexpected))

    print("Full-model architectural intersection (not an ImageNet-pretraining claim):")
    print("  expected OBP-AIFI-only missing keys:")
    for key in expected_missing:
        print(f"    {key}")
    print("  expected obsolete original-AIFI FFN checkpoint keys:")
    for key in expected_unexpected:
        print(f"    {key}")
    print(f"  other missing keys: {other_missing}")
    print(f"  other unexpected keys: {other_unexpected}")
    print(f"  shape mismatches: {shape_mismatches}")

    if other_missing or other_unexpected or shape_mismatches:
        raise AssertionError("Unexpected non-OBP incompatibility exists in the full-model checkpoint intersection.")


def main() -> None:
    """Build both variants and run the complete ImageNet backbone compatibility audit."""
    print(f"Project root: {ROOT}")
    print(f"Imported ultralytics from: {ultralytics.__file__}")
    if not Path(ultralytics.__file__).resolve().is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"ImageNet backbone checkpoint not found: {WEIGHTS}")

    baseline = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False)
    obp = RTDETRDetectionModel(str(OBP_CFG), ch=3, nc=80, verbose=False)
    if not isinstance(baseline.model[9], AIFI) or isinstance(baseline.model[9], OBPAIFI):
        raise AssertionError("Baseline layer 9 is not the original AIFI.")
    if not isinstance(obp.model[9], OBPAIFI):
        raise AssertionError("OBP model layer 9 is not OBPAIFI.")
    compare_backbone_architectures(baseline, obp)

    source = checkpoint_state(WEIGHTS)
    audit_checkpoint_loading(obp, source)
    report_full_model_intersection(obp, source)

    # Exercise the existing Ultralytics model-level loading interface with the checkpoint object.
    checkpoint = torch_load(str(WEIGHTS), map_location="cpu")
    interface_model = RTDETRDetectionModel(str(OBP_CFG), ch=3, nc=80, verbose=False)
    interface_model.load(checkpoint, verbose=False)
    print("Existing RTDETRDetectionModel.load(checkpoint) interface: PASS")
    print("Audit conclusion:")
    print("  Backbone: 100% covered by the ImageNet-initialized R18-Lite checkpoint.")
    print("  Original AIFI/neck/decoder: checkpoint tensors are random initialization, not ImageNet pretraining.")
    print("  OBP-AIFI mixer and gamma: newly initialized parameters; absence from the checkpoint is expected.")
    print("RT-DETR-R18-Lite OBP-AIFI ImageNet backbone audit passed.")


if __name__ == "__main__":
    main()
