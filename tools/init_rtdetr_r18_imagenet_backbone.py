"""Initialize RT-DETR-ResNet18 backbone from torchvision ImageNet weights."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
CFG = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18.yaml"
OUT = ROOT / "weights" / "rtdetr_r18_imagenet_backbone_init.pt"
TORCH_HOME = ROOT / "weights" / "torch_home"

# Prefer this repository's Ultralytics package over any pip-installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))
# Keep downloaded torchvision weights inside the ignored project weights directory.
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))

from torchvision import models  # noqa: E402

from ultralytics import RTDETR, __version__  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG_DICT  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402


def load_torchvision_resnet18() -> torch.nn.Module:
    """Load torchvision ResNet18 with ImageNet weights across old/new torchvision APIs."""
    if hasattr(models, "ResNet18_Weights"):
        weights_enum = models.ResNet18_Weights
        weights = getattr(weights_enum, "IMAGENET1K_V1", None) or weights_enum.DEFAULT
        print(f"Loading torchvision ResNet18 weights: {weights}")
        return models.resnet18(weights=weights)

    print("Loading torchvision ResNet18 weights: pretrained=True")
    return models.resnet18(pretrained=True)


def print_state_dict_keys(title: str, state_dict: dict[str, torch.Tensor], prefixes: Iterable[str] | None = None) -> None:
    """Print selected state_dict keys and tensor shapes."""
    print(title)
    for key, value in state_dict.items():
        if prefixes is None or key.startswith(tuple(prefixes)):
            print(f"  {key}: {tuple(value.shape)}")


def build_backbone_mapping() -> dict[str, str]:
    """Map torchvision ResNet18 backbone keys to Ultralytics RT-DETR-R18 backbone keys."""
    mapping = {
        "conv1.weight": "model.0.layer.0.conv.weight",
        "bn1.weight": "model.0.layer.0.bn.weight",
        "bn1.bias": "model.0.layer.0.bn.bias",
        "bn1.running_mean": "model.0.layer.0.bn.running_mean",
        "bn1.running_var": "model.0.layer.0.bn.running_var",
        "bn1.num_batches_tracked": "model.0.layer.0.bn.num_batches_tracked",
    }

    module_by_layer = {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
    block_map = {
        "conv1.weight": "cv1.conv.weight",
        "bn1.weight": "cv1.bn.weight",
        "bn1.bias": "cv1.bn.bias",
        "bn1.running_mean": "cv1.bn.running_mean",
        "bn1.running_var": "cv1.bn.running_var",
        "bn1.num_batches_tracked": "cv1.bn.num_batches_tracked",
        "conv2.weight": "cv2.conv.weight",
        "bn2.weight": "cv2.bn.weight",
        "bn2.bias": "cv2.bn.bias",
        "bn2.running_mean": "cv2.bn.running_mean",
        "bn2.running_var": "cv2.bn.running_var",
        "bn2.num_batches_tracked": "cv2.bn.num_batches_tracked",
    }
    downsample_map = {
        "downsample.0.weight": "shortcut.0.conv.weight",
        "downsample.1.weight": "shortcut.0.bn.weight",
        "downsample.1.bias": "shortcut.0.bn.bias",
        "downsample.1.running_mean": "shortcut.0.bn.running_mean",
        "downsample.1.running_var": "shortcut.0.bn.running_var",
        "downsample.1.num_batches_tracked": "shortcut.0.bn.num_batches_tracked",
    }

    for layer_name, module_idx in module_by_layer.items():
        for block_idx in (0, 1):
            for source_suffix, target_suffix in block_map.items():
                source_key = f"{layer_name}.{block_idx}.{source_suffix}"
                target_key = f"model.{module_idx}.layer.{block_idx}.{target_suffix}"
                mapping[source_key] = target_key

            if layer_name != "layer1" and block_idx == 0:
                for source_suffix, target_suffix in downsample_map.items():
                    source_key = f"{layer_name}.{block_idx}.{source_suffix}"
                    target_key = f"model.{module_idx}.layer.{block_idx}.{target_suffix}"
                    mapping[source_key] = target_key

    return mapping


def copy_backbone_weights(
    source_state: dict[str, torch.Tensor],
    target_model: torch.nn.Module,
    mapping: dict[str, str],
) -> tuple[int, list[str], list[str]]:
    """Copy explicitly mapped backbone weights and report every decision."""
    target_state = target_model.state_dict()
    copied = 0
    skipped: list[str] = []
    shape_mismatches: list[str] = []

    print("Backbone transfer log:")
    for source_key, target_key in mapping.items():
        if source_key not in source_state:
            skipped.append(f"{source_key} -> {target_key}: missing source")
            print(f"  {source_key} -> {target_key}, shape=N/A, status=SKIP missing source")
            continue
        if target_key not in target_state:
            skipped.append(f"{source_key} -> {target_key}: missing target")
            print(f"  {source_key} -> {target_key}, shape={tuple(source_state[source_key].shape)}, status=SKIP missing target")
            continue

        source_tensor = source_state[source_key]
        target_tensor = target_state[target_key]
        if source_tensor.shape != target_tensor.shape:
            shape_mismatches.append(
                f"{source_key} -> {target_key}: source {tuple(source_tensor.shape)} target {tuple(target_tensor.shape)}"
            )
            print(
                f"  {source_key} -> {target_key}, shape={tuple(source_tensor.shape)} != "
                f"{tuple(target_tensor.shape)}, status=SKIP shape mismatch"
            )
            continue

        target_state[target_key].copy_(source_tensor)
        copied += 1
        print(f"  {source_key} -> {target_key}, shape={tuple(source_tensor.shape)}, status=COPIED")

    unused_source_keys = sorted(key for key in source_state if key not in mapping)
    for key in unused_source_keys:
        skipped.append(f"{key}: not a ResNet18 backbone key for RT-DETR-R18")

    if skipped:
        print("Skipped keys:")
        for item in skipped:
            print(f"  {item}")

    if shape_mismatches:
        print("Shape mismatches:")
        for item in shape_mismatches:
            print(f"  {item}")

    return copied, skipped, shape_mismatches


def save_ultralytics_checkpoint(model: RTDETRDetectionModel, out_path: Path) -> None:
    """Save a full Ultralytics checkpoint that can be loaded as model=*.pt."""
    model.eval()
    model.args = {**DEFAULT_CFG_DICT, "model": str(CFG), "task": "detect"}
    model.task = "detect"
    model.pt_path = str(out_path)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)


def main() -> None:
    """Build RT-DETR-R18, transfer ImageNet backbone weights, save, and smoke-test."""
    print(f"Project root: {ROOT}")
    print(f"Using Ultralytics path: {ULTRALYTICS_ROOT}")
    print(f"RT-DETR-R18 config: {CFG}")
    print(f"Save path: {OUT}")

    import ultralytics

    print(f"Imported ultralytics from: {ultralytics.__file__}")
    if not Path(ultralytics.__file__).resolve().is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")

    rtdetr_model = RTDETRDetectionModel(str(CFG), ch=3, nc=80, verbose=False)
    torchvision_resnet18 = load_torchvision_resnet18()

    source_state = torchvision_resnet18.state_dict()
    target_state = rtdetr_model.state_dict()
    backbone_prefixes = tuple(f"model.{idx}." for idx in range(5))
    backbone_target_keys = [key for key in target_state if key.startswith(backbone_prefixes)]

    print_state_dict_keys(
        "Torchvision ResNet18 state_dict keys:",
        source_state,
        prefixes=("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.", "fc."),
    )
    print_state_dict_keys("RT-DETR-R18 backbone target keys:", target_state, prefixes=backbone_prefixes)

    mapping = build_backbone_mapping()
    copied, skipped, shape_mismatches = copy_backbone_weights(source_state, rtdetr_model, mapping)
    save_ultralytics_checkpoint(rtdetr_model, OUT)

    print("Summary:")
    print(f"  torchvision ResNet18 weight count: {len(source_state)}")
    print(f"  RT-DETR-R18 backbone target weight count: {len(backbone_target_keys)}")
    print(f"  successful transfers: {copied}")
    print(f"  skipped count: {len(skipped)}")
    print(f"  shape mismatch count: {len(shape_mismatches)}")
    print(f"  saved path: {OUT}")

    print("Running saved checkpoint smoke test...")
    loaded = RTDETR(str(OUT))
    loaded.info(verbose=False)
    print("RT-DETR-R18 ImageNet backbone initialization smoke test passed.")


if __name__ == "__main__":
    main()
