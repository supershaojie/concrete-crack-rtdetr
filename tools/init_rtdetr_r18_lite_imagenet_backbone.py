"""Initialize RT-DETR-ResNet18-Lite backbone from torchvision ImageNet weights."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
CFG = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18-lite.yaml"
OUT = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
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


def build_backbone_mapping() -> dict[str, str]:
    """Map torchvision ResNet18 BasicBlock keys to RT-DETR-R18-Lite Blocks keys."""
    mapping: dict[str, str] = {}
    stage_map = {"layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7}
    conv_bn_map = {
        "conv1.weight": "branch2a.conv.weight",
        "bn1.weight": "branch2a.norm.weight",
        "bn1.bias": "branch2a.norm.bias",
        "bn1.running_mean": "branch2a.norm.running_mean",
        "bn1.running_var": "branch2a.norm.running_var",
        "bn1.num_batches_tracked": "branch2a.norm.num_batches_tracked",
        "conv2.weight": "branch2b.conv.weight",
        "bn2.weight": "branch2b.norm.weight",
        "bn2.bias": "branch2b.norm.bias",
        "bn2.running_mean": "branch2b.norm.running_mean",
        "bn2.running_var": "branch2b.norm.running_var",
        "bn2.num_batches_tracked": "branch2b.norm.num_batches_tracked",
    }
    downsample_map = {
        "downsample.0.weight": "short.conv.conv.weight",
        "downsample.1.weight": "short.conv.norm.weight",
        "downsample.1.bias": "short.conv.norm.bias",
        "downsample.1.running_mean": "short.conv.norm.running_mean",
        "downsample.1.running_var": "short.conv.norm.running_var",
        "downsample.1.num_batches_tracked": "short.conv.norm.num_batches_tracked",
    }

    for source_stage, target_stage in stage_map.items():
        for block_idx in (0, 1):
            target_prefix = f"model.{target_stage}.blocks.{block_idx}"
            for source_suffix, target_suffix in conv_bn_map.items():
                mapping[f"{source_stage}.{block_idx}.{source_suffix}"] = f"{target_prefix}.{target_suffix}"

            if source_stage != "layer1" and block_idx == 0:
                for source_suffix, target_suffix in downsample_map.items():
                    mapping[f"{source_stage}.{block_idx}.{source_suffix}"] = f"{target_prefix}.{target_suffix}"

    return mapping


def copy_shape_matched_backbone_weights(
    source_state: dict[str, torch.Tensor],
    target_model: torch.nn.Module,
    mapping: dict[str, str],
) -> tuple[int, int, int, list[str]]:
    """Copy mapped tensors with identical shapes and return transfer statistics."""
    target_state = target_model.state_dict()
    transferred_tensors = 0
    mapped_skipped_tensors = 0
    transferred_count = 0
    copied_target_keys: set[str] = set()
    skipped_messages: list[str] = []

    print("Backbone transfer log:")
    for source_key, target_key in mapping.items():
        if source_key not in source_state:
            mapped_skipped_tensors += 1
            skipped_messages.append(f"{source_key} -> {target_key}: missing source")
            print(f"  {source_key} -> {target_key}, status=SKIP missing source")
            continue
        if target_key not in target_state:
            mapped_skipped_tensors += 1
            skipped_messages.append(f"{source_key} -> {target_key}: missing target")
            print(f"  {source_key} -> {target_key}, status=SKIP missing target")
            continue

        source_tensor = source_state[source_key]
        target_tensor = target_state[target_key]
        if source_tensor.shape != target_tensor.shape:
            mapped_skipped_tensors += 1
            skipped_messages.append(
                f"{source_key} -> {target_key}: source {tuple(source_tensor.shape)} target {tuple(target_tensor.shape)}"
            )
            print(
                f"  {source_key} -> {target_key}, shape={tuple(source_tensor.shape)} != "
                f"{tuple(target_tensor.shape)}, status=SKIP shape mismatch"
            )
            continue

        target_tensor.copy_(source_tensor)
        transferred_tensors += 1
        transferred_count += source_tensor.numel()
        copied_target_keys.add(target_key)
        print(f"  {source_key} -> {target_key}, shape={tuple(source_tensor.shape)}, status=COPIED")

    target_backbone_prefixes = tuple(f"model.{idx}." for idx in range(8))
    default_backbone_keys = [
        key for key in target_state if key.startswith(target_backbone_prefixes) and key not in copied_target_keys
    ]
    print("Backbone tensors left at default initialization:")
    for key in default_backbone_keys:
        print(f"  {key}: {tuple(target_state[key].shape)}")
    if mapped_skipped_tensors:
        print(f"Mapped tensors skipped before copy: {mapped_skipped_tensors}")

    return transferred_tensors, len(default_backbone_keys), transferred_count, skipped_messages


def save_ultralytics_checkpoint(model: RTDETRDetectionModel, out_path: Path) -> None:
    """Save a full Ultralytics checkpoint that can be loaded by RTDETR(path)."""
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
    """Build RT-DETR-R18-Lite, transfer ImageNet backbone weights, save, and verify."""
    print(f"Project root: {ROOT}")
    print(f"Using Ultralytics path: {ULTRALYTICS_ROOT}")
    print(f"RT-DETR-R18-Lite config: {CFG}")
    print(f"Save path: {OUT}")

    import ultralytics

    print(f"Imported ultralytics from: {ultralytics.__file__}")
    if not Path(ultralytics.__file__).resolve().is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")

    rtdetr_model = RTDETRDetectionModel(str(CFG), ch=3, nc=80, verbose=False)
    total_parameters = sum(parameter.numel() for parameter in rtdetr_model.parameters())
    torchvision_resnet18 = load_torchvision_resnet18()

    mapping = build_backbone_mapping()
    transferred_tensors, skipped_tensors, transferred_count, skipped_messages = copy_shape_matched_backbone_weights(
        torchvision_resnet18.state_dict(),
        rtdetr_model,
        mapping,
    )
    save_ultralytics_checkpoint(rtdetr_model, OUT)

    print("Summary:")
    print(f"  total model parameters: {total_parameters}")
    print(f"  candidate mapped tensors: {len(mapping)}")
    print(f"  successfully transferred tensors: {transferred_tensors}")
    print(f"  skipped tensors: {skipped_tensors}")
    print(f"  transferred parameter count: {transferred_count}")
    if skipped_messages:
        print("  skipped mapped tensor details:")
        for item in skipped_messages:
            print(f"    {item}")
    print(f"  saved path: {OUT}")

    print("Running saved checkpoint verification...")
    loaded = RTDETR(str(OUT))
    info = loaded.info(verbose=True, imgsz=640)
    print(f"verified model.info tuple: {info}")
    print("RT-DETR-R18-Lite ImageNet backbone initialization smoke test passed.")


if __name__ == "__main__":
    main()
