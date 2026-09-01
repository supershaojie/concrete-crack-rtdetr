"""Audit RT-DETR-R18-Lite CSCEF structure and ImageNet backbone checkpoint compatibility."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
BASE_CFG = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18-lite.yaml"
CSCEF_CFG = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18-lite-cscef.yaml"
IMAGENET_CHECKPOINT = ROOT / "weights" / "rtdetr_r18_lite_imagenet_backbone_init.pt"
TORCHVISION_CACHE = ROOT / "weights" / "torch_home" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
BACKBONE_PREFIXES = tuple(f"model.{index}." for index in range(8))

# Prefer this repository's Ultralytics package over any pip-installed package.
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils.patches import torch_load  # noqa: E402
from ultralytics.utils.torch_utils import intersect_dicts  # noqa: E402

from init_rtdetr_r18_lite_imagenet_backbone import build_backbone_mapping  # noqa: E402


def select_backbone(items):
    """Select model layers 0-7, which are the unchanged R18-Lite backbone."""
    return {key: value for key, value in items if key.startswith(BACKBONE_PREFIXES)}


def main() -> None:
    """Run structural and available local-checkpoint compatibility checks without writing files."""
    imported_path = Path(ultralytics.__file__).resolve()
    if not imported_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")

    base_model = RTDETRDetectionModel(str(BASE_CFG), ch=3, nc=80, verbose=False)
    cscef_model = RTDETRDetectionModel(str(CSCEF_CFG), ch=3, nc=80, verbose=False)
    base_backbone_parameters = select_backbone(base_model.named_parameters())
    cscef_backbone_parameters = select_backbone(cscef_model.named_parameters())
    base_backbone_state = select_backbone(base_model.state_dict().items())
    cscef_backbone_state = select_backbone(cscef_model.state_dict().items())

    parameter_shapes_match = {
        key: tuple(value.shape) for key, value in base_backbone_parameters.items()
    } == {key: tuple(value.shape) for key, value in cscef_backbone_parameters.items()}
    state_shapes_match = {key: tuple(value.shape) for key, value in base_backbone_state.items()} == {
        key: tuple(value.shape) for key, value in cscef_backbone_state.items()
    }
    base_backbone_numel = sum(value.numel() for value in base_backbone_parameters.values())
    cscef_backbone_numel = sum(value.numel() for value in cscef_backbone_parameters.values())

    print(f"ultralytics_version={ultralytics.__version__}")
    print(f"ultralytics_file={imported_path}")
    print(f"base_backbone_parameter_keys={len(base_backbone_parameters)}")
    print(f"cscef_backbone_parameter_keys={len(cscef_backbone_parameters)}")
    print(f"base_backbone_parameter_numel={base_backbone_numel}")
    print(f"cscef_backbone_parameter_numel={cscef_backbone_numel}")
    print(f"backbone_parameter_names_and_shapes_identical={parameter_shapes_match}")
    print(f"base_backbone_state_keys={len(base_backbone_state)}")
    print(f"cscef_backbone_state_keys={len(cscef_backbone_state)}")
    print(f"backbone_state_names_and_shapes_identical={state_shapes_match}")

    if not parameter_shapes_match or not state_shapes_match:
        raise RuntimeError("Base and CSCEF backbone structures differ.")
    if not IMAGENET_CHECKPOINT.exists():
        print(f"imagenet_checkpoint_missing={IMAGENET_CHECKPOINT}")
        print("actual_checkpoint_load_audit=SKIPPED")
        return

    checkpoint = torch_load(IMAGENET_CHECKPOINT, map_location="cpu")
    source_model = checkpoint.get("ema") or checkpoint["model"]
    source_state = source_model.float().state_dict()
    target_state = cscef_model.state_dict()
    transferred_state = intersect_dicts(source_state, target_state)
    target_parameter_keys = dict(cscef_model.named_parameters())
    loaded_backbone_state_keys = sorted(set(cscef_backbone_state) & set(transferred_state))
    loaded_backbone_parameter_keys = sorted(set(cscef_backbone_parameters) & set(transferred_state))
    loaded_backbone_numel = sum(target_parameter_keys[key].numel() for key in loaded_backbone_parameter_keys)
    backbone_parameter_ratio = loaded_backbone_numel / cscef_backbone_numel
    unexpected_backbone_missing = sorted(set(cscef_backbone_state) - set(transferred_state))
    backbone_shape_mismatches = sorted(
        key
        for key, value in cscef_backbone_state.items()
        if key in source_state and tuple(source_state[key].shape) != tuple(value.shape)
    )
    missing_target_state = sorted(set(target_state) - set(transferred_state))
    missing_cscef_state = [key for key in missing_target_state if key.startswith("model.18.")]

    cscef_model.load(source_model, verbose=False)
    loaded_state = cscef_model.state_dict()
    load_values_verified = all(
        loaded_state[key].float().equal(source_state[key].float()) for key in loaded_backbone_state_keys
    )

    print(f"imagenet_checkpoint={IMAGENET_CHECKPOINT}")
    print(f"transferred_total_state_keys={len(transferred_state)}")
    print(f"loaded_backbone_state_keys={len(loaded_backbone_state_keys)}/{len(cscef_backbone_state)}")
    print(f"loaded_backbone_parameter_keys={len(loaded_backbone_parameter_keys)}/{len(cscef_backbone_parameters)}")
    print(f"loaded_backbone_parameter_numel={loaded_backbone_numel}/{cscef_backbone_numel}")
    print(f"loaded_backbone_parameter_ratio={backbone_parameter_ratio:.6f}")
    print(f"loaded_backbone_values_verified={load_values_verified}")
    print(f"unexpected_backbone_missing={unexpected_backbone_missing}")
    print(f"backbone_shape_mismatches={backbone_shape_mismatches}")
    print(f"missing_cscef_state_keys={missing_cscef_state}")
    print(f"other_missing_target_state_key_count={len(missing_target_state) - len(missing_cscef_state)}")

    if TORCHVISION_CACHE.exists():
        torchvision_state = torch_load(TORCHVISION_CACHE, map_location="cpu", weights_only=True)
        mapping = build_backbone_mapping()
        mapped_missing_source = []
        mapped_missing_target = []
        mapped_shape_mismatches = []
        mapped_value_mismatches = []
        verified_mapped_target_keys = []
        verified_mapped_parameter_keys = []
        verified_mapped_numel = 0
        for source_key, target_key in mapping.items():
            if source_key not in torchvision_state:
                mapped_missing_source.append(source_key)
                continue
            if target_key not in source_state:
                mapped_missing_target.append(target_key)
                continue
            source_value = torchvision_state[source_key]
            checkpoint_value = source_state[target_key]
            if source_value.shape != checkpoint_value.shape:
                mapped_shape_mismatches.append(
                    f"{source_key} -> {target_key}: {tuple(source_value.shape)} != {tuple(checkpoint_value.shape)}"
                )
                continue
            # The initialization script stores the final checkpoint in FP16, then loading converts it back to FP32.
            saved_source_value = source_value.to(torch.float16).to(checkpoint_value.dtype)
            if not checkpoint_value.equal(saved_source_value):
                mapped_value_mismatches.append(f"{source_key} -> {target_key}")
                continue
            verified_mapped_target_keys.append(target_key)
            verified_mapped_numel += checkpoint_value.numel()
            if target_key in target_parameter_keys:
                verified_mapped_parameter_keys.append(target_key)

        verified_mapped_parameter_numel = sum(
            target_parameter_keys[key].numel() for key in verified_mapped_parameter_keys
        )

        print(f"torchvision_imagenet_cache={TORCHVISION_CACHE}")
        print(f"imagenet_mapping_verified_keys={len(verified_mapped_target_keys)}/{len(mapping)}")
        print(f"imagenet_mapping_verified_numel={verified_mapped_numel}")
        print(
            f"imagenet_mapping_verified_parameter_keys={len(verified_mapped_parameter_keys)}/"
            f"{len(cscef_backbone_parameters)}"
        )
        print(
            f"imagenet_mapping_verified_parameter_numel={verified_mapped_parameter_numel}/"
            f"{cscef_backbone_numel}"
        )
        print(
            f"imagenet_mapping_verified_parameter_ratio="
            f"{verified_mapped_parameter_numel / cscef_backbone_numel:.6f}"
        )
        print(f"imagenet_mapping_missing_source_keys={mapped_missing_source}")
        print(f"imagenet_mapping_missing_target_keys={mapped_missing_target}")
        print(f"imagenet_mapping_shape_mismatches={mapped_shape_mismatches}")
        print(f"imagenet_mapping_value_mismatches={mapped_value_mismatches}")
        print("verified_imagenet_mapped_target_keys:")
        for key in verified_mapped_target_keys:
            print(f"  {key}")
        if mapped_missing_target or mapped_shape_mismatches or mapped_value_mismatches:
            raise RuntimeError("The saved checkpoint does not match the cached torchvision ImageNet mapping.")
    else:
        print(f"torchvision_imagenet_cache_missing={TORCHVISION_CACHE}")
        print("direct_torchvision_mapping_audit=SKIPPED")

    print("actual_loaded_backbone_state_keys:")
    for key in loaded_backbone_state_keys:
        print(f"  {key}")

    if unexpected_backbone_missing or backbone_shape_mismatches or backbone_parameter_ratio != 1.0:
        raise RuntimeError("ImageNet checkpoint did not fully cover the CSCEF model backbone.")
    print("CSCEF ImageNet backbone compatibility audit passed.")


if __name__ == "__main__":
    main()
