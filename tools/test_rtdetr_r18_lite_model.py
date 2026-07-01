"""Minimal RT-DETR ResNet18-lite model construction smoke test."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
sys.path.insert(0, str(ULTRALYTICS_ROOT))

import ultralytics  # noqa: E402
from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402


def main():
    """Build RT-DETR ResNet18-lite from YAML and run model.info()."""
    cfg = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18-lite.yaml"
    ultralytics_path = Path(ultralytics.__file__).resolve()
    if not ultralytics_path.is_relative_to(ULTRALYTICS_ROOT.resolve()):
        raise RuntimeError("Imported ultralytics is not from this repository's ultralytics-main directory.")

    model = RTDETRDetectionModel(str(cfg), ch=3, nc=80, verbose=False)
    layers, parameters, gradients, gflops = model.info(verbose=True, imgsz=640)
    print(f"layers={layers}")
    print(f"parameters={parameters}")
    print(f"gradients={gradients}")
    print(f"GFLOPs={gflops:.3f}")
    print("RT-DETR ResNet18-lite model.info() smoke test passed.")


if __name__ == "__main__":
    main()
