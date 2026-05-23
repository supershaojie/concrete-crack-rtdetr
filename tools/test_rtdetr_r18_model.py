"""Minimal RT-DETR ResNet18 model construction smoke test."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "ultralytics-main"
sys.path.insert(0, str(ULTRALYTICS_ROOT))

from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402


def main():
    """Build RT-DETR ResNet18 from YAML and run model.info()."""
    cfg = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-resnet18.yaml"
    model = RTDETRDetectionModel(str(cfg), ch=3, nc=80, verbose=False)
    model.info(verbose=True)
    print("RT-DETR ResNet18 model.info() smoke test passed.")


if __name__ == "__main__":
    main()
