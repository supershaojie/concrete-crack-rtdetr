from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DET_DIR = ROOT / "datasets" / "crack_det"
LOG_DIR = ROOT / "logs"
OUT_DIR = LOG_DIR / "vis_check" / "crack_det"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SEED = 42

# 每个 split 最多抽样数量
MAX_TRAIN_ORIGINAL = 30
MAX_PER_AUG_TYPE = 20
MAX_VAL = 60
MAX_TEST = 60


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []

    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    buf.tofile(str(path))


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []

    if not label_path.exists():
        return boxes

    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        boxes.append((cls, x, y, w, h))

    return boxes


def yolo_to_xyxy(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h

    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2


def draw_labels(img_path: Path, label_path: Path, out_path: Path) -> tuple[int, bool]:
    img = imread_unicode(img_path)
    img_h, img_w = img.shape[:2]

    boxes = read_yolo_labels(label_path)

    valid = True

    for cls, x, y, w, h in boxes:
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            valid = False
            continue

        x1, y1, x2, y2 = yolo_to_xyxy(x, y, w, h, img_w, img_h)

        if x2 <= x1 or y2 <= y1:
            valid = False
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label_text = f"crack {w:.3f}x{h:.3f}"
        cv2.putText(
            img,
            label_text,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    info_text = f"{img_path.name} | boxes: {len(boxes)}"
    cv2.putText(
        img,
        info_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    imwrite_unicode(out_path, img)

    return len(boxes), valid


def get_aug_type(stem: str) -> str:
    aug_types = [
        "original",
        "hflip",
        "vflip",
        "brightness_contrast",
        "clahe_gamma",
        "noise",
        "blur",
        "rotate_pos",
        "rotate_neg",
        "brightness_up",
        "brightness_down",
        "contrast_up",
        "gaussian_noise",
        "clahe",
    ]

    for aug_type in aug_types:
        if stem.endswith(f"_{aug_type}"):
            return aug_type

    return "original"


def sample_train_images(images: list[Path]) -> list[tuple[str, Path]]:
    rng = random.Random(SEED)

    grouped: dict[str, list[Path]] = defaultdict(list)

    for img_path in images:
        aug_type = get_aug_type(img_path.stem)
        grouped[aug_type].append(img_path)

    sampled: list[tuple[str, Path]] = []

    for aug_type, group_images in grouped.items():
        group_images = sorted(group_images)

        if aug_type == "original":
            sample_n = min(MAX_TRAIN_ORIGINAL, len(group_images))
        else:
            sample_n = min(MAX_PER_AUG_TYPE, len(group_images))

        sampled_images = rng.sample(group_images, sample_n)
        sampled.extend((aug_type, p) for p in sampled_images)

    return sampled


def sample_split_images(split: str, images: list[Path]) -> list[tuple[str, Path]]:
    rng = random.Random(SEED)

    if split == "train":
        return sample_train_images(images)

    if split == "val":
        sample_n = min(MAX_VAL, len(images))
    elif split == "test":
        sample_n = min(MAX_TEST, len(images))
    else:
        sample_n = min(50, len(images))

    sampled = rng.sample(images, sample_n)
    return [(get_aug_type(p.stem), p) for p in sampled]


def visualize_split(split: str) -> tuple[int, int, int]:
    img_dir = DET_DIR / "images" / split
    lab_dir = DET_DIR / "labels" / split

    images = list_images(img_dir)

    if not images:
        print(f"[WARN] No images found in {img_dir}")
        return 0, 0, 0

    sampled = sample_split_images(split, images)

    total_visualized = 0
    total_boxes = 0
    invalid_count = 0

    for aug_type, img_path in sampled:
        label_path = lab_dir / f"{img_path.stem}.txt"

        out_path = OUT_DIR / split / aug_type / img_path.name

        box_count, valid = draw_labels(img_path, label_path, out_path)

        total_visualized += 1
        total_boxes += box_count

        if not valid:
            invalid_count += 1

    return total_visualized, total_boxes, invalid_count


def main() -> None:
    if not DET_DIR.exists():
        raise FileNotFoundError(
            f"Final dataset not found: {DET_DIR}\n"
            "Please run tools/03_augment_dataset.py first."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    reset_dir(OUT_DIR)

    log_lines = []
    log_lines.append("Visualization check log")
    log_lines.append("=======================")
    log_lines.append("")
    log_lines.append(f"Dataset: {DET_DIR}")
    log_lines.append(f"Output: {OUT_DIR}")
    log_lines.append("")

    for split in ["train", "val", "test"]:
        total_visualized, total_boxes, invalid_count = visualize_split(split)

        print(f"{split}: visualized={total_visualized}, boxes={total_boxes}, invalid={invalid_count}")

        log_lines.append(f"{split}:")
        log_lines.append(f"  visualized: {total_visualized}")
        log_lines.append(f"  boxes: {total_boxes}")
        log_lines.append(f"  invalid: {invalid_count}")
        log_lines.append("")

    log_path = LOG_DIR / "05_visualize_labels_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print("")
    print("Visualization finished.")
    print(f"Output: {OUT_DIR}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
