from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "datasets" / "crack_raw" / "split_raw"
DET_DIR = ROOT / "datasets" / "crack_det"
CONFIGS_DIR = ROOT / "configs"
LOG_DIR = ROOT / "logs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
JPEG_QUALITY = 95


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

    ok, buf = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    buf.tofile(str(path))


def read_labels(path: Path) -> list[list]:
    labels: list[list] = []

    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(f"Invalid label line in {path}: {line}")

        cls = parts[0]
        x, y, w, h = map(float, parts[1:])
        labels.append([cls, x, y, w, h])

    return labels


def write_labels(path: Path, labels: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for cls, x, y, w, h in labels:
        x = min(max(float(x), 0.0), 1.0)
        y = min(max(float(y), 0.0), 1.0)
        w = min(max(float(w), 0.0), 1.0)
        h = min(max(float(h), 0.0), 1.0)

        if w <= 0 or h <= 0:
            continue

        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def yolo_to_xyxy(label: list, img_w: int, img_h: int) -> tuple[str, float, float, float, float]:
    cls, x, y, w, h = label

    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    return cls, x1, y1, x2, y2


def xyxy_to_yolo(
    cls: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> list | None:
    x1 = min(max(x1, 0), img_w - 1)
    y1 = min(max(y1, 0), img_h - 1)
    x2 = min(max(x2, 0), img_w - 1)
    y2 = min(max(y2, 0), img_h - 1)

    bw = x2 - x1
    bh = y2 - y1

    if bw < 2 or bh < 2:
        return None

    cx = x1 + bw / 2
    cy = y1 + bh / 2

    return [cls, cx / img_w, cy / img_h, bw / img_w, bh / img_h]


def hflip(img: np.ndarray, labels: list[list]) -> tuple[np.ndarray, list[list]]:
    out = cv2.flip(img, 1)

    new_labels = []
    for cls, x, y, w, h in labels:
        new_labels.append([cls, 1.0 - x, y, w, h])

    return out, new_labels


def vflip(img: np.ndarray, labels: list[list]) -> tuple[np.ndarray, list[list]]:
    out = cv2.flip(img, 0)

    new_labels = []
    for cls, x, y, w, h in labels:
        new_labels.append([cls, x, 1.0 - y, w, h])

    return out, new_labels


def affine_transform(
    img: np.ndarray,
    labels: list[list],
    angle: float,
    scale: float = 1.0,
    tx_ratio: float = 0.0,
    ty_ratio: float = 0.0,
) -> tuple[np.ndarray, list[list]]:
    img_h, img_w = img.shape[:2]

    center = (img_w / 2, img_h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += tx_ratio * img_w
    matrix[1, 2] += ty_ratio * img_h

    out = cv2.warpAffine(
        img,
        matrix,
        (img_w, img_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    new_labels = []

    for label in labels:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(label, img_w, img_h)

        corners = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )

        ones = np.ones((4, 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        transformed = corners_h @ matrix.T

        nx1 = float(transformed[:, 0].min())
        ny1 = float(transformed[:, 1].min())
        nx2 = float(transformed[:, 0].max())
        ny2 = float(transformed[:, 1].max())

        new_label = xyxy_to_yolo(cls, nx1, ny1, nx2, ny2, img_w, img_h)

        if new_label is not None:
            new_labels.append(new_label)

    if labels and not new_labels:
        return img, labels

    return out, new_labels


def brightness_up(img: np.ndarray, labels: list[list]) -> tuple[np.ndarray, list[list]]:
    out = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    return out, [label.copy() for label in labels]


def brightness_down(img: np.ndarray, labels: list[list]) -> tuple[np.ndarray, list[list]]:
    out = cv2.convertScaleAbs(img, alpha=1.0, beta=-30)
    return out, [label.copy() for label in labels]


def contrast_up(img: np.ndarray, labels: list[list]) -> tuple[np.ndarray, list[list]]:
    out = cv2.convertScaleAbs(img, alpha=1.35, beta=0)
    return out, [label.copy() for label in labels]


def gaussian_noise(img: np.ndarray, labels: list[list]) -> tuple[np.ndarray, list[list]]:
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 8, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, [label.copy() for label in labels]


def clahe_enhance(img: np.ndarray, labels: list[list]) -> tuple[np.ndarray, list[list]]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return out, [label.copy() for label in labels]


def rotate_pos(img: np.ndarray, labels: list[list]) -> tuple[np.ndarray, list[list]]:
    return affine_transform(img, labels, angle=8.0)


def rotate_neg(img: np.ndarray, labels: list[list]) -> tuple[np.ndarray, list[list]]:
    return affine_transform(img, labels, angle=-8.0)


AUGMENTATIONS = [
    ("hflip", hflip),
    ("vflip", vflip),
    ("rotate_pos", rotate_pos),
    ("rotate_neg", rotate_neg),
    ("brightness_up", brightness_up),
    ("brightness_down", brightness_down),
    ("contrast_up", contrast_up),
    ("gaussian_noise", gaussian_noise),
    ("clahe", clahe_enhance),
]


def copy_original_split(split: str, manifest_rows: list[list[str]]) -> int:
    src_img_dir = SPLIT_DIR / "images" / split
    src_lab_dir = SPLIT_DIR / "labels" / split

    dst_img_dir = DET_DIR / "images" / split
    dst_lab_dir = DET_DIR / "labels" / split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lab_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for img_path in list_images(src_img_dir):
        label_path = src_lab_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"[WARN] Missing label, skipped: {img_path.name}")
            continue

        dst_img_path = dst_img_dir / img_path.name
        dst_lab_path = dst_lab_dir / label_path.name

        shutil.copy2(img_path, dst_img_path)
        shutil.copy2(label_path, dst_lab_path)

        manifest_rows.append(
            [
                split,
                "original",
                img_path.name,
                dst_img_path.name,
                label_path.name,
                dst_lab_path.name,
            ]
        )

        count += 1

    return count


def write_yaml() -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    yaml_text = (
        f"path: {DET_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: crack\n"
    )

    (DET_DIR / "crack.yaml").write_text(yaml_text, encoding="utf-8")
    (CONFIGS_DIR / "crack.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLIT_DIR.exists():
        raise FileNotFoundError(
            f"Split dataset not found: {SPLIT_DIR}\n"
            "Please run tools/02_split_dataset.py first."
        )

    reset_dir(DET_DIR)

    manifest_rows: list[list[str]] = []

    train_original = copy_original_split("train", manifest_rows)
    val_original = copy_original_split("val", manifest_rows)
    test_original = copy_original_split("test", manifest_rows)

    train_img_dir = SPLIT_DIR / "images" / "train"
    train_lab_dir = SPLIT_DIR / "labels" / "train"

    dst_train_img_dir = DET_DIR / "images" / "train"
    dst_train_lab_dir = DET_DIR / "labels" / "train"

    train_images = list_images(train_img_dir)

    created = 0
    failed = 0

    for img_path in train_images:
        label_path = train_lab_dir / f"{img_path.stem}.txt"

        try:
            img = imread_unicode(img_path)
            labels = read_labels(label_path)

            for aug_name, aug_func in AUGMENTATIONS:
                aug_img, aug_labels = aug_func(img, labels)

                out_stem = f"{img_path.stem}_{aug_name}"
                out_img_path = dst_train_img_dir / f"{out_stem}.jpg"
                out_lab_path = dst_train_lab_dir / f"{out_stem}.txt"

                imwrite_unicode(out_img_path, aug_img)
                write_labels(out_lab_path, aug_labels)

                manifest_rows.append(
                    [
                        "train",
                        aug_name,
                        img_path.name,
                        out_img_path.name,
                        label_path.name,
                        out_lab_path.name,
                    ]
                )

                created += 1

        except Exception as e:
            failed += 1
            print(f"[WARN] Failed to augment {img_path.name}: {e}")

    write_yaml()

    manifest_path = DET_DIR / "augmentation_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "augmentation",
                "source_image",
                "output_image",
                "source_label",
                "output_label",
            ]
        )
        writer.writerows(manifest_rows)

    final_train = len(list_images(DET_DIR / "images" / "train"))
    final_val = len(list_images(DET_DIR / "images" / "val"))
    final_test = len(list_images(DET_DIR / "images" / "test"))
    final_total = final_train + final_val + final_test

    log_path = LOG_DIR / "03_augment_dataset_log.txt"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Dataset augmentation log\n")
        f.write("========================\n\n")
        f.write("Augmentation strategy: deterministic single-operation offline augmentation\n")
        f.write("Only the training set is augmented. Validation and test sets remain unchanged.\n\n")
        f.write(f"Input split dir: {SPLIT_DIR}\n")
        f.write(f"Output det dir: {DET_DIR}\n\n")
        f.write(f"Train original: {train_original}\n")
        f.write(f"Val original: {val_original}\n")
        f.write(f"Test original: {test_original}\n\n")
        f.write(f"Augmentations per train image: {len(AUGMENTATIONS)}\n")
        f.write(f"Augmentation types: {', '.join(name for name, _ in AUGMENTATIONS)}\n")
        f.write(f"Augmented created: {created}\n")
        f.write(f"Augmentation failed: {failed}\n\n")
        f.write(f"Final train: {final_train}\n")
        f.write(f"Final val: {final_val}\n")
        f.write(f"Final test: {final_test}\n")
        f.write(f"Final total: {final_total}\n")
        f.write(f"Manifest: {manifest_path}\n")

    print("Augmentation finished.")
    print(f"Strategy: deterministic single-operation offline augmentation")
    print(f"Train original: {train_original}")
    print(f"Val original: {val_original}")
    print(f"Test original: {test_original}")
    print(f"Augmentations per train image: {len(AUGMENTATIONS)}")
    print(f"Augmented created: {created}")
    print(f"Augmentation failed: {failed}")
    print(f"Final train: {final_train}")
    print(f"Final val: {final_val}")
    print(f"Final test: {final_test}")
    print(f"Final total: {final_total}")
    print(f"Manifest: {manifest_path}")
    print(f"YAML: {CONFIGS_DIR / 'crack.yaml'}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()