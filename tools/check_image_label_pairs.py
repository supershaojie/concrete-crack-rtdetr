from __future__ import annotations

from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "datasets" / "crack_raw" / "images_renamed"
LABEL_DIR = ROOT / "datasets" / "crack_raw" / "labels_raw"
LOG_DIR = ROOT / "logs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALLOWED_CLASSES = {"0"}


def check_label_file(label_path: Path) -> list[str]:
    errors: list[str] = []

    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(f"{label_path.name} line {line_no}: expected 5 values, got {len(parts)}")
            continue

        cls, x, y, w, h = parts

        if cls not in ALLOWED_CLASSES:
            errors.append(f"{label_path.name} line {line_no}: invalid class id {cls}")

        try:
            x, y, w, h = map(float, (x, y, w, h))
        except ValueError:
            errors.append(f"{label_path.name} line {line_no}: bbox values are not numbers")
            continue

        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(
                f"{label_path.name} line {line_no}: bbox out of range "
                f"x={x}, y={y}, w={w}, h={h}"
            )

    return errors


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label folder not found: {LABEL_DIR}")

    image_files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    label_files = sorted(
        p for p in LABEL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )

    image_stems = Counter(p.stem for p in image_files)
    label_stems = Counter(p.stem for p in label_files)

    errors: list[str] = []
    warnings: list[str] = []

    duplicate_images = [stem for stem, count in image_stems.items() if count > 1]
    duplicate_labels = [stem for stem, count in label_stems.items() if count > 1]

    for stem in duplicate_images:
        errors.append(f"duplicate image stem: {stem}")

    for stem in duplicate_labels:
        errors.append(f"duplicate label stem: {stem}")

    for img_path in image_files:
        label_path = LABEL_DIR / f"{img_path.stem}.txt"

        if not label_path.exists():
            errors.append(f"missing label for image: {img_path.name}")
            continue

        label_errors = check_label_file(label_path)

        if label_errors:
            errors.extend(label_errors)

    for label_path in label_files:
        if label_path.stem not in image_stems:
            errors.append(f"label has no matching image: {label_path.name}")

    empty_labels = []
    total_boxes = 0

    for label_path in label_files:
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels.append(label_path.name)

        total_boxes += len(lines)

    if empty_labels:
        warnings.append(f"empty label files: {len(empty_labels)}")

    matched = len([img for img in image_files if (LABEL_DIR / f"{img.stem}.txt").exists()])

    log_path = LOG_DIR / "01_check_dataset_log.txt"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Dataset check log\n")
        f.write("=================\n\n")
        f.write(f"Image dir: {IMAGE_DIR}\n")
        f.write(f"Label dir: {LABEL_DIR}\n\n")
        f.write(f"Images: {len(image_files)}\n")
        f.write(f"Labels: {len(label_files)}\n")
        f.write(f"Matched image-label pairs: {matched}\n")
        f.write(f"Total boxes: {total_boxes}\n")
        f.write(f"Empty label files: {len(empty_labels)}\n\n")

        if warnings:
            f.write("Warnings:\n")
            for item in warnings:
                f.write(f"- {item}\n")
            f.write("\n")

        if errors:
            f.write("Errors:\n")
            for item in errors:
                f.write(f"- {item}\n")
        else:
            f.write("No errors found.\n")

    print("Dataset check finished.")
    print(f"Images: {len(image_files)}")
    print(f"Labels: {len(label_files)}")
    print(f"Matched pairs: {matched}")
    print(f"Total boxes: {total_boxes}")
    print(f"Empty label files: {len(empty_labels)}")
    print(f"Log saved to: {log_path}")

    if warnings:
        print(f"Warnings: {len(warnings)}")

    if errors:
        print(f"Errors: {len(errors)}")
        print("Please fix errors before splitting dataset.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()