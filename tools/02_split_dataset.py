from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "datasets" / "crack_raw" / "images_renamed"
LABEL_DIR = ROOT / "datasets" / "crack_raw" / "labels_raw"
OUT_DIR = ROOT / "datasets" / "crack_raw" / "split_raw"
LOG_DIR = ROOT / "logs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 7:2:1 split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

SEED = 42


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def validate_label(label_path: Path) -> list[str]:
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

        if cls != "0":
            errors.append(f"{label_path.name} line {line_no}: class id should be 0, got {cls}")

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


def collect_valid_pairs() -> tuple[list[tuple[Path, Path]], list[str]]:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image dir not found: {IMAGE_DIR}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label dir not found: {LABEL_DIR}")

    pairs: list[tuple[Path, Path]] = []
    errors: list[str] = []

    images = list_images(IMAGE_DIR)

    for img_path in images:
        label_path = LABEL_DIR / f"{img_path.stem}.txt"

        if not label_path.exists():
            errors.append(f"missing label: {img_path.name}")
            continue

        label_errors = validate_label(label_path)

        if label_errors:
            errors.extend(label_errors)
            continue

        pairs.append((img_path, label_path))

    label_files = sorted(
        p for p in LABEL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )
    image_stems = {p.stem for p in images}

    for label_path in label_files:
        if label_path.stem not in image_stems:
            errors.append(f"label has no matching image: {label_path.name}")

    return pairs, errors


def copy_pairs(pairs: list[tuple[Path, Path]], split: str) -> None:
    img_out = OUT_DIR / "images" / split
    lab_out = OUT_DIR / "labels" / split

    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in pairs:
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(label_path, lab_out / label_path.name)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Image dir: {IMAGE_DIR}")
    print(f"Label dir: {LABEL_DIR}")
    print(f"Output dir: {OUT_DIR}")

    pairs, errors = collect_valid_pairs()

    if errors:
        log_path = LOG_DIR / "02_split_dataset_errors.txt"
        log_path.write_text("\n".join(errors), encoding="utf-8")

        print(f"Found errors: {len(errors)}")
        print(f"Error log saved to: {log_path}")
        print("Please fix these errors before splitting dataset.")
        raise SystemExit(1)

    if not pairs:
        raise RuntimeError("No valid image-label pairs found.")

    random.seed(SEED)
    random.shuffle(pairs)

    total = len(pairs)

    n_train = round(total * TRAIN_RATIO)
    n_val = round(total * VAL_RATIO)
    n_test = total - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    reset_dir(OUT_DIR)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(test_pairs, "test")

    manifest_path = OUT_DIR / "split_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "label"])

        for split_name, split_pairs in [
            ("train", train_pairs),
            ("val", val_pairs),
            ("test", test_pairs),
        ]:
            for img_path, label_path in split_pairs:
                writer.writerow([split_name, img_path.name, label_path.name])

    log_path = LOG_DIR / "02_split_dataset_log.txt"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Dataset split log\n")
        f.write("=================\n\n")
        f.write(f"Image dir: {IMAGE_DIR}\n")
        f.write(f"Label dir: {LABEL_DIR}\n")
        f.write(f"Output dir: {OUT_DIR}\n")
        f.write(f"Seed: {SEED}\n\n")
        f.write(f"Train ratio: {TRAIN_RATIO}\n")
        f.write(f"Val ratio: {VAL_RATIO}\n")
        f.write(f"Test ratio: {TEST_RATIO}\n\n")
        f.write(f"Total: {total}\n")
        f.write(f"Train: {len(train_pairs)}\n")
        f.write(f"Val: {len(val_pairs)}\n")
        f.write(f"Test: {len(test_pairs)}\n")

    print("\nSplit finished.")
    print(f"Total: {total}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val: {len(val_pairs)}")
    print(f"Test: {len(test_pairs)}")
    print(f"Output: {OUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()