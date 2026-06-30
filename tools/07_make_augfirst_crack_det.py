from __future__ import annotations

import csv
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "datasets" / "crack_raw" / "images_renamed"
LABEL_DIR = ROOT / "datasets" / "crack_raw" / "labels_raw"
DET_DIR = ROOT / "datasets" / "crack_det"
CONFIGS_DIR = ROOT / "configs"
LOG_DIR = ROOT / "logs"

TMP_IMG_DIR = DET_DIR / "_tmp_augfirst" / "images"
TMP_LAB_DIR = DET_DIR / "_tmp_augfirst" / "labels"

AUTODL_DATASET_PATH = "/root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
SEED = 42
JPEG_QUALITY = 95
MAX_WORKERS = min(4, max(1, os.cpu_count() or 1))

cv2.setNumThreads(1)

AUGMENTATION_TYPES = (
    "original",
    "hflip",
    "vflip",
    "brightness_contrast",
    "clahe_gamma",
    "noise",
    "blur",
)


@dataclass(frozen=True)
class YoloLabel:
    cls: int
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class SourcePair:
    index: int
    image_path: Path
    label_path: Path
    labels: tuple[YoloLabel, ...]


@dataclass
class SampleRecord:
    image_name: str
    label_name: str
    source_image: str
    source_label: str
    augmentation_type: str
    tmp_image_path: Path
    tmp_label_path: Path
    split: str = ""


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def rel_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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


def validate_label_file(label_path: Path) -> tuple[list[YoloLabel], list[str]]:
    labels: list[YoloLabel] = []
    errors: list[str] = []

    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()
        line_errors: list[str] = []

        if len(parts) != 5:
            errors.append(f"{label_path.name} line {line_no}: expected 5 columns, got {len(parts)}")
            continue

        try:
            cls_float = float(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{label_path.name} line {line_no}: non-numeric label value")
            continue

        if not cls_float.is_integer() or int(cls_float) != 0:
            errors.append(f"{label_path.name} line {line_no}: class must be 0, got {parts[0]}")
            continue

        if not (0.0 <= x <= 1.0):
            line_errors.append(f"{label_path.name} line {line_no}: x_center out of [0, 1], got {x}")

        if not (0.0 <= y <= 1.0):
            line_errors.append(f"{label_path.name} line {line_no}: y_center out of [0, 1], got {y}")

        if not (0.0 <= w <= 1.0):
            line_errors.append(f"{label_path.name} line {line_no}: width out of [0, 1], got {w}")

        if not (0.0 <= h <= 1.0):
            line_errors.append(f"{label_path.name} line {line_no}: height out of [0, 1], got {h}")

        if w <= 0.0:
            line_errors.append(f"{label_path.name} line {line_no}: width must be > 0, got {w}")

        if h <= 0.0:
            line_errors.append(f"{label_path.name} line {line_no}: height must be > 0, got {h}")

        if line_errors:
            errors.extend(line_errors)
        else:
            labels.append(YoloLabel(0, x, y, w, h))

    return labels, errors


def collect_source_pairs() -> list[SourcePair]:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image dir not found: {IMAGE_DIR}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label dir not found: {LABEL_DIR}")

    images = list_images(IMAGE_DIR)

    if not images:
        raise RuntimeError(f"No images found in: {IMAGE_DIR}")

    label_files = sorted(
        p for p in LABEL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in label_files}

    errors: list[str] = []

    for stem in sorted(image_stems - label_stems):
        errors.append(f"missing label for image: {stem}")

    for stem in sorted(label_stems - image_stems):
        errors.append(f"label has no matching image: {stem}.txt")

    pairs: list[SourcePair] = []

    for index, img_path in enumerate(images, start=1):
        label_path = LABEL_DIR / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        labels, label_errors = validate_label_file(label_path)

        if label_errors:
            errors.extend(label_errors)
            continue

        pairs.append(
            SourcePair(
                index=index,
                image_path=img_path,
                label_path=label_path,
                labels=tuple(labels),
            )
        )

    if errors:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        error_log = LOG_DIR / "07_make_augfirst_crack_det_errors.txt"
        error_log.write_text("\n".join(errors), encoding="utf-8")
        raise RuntimeError(
            f"Input validation failed with {len(errors)} error(s). "
            f"See: {error_log}"
        )

    if not pairs:
        raise RuntimeError("No valid image-label pairs found.")

    return pairs


def labels_to_lines(labels: list[YoloLabel] | tuple[YoloLabel, ...]) -> list[str]:
    lines: list[str] = []

    for label in labels:
        x = min(max(label.x, 0.0), 1.0)
        y = min(max(label.y, 0.0), 1.0)
        w = min(max(label.w, 0.0), 1.0)
        h = min(max(label.h, 0.0), 1.0)

        if w <= 0.0 or h <= 0.0:
            continue

        lines.append(f"{label.cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return lines


def write_labels(path: Path, labels: list[YoloLabel] | tuple[YoloLabel, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = labels_to_lines(labels)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_labels(labels: tuple[YoloLabel, ...]) -> list[YoloLabel]:
    return [YoloLabel(label.cls, label.x, label.y, label.w, label.h) for label in labels]


def hflip_labels(labels: tuple[YoloLabel, ...]) -> list[YoloLabel]:
    return [YoloLabel(label.cls, 1.0 - label.x, label.y, label.w, label.h) for label in labels]


def vflip_labels(labels: tuple[YoloLabel, ...]) -> list[YoloLabel]:
    return [YoloLabel(label.cls, label.x, 1.0 - label.y, label.w, label.h) for label in labels]


def make_brightness_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.90, 1.12)
    beta = rng.uniform(-12.0, 12.0)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, table)


def make_clahe_gamma(img: np.ndarray, rng: random.Random) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clip_limit = rng.uniform(1.2, 1.8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    out = cv2.merge([l_channel, a_channel, b_channel])
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    gamma = rng.uniform(0.90, 1.10)
    return apply_gamma(out, gamma)


def make_noise(
    img: np.ndarray,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> np.ndarray:
    sigma = rng.uniform(2.0, 5.0)
    noise = np_rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def make_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    blur_sigma = rng.uniform(0.35, 0.75)
    return cv2.GaussianBlur(img, (3, 3), blur_sigma)


def write_sample_image(
    out_path: Path,
    img: np.ndarray,
    source_image: Path,
    augmentation_type: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if augmentation_type == "original" and source_image.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source_image, out_path)
        return

    imwrite_unicode(out_path, img)


def make_sample(
    pair: SourcePair,
    img: np.ndarray,
    augmentation_type: str,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, list[YoloLabel], str | None]:
    if augmentation_type == "original":
        return img, copy_labels(pair.labels), None

    if augmentation_type == "hflip":
        return cv2.flip(img, 1), hflip_labels(pair.labels), None

    if augmentation_type == "vflip":
        return cv2.flip(img, 0), vflip_labels(pair.labels), None

    if augmentation_type == "brightness_contrast":
        return make_brightness_contrast(img, rng), copy_labels(pair.labels), None

    if augmentation_type == "clahe_gamma":
        return make_clahe_gamma(img, rng), copy_labels(pair.labels), None

    if augmentation_type == "noise":
        return make_noise(img, rng, np_rng), copy_labels(pair.labels), None

    if augmentation_type == "blur":
        return make_blur(img, rng), copy_labels(pair.labels), None

    raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def ensure_safe_output_dir() -> None:
    expected = ROOT / "datasets" / "crack_det"

    if DET_DIR.resolve() != expected.resolve():
        raise RuntimeError(f"Refusing to reset unexpected output dir: {DET_DIR}")

    raw_root = (ROOT / "datasets" / "crack_raw").resolve()
    det_root = DET_DIR.resolve()

    if det_root == raw_root or raw_root in det_root.parents:
        raise RuntimeError(f"Refusing to write inside raw data dir: {DET_DIR}")


def generate_all_samples(pairs: list[SourcePair]) -> tuple[list[SampleRecord], list[str]]:
    ensure_safe_output_dir()
    reset_dir(DET_DIR)
    TMP_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_LAB_DIR.mkdir(parents=True, exist_ok=True)

    records: list[SampleRecord] = []
    skipped: list[str] = []
    worker_count = min(MAX_WORKERS, len(pairs))

    print(f"Workers: {worker_count}", flush=True)

    if worker_count == 1:
        for completed, pair in enumerate(pairs, start=1):
            pair_records, pair_skipped = process_pair(pair)
            records.extend(pair_records)
            skipped.extend(pair_skipped)

            if completed % 50 == 0 or completed == len(pairs):
                print(f"Processed source images: {completed}/{len(pairs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(process_pair, pair) for pair in pairs]

            for completed, future in enumerate(as_completed(futures), start=1):
                pair_records, pair_skipped = future.result()
                records.extend(pair_records)
                skipped.extend(pair_skipped)

                if completed % 50 == 0 or completed == len(pairs):
                    print(f"Processed source images: {completed}/{len(pairs)}", flush=True)

    skipped.sort()

    if not records:
        raise RuntimeError("No samples were generated.")

    return records, skipped


def process_pair(pair: SourcePair) -> tuple[list[SampleRecord], list[str]]:
    cv2.setNumThreads(1)

    rng_seed = SEED + pair.index * 104729
    rng = random.Random(rng_seed)
    np_rng = np.random.default_rng(rng_seed)
    img = imread_unicode(pair.image_path)

    records: list[SampleRecord] = []
    skipped: list[str] = []

    for augmentation_type in AUGMENTATION_TYPES:
        out_img, out_labels, skip_reason = make_sample(
            pair=pair,
            img=img,
            augmentation_type=augmentation_type,
            rng=rng,
            np_rng=np_rng,
        )

        out_stem = f"crack_{pair.index:06d}_{augmentation_type}"
        image_name = f"{out_stem}.jpg"
        label_name = f"{out_stem}.txt"
        tmp_image_path = TMP_IMG_DIR / image_name
        tmp_label_path = TMP_LAB_DIR / label_name

        if skip_reason is not None:
            skipped.append(
                f"{pair.image_path.name},{pair.label_path.name},"
                f"{augmentation_type},{skip_reason}"
            )
            continue

        write_sample_image(tmp_image_path, out_img, pair.image_path, augmentation_type)
        write_labels(tmp_label_path, out_labels)

        records.append(
            SampleRecord(
                image_name=image_name,
                label_name=label_name,
                source_image=rel_path(pair.image_path),
                source_label=rel_path(pair.label_path),
                augmentation_type=augmentation_type,
                tmp_image_path=tmp_image_path,
                tmp_label_path=tmp_label_path,
            )
        )

    return records, skipped


def split_records(records: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
    rng = random.Random(SEED)
    records = sorted(records, key=lambda record: record.image_name)
    rng.shuffle(records)

    total = len(records)
    n_train = int(total * 0.7)
    n_val = int(total * 0.1)

    split_map = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }

    for split, split_records_ in split_map.items():
        for record in split_records_:
            record.split = split

    return split_map


def move_records_to_splits(split_map: dict[str, list[SampleRecord]]) -> None:
    for split in SPLITS:
        (DET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, records in split_map.items():
        img_dir = DET_DIR / "images" / split
        lab_dir = DET_DIR / "labels" / split

        for record in records:
            shutil.move(str(record.tmp_image_path), str(img_dir / record.image_name))
            shutil.move(str(record.tmp_label_path), str(lab_dir / record.label_name))

    tmp_root = DET_DIR / "_tmp_augfirst"

    if tmp_root.exists():
        shutil.rmtree(tmp_root)


def count_split_files(split: str) -> tuple[int, int]:
    img_dir = DET_DIR / "images" / split
    lab_dir = DET_DIR / "labels" / split

    image_count = len(list_images(img_dir))
    label_count = len([
        p for p in lab_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ])

    return image_count, label_count


def verify_final_dataset() -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}

    for split in SPLITS:
        img_dir = DET_DIR / "images" / split
        lab_dir = DET_DIR / "labels" / split

        images = list_images(img_dir)
        labels = sorted(
            p for p in lab_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        )

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)

        if len(images) != len(labels) or missing_labels or missing_images:
            details = [
                f"{split}: images={len(images)}, labels={len(labels)}",
                f"{split}: images without labels={missing_labels[:20]}",
                f"{split}: labels without images={missing_images[:20]}",
            ]
            raise RuntimeError("Final dataset image/label mismatch:\n" + "\n".join(details))

        counts[split] = (len(images), len(labels))

    return counts


def write_manifest(split_map: dict[str, list[SampleRecord]]) -> Path:
    manifest_path = DET_DIR / "split_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "image_name",
            "label_name",
            "source_image",
            "source_label",
            "augmentation_type",
        ])

        for split in SPLITS:
            for record in split_map[split]:
                writer.writerow([
                    record.split,
                    record.image_name,
                    record.label_name,
                    record.source_image,
                    record.source_label,
                    record.augmentation_type,
                ])

    return manifest_path


def dataset_yaml(path_value: str) -> str:
    return (
        f"path: {path_value}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: crack\n"
    )


def write_yaml_files() -> tuple[Path, Path, Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    local_yaml = dataset_yaml(DET_DIR.as_posix())
    autodl_yaml = dataset_yaml(AUTODL_DATASET_PATH)

    dataset_yaml_path = DET_DIR / "crack.yaml"
    local_config_path = CONFIGS_DIR / "crack.yaml"
    autodl_config_path = CONFIGS_DIR / "crack_autodl.yaml"

    dataset_yaml_path.write_text(local_yaml, encoding="utf-8")
    local_config_path.write_text(local_yaml, encoding="utf-8")
    autodl_config_path.write_text(autodl_yaml, encoding="utf-8")

    return dataset_yaml_path, local_config_path, autodl_config_path


def write_log(
    pairs: list[SourcePair],
    records: list[SampleRecord],
    skipped: list[str],
    counts: dict[str, tuple[int, int]],
    manifest_path: Path,
    yaml_paths: tuple[Path, Path, Path],
) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "07_make_augfirst_crack_det_log.txt"

    lines = [
        "Aug-first crack_det generation log",
        "==================================",
        "",
        f"Seed: {SEED}",
        f"Input images: {IMAGE_DIR}",
        f"Input labels: {LABEL_DIR}",
        f"Output dir: {DET_DIR}",
        f"Raw images: {len(pairs)}",
        f"Generated total: {len(records)}",
        f"Skipped augmented samples: {len(skipped)}",
        "",
    ]

    for split in SPLITS:
        image_count, label_count = counts[split]
        lines.append(f"{split}: images={image_count}, labels={label_count}")

    lines.extend([
        "",
        f"Manifest: {manifest_path}",
        f"Dataset yaml: {yaml_paths[0]}",
        f"Local config yaml: {yaml_paths[1]}",
        f"AutoDL config yaml: {yaml_paths[2]}",
        "",
        "Skipped samples:",
    ])

    if skipped:
        lines.extend(skipped)
    else:
        lines.append("None")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def main() -> None:
    print(f"Image dir: {IMAGE_DIR}")
    print(f"Label dir: {LABEL_DIR}")
    print(f"Output dir: {DET_DIR}")

    pairs = collect_source_pairs()
    records, skipped = generate_all_samples(pairs)
    split_map = split_records(records)
    move_records_to_splits(split_map)
    manifest_path = write_manifest(split_map)
    yaml_paths = write_yaml_files()
    counts = verify_final_dataset()
    log_path = write_log(pairs, records, skipped, counts, manifest_path, yaml_paths)

    total_images = sum(image_count for image_count, _ in counts.values())
    total_labels = sum(label_count for _, label_count in counts.values())

    print("")
    print("Aug-first dataset generation finished.")
    print(f"Raw images: {len(pairs)}")
    print(f"Generated total images/labels: {total_images}/{total_labels}")

    for split in SPLITS:
        image_count, label_count = counts[split]
        print(f"{split}: images={image_count}, labels={label_count}")

    print(f"Skipped augmented samples: {len(skipped)}")
    print(f"Output dir: {DET_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"Dataset yaml: {yaml_paths[0]}")
    print(f"Local config yaml: {yaml_paths[1]}")
    print(f"AutoDL config yaml: {yaml_paths[2]}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
