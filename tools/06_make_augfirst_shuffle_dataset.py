from __future__ import annotations

import csv
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = ROOT / "datasets" / "crack_det"
OUT_DIR = ROOT / "datasets" / "crack_det_augfirst_shuffle"
CONFIGS_DIR = ROOT / "configs"

AUTODL_DATASET_PATH = "/root/autodl-tmp/projects/Crack_RTDETR/datasets/crack_det_augfirst_shuffle"

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
COPY_WORKERS = 8


@dataclass(frozen=True)
class Sample:
    original_split: str
    image_path: Path
    label_path: Path

    @property
    def image_name(self) -> str:
        return self.image_path.name

    @property
    def label_name(self) -> str:
        return self.label_path.name


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []

    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def list_labels(path: Path) -> list[Path]:
    if not path.exists():
        return []

    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )


def find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)

    return sorted(duplicates)


def collect_samples() -> list[Sample]:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_DIR}")

    samples: list[Sample] = []
    errors: list[str] = []

    for split in SPLITS:
        image_dir = SOURCE_DIR / "images" / split
        label_dir = SOURCE_DIR / "labels" / split

        if not image_dir.exists():
            errors.append(f"Missing image dir: {image_dir}")
            continue

        if not label_dir.exists():
            errors.append(f"Missing label dir: {label_dir}")
            continue

        images = list_images(image_dir)
        labels = list_labels(label_dir)

        if len(images) != len(labels):
            errors.append(
                f"{split}: image/label count mismatch "
                f"({len(images)} images, {len(labels)} labels)"
            )

        duplicate_image_stems = find_duplicates([p.stem for p in images])
        duplicate_label_stems = find_duplicates([p.stem for p in labels])

        for stem in duplicate_image_stems:
            errors.append(f"{split}: duplicate image stem: {stem}")

        for stem in duplicate_label_stems:
            errors.append(f"{split}: duplicate label stem: {stem}")

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                errors.append(f"{split}: missing label for image: {image_path.name}")
                continue

            samples.append(
                Sample(
                    original_split=split,
                    image_path=image_path,
                    label_path=label_path,
                )
            )

        for label_path in labels:
            if label_path.stem not in image_stems:
                errors.append(f"{split}: label has no matching image: {label_path.name}")

        for image_path in images:
            if image_path.stem not in label_stems:
                errors.append(f"{split}: image has no matching label: {image_path.name}")

    duplicate_image_names = find_duplicates([sample.image_name for sample in samples])
    duplicate_label_names = find_duplicates([sample.label_name for sample in samples])

    for image_name in duplicate_image_names:
        errors.append(f"Duplicate image name across source splits: {image_name}")

    for label_name in duplicate_label_names:
        errors.append(f"Duplicate label name across source splits: {label_name}")

    if errors:
        print("Dataset validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    if not samples:
        raise RuntimeError("No image-label pairs found.")

    return samples


def make_split(samples: list[Sample]) -> dict[str, list[Sample]]:
    shuffled = samples.copy()
    random.seed(SEED)
    random.shuffle(shuffled)

    total = len(shuffled)
    n_train = round(total * TRAIN_RATIO)
    n_val = round(total * VAL_RATIO)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def reset_output_dir() -> None:
    output_root = OUT_DIR.resolve()
    source_root = SOURCE_DIR.resolve()
    datasets_root = (ROOT / "datasets").resolve()

    if output_root == source_root:
        raise RuntimeError("Refusing to overwrite the source dataset.")

    if not output_root.is_relative_to(datasets_root):
        raise RuntimeError(f"Refusing to reset unexpected output path: {OUT_DIR}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split in SPLITS:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_one(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


def copy_split(split_samples: dict[str, list[Sample]]) -> None:
    copy_jobs: list[tuple[Path, Path]] = []

    for split, samples in split_samples.items():
        image_out_dir = OUT_DIR / "images" / split
        label_out_dir = OUT_DIR / "labels" / split

        for sample in samples:
            copy_jobs.append((sample.image_path, image_out_dir / sample.image_name))
            copy_jobs.append((sample.label_path, label_out_dir / sample.label_name))

    with ThreadPoolExecutor(max_workers=COPY_WORKERS) as executor:
        futures = [
            executor.submit(copy_one, src, dst)
            for src, dst in copy_jobs
        ]

        for future in as_completed(futures):
            future.result()


def write_manifest(split_samples: dict[str, list[Sample]]) -> Path:
    manifest_path = OUT_DIR / "split_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "new_split",
                "image_name",
                "label_name",
                "original_split",
                "source_path",
            ]
        )

        for split in SPLITS:
            for sample in split_samples[split]:
                writer.writerow(
                    [
                        split,
                        sample.image_name,
                        sample.label_name,
                        sample.original_split,
                        sample.image_path.as_posix(),
                    ]
                )

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


def write_yaml_files() -> list[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    local_yaml = dataset_yaml(OUT_DIR.as_posix())
    autodl_yaml = dataset_yaml(AUTODL_DATASET_PATH)

    output_paths = [
        OUT_DIR / "crack.yaml",
        CONFIGS_DIR / "crack_augfirst_shuffle.yaml",
        CONFIGS_DIR / "crack_augfirst_shuffle_autodl.yaml",
    ]

    output_paths[0].write_text(local_yaml, encoding="utf-8")
    output_paths[1].write_text(local_yaml, encoding="utf-8")
    output_paths[2].write_text(autodl_yaml, encoding="utf-8")

    return output_paths


def count_split(split: str) -> tuple[int, int]:
    image_count = len(list_images(OUT_DIR / "images" / split))
    label_count = len(list_labels(OUT_DIR / "labels" / split))
    return image_count, label_count


def validate_output_counts() -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    errors: list[str] = []

    for split in SPLITS:
        image_count, label_count = count_split(split)
        counts[split] = (image_count, label_count)

        if image_count != label_count:
            errors.append(
                f"{split}: image/label count mismatch "
                f"({image_count} images, {label_count} labels)"
            )

    if errors:
        print("Output validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    return counts


def main() -> None:
    print(f"Source dataset: {SOURCE_DIR}")
    print(f"Output dataset: {OUT_DIR}")
    print(f"Seed: {SEED}")
    print(f"Copy workers: {COPY_WORKERS}")

    samples = collect_samples()
    split_samples = make_split(samples)

    reset_output_dir()
    copy_split(split_samples)
    manifest_path = write_manifest(split_samples)
    yaml_paths = write_yaml_files()
    counts = validate_output_counts()

    print("\nAug-first shuffle dataset created.")
    print(f"Total samples: {len(samples)}")
    print(f"Manifest: {manifest_path}")
    for yaml_path in yaml_paths:
        print(f"YAML: {yaml_path}")

    print("\nSplit counts:")
    for split in SPLITS:
        image_count, label_count = counts[split]
        print(f"{split}: images={image_count}, labels={label_count}")


if __name__ == "__main__":
    main()
