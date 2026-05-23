# -*- coding: utf-8 -*-
"""
check_image_label_pairs.py

功能：
检查 images_renamed 和 labels_raw 是否对应。

主要检查：
1. 有图片但没有标签
2. 有标签但没有图片
3. YOLO 标签格式是否正常
4. 前 700 张是否都有标签
5. 701~1440 是否还没有标签或部分有标签
"""

from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "images_renamed"
LABEL_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "labels_raw"
META_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "meta"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_OLD_LABEL_NUM = 700


def check_yolo_label(label_path: Path):
    """
    检查 YOLO bbox 标签格式：
    class x_center y_center width height

    每行 5 列；
    class 是整数；
    坐标在 0~1 范围内。
    """
    errors = []

    text = label_path.read_text(encoding="utf-8").strip()

    # 空标签不一定是错，有些图片可能无目标。
    # 但裂缝检测通常每张图都有裂缝，所以这里只记录为空。
    if text == "":
        errors.append("empty_label")
        return errors

    lines = text.splitlines()

    for line_idx, line in enumerate(lines, start=1):
        parts = line.strip().split()

        if len(parts) != 5:
            errors.append(f"line_{line_idx}: column_count_not_5")
            continue

        cls, x, y, w, h = parts

        try:
            cls_int = int(cls)
        except Exception:
            errors.append(f"line_{line_idx}: class_not_int")
            continue

        try:
            vals = [float(x), float(y), float(w), float(h)]
        except Exception:
            errors.append(f"line_{line_idx}: bbox_not_float")
            continue

        for name, value in zip(["x", "y", "w", "h"], vals):
            if value < 0 or value > 1:
                errors.append(f"line_{line_idx}: {name}_out_of_range_{value}")

        if vals[2] <= 0 or vals[3] <= 0:
            errors.append(f"line_{line_idx}: width_or_height_le_0")

    return errors


def main():
    print("=" * 80)
    print("检查图片和标签是否对应")
    print("=" * 80)
    print(f"图片目录: {IMAGE_DIR}")
    print(f"标签目录: {LABEL_DIR}")
    print("=" * 80)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"图片目录不存在: {IMAGE_DIR}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"标签目录不存在: {LABEL_DIR}")

    META_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda x: x.name.lower()
    )

    label_files = sorted(
        [p for p in LABEL_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda x: x.name.lower()
    )

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    images_without_labels = sorted(image_stems - label_stems)
    labels_without_images = sorted(label_stems - image_stems)
    matched = sorted(image_stems & label_stems)

    print(f"图片数量: {len(image_files)}")
    print(f"标签数量: {len(label_files)}")
    print(f"图片和标签匹配数量: {len(matched)}")
    print(f"有图片但没有标签数量: {len(images_without_labels)}")
    print(f"有标签但没有图片数量: {len(labels_without_images)}")

    print("\n前 20 个没有标签的图片：")
    for stem in images_without_labels[:20]:
        print(f"  {stem}")

    print("\n前 20 个没有对应图片的标签：")
    for stem in labels_without_images[:20]:
        print(f"  {stem}")

    # 检查前 700
    missing_in_first_700 = []
    for i in range(1, EXPECTED_OLD_LABEL_NUM + 1):
        stem = f"crack_{i:05d}"
        if stem not in image_stems:
            missing_in_first_700.append((stem, "missing_image"))
        elif stem not in label_stems:
            missing_in_first_700.append((stem, "missing_label"))

    print("\n前 700 张检查：")
    if not missing_in_first_700:
        print("  前 700 张图片和标签文件名全部对应。")
    else:
        print(f"  前 700 张存在问题数量: {len(missing_in_first_700)}")
        for item in missing_in_first_700[:50]:
            print(" ", item)

    # 检查标签格式
    label_errors = []
    for label_path in label_files:
        errors = check_yolo_label(label_path)
        if errors:
            label_errors.append({
                "label": label_path.name,
                "errors": "; ".join(errors)
            })

    print("\n标签格式检查：")
    print(f"  存在格式问题的标签数量: {len(label_errors)}")

    # 输出报告
    report_csv = META_DIR / "image_label_pair_check_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "detail"])

        for stem in images_without_labels:
            writer.writerow(["image_without_label", stem, ""])

        for stem in labels_without_images:
            writer.writerow(["label_without_image", stem, ""])

        for stem, detail in missing_in_first_700:
            writer.writerow(["first_700_problem", stem, detail])

        for item in label_errors:
            writer.writerow(["label_format_error", item["label"], item["errors"]])

    print(f"\n检查报告已保存: {report_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
