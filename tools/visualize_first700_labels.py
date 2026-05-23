# -*- coding: utf-8 -*-
"""
visualize_first700_labels.py

功能：
把前 700 张已经标注过的图片和标签画出来，保存到 vis_check_first700。
用于检查旧标签是否和当前图片一一对应。

如果框都能正常框住裂缝，说明旧标签可以继续用。
如果大量框偏移或者框到不相关位置，说明图片顺序可能变了，不能直接接着用。
"""

from pathlib import Path
import cv2
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "images_renamed"
LABEL_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "labels_raw"
OUT_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "vis_check_first700"

IMAGE_EXT = ".jpg"

# 是否只抽样检查
SAMPLE_ONLY = True

# 抽样数量
SAMPLE_NUM = 80

# 固定随机种子
RANDOM_SEED = 2026

# 前多少张属于旧标注
OLD_NUM = 700


def read_yolo_label(label_path: Path):
    boxes = []

    if not label_path.exists():
        return boxes

    text = label_path.read_text(encoding="utf-8").strip()
    if text == "":
        return boxes

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])

        boxes.append((cls, x, y, w, h))

    return boxes


def draw_boxes(img, boxes):
    h_img, w_img = img.shape[:2]

    for cls, x, y, w, h in boxes:
        x1 = int((x - w / 2) * w_img)
        y1 = int((y - h / 2) * h_img)
        x2 = int((x + w / 2) * w_img)
        y2 = int((y + h / 2) * h_img)

        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img,
            str(cls),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    return img


def main():
    print("=" * 80)
    print("可视化检查前 700 张旧标签")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    indices = list(range(1, OLD_NUM + 1))

    if SAMPLE_ONLY:
        random.seed(RANDOM_SEED)
        indices = random.sample(indices, min(SAMPLE_NUM, len(indices)))
        indices = sorted(indices)

    print(f"输出目录: {OUT_DIR}")
    print(f"检查数量: {len(indices)}")

    count = 0
    missing = 0

    for i in indices:
        stem = f"crack_{i:05d}"
        image_path = IMAGE_DIR / f"{stem}{IMAGE_EXT}"
        label_path = LABEL_DIR / f"{stem}.txt"

        if not image_path.exists() or not label_path.exists():
            print(f"[缺失] {stem} image_exists={image_path.exists()} label_exists={label_path.exists()}")
            missing += 1
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[错误] 无法读取图片: {image_path}")
            continue

        boxes = read_yolo_label(label_path)
        img = draw_boxes(img, boxes)

        cv2.putText(
            img,
            stem,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2
        )

        out_path = OUT_DIR / f"{stem}_vis.jpg"
        cv2.imwrite(str(out_path), img)

        count += 1

    print(f"完成，可视化输出数量: {count}")
    print(f"缺失数量: {missing}")
    print(f"请打开这个文件夹人工检查: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
