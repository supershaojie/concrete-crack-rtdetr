# -*- coding: utf-8 -*-
"""
05_visualize_labels.py

功能：
1. 从 augmented/train, val, test 中随机抽样
2. 读取 YOLO 标签并画框
3. 将可视化结果保存到 vis_check 目录
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUG_ROOT = PROJECT_ROOT / "datasets" / "crack_raw" / "augmented"
VIS_ROOT = PROJECT_ROOT / "datasets" / "crack_raw" / "vis_check"

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 每个集合抽样多少张
NUM_SAMPLES_PER_SPLIT = 20

RANDOM_SEED = 42


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path):
    if path.exists():
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def get_image_files(images_dir: Path):
    files = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def read_yolo_label(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except UnicodeDecodeError:
        with open(label_path, "r", encoding="gbk") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        boxes.append([cls_id, x, y, w, h])
    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    cls_id, x, y, w, h = box
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return cls_id, x1, y1, x2, y2


def draw_boxes(image_path: Path, label_path: Path, save_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    img_w, img_h = image.size
    boxes = read_yolo_label(label_path)

    for box in boxes:
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box, img_w, img_h)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(y1 - 12, 0)), f"class {cls_id}", fill="yellow")

    ensure_dir(save_path.parent)
    image.save(save_path)


def process_split(split_name: str):
    img_dir = AUG_ROOT / "images" / split_name
    lbl_dir = AUG_ROOT / "labels" / split_name
    out_dir = VIS_ROOT / split_name

    ensure_dir(out_dir)
    clear_dir(out_dir)

    image_files = get_image_files(img_dir)
    if len(image_files) == 0:
        print(f"[警告] {split_name} 没有图片")
        return

    sample_num = min(NUM_SAMPLES_PER_SPLIT, len(image_files))
    sampled = random.sample(image_files, sample_num)

    print(f"{split_name}: 抽样 {sample_num} 张进行可视化检查")

    for img_path in sampled:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        save_path = out_dir / img_path.name
        draw_boxes(img_path, label_path, save_path)


def main():
    print("=" * 80)
    print("开始可视化检查增强后的标签")
    print("=" * 80)

    random.seed(RANDOM_SEED)

    for split_name in SPLITS:
        process_split(split_name)

    print("-" * 80)
    print(f"可视化结果已保存到: {VIS_ROOT}")
    print("请打开图片人工检查 bbox 是否正确。")
    print("=" * 80)


if __name__ == "__main__":
    main()
