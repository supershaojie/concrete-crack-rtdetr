# -*- coding: utf-8 -*-
"""
03_augment_dataset.py

功能：
1. 读取 split_raw 中的 train / val / test 数据
2. 对每张图生成以下 6 个版本：
   - 原图
   - 水平翻转 (hf)
   - 垂直翻转 (vf)
   - 亮度增强 (bu)
   - 亮度减弱 (bd)
   - 对比度增强 (cu)
3. 将增强后的图片和标签保存到 augmented 目录中
4. 自动同步修改翻转后的 YOLO 标签

输入目录：
datasets/crack_raw/split_raw/images/{train,val,test}
datasets/crack_raw/split_raw/labels/{train,val,test}

输出目录：
datasets/crack_raw/augmented/images/{train,val,test}
datasets/crack_raw/augmented/labels/{train,val,test}
"""

from pathlib import Path
from PIL import Image, ImageEnhance
import shutil


# =========================
# 1. 路径配置
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_ROOT = PROJECT_ROOT / "datasets" / "crack_raw" / "split_raw"
AUG_ROOT = PROJECT_ROOT / "datasets" / "crack_raw" / "augmented"

IMAGE_SETS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 亮度和对比度增强系数
BRIGHTNESS_UP_FACTOR = 1.25
BRIGHTNESS_DOWN_FACTOR = 0.75
CONTRAST_UP_FACTOR = 1.3


# =========================
# 2. 工具函数
# =========================

def ensure_dir(path: Path):
    """创建目录"""
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path):
    """清空目录中的内容"""
    if path.exists():
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def get_image_files(images_dir: Path):
    """获取目录下所有图片文件"""
    image_files = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            image_files.append(p)
    return sorted(image_files)


def read_yolo_label(label_path: Path):
    """
    读取 YOLO 标签
    返回：
        boxes: list of [cls_id, x, y, w, h]
    """
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


def write_yolo_label(label_path: Path, boxes):
    """写入 YOLO 标签"""
    with open(label_path, "w", encoding="utf-8") as f:
        for box in boxes:
            cls_id, x, y, w, h = box
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def transform_boxes_hflip(boxes):
    """水平翻转标签：x -> 1 - x"""
    new_boxes = []
    for cls_id, x, y, w, h in boxes:
        new_x = 1.0 - x
        new_boxes.append([cls_id, new_x, y, w, h])
    return new_boxes


def transform_boxes_vflip(boxes):
    """垂直翻转标签：y -> 1 - y"""
    new_boxes = []
    for cls_id, x, y, w, h in boxes:
        new_y = 1.0 - y
        new_boxes.append([cls_id, x, new_y, w, h])
    return new_boxes


def save_image(image: Image.Image, save_path: Path):
    """保存图片"""
    ensure_dir(save_path.parent)

    # 对 JPEG 图片更稳妥地转成 RGB
    if save_path.suffix.lower() in {".jpg", ".jpeg"}:
        if image.mode != "RGB":
            image = image.convert("RGB")

    image.save(save_path)


def augment_one_image(image_path: Path, label_path: Path, out_img_dir: Path, out_lbl_dir: Path):
    """
    对单张图片做增强并保存
    输出 6 个版本：
    1. 原图
    2. hf
    3. vf
    4. bu
    5. bd
    6. cu
    """
    stem = image_path.stem
    suffix = image_path.suffix

    # 读取图片和标签
    image = Image.open(image_path)
    boxes = read_yolo_label(label_path)

    # ---------- 1. 原图 ----------
    save_image(image, out_img_dir / f"{stem}{suffix}")
    write_yolo_label(out_lbl_dir / f"{stem}.txt", boxes)

    # ---------- 2. 水平翻转 ----------
    img_hf = image.transpose(Image.FLIP_LEFT_RIGHT)
    boxes_hf = transform_boxes_hflip(boxes)
    save_image(img_hf, out_img_dir / f"{stem}_aug_hf{suffix}")
    write_yolo_label(out_lbl_dir / f"{stem}_aug_hf.txt", boxes_hf)

    # ---------- 3. 垂直翻转 ----------
    img_vf = image.transpose(Image.FLIP_TOP_BOTTOM)
    boxes_vf = transform_boxes_vflip(boxes)
    save_image(img_vf, out_img_dir / f"{stem}_aug_vf{suffix}")
    write_yolo_label(out_lbl_dir / f"{stem}_aug_vf.txt", boxes_vf)

    # ---------- 4. 亮度增强 ----------
    enhancer_bu = ImageEnhance.Brightness(image)
    img_bu = enhancer_bu.enhance(BRIGHTNESS_UP_FACTOR)
    save_image(img_bu, out_img_dir / f"{stem}_aug_bu{suffix}")
    write_yolo_label(out_lbl_dir / f"{stem}_aug_bu.txt", boxes)

    # ---------- 5. 亮度减弱 ----------
    enhancer_bd = ImageEnhance.Brightness(image)
    img_bd = enhancer_bd.enhance(BRIGHTNESS_DOWN_FACTOR)
    save_image(img_bd, out_img_dir / f"{stem}_aug_bd{suffix}")
    write_yolo_label(out_lbl_dir / f"{stem}_aug_bd.txt", boxes)

    # ---------- 6. 对比度增强 ----------
    enhancer_cu = ImageEnhance.Contrast(image)
    img_cu = enhancer_cu.enhance(CONTRAST_UP_FACTOR)
    save_image(img_cu, out_img_dir / f"{stem}_aug_cu{suffix}")
    write_yolo_label(out_lbl_dir / f"{stem}_aug_cu.txt", boxes)


def process_one_split(split_name: str):
    """处理一个数据集划分：train / val / test"""
    in_img_dir = SPLIT_ROOT / "images" / split_name
    in_lbl_dir = SPLIT_ROOT / "labels" / split_name

    out_img_dir = AUG_ROOT / "images" / split_name
    out_lbl_dir = AUG_ROOT / "labels" / split_name

    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    # 清空旧结果
    clear_dir(out_img_dir)
    clear_dir(out_lbl_dir)

    image_files = get_image_files(in_img_dir)

    print(f"\n开始增强 {split_name} 集，共 {len(image_files)} 张原图...")

    count = 0
    for image_path in image_files:
        stem = image_path.stem
        label_path = in_lbl_dir / f"{stem}.txt"

        if not label_path.exists():
            print(f"[跳过] 找不到对应标签: {label_path.name}")
            continue

        augment_one_image(image_path, label_path, out_img_dir, out_lbl_dir)
        count += 1

        if count % 50 == 0:
            print(f"  已处理 {count} / {len(image_files)}")

    final_img_num = len(list(out_img_dir.iterdir()))
    final_lbl_num = len(list(out_lbl_dir.iterdir()))

    print(f"{split_name} 集增强完成：")
    print(f"  原始样本数: {count}")
    print(f"  增强后图片数: {final_img_num}")
    print(f"  增强后标签数: {final_lbl_num}")


# =========================
# 3. 主函数
# =========================

def main():
    print("=" * 80)
    print("开始对 split_raw 数据集进行增强")
    print("=" * 80)
    print(f"输入目录: {SPLIT_ROOT}")
    print(f"输出目录: {AUG_ROOT}")
    print(f"亮度增强系数: {BRIGHTNESS_UP_FACTOR}")
    print(f"亮度减弱系数: {BRIGHTNESS_DOWN_FACTOR}")
    print(f"对比度增强系数: {CONTRAST_UP_FACTOR}")
    print("-" * 80)

    # 检查输入目录是否存在
    for split_name in IMAGE_SETS:
        in_img_dir = SPLIT_ROOT / "images" / split_name
        in_lbl_dir = SPLIT_ROOT / "labels" / split_name

        if not in_img_dir.exists():
            raise FileNotFoundError(f"输入图片目录不存在: {in_img_dir}")
        if not in_lbl_dir.exists():
            raise FileNotFoundError(f"输入标签目录不存在: {in_lbl_dir}")

    # 处理 train / val / test
    for split_name in IMAGE_SETS:
        process_one_split(split_name)

    print("\n" + "=" * 80)
    print("所有数据集增强完成。")
    print("你可以进入下一步：检查增强结果 / 组装最终数据集。")
    print("=" * 80)


if __name__ == "__main__":
    main()
