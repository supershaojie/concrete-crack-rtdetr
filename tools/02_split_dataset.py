# -*- coding: utf-8 -*-
"""
02_split_dataset.py

功能：
1. 将 crack_raw 下的原始图片和标签按 7:2:1 划分为 train / val / test
2. 划分后复制到 split_raw 目录中
3. 固定随机种子，保证可复现
4. 输出划分统计信息

输入目录：
datasets/crack_raw/images_renamed
datasets/crack_raw/labels_raw

输出目录：
datasets/crack_raw/split_raw/images/train
datasets/crack_raw/split_raw/images/val
datasets/crack_raw/split_raw/images/test
datasets/crack_raw/split_raw/labels/train
datasets/crack_raw/split_raw/labels/val
datasets/crack_raw/split_raw/labels/test
"""

import random
import shutil
from pathlib import Path


# =========================
# 1. 路径配置
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "images_renamed"
LABELS_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "labels_raw"

SPLIT_ROOT = PROJECT_ROOT / "datasets" / "crack_raw" / "split_raw"

TRAIN_IMG_DIR = SPLIT_ROOT / "images" / "train"
VAL_IMG_DIR = SPLIT_ROOT / "images" / "val"
TEST_IMG_DIR = SPLIT_ROOT / "images" / "test"

TRAIN_LBL_DIR = SPLIT_ROOT / "labels" / "train"
VAL_LBL_DIR = SPLIT_ROOT / "labels" / "val"
TEST_LBL_DIR = SPLIT_ROOT / "labels" / "test"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 随机种子，保证每次划分结果一致
RANDOM_SEED = 42


# =========================
# 2. 工具函数
# =========================

def ensure_dir(path: Path):
    """如果目录不存在就创建"""
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path):
    """清空目录中的所有文件和子目录"""
    if path.exists():
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def get_image_files(images_dir: Path):
    """获取所有图片文件"""
    image_files = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            image_files.append(p)
    return sorted(image_files)


def copy_one_sample(image_path: Path, label_path: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    """复制单个样本（图片 + 标签）"""
    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)

    shutil.copy2(image_path, dst_img_dir / image_path.name)
    shutil.copy2(label_path, dst_lbl_dir / label_path.name)


def print_split_info(name: str, sample_list):
    """打印某个集合的信息"""
    print(f"{name} 数量: {len(sample_list)}")
    if len(sample_list) > 0:
        print(f"  第一个样本: {sample_list[0][0].name} | {sample_list[0][1].name}")
        print(f"  最后一个样本: {sample_list[-1][0].name} | {sample_list[-1][1].name}")


# =========================
# 3. 主函数
# =========================

def main():
    print("=" * 80)
    print("开始按 7:2:1 划分原始裂缝数据集")
    print("=" * 80)

    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"图片目录: {IMAGES_DIR}")
    print(f"标签目录: {LABELS_DIR}")
    print(f"输出目录: {SPLIT_ROOT}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"划分比例: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print("-" * 80)

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"图片目录不存在: {IMAGES_DIR}")

    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"标签目录不存在: {LABELS_DIR}")

    # 创建输出目录
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR, TEST_LBL_DIR]:
        ensure_dir(d)

    # 清空旧划分结果，避免重复叠加
    print("正在清空 split_raw 中旧的划分结果...")
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR, TEST_LBL_DIR]:
        clear_dir(d)

    # 收集匹配样本
    image_files = get_image_files(IMAGES_DIR)

    matched_samples = []
    missing_label_images = []

    for img_path in image_files:
        stem = img_path.stem
        label_path = LABELS_DIR / f"{stem}.txt"
        if label_path.exists():
            matched_samples.append((img_path, label_path))
        else:
            missing_label_images.append(img_path.name)

    if missing_label_images:
        print("发现没有对应标签的图片，以下样本将被跳过：")
        for name in missing_label_images[:20]:
            print(f"  - {name}")
        print("-" * 80)

    total_samples = len(matched_samples)
    print(f"可参与划分的有效样本数: {total_samples}")

    if total_samples == 0:
        raise RuntimeError("没有可用于划分的有效样本，请检查图片和标签目录。")

    # 打乱样本
    random.seed(RANDOM_SEED)
    random.shuffle(matched_samples)

    # 计算划分数量
    train_count = round(total_samples * TRAIN_RATIO)
    val_count = round(total_samples * VAL_RATIO)
    test_count = total_samples - train_count - val_count

    print(f"计划划分结果:")
    print(f"  train: {train_count}")
    print(f"  val  : {val_count}")
    print(f"  test : {test_count}")
    print("-" * 80)

    # 划分
    train_samples = matched_samples[:train_count]
    val_samples = matched_samples[train_count:train_count + val_count]
    test_samples = matched_samples[train_count + val_count:]

    # 打印划分概况
    print_split_info("Train", train_samples)
    print_split_info("Val", val_samples)
    print_split_info("Test", test_samples)
    print("-" * 80)

    # 复制 train
    print("正在复制 train 集...")
    for img_path, lbl_path in train_samples:
        copy_one_sample(img_path, lbl_path, TRAIN_IMG_DIR, TRAIN_LBL_DIR)

    # 复制 val
    print("正在复制 val 集...")
    for img_path, lbl_path in val_samples:
        copy_one_sample(img_path, lbl_path, VAL_IMG_DIR, VAL_LBL_DIR)

    # 复制 test
    print("正在复制 test 集...")
    for img_path, lbl_path in test_samples:
        copy_one_sample(img_path, lbl_path, TEST_IMG_DIR, TEST_LBL_DIR)

    print("-" * 80)

    # 最终统计
    train_img_num = len(list(TRAIN_IMG_DIR.iterdir()))
    val_img_num = len(list(VAL_IMG_DIR.iterdir()))
    test_img_num = len(list(TEST_IMG_DIR.iterdir()))

    train_lbl_num = len(list(TRAIN_LBL_DIR.iterdir()))
    val_lbl_num = len(list(VAL_LBL_DIR.iterdir()))
    test_lbl_num = len(list(TEST_LBL_DIR.iterdir()))

    print("=" * 80)
    print("划分完成，最终目录统计如下：")
    print("=" * 80)
    print(f"train images: {train_img_num}")
    print(f"train labels: {train_lbl_num}")
    print(f"val images  : {val_img_num}")
    print(f"val labels  : {val_lbl_num}")
    print(f"test images : {test_img_num}")
    print(f"test labels : {test_lbl_num}")
    print("-" * 80)

    # 一致性检查
    if train_img_num != train_lbl_num:
        print("警告：train 图片数与标签数不一致！")
    if val_img_num != val_lbl_num:
        print("警告：val 图片数与标签数不一致！")
    if test_img_num != test_lbl_num:
        print("警告：test 图片数与标签数不一致！")

    if (
        train_img_num == train_lbl_num == train_count and
        val_img_num == val_lbl_num == val_count and
        test_img_num == test_lbl_num == test_count
    ):
        print("结论：数据集已成功按 7:2:1 划分，可以进入下一步：数据增强。")
    else:
        print("结论：划分已执行，但请检查是否存在数量不一致问题。")

    print("=" * 80)


if __name__ == "__main__":
    main()
