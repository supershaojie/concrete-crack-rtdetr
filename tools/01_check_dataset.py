# -*- coding: utf-8 -*-
"""
01_check_dataset.py

功能：
1. 检查原始图片与 YOLO 标签是否一一对应
2. 检查标签格式是否正确
3. 检查类别 id 是否符合要求
4. 检查 bbox 坐标是否在 [0, 1] 范围内
5. 检查 bbox 宽高是否大于 0
6. 统计图片数量、标签数量、目标框数量、空标签数量等

适用数据结构：
Crack_RTDETR/
├── datasets/
│   └── crack_raw/
│       ├── images_renamed/
│       └── labels_raw/
└── tools/
    └── 01_check_dataset.py

YOLO 标签格式：
class_id x_center y_center width height
例如：
0 0.5123 0.4388 0.1200 0.0350
"""

from pathlib import Path
from collections import Counter


# =========================
# 1. 路径配置
# =========================

# 项目根目录：当前脚本在 tools/ 下，所以 parents[1] 是 Crack_RTDETR 根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 原始图片目录
IMAGES_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "images_renamed"

# 原始标签目录
LABELS_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "labels_raw"

# 支持的图片后缀
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 你的类别数
# 如果你只有一个类别 crack，那么类别 id 应该只有 0
NUM_CLASSES = 1

# 是否允许空标签文件
# 对于纯裂缝检测数据集，如果每张图都有裂缝，建议 False
# 如果你的数据里包含无裂缝负样本，则可以改成 True
ALLOW_EMPTY_LABEL = False


# =========================
# 2. 工具函数
# =========================

def get_image_files(images_dir: Path):
    """获取图片文件列表"""
    image_files = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            image_files.append(p)
    return sorted(image_files)


def get_label_files(labels_dir: Path):
    """获取标签 txt 文件列表"""
    label_files = []
    for p in labels_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".txt":
            label_files.append(p)
    return sorted(label_files)


def read_label_file(label_path: Path):
    """
    读取 YOLO 标签文件
    返回：
        lines: 非空行列表
    """
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
    except UnicodeDecodeError:
        with open(label_path, "r", encoding="gbk") as f:
            lines = [line.strip() for line in f.readlines()]

    # 去掉完全空白行
    lines = [line for line in lines if line != ""]
    return lines


def check_one_label(label_path: Path, num_classes: int):
    """
    检查单个标签文件

    返回：
        is_ok: bool
        errors: list[str]
        warnings: list[str]
        box_count: int
        class_counter: Counter
    """
    errors = []
    warnings = []
    class_counter = Counter()
    box_count = 0

    lines = read_label_file(label_path)

    if len(lines) == 0:
        warnings.append("空标签文件")
        return True, errors, warnings, 0, class_counter

    for line_idx, line in enumerate(lines, start=1):
        parts = line.split()

        # YOLO 检测标签必须是 5 列
        if len(parts) != 5:
            errors.append(
                f"第 {line_idx} 行列数错误：应为5列，实际为{len(parts)}列，内容：{line}"
            )
            continue

        # 尝试解析
        try:
            cls_float = float(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(
                f"第 {line_idx} 行存在非数字内容：{line}"
            )
            continue

        # 类别 id 应该是整数
        if not cls_float.is_integer():
            errors.append(
                f"第 {line_idx} 行类别 id 不是整数：{parts[0]}"
            )
            continue

        cls_id = int(cls_float)

        # 类别范围检查
        if cls_id < 0 or cls_id >= num_classes:
            errors.append(
                f"第 {line_idx} 行类别 id 越界：{cls_id}，合法范围应为 [0, {num_classes - 1}]"
            )

        # 坐标范围检查
        vals = {
            "x_center": x,
            "y_center": y,
            "width": w,
            "height": h
        }

        for name, value in vals.items():
            if value < 0 or value > 1:
                errors.append(
                    f"第 {line_idx} 行 {name}={value} 越界，应在 [0, 1] 内，内容：{line}"
                )

        # 宽高检查
        if w <= 0:
            errors.append(
                f"第 {line_idx} 行 width={w} 非法，应 > 0，内容：{line}"
            )

        if h <= 0:
            errors.append(
                f"第 {line_idx} 行 height={h} 非法，应 > 0，内容：{line}"
            )

        # bbox 边界进一步检查
        # YOLO格式中 x,y,w,h 均归一化
        # 左边界 = x - w/2，右边界 = x + w/2
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        eps = 1e-6
        if x1 < -eps or y1 < -eps or x2 > 1 + eps or y2 > 1 + eps:
            errors.append(
                f"第 {line_idx} 行 bbox 超出图像边界："
                f"x1={x1:.6f}, y1={y1:.6f}, x2={x2:.6f}, y2={y2:.6f}，内容：{line}"
            )

        box_count += 1
        class_counter[cls_id] += 1

    return len(errors) == 0, errors, warnings, box_count, class_counter


# =========================
# 3. 主函数
# =========================

def main():
    print("=" * 80)
    print("开始检查裂缝检测原始数据集")
    print("=" * 80)

    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"图片目录: {IMAGES_DIR}")
    print(f"标签目录: {LABELS_DIR}")
    print(f"类别数 NUM_CLASSES: {NUM_CLASSES}")
    print(f"是否允许空标签 ALLOW_EMPTY_LABEL: {ALLOW_EMPTY_LABEL}")
    print("-" * 80)

    # 目录存在性检查
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"图片目录不存在: {IMAGES_DIR}")

    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"标签目录不存在: {LABELS_DIR}")

    image_files = get_image_files(IMAGES_DIR)
    label_files = get_label_files(LABELS_DIR)

    print(f"检测到图片数量: {len(image_files)}")
    print(f"检测到标签数量: {len(label_files)}")
    print("-" * 80)

    if len(image_files) == 0:
        print("错误：图片目录为空，请检查 images_renamed。")
        return

    if len(label_files) == 0:
        print("错误：标签目录为空，请检查 labels_raw。")
        return

    # stem 是不带后缀的文件名
    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    # 图片没有对应标签
    images_without_labels = sorted(image_stems - label_stems)

    # 标签没有对应图片
    labels_without_images = sorted(label_stems - image_stems)

    if images_without_labels:
        print(f"有 {len(images_without_labels)} 张图片没有对应标签文件，示例：")
        for name in images_without_labels[:20]:
            print(f"  图片无标签: {name}")
        print("-" * 80)

    if labels_without_images:
        print(f"有 {len(labels_without_images)} 个标签没有对应图片，示例：")
        for name in labels_without_images[:20]:
            print(f"  标签无图片: {name}")
        print("-" * 80)

    # 只检查有图片且有标签的样本
    matched_stems = sorted(image_stems & label_stems)
    print(f"图片和标签匹配成功的样本数: {len(matched_stems)}")
    print("-" * 80)

    total_boxes = 0
    empty_label_count = 0
    bad_label_count = 0
    warning_label_count = 0
    global_class_counter = Counter()

    bad_examples = []
    warning_examples = []

    for stem in matched_stems:
        label_path = LABELS_DIR / f"{stem}.txt"

        is_ok, errors, warnings, box_count, class_counter = check_one_label(
            label_path=label_path,
            num_classes=NUM_CLASSES
        )

        total_boxes += box_count
        global_class_counter.update(class_counter)

        if len(warnings) > 0:
            warning_label_count += 1
            warning_examples.append((label_path.name, warnings))

        if box_count == 0:
            empty_label_count += 1
            if not ALLOW_EMPTY_LABEL:
                is_ok = False
                errors.append("当前设置不允许空标签，但该文件没有任何目标框")

        if not is_ok:
            bad_label_count += 1
            bad_examples.append((label_path.name, errors))

    # 输出错误样本
    if bad_examples:
        print("发现标签错误文件，前 30 个如下：")
        for file_name, errors in bad_examples[:30]:
            print(f"\n[错误文件] {file_name}")
            for e in errors[:10]:
                print(f"  - {e}")
        print("-" * 80)

    # 输出警告样本
    if warning_examples:
        print("发现标签警告文件，前 20 个如下：")
        for file_name, warnings in warning_examples[:20]:
            print(f"\n[警告文件] {file_name}")
            for w in warnings:
                print(f"  - {w}")
        print("-" * 80)

    # 汇总
    print("=" * 80)
    print("数据集检查汇总")
    print("=" * 80)
    print(f"图片总数: {len(image_files)}")
    print(f"标签总数: {len(label_files)}")
    print(f"匹配样本数: {len(matched_stems)}")
    print(f"图片无标签数: {len(images_without_labels)}")
    print(f"标签无图片数: {len(labels_without_images)}")
    print(f"目标框总数: {total_boxes}")
    print(f"空标签文件数: {empty_label_count}")
    print(f"存在警告的标签文件数: {warning_label_count}")
    print(f"存在错误的标签文件数: {bad_label_count}")

    print("-" * 80)
    print("类别目标框统计:")
    if len(global_class_counter) == 0:
        print("  未统计到任何目标框")
    else:
        for cls_id in sorted(global_class_counter.keys()):
            print(f"  class {cls_id}: {global_class_counter[cls_id]} boxes")

    print("-" * 80)

    # 最终结论
    has_pair_error = len(images_without_labels) > 0 or len(labels_without_images) > 0
    has_label_error = bad_label_count > 0

    if not has_pair_error and not has_label_error:
        print("结论：数据集基础检查通过，可以进入下一步：划分 train/val/test。")
    else:
        print("结论：数据集存在问题，请先修正后再进入划分和增强。")

    print("=" * 80)


if __name__ == "__main__":
    main()
