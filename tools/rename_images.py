# -*- coding: utf-8 -*-
"""
rename_images.py

功能：
1. 从 datasets/crack_raw/images_collected 中读取所有原始图片
2. 统一重命名为 crack_00001.jpg, crack_00002.jpg, ...
3. 输出到 datasets/crack_raw/images_renamed
4. 生成重命名映射表 datasets/crack_raw/meta/rename_map.csv

适用于当前情况：
- 简单样本和复杂样本不分开
- 所有图片都放在 images_collected 里
- 后续使用 LabelImg 标注 images_renamed 或 images_selected 中的图片

推荐流程：
1. 原始图片放入 images_collected
2. 运行本脚本，生成 images_renamed
3. 检查 images_renamed 是否正常
4. 可以把 images_renamed 的图片复制到 images_selected，或者直接用 images_renamed 作为待标注图片
"""

from pathlib import Path
import shutil
import csv
import hashlib
from datetime import datetime
from PIL import Image


# =========================
# 配置区
# =========================

# 项目根目录：当前脚本在 tools 目录下，所以 parents[1] 是 Crack_RTDETR
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 原始图片目录
SRC_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "images_collected"

# 重命名后图片输出目录
DST_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "images_renamed"

# 元信息目录
META_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "meta"

# 映射表路径
MAP_CSV = META_DIR / "rename_map.csv"

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"

# 日志文件
LOG_FILE = LOG_DIR / "rename_images_log.txt"

# 图片名前缀
PREFIX = "crack"

# 编号位数
DIGITS = 5

# 支持的图片后缀
IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}

# 是否递归读取子文件夹
# 你现在是所有图片都放在 images_collected 里，一般 False 就够了
# 如果 images_collected 下面还有子文件夹，则改成 True
RECURSIVE = False

# 是否清空输出目录
# True：每次运行前清空 images_renamed
# False：如果输出目录已有文件，会报错停止，避免覆盖
CLEAR_DST_DIR = True

# 是否统一输出为 .jpg
# True：不管原图是 png/jpeg/bmp，输出都保存成 jpg
# False：保留原始后缀
CONVERT_TO_JPG = True

# JPG 保存质量
JPG_QUALITY = 95

# 是否计算文件 md5
# True：更方便排查重复图片，但速度稍慢
CALCULATE_MD5 = True


def write_log(message: str):
    """
    同时打印和写入日志。
    """
    print(message)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def file_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    计算文件 MD5。
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def collect_images(src_dir: Path):
    """
    收集图片文件。
    """
    if RECURSIVE:
        files = [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    # 按文件名排序，保证每次运行顺序稳定
    files = sorted(files, key=lambda x: str(x.name).lower())
    return files


def check_image_readable(image_path: Path):
    """
    检查图片是否可以被 PIL 正常读取。
    返回：
    success, width, height, mode, error_message
    """
    try:
        with Image.open(image_path) as img:
            img.verify()

        # verify 后需要重新打开才能读取尺寸等信息
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode

        return True, width, height, mode, ""

    except Exception as e:
        return False, None, None, None, str(e)


def save_as_jpg(src_path: Path, dst_path: Path):
    """
    将图片保存为 jpg。
    """
    with Image.open(src_path) as img:
        # 处理透明通道，统一转 RGB
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        img.save(dst_path, "JPEG", quality=JPG_QUALITY)


def prepare_output_dir(dst_dir: Path):
    """
    准备输出目录。
    """
    if dst_dir.exists():
        existing_files = [p for p in dst_dir.iterdir() if p.is_file()]
        if existing_files:
            if CLEAR_DST_DIR:
                write_log(f"[INFO] 清空输出目录: {dst_dir}")
                shutil.rmtree(dst_dir)
                dst_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise RuntimeError(
                    f"输出目录已有文件，为避免覆盖，脚本已停止: {dst_dir}\n"
                    f"如果你确认要清空，请把 CLEAR_DST_DIR 改成 True。"
                )
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)


def main():
    # 清空旧日志
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    start_time = datetime.now()

    write_log("=" * 100)
    write_log("rename_images.py 开始运行")
    write_log("=" * 100)
    write_log(f"运行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(f"项目根目录: {PROJECT_ROOT}")
    write_log(f"原始图片目录: {SRC_DIR}")
    write_log(f"输出图片目录: {DST_DIR}")
    write_log(f"映射表路径: {MAP_CSV}")
    write_log(f"是否递归读取: {RECURSIVE}")
    write_log(f"是否清空输出目录: {CLEAR_DST_DIR}")
    write_log(f"是否统一转 JPG: {CONVERT_TO_JPG}")
    write_log("=" * 100)

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"原始图片目录不存在: {SRC_DIR}")

    META_DIR.mkdir(parents=True, exist_ok=True)

    # 收集图片
    image_files = collect_images(SRC_DIR)

    if len(image_files) == 0:
        write_log(f"[ERROR] 没有在目录中找到图片: {SRC_DIR}")
        return

    write_log(f"[INFO] 找到原始图片数量: {len(image_files)}")

    # 准备输出目录
    prepare_output_dir(DST_DIR)

    records = []
    error_records = []
    md5_count = {}

    valid_count = 0

    for idx, src_path in enumerate(image_files, start=1):
        src_suffix = src_path.suffix.lower()

        if CONVERT_TO_JPG:
            dst_suffix = ".jpg"
        else:
            dst_suffix = src_suffix

        new_name = f"{PREFIX}_{idx:0{DIGITS}d}{dst_suffix}"
        dst_path = DST_DIR / new_name

        readable, width, height, mode, error_msg = check_image_readable(src_path)

        if not readable:
            write_log(f"[ERROR] 图片无法读取，跳过: {src_path.name} | {error_msg}")
            error_records.append({
                "old_name": src_path.name,
                "old_path": str(src_path),
                "error": error_msg
            })
            continue

        try:
            if CONVERT_TO_JPG:
                save_as_jpg(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

            valid_count += 1

            if CALCULATE_MD5:
                md5_value = file_md5(src_path)
                md5_count[md5_value] = md5_count.get(md5_value, 0) + 1
            else:
                md5_value = ""

            records.append({
                "index": idx,
                "old_name": src_path.name,
                "new_name": new_name,
                "old_path": str(src_path),
                "new_path": str(dst_path),
                "old_suffix": src_suffix,
                "new_suffix": dst_suffix,
                "width": width,
                "height": height,
                "mode": mode,
                "md5": md5_value
            })

            if idx <= 20:
                write_log(f"[OK] {src_path.name} -> {new_name}")

        except Exception as e:
            write_log(f"[ERROR] 复制/转换失败: {src_path.name} | {e}")
            error_records.append({
                "old_name": src_path.name,
                "old_path": str(src_path),
                "error": str(e)
            })

    # 写入映射表
    with open(MAP_CSV, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "index",
            "old_name",
            "new_name",
            "old_path",
            "new_path",
            "old_suffix",
            "new_suffix",
            "width",
            "height",
            "mode",
            "md5"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # 写入错误表
    error_csv = META_DIR / "rename_errors.csv"
    if error_records:
        with open(error_csv, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["old_name", "old_path", "error"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(error_records)

    # 统计重复 MD5
    duplicate_md5_count = 0
    if CALCULATE_MD5:
        duplicate_md5_count = sum(1 for _, count in md5_count.items() if count > 1)

    end_time = datetime.now()
    elapsed = end_time - start_time

    write_log("=" * 100)
    write_log("rename_images.py 运行完成")
    write_log("=" * 100)
    write_log(f"原始图片总数: {len(image_files)}")
    write_log(f"成功输出数量: {valid_count}")
    write_log(f"失败数量: {len(error_records)}")
    write_log(f"输出目录: {DST_DIR}")
    write_log(f"映射表: {MAP_CSV}")

    if error_records:
        write_log(f"[WARNING] 存在无法处理的图片，错误表已保存: {error_csv}")

    if CALCULATE_MD5:
        write_log(f"检测到重复 MD5 数量: {duplicate_md5_count}")
        if duplicate_md5_count > 0:
            write_log("[WARNING] 可能存在重复图片，请查看 rename_map.csv 中的 md5 列。")

    write_log(f"耗时: {elapsed}")
    write_log("=" * 100)

    write_log("\n输出文件命名示例：")
    write_log(f"  {PREFIX}_00001.jpg")
    write_log(f"  {PREFIX}_00002.jpg")
    write_log(f"  ...")
    write_log(f"  {PREFIX}_{valid_count:0{DIGITS}d}.jpg")


if __name__ == "__main__":
    main()
