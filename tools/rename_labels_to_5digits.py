# -*- coding: utf-8 -*-
"""
rename_labels_to_5digits.py

功能：
将 labels_raw 中的 YOLO 标签文件从四位数改成五位数。

例如：
crack_0001.txt -> crack_00001.txt
crack_0682.txt -> crack_00682.txt
crack_0700.txt -> crack_00700.txt

适用于：
图片已经是 crack_00001.jpg ~ crack_01440.jpg
但旧标签还是 crack_0001.txt ~ crack_0700.txt 的情况。
"""

from pathlib import Path
import re
import csv
import shutil
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "labels_raw"
META_DIR = PROJECT_ROOT / "datasets" / "crack_raw" / "meta"

PREFIX = "crack"
DIGITS = 5

# True：只预览，不真正改名
# False：真正改名
DRY_RUN = False

# 是否备份 labels_raw
BACKUP = True


def extract_number(path: Path):
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return None
    return int(nums[-1])


def main():
    print("=" * 80)
    print("标签文件改成五位数")
    print("=" * 80)
    print(f"标签目录: {LABEL_DIR}")
    print(f"DRY_RUN: {DRY_RUN}")
    print("=" * 80)

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"标签目录不存在: {LABEL_DIR}")

    META_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(
        [p for p in LABEL_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda x: x.name.lower()
    )

    if not label_files:
        print("没有找到 txt 标签文件。")
        return

    print(f"找到标签数量: {len(label_files)}")

    # 备份
    if BACKUP and not DRY_RUN:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = LABEL_DIR.parent / f"labels_raw_backup_before_5digits_{time_str}"
        shutil.copytree(LABEL_DIR, backup_dir)
        print(f"已备份标签目录到: {backup_dir}")

    pairs = []

    for old_path in label_files:
        num = extract_number(old_path)
        if num is None:
            print(f"[跳过] 无法提取编号: {old_path.name}")
            continue

        new_name = f"{PREFIX}_{num:0{DIGITS}d}.txt"
        new_path = LABEL_DIR / new_name

        pairs.append((old_path, new_path))

    print("\n重命名预览：")
    for old_path, new_path in pairs[:30]:
        print(f"{old_path.name} -> {new_path.name}")

    if len(pairs) > 30:
        print(f"... 还有 {len(pairs) - 30} 个")

    map_csv = META_DIR / "rename_labels_to_5digits_map.csv"

    if DRY_RUN:
        print("\n当前是 DRY_RUN=True，只预览，不改名。")
        return

    # 使用临时名避免冲突
    temp_pairs = []
    for old_path, new_path in pairs:
        if old_path.name == new_path.name:
            continue

        temp_path = LABEL_DIR / f"__tmp__{old_path.name}"
        old_path.rename(temp_path)
        temp_pairs.append((temp_path, new_path, old_path.name, new_path.name))

    for temp_path, new_path, old_name, new_name in temp_pairs:
        if new_path.exists():
            raise FileExistsError(f"目标标签已存在，停止避免覆盖: {new_path}")
        temp_path.rename(new_path)

    with open(map_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["old_name", "new_name"])
        for old_path, new_path in pairs:
            writer.writerow([old_path.name, new_path.name])

    print("\n标签重命名完成。")
    print(f"映射表保存到: {map_csv}")


if __name__ == "__main__":
    main()
