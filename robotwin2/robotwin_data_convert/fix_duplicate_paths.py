#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复重复路径结构

将:
  /path/adjust_bottle/aloha-agilex_clean_50/aloha-agilex_clean_50/...
变为:
  /path/adjust_bottle/aloha-agilex_clean_50/...
"""

import os
import shutil
from pathlib import Path


def fix_duplicate_paths(root_dir: str, dry_run: bool = False) -> None:
    """
    修复重复路径

    如果存在 /path/X/X/，将其移动到 /path/X/

    Args:
        root_dir: 根目录
        dry_run: 只显示，不实际操作
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误: 路径不存在: {root_dir}")
        return

    # 遍历所有目录
    fixed_count = 0
    skipped_count = 0

    for parent in sorted(root_path.rglob("*")):
        if not parent.is_dir():
            continue

        # 检查是否有同名子目录
        for child in parent.iterdir():
            if child.is_dir() and child.name == parent.name:
                duplicate_dir = child

                print(f"\n发现重复路径: {duplicate_dir}")
                print(f"  移动内容到: {parent}")

                if dry_run:
                    print("  (dry run, 跳过)")
                    fixed_count += 1
                    continue

                # 移动子目录中的所有内容到父目录
                moved = []
                for item in duplicate_dir.iterdir():
                    dest = parent / item.name
                    if dest.exists():
                        print(f"  跳过已存在: {item.name} -> {dest}")
                        removed = False
                    else:
                        if dry_run:
                            moved.append(item.name)
                        else:
                            shutil.move(str(item), str(dest))
                            moved.append(item.name)
                            removed = True

                # 删除空的重复目录
                if not dry_run and duplicate_dir.exists():
                    try:
                        shutil.rmtree(duplicate_dir)
                        print(f"  删除空目录: {duplicate_dir}")
                    except Exception as e:
                        print(f"  警告: 无法删除 {duplicate_dir}: {e}")

                if moved:
                    print(f"  已移动 {len(moved)} 项: {', '.join(moved)}")
                    fixed_count += 1
                else:
                    skipped_count += 1

                # 每个父目录只处理一个重复子目录
                break

    print("\n" + "=" * 50)
    print(f"修复完成!")
    print(f"  已处理: {fixed_count} 个")
    print(f"  已跳过: {skipped_count} 个")
    print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="修复解压后的重复路径结构"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="数据集根目录"
    )
    parser.add_argument(
        "--dry_run",
        "-n",
        action="store_true",
        help="只显示将要修复的路径，不实际操作"
    )

    args = parser.parse_args()

    fix_duplicate_paths(args.root_dir, args.dry_run)


if __name__ == "__main__":
    main()