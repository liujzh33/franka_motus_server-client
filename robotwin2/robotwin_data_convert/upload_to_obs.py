#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上传本地数据到华为云 OBS

将本地文件夹上传到 obs:// 路径，使用 moxing 库。
"""

import os
import moxing as mox
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_total_size(path: Path) -> int:
    """获取目录总大小（字节）"""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def upload_to_obs(
    local_dir: str,
    obs_path: str,
    overwrite: bool = True,
    dry_run: bool = False
) -> None:
    """
    上传本地文件夹到 OBS

    Args:
        local_dir: 本地目录路径
        obs_path: OBS 路径（必须以 obs:// 开头）
        overwrite: 是否覆盖已存在的文件
        dry_run: 只显示将要上传的文件，不实际执行
    """
    local_path = Path(local_dir)

    if not local_path.exists():
        raise ValueError(f"本地路径不存在: {local_dir}")

    if not obs_path.startswith("obs://"):
        raise ValueError(f"OBS 路径必须以 obs:// 开头: {obs_path}")

    # 确保 OBS 路径以 / 结尾
    if not obs_path.endswith('/'):
        obs_path += '/'

    # 统计文件
    files_to_upload: List[tuple] = []  # (src, dst)
    total_size = 0

    logger.info(f"扫描本地目录: {local_dir}")
    for root, dirs, files in os.walk(local_path):
        for f in files:
            src = Path(root) / f
            # 计算相对路径
            rel_path = src.relative_to(local_path)
            dst = obs_path + str(rel_path)

            files_to_upload.append((str(src), dst))
            total_size += src.stat().st_size

    # 显示统计信息
    total_size_gb = total_size / (1024 ** 3)
    logger.info(f"找到 {len(files_to_upload)} 个文件, 总大小: {total_size_gb:.2f} GB")

    if dry_run:
        logger.info("=== DRY RUN 模式，不实际上传 ===")
        for src, dst in files_to_upload[:10]:  # 只显示前10个
            logger.info(f"  {src} -> {dst}")
        if len(files_to_upload) > 10:
            logger.info(f"  ... 还有 {len(files_to_upload) - 10} 个文件")
        return

    # 上传
    logger.info(f"开始上传到: {obs_path}")
    success_count = 0
    fail_count = 0

    for i, (src, dst) in enumerate(files_to_upload):
        try:
            mox.file.copy(src, dst)
            success_count += 1
            if (i + 1) % 100 == 0:
                logger.info(f"进度: {i + 1}/{len(files_to_upload)}")
        except Exception as e:
            logger.error(f"上传失败: {src} -> {dst}, 错误: {e}")
            fail_count += 1

    logger.info("=" * 50)
    logger.info(f"上传完成!")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="上传本地数据到华为云 OBS"
    )
    parser.add_argument(
        "local_dir",
        type=str,
        help="本地目录路径"
    )
    parser.add_argument(
        "obs_path",
        type=str,
        help="OBS 路径（obs://...）"
    )
    parser.add_argument(
        "--dry_run",
        "-n",
        action="store_true",
        help="只显示将要上传的文件，不实际执行"
    )

    args = parser.parse_args()

    upload_to_obs(args.local_dir, args.obs_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()