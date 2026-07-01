#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上传本地文件夹到 OBS

将 /home/ma-user/work/wwx1484778/Our 上传到 obs://yw-2030-gy/external/wwx1484778/
"""

import os
import moxing as mox
from pathlib import Path


def upload_dir_to_obs(
    local_dir: str,
    obs_dir: str,
    overwrite: bool = True
) -> None:
    """
    上传本地目录到 OBS

    Args:
        local_dir: 本地目录路径
        obs_dir: OBS 路径
        overwrite: 是否覆盖已存在的文件
    """
    local_path = Path(local_dir)

    if not local_path.exists():
        raise ValueError(f"本地路径不存在: {local_dir}")

    if not obs_dir.startswith("obs://"):
        raise ValueError(f"OBS 路径必须以 obs:// 开头: {obs_dir}")

    # 确保 OBS 路径以 / 结尾
    if not obs_dir.endswith('/'):
        obs_dir += '/'

    # 统计文件
    files_to_upload = []
    total_size = 0

    for root, dirs, files in os.walk(local_path):
        # 跳过隐藏目录和常见的不需要上传的目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', '.git', 'node_modules']]
        # 跳过隐藏文件
        files = [f for f in files if not f.startswith('.')]

        for f in files:
            src = Path(root) / f
            rel_path = src.relative_to(local_path)
            dst = obs_dir + str(rel_path)
            files_to_upload.append((str(src), dst))
            total_size += src.stat().st_size

    total_size_gb = total_size / (1024 ** 3)
    print(f"找到 {len(files_to_upload)} 个文件, 总大小: {total_size_gb:.2f} GB")
    print(f"目标 OBS 路径: {obs_dir}")

    # 上传
    success_count = 0
    fail_count = 0

    for i, (src, dst) in enumerate(files_to_upload):
        try:
            mox.file.copy(src, dst)
            success_count += 1
            if (i + 1) % 100 == 0:
                progress_pct = (i + 1) / len(files_to_upload) * 100
                print(f"进度: {i + 1}/{len(files_to_upload)} ({progress_pct:.1f}%)")
        except Exception as e:
            print(f"上传失败: {src} -> {dst}, 错误: {e}")
            fail_count += 1

    print("=" * 50)
    print(f"上传完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print("=" * 50)


if __name__ == "__main__":
    import sys

    # 配置参数
    local_dir = "/home/ma-user/work/wwx1484778/Our/Motus"
    obs_dir = "obs://yw-2030-gy/external/wwx1484778/Our/"

    # 可以通过命令行参数指定
    if len(sys.argv) > 1:
        local_dir = sys.argv[1]
    if len(sys.argv) > 2:
        obs_dir = sys.argv[2]

    # 在 OBS 路径后加上本地目录的文件夹名，保留目录结构
    folder_name = Path(local_dir).name
    if not obs_dir.endswith('/'):
        obs_dir += '/'
    obs_dir = obs_dir + folder_name + '/'

    print(f"上传: {local_dir} -> {obs_dir}")
    upload_dir_to_obs(local_dir, obs_dir)