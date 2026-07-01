#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 OBS 下载文件夹到本地

将 obs://yw-2030-gy/data/opensource/RoboTwin2_0/RoboTwin_emb 下载到 /cache/wwx1484778/RoboTwin_emb
"""

import os
import moxing as mox
from pathlib import Path


def download_obs_to_local(
    obs_dir: str,
    local_dir: str,
    overwrite: bool = True
) -> None:
    """
    从 OBS 下载目录到本地

    Args:
        obs_dir: OBS 路径
        local_dir: 本地目录路径
        overwrite: 是否覆盖已存在的文件
    """
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    if not obs_dir.startswith("obs://"):
        raise ValueError(f"OBS 路径必须以 obs:// 开头: {obs_dir}")

    # 确保 OBS 路径以 / 结尾
    if not obs_dir.endswith('/'):
        obs_dir += '/'

    # 递归获取所有文件
    files_to_download = []
    total_size = 0

    print(f"扫描 OBS 目录: {obs_dir}")

    # 递归扫描
    stack = [(obs_dir, "")]

    while stack:
        current_obs_dir, rel_prefix = stack.pop()

        try:
            items = mox.file.list_directory(current_obs_dir)
        except Exception as e:
            print(f"  警告: 无法列出 {current_obs_dir}: {e}")
            continue

        for item in items:
            item_name = item.rstrip('/')

            # 跳过隐藏文件
            if item_name.startswith('.'):
                continue

            full_obs_path = current_obs_dir + item_name
            rel_path = rel_prefix + item_name

            try:
                sub_items = mox.file.list_directory(full_obs_path + '/')
                # 是文件夹，继续递归
                stack.append((full_obs_path + '/', rel_path + '/'))
            except:
                # 是文件
                files_to_download.append((full_obs_path, rel_path))

                # 获取文件大小
                try:
                    stat = mox.file.stat(full_obs_path)
                    total_size += stat.size
                except:
                    pass

    total_size_gb = total_size / (1024 ** 3)
    total_size_tb = total_size_gb / 1024
    print(f"找到 {len(files_to_download)} 个文件, 总大小: {total_size_gb:.2f} GB ({total_size_tb:.2f} TB)")
    print(f"目标本地路径: {local_dir}")

    # 下载
    success_count = 0
    fail_count = 0

    for i, (obs_path, rel_path) in enumerate(files_to_download):
        try:
            local_file = local_path / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # 检查本地文件是否已存在
            if local_file.exists():
                success_count += 1
                if (i + 1) % 100 == 0:
                    print(f"进度: {i + 1}/{len(files_to_download)} ({(i+1)/len(files_to_download)*100:.1f}%) - 跳过已存在文件")
                continue

            mox.file.copy(obs_path, str(local_file))
            success_count += 1

            if (i + 1) % 100 == 0:
                print(f"进度: {i + 1}/{len(files_to_download)} ({(i+1)/len(files_to_download)*100:.1f}%)")

        except Exception as e:
            print(f"下载失败: {obs_path} -> {rel_path}, 错误: {e}")
            fail_count += 1

    print("=" * 50)
    print(f"下载完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print("=" * 50)


def list_obs_structure():
    """查看 OBS 目录结构"""
    obs_dir = "obs://yw-2030-gy/data/opensource/RoboTwin2_0/RoboTwin_emb"
    if not obs_dir.endswith('/'):
        obs_dir += '/'

    print(f"OBS 目录结构: {obs_dir}")
    items = mox.file.list_directory(obs_dir)
    for item in items:
        item_name = item.rstrip('/')
        print(f"  - {item_name}")


if __name__ == "__main__":
    import sys

    # 先查看 OBS 结构
    print("=" * 50)
    list_obs_structure()
    print("=" * 50)

    if '--list' in sys.argv:
        sys.exit(0)

    # 配置参数
    obs_dir = "obs://yw-2030-gy/data/opensource/RoboTwin2_0/RoboTwin_emb"
    local_dir = "/cache/wwx1484778/RoboTwin_emb"

    # 下载
    download_obs_to_local(obs_dir, local_dir)