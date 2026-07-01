#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除 OBS 上 Motus 文件夹下的所有内容
"""

import moxing as mox


def delete_all_in_obs_directory(obs_dir: str) -> None:
    """
    递归删除 OBS 目录下的所有内容

    Args:
        obs_dir: OBS 路径
    """
    if not obs_dir.startswith("obs://"):
        raise ValueError(f"OBS 路径必须以 obs:// 开头: {obs_dir}")

    obs_dir = obs_dir.rstrip('/') + '/'

    print(f"目标 OBS 路径: {obs_dir}")
    print("正在扫描目录...")

    # 扫描所有文件
    files_to_delete = []
    folders_to_scan = [obs_dir]
    folders_to_delete = []

    while folders_to_scan:
        current_dir = folders_to_scan.pop()

        try:
            items = mox.file.list_directory(current_dir)
        except:
            continue

        for item in items:
            item_name = item.rstrip('/')
            full_path = current_dir + item_name

            # 判断是文件还是目录
            try:
                mox.file.list_directory(full_path + '/')
                # 是文件夹
                folders_to_delete.append(full_path)
                folders_to_scan.append(full_path + '/')
            except:
                # 是文件
                files_to_delete.append(full_path)

    print(f"找到 {len(files_to_delete)} 个文件, {len(folders_to_delete)} 个文件夹\n")

    # 确认删除
    print(f"即将删除 {len(files_to_delete)} 个文件和 {len(folders_to_delete)} 个文件夹")
    confirm = input("确认删除? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消")
        return

    # 删除所有文件
    print("\n开始删除文件...")
    for i, full_path in enumerate(files_to_delete):
        try:
            mox.file.remove(full_path)
            if (i + 1) % 50 == 0:
                print(f"  进度: {i + 1}/{len(files_to_delete)}")
        except Exception as e:
            pass

    # 从深到浅删除文件夹
    sorted_folders = sorted(folders, key=lambda x: x.count('/'), reverse=True)

    print("\n开始删除文件夹...")
    for folder in sorted_folders:
        try:
            mox.file.remove(folder.rstrip('/'))
        except:
            pass

    print("\n删除完成!")


if __name__ == "__main__":
    obs_dir = "obs://yw-2030-gy/external/wwx1484778/Motus"
    delete_all_in_obs_directory(obs_dir)
