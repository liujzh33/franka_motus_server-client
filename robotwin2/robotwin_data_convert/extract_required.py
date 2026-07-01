#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobotTwin2.0 Dataset Extractor for Huawei Cloud OBS

只解压 RobotTwin2.0 数据集中需要的 ZIP 文件:
- aloha-agilex_clean_50.zip
- aloha-agilex_randomized_500.zip

支持 obs:// 路径，使用 moxing 库。
"""

import os
import zipfile
import io
from pathlib import Path
from typing import List


def is_obs_path(path: str) -> bool:
    """判断是否是 OBS 路径"""
    return path.startswith("obs://")


def list_obs_files(obs_dir: str, moxing) -> List[str]:
    """
    列出 OBS 目录下 dataset 中的所有 ZIP 文件

    使用分步 list_directory 而非 walk，避免 obs_client 配置问题。

    Args:
        obs_dir: obs:// 路径
        moxing: moxing 模块

    Returns:
        ZIP 文件路径列表
    """
    zip_files = []

    # 确保路径以 / 结尾
    if not obs_dir.endswith('/'):
        obs_dir = obs_dir + '/'

    # 1. 检查 dataset 目录
    dataset_dir = obs_dir + 'dataset/'
    if not moxing.file.exists(dataset_dir):
        print(f"警告: dataset 目录不存在: {dataset_dir}")
        return zip_files

    # 2. 获取所有任务目录
    try:
        tasks = moxing.file.list_directory(dataset_dir)
    except Exception as e:
        print(f"无法列出目录 {dataset_dir}: {e}")
        return zip_files

    # 3. 遍历每个任务，查找 ZIP 文件
    for task in tasks:
        task_dir = dataset_dir + task + '/'
        try:
            files = moxing.file.list_directory(task_dir)
            for f in files:
                if f.endswith('.zip'):
                    zip_files.append(task_dir + f)
        except Exception as e:
            print(f"警告: 无法列出 {task_dir}: {e}")
            continue

    return zip_files


def extract_obs_zip(
    obs_zip_path: str,
    output_dir: str,
    moxing,
    keep_zip_in_obs: bool = False
) -> bool:
    """
    从 OBS 解压 ZIP 文件到本地

    Args:
        obs_zip_path: obs:// 下的 ZIP 文件路径
        output_dir: 本地输出目录
        moxing: moxing 模块
        keep_zip_in_obs: 是否保留 OBS 中的 ZIP 文件

    Returns:
        是否成功
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  正在下载: {obs_zip_path}")
        # 从 OBS 下载 ZIP 到本地临时文件 (copy(src, dst): 从 src 复制到 dst)
        moxing.file.copy(obs_zip_path, os.path.basename(obs_zip_path))

        # 读取 ZIP 文件并解压
        print(f"  正在解压...")
        with zipfile.ZipFile(os.path.basename(obs_zip_path), 'r') as zip_ref:
            # ZIP 内部已经有了 aloha-agilex_clean_50/ 这一层
            # 提取时跳过根目录，直接提取内容
            members = zip_ref.namelist()
            # 获取根目录名（假设所有文件都在同一个根目录下）
            if members:
                root_dir = members[0].split('/')[0]
                # 提取所有文件，去掉根目录前缀
                for member in members:
                    # 跳过根目录本身
                    if member == root_dir + '/':
                        continue
                    # 获取相对路径（去掉根目录）
                    rel_path = member.replace(root_dir + '/', '', 1)
                    # 跳过 __MACOSX 等隐藏目录
                    if '__MACOSX' in rel_path:
                        continue
                    # 写入文件
                    if member.endswith('/'):
                        # 目录
                        (output_path / rel_path).mkdir(parents=True, exist_ok=True)
                    else:
                        # 文件
                        (output_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path / rel_path, 'wb') as f:
                            f.write(zip_ref.read(member))

        # 删除本地临时文件
        local_zip = Path(os.path.basename(obs_zip_path))
        if local_zip.exists():
            local_zip.unlink()

        print(f"  ✓ 解压到: {output_path}")

        # 删除 OBS 中的 ZIP 文件
        if not keep_zip_in_obs:
            try:
                moxing.file.remove(obs_zip_path)
                print(f"  ✓ 已删除 OBS 中的 ZIP")
            except Exception as e:
                print(f"  警告: 删除 OBS ZIP 失败: {e}")

        return True

    except Exception as e:
        print(f"  ✗ 解压失败: {e}")
        return False


def extract_required_zips(
    root_dir: str,
    output_dir: str = None,
    keep_zip: bool = False,
    dry_run: bool = False
) -> None:
    """
    只解压需要的 ZIP 文件

    Args:
        root_dir: 数据集根目录（本地或 obs://）
        output_dir: 解压输出目录（对于 obs:// 必须指定本地路径）
        keep_zip: 是否保留 ZIP 文件
        dry_run: 只显示将要解压的文件，不实际解压
    """
    required_zips = [
        "aloha-agilex_clean_50.zip",
        "aloha-agilex_randomized_500.zip"
    ]

    if is_obs_path(root_dir):
        print(f"检测到 OBS 路径: {root_dir}")
        extract_from_obs(root_dir, output_dir, required_zips, keep_zip, dry_run)
    else:
        extract_from_local(root_dir, output_dir, required_zips, keep_zip, dry_run)


def extract_from_obs(
    obs_dir: str,
    output_dir: str,
    required_zips: List[str],
    keep_zip: bool,
    dry_run: bool
) -> None:
    """从 OBS 解压"""
    try:
        import moxing as mox
    except ImportError:
        print("错误: 需要安装 moxing 库")
        print("在 ModelArts 环境中运行，或: pip install moxing")
        return

    if not output_dir:
        print("错误: obs:// 路径需要指定本地 output_dir")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 列出所有文件
    print(f"\n正在列出 {obs_dir} 下的文件...")
    files = list_obs_files(obs_dir, mox)

    # 筛选需要的 ZIP 文件
    zip_files = []
    for f in files:
        f_name = os.path.basename(f)
        if f_name in required_zips:
            zip_files.append(f)

    if not zip_files:
        print(f"没有找到需要的文件: {required_zips}")
        return

    print(f"\n找到 {len(zip_files)} 个需要解压的文件:")
    for z in zip_files:
        print(f"  - {os.path.basename(z)}")

    if dry_run:
        print("\n(不实际解压，使用 --no-dry_run 解压)")
        return

    print(f"\n开始解压到: {output_dir}\n")

    # 解压每个文件
    success_count = 0
    for obs_zip_path in zip_files:
        # OBS 路径格式: obs://.../dataset/{task_name}/{sub_dir}.zip
        # 需要提取 task_name 和 sub_dir
        # 例如: dataset/adjust_bottle/aloha-agilex_clean_50.zip
        zip_name = os.path.basename(obs_zip_path)  # aloha-agilex_clean_50.zip
        sub_dir = zip_name.replace('.zip', '')      # aloha-agilex_clean_50

        # 从路径中提取 task_name
        parts = obs_zip_path.replace('\\', '/').split('/')
        # 期望: [..., 'dataset', 'adjust_bottle', 'aloha-agilex_clean_50.zip']
        if 'dataset' in parts:
            dataset_idx = parts.index('dataset')
            if dataset_idx + 2 < len(parts):
                task_name = parts[dataset_idx + 1]  # adjust_bottle
            else:
                task_name = 'unknown'
        else:
            task_name = 'unknown'

        # 解压到: {output_dir}/{task_name}/{sub_dir}/
        extract_dir = output_path / task_name / sub_dir

        if extract_obs_zip(obs_zip_path, extract_dir, mox, keep_zip_in_obs=keep_zip):
            success_count += 1
        print()

    # 统计
    if success_count > 0:
        total_size = sum(
            f.stat().st_size
            for f in output_path.rglob("*")
            if f.is_file()
        )
        size_gb = total_size / (1024 ** 3)

        print("=" * 50)
        print("解压完成!")
        print(f"成功: {success_count}/{len(zip_files)} 个文件")
        print(f"输出目录: {output_dir}")
        print(f"总大小: {size_gb:.2f} GB")
        print("=" * 50)


def extract_from_local(
    root_dir: str,
    output_dir: str,
    required_zips: List[str],
    keep_zip: bool,
    dry_run: bool
) -> None:
    """从本地解压"""
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误: 路径不存在: {root_dir}")
        return

    output_path = Path(output_dir) if output_dir else root_path

    # 查找所有匹配的 ZIP 文件
    zip_files = []
    for zip_name in required_zips:
        found = list(root_path.rglob(zip_name))
        if found:
            zip_files.extend(found)
            print(f"  找到 {len(found)} 个: {zip_name}")
        else:
            print(f"  未找到: {zip_name}")

    if not zip_files:
        print("没有找到需要的 ZIP 文件")
        return

    print(f"\n共找到 {len(zip_files)} 个需要解压的文件\n")

    if dry_run:
        print("(不实际解压，使用 --no-dry_run 解压)")
        return

    # 解压
    for zip_path in zip_files:
        print(f"解压: {zip_path}")

        rel_path = zip_path.relative_to(root_path)
        extract_dir = output_path / rel_path.parent
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"  ✓ 解压到: {extract_dir}")
        except Exception as e:
            print(f"  ✗ 解压失败: {e}")
            continue

        if not keep_zip:
            zip_path.unlink()
            print(f"  ✓ 删除 ZIP 文件")
        print()

    # 统计
    total_size = sum(
        f.stat().st_size
        for f in output_path.rglob("*")
        if f.is_file()
    )
    size_gb = total_size / (1024 ** 3)

    print("=" * 50)
    print("解压完成!")
    print(f"输出目录: {output_path}")
    print(f"总大小: {size_gb:.2f} GB")
    print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="只解压 RobotTwin2.0 需要的 ZIP 文件 (支持 obs://)"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="数据集根目录 (本地路径 或 obs://...)"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="解压输出目录（obs:// 必须指定，本地路径可选）"
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="解压后保留 ZIP 文件"
    )
    parser.add_argument(
        "--dry_run",
        "-n",
        action="store_true",
        help="只显示将要解压的文件，不实际解压"
    )

    args = parser.parse_args()

    extract_required_zips(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        keep_zip=args.keep_zip,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()