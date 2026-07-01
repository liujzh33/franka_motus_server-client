#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 OBS 下载文件夹到本地 (优化版)

使用 mox.file.copy_parallel 实现并行下载，大幅提升速度
"""

import moxing as mox
from pathlib import Path


def download_obs_to_local_parallel(
    obs_dir: str,
    local_dir: str,
    overwrite: bool = True
) -> None:
    """
    从 OBS 下载目录或文件到本地 (并行版本)

    Args:
        obs_dir: OBS 路径 (如 obs://yw-2030-gy/external/wwx1484778/)
        local_dir: 本地目录路径
        overwrite: 是否覆盖已存在的文件
    """
    local_path_obj = Path(local_dir)

    if not obs_dir.startswith("obs://"):
        raise ValueError(f"OBS 路径必须以 obs:// 开头: {obs_dir}")

    # 判断是文件还是文件夹
    # 有真实文件扩展名才当作文件，否则当作目录
    real_extensions = {'.pt', '.pth', '.json', '.yaml', '.yml', '.bin', '.safetensors', '.pkl', '.ckpt', '.h5', '.hdf5', '.npz', '.csv', '.txt', '.md', '.sh', '.py', '.log'}
    has_real_ext = any(local_path_obj.suffix.lower() == ext for ext in real_extensions)

    if has_real_ext:
        # 有真实扩展名，当作文件下载
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载文件: {obs_dir} -> {local_dir}")
        mox.file.copy(obs_dir, str(local_dir))
        print(f"文件下载完成: {local_dir}")
    else:
        # 没有真实扩展名，当作目录下载
        local_path_obj.mkdir(parents=True, exist_ok=True)
        print(f"开始下载 (并行): {obs_dir} -> {local_dir}")
        mox.file.copy_parallel(obs_dir, str(local_dir))
        print(f"下载完成: {local_dir}")


def list_obs_structure():
    """查看 OBS 目录结构"""
    obs_dir = "obs://yw-2030-gy/external/wwx1484778"
    if not obs_dir.endswith('/'):
        obs_dir += '/'

    print(f"OBS 目录结构: {obs_dir}")
    items = mox.file.list_directory(obs_dir)
    for item in items:
        item_name = item.rstrip('/')
        print(f"  - {item_name}")


if __name__ == "__main__":
    # 要下载的目录列表
    folders_to_download = [
        "Qwen3-VL-2B-Instruct",
        "Wan2.2-TI2V-5B",
        "ActionExpert",
        "InternVLA-qwen3vl",
        "Motus_Wan2_2_5B_pretrain",
        "Motus_stage2",
        "train_log/checkpoints_0430_pretrain/pretrain_multi_source/motus_wan_vlm_multi_source_pretrain_bs8_lr5e-05/checkpoint_step_80000/pytorch_model/mp_rank_00_model_states.pt",
        "train_log/checkpoints_0501_15w_pretrain/pretrain_multi_source_15w/motus_wan_vlm_multi_source_pretrain_bs8_lr5e-05/checkpoint_step_150000/pytorch_model/mp_rank_00_model_states.pt"

    ]
    obs_base_dir = "obs://yw-2030-gy/external/wwx1484778/"
    local_base_dir = "/cache/wwx1484778/motus_weights"

    print("=" * 50)
    print(f"将要下载的目录: {', '.join(folders_to_download)}")
    print("=" * 50)

    # 逐个下载指定目录
    success_folders = 0
    for folder in folders_to_download:
        obs_dir = obs_base_dir + folder
        local_dir = local_base_dir + "/" + folder

        print(f"\n{'=' * 50}")
        print(f"下载目录: {folder}")
        print(f"  OBS: {obs_dir}")
        print(f"  Local: {local_dir}")
        print(f"{'=' * 50}")

        try:
            download_obs_to_local_parallel(obs_dir, local_dir)
            success_folders += 1
        except Exception as e:
            print(f"下载失败: {e}")

    print("\n" + "=" * 50)
    print(f"下载任务完成! 成功: {success_folders}/{len(folders_to_download)}")
    print("=" * 50)
