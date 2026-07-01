#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU parallel T5 embedding generation.

Usage:
    # 4-GPU parallel, encode all meta files under RoboTwin_processed_motus
    python t5_encode_multigpu.py --target_root /root/.cache/RoboTwin_processed_motus --gpus 1,2,3,4

    # Specific tasks only
    python t5_encode_multigpu.py --target_root /root/.cache/RoboTwin_processed_motus --gpus 0,1 --tasks adjust_bottle beat_block_hammer

    # 8-GPU, larger batch size
    python t5_encode_multigpu.py --target_root /root/.cache/RoboTwin_processed_motus --gpus 0,1,2,3,4,5,6,7 --batch_size 4
"""

import os
import sys
import argparse
import logging
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def worker(rank: int, gpu_id: int, meta_files: list, wan_repo_path: str, t5_max_length: int, batch_size: int):
    """Worker function: each GPU processes its shard of meta files."""
    logger = logging.getLogger(f"gpu{gpu_id}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Add WAN bak module path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wan_module_path = os.path.abspath(os.path.join(script_dir, "bak"))
    if wan_module_path not in sys.path:
        sys.path.insert(0, wan_module_path)

    from wan.modules.t5 import T5EncoderModel

    encoder = T5EncoderModel(
        text_len=t5_max_length,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=os.path.join(wan_repo_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(wan_repo_path, "google/umt5-xxl"),
    )
    logger.info(f"T5 encoder loaded on GPU {gpu_id}")

    success = 0
    fail = 0

    for meta_path, t5_path in tqdm(meta_files, desc=f"GPU {gpu_id}", position=rank, leave=True):
        if os.path.exists(t5_path):
            success += 1
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                content = f.read()

            prompts = [line for line in content.split("\n") if line.strip()]
            if not prompts:
                fail += 1
                continue

            # Batch encode: process in chunks of batch_size to utilize GPU better
            all_encoded = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                encoded = encoder(batch, device)
                for enc in encoded:
                    if isinstance(enc, torch.Tensor):
                        all_encoded.append(enc.cpu())
                    else:
                        all_encoded.append(torch.from_numpy(enc).cpu())

            os.makedirs(os.path.dirname(t5_path), exist_ok=True)
            torch.save(all_encoded, t5_path)
            success += 1

        except Exception as e:
            logger.error(f"Failed {meta_path}: {e}")
            fail += 1

    logger.info(f"GPU {gpu_id} done: {success} success, {fail} failed")
    return success, fail


def _spawn_worker(rank, gpu_ids, shards, wan_repo_path, t5_max_length, batch_size):
    worker(rank, gpu_ids[rank], shards[rank], wan_repo_path, t5_max_length, batch_size)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU T5 embedding generation")
    parser.add_argument("--target_root", type=str, default="/root/.cache/RoboTwin_processed_motus")
    parser.add_argument("--wan_repo_path", type=str, default="/cache/wwx1484778/motus_weights/Wan2.2-TI2V-5B")
    parser.add_argument("--gpus", type=str, default="1,2,3,4", help="Comma-separated GPU IDs")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific task names to encode (default: all tasks)")
    parser.add_argument("--batch_size", type=int, default=2, help="Prompts per T5 forward pass")
    parser.add_argument("--t5_max_length", type=int, default=512)
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    num_gpus = len(gpu_ids)

    # Collect all meta files
    target_path = Path(args.target_root)
    meta_files = []
    for split_dir in sorted(target_path.iterdir()):
        if not split_dir.is_dir():
            continue
        for task_dir in sorted(split_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            # Filter by task name if --tasks specified
            if args.tasks is not None and task_dir.name not in args.tasks:
                continue
            metas_dir = task_dir / "metas"
            umt5_dir = task_dir / "umt5_wan"
            if not metas_dir.exists():
                continue
            umt5_dir.mkdir(exist_ok=True)
            for meta_file in sorted(metas_dir.glob("*.txt")):
                t5_file = umt5_dir / f"{meta_file.stem}.pt"
                meta_files.append((str(meta_file), str(t5_file)))

    print(f"Total meta files: {len(meta_files)}")
    print(f"Using GPUs: {gpu_ids}")

    if not meta_files:
        print("No meta files found, nothing to do.")
        return

    # Shard across GPUs
    shards = [[] for _ in range(num_gpus)]
    for i, item in enumerate(meta_files):
        shards[i % num_gpus].append(item)

    for i, (gid, shard) in enumerate(zip(gpu_ids, shards)):
        print(f"  GPU {gid}: {len(shard)} files")

    mp.spawn(
        _spawn_worker,
        args=(gpu_ids, shards, args.wan_repo_path, args.t5_max_length, args.batch_size),
        nprocs=num_gpus,
        join=True,
    )

    print("All GPUs finished!")


if __name__ == "__main__":
    main()
