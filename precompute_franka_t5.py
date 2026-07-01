#!/usr/bin/env python3
"""
为 franka LeRobot v3.0 数据集预计算 T5 文本 embedding。
v3.0 格式没有 episodes.jsonl，改为从 meta/episodes/ parquet 读取任务文本。
输出: {dataset_root}/t5_embedding/episode_XXXXXX.pt
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import torch
import pyarrow.parquet as pq
from pathlib import Path

WAN_PATH = "/home/ma-user/work/wx1513998/pretrained_models"
T5_TEXT_LEN = 512
T5_FOLDER = "t5_embedding"

DATASETS = [
    "/home/ma-user/work/wx1513998/franka_data/place_objects_into_the_box_rc_0524",
    "/home/ma-user/work/wx1513998/franka_data/stack_bowls_rc",
]

def load_t5_encoder(wan_path, text_len, device="cuda"):
    """加载 WAN T5 编码器"""
    import sys
    bak_root = "/home/ma-user/work/wx1513998/Motus_initial/bak"
    if bak_root not in sys.path:
        sys.path.insert(0, bak_root)
    from wan.modules.t5 import T5EncoderModel

    ckpt = os.path.join(wan_path, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth")
    tok = os.path.join(wan_path, "Wan2.2-TI2V-5B", "google/umt5-xxl")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"Loading T5 encoder: ckpt={ckpt}, tok={tok}, device={device}")
    encoder = T5EncoderModel(
        text_len=text_len,
        dtype=dtype,
        device=device,
        checkpoint_path=ckpt,
        tokenizer_path=tok,
    )
    return encoder

def get_episodes_tasks(dataset_root):
    """从 v3.0 episodes parquet 读取每个 episode 的任务文本"""
    eps_dir = Path(dataset_root) / "meta" / "episodes"
    if not eps_dir.exists():
        raise FileNotFoundError(f"episodes dir not found: {eps_dir}")
    
    # 读取所有 episodes parquet 文件
    parquet_files = sorted(eps_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {eps_dir}")
    
    t = pq.read_table(parquet_files[0])
    df = t.to_pandas()
    
    episodes = []
    for _, row in df.iterrows():
        ep_idx = int(row["episode_index"])
        tasks = row["tasks"]
        # tasks 可能是 numpy.ndarray, list, 或 string
        if hasattr(tasks, '__len__') and not isinstance(tasks, str):
            task_text = tasks[0] if len(tasks) > 0 else str(tasks)
        elif isinstance(tasks, str):
            import ast
            try:
                parsed = ast.literal_eval(tasks)
                task_text = parsed[0] if len(parsed) > 0 else tasks
            except Exception:
                task_text = tasks
        else:
            task_text = str(tasks)
        episodes.append((ep_idx, task_text))
    
    episodes.sort(key=lambda x: x[0])
    return episodes

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    encoder = load_t5_encoder(WAN_PATH, T5_TEXT_LEN, device)
    
    for dataset_root in DATASETS:
        print(f"\n===== Processing {dataset_root} =====")
        episodes = get_episodes_tasks(dataset_root)
        print(f"  Total episodes: {len(episodes)}")
        
        # 编码所有唯一的任务文本
        unique_tasks = list(set(task for _, task in episodes))
        print(f"  Unique tasks: {unique_tasks}")
        
        task_to_emb = {}
        for task in unique_tasks:
            with torch.no_grad():
                t5_out = encoder([task], device)
            if isinstance(t5_out, list):
                emb = t5_out[0]
            elif isinstance(t5_out, torch.Tensor):
                emb = t5_out
            if isinstance(emb, torch.Tensor) and emb.ndim == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            task_to_emb[task] = emb.cpu()
            print(f"  Encoded task: '{task}' -> shape {emb.shape}")
        
        # 保存每个 episode 的 embedding
        out_dir = Path(dataset_root) / T5_FOLDER
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for ep_idx, task_text in episodes:
            out_path = out_dir / f"episode_{ep_idx:06d}.pt"
            emb = task_to_emb[task_text]
            torch.save(emb.detach().cpu(), out_path)
        
        print(f"  Saved {len(episodes)} embeddings to {out_dir}")
    
    print("\nAll T5 embeddings computed.")

if __name__ == "__main__":
    main()
