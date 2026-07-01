#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert RoboTwin_processed_pi0 (HDF5 + instructions.json) → RoboTwin_processed_motus format.

RoboTwin_processed_motus is the same format as RoboTwin_emb, suitable for Motus training.

Source format (RoboTwin_processed_pi0):
    {task}/
    ├── aloha-agilex_clean_50-50/
    │   ├── episode_0/
    │   │   ├── episode_0.hdf5  (action, qpos, 3-cam images)
    │   │   └── instructions.json (instructions, subtasks, phase_info)
    │   ├── episode_1/...
    │   └── ...
    └── aloha-agilex_randomized_500-500/
        ├── episode_0/...
        └── ...

Target format (RoboTwin_processed_motus):
    clean/  or  randomized/
    └── {task}/
        ├── videos/{N}.mp4        (T-shape concatenated 3-cam video)
        ├── qpos/{N}.pt           (T, 14) float32 tensor
        ├── metas/{N}.txt         (prefixed instruction text, one per line)
        ├── umt5_wan/{N}.pt       (list of T5 embeddings, one per instruction)
        └── phase_info/{N}.json   (phase_info + subtasks for subtask prediction)
"""

import os
import sys
import json
import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Set multiprocessing start method before CUDA imports
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

import torch
import numpy as np
import h5py
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Meta prefix used in RoboTwin_emb format
META_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)

# HDF5 key mapping (processed_pi0 format)
HDF5_CAM_KEYS = {
    "head": "observations/images/cam_high",
    "left_wrist": "observations/images/cam_left_wrist",
    "right_wrist": "observations/images/cam_right_wrist",
}
HDF5_QPOS_KEY = "observations/qpos"
HDF5_ACTION_KEY = "action"

# Subdirectory name → output split name
DEMO_TYPE_MAPPING = {
    "aloha-agilex_clean_50-50": "clean",
    "aloha-agilex_randomized_500-500": "randomized",
}


def decode_compressed_image(compressed_data: bytes) -> Optional[np.ndarray]:
    """Decode compressed image bytes from HDF5 to BGR numpy array."""
    try:
        np_array = np.frombuffer(compressed_data, dtype=np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.warning(f"Failed to decode image: {e}")
        return None


def create_tshape_video(
    hdf5_path: str,
    output_path: str,
    fps: int = 30,
    target_width: int = 320,
    target_height: int = 360,
) -> bool:
    """
    Extract 3-camera images from HDF5, concatenate in T-shape layout,
    and write to MP4 video.

    T-shape layout:
        ┌─────────────────────────┐
        │      Head Camera        │
        ├────────────┬────────────┤
        │ Left Wrist │ Right Wrist│
        └────────────┴────────────┘
    """
    try:
        with h5py.File(hdf5_path, "r") as f:
            # Check camera keys exist
            for cam_name, hdf5_key in HDF5_CAM_KEYS.items():
                if hdf5_key not in f:
                    logger.error(f"Camera key {hdf5_key} not found in {hdf5_path}")
                    return False

            num_frames = f[HDF5_CAM_KEYS["head"]].shape[0]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
            if not writer.isOpened():
                logger.error(f"Failed to open video writer for {output_path}")
                return False

            for i in range(num_frames):
                head_img = decode_compressed_image(f[HDF5_CAM_KEYS["head"]][i])
                left_img = decode_compressed_image(f[HDF5_CAM_KEYS["left_wrist"]][i])
                right_img = decode_compressed_image(f[HDF5_CAM_KEYS["right_wrist"]][i])

                if head_img is None or left_img is None or right_img is None:
                    logger.warning(f"Failed to decode frame {i}, skipping")
                    continue

                # T-shape concatenation
                orig_h, orig_w = head_img.shape[:2]
                half_h, half_w = orig_h // 2, orig_w // 2

                left_resized = cv2.resize(left_img, (half_w, half_h))
                right_resized = cv2.resize(right_img, (half_w, half_h))
                bottom_row = np.hstack([left_resized, right_resized])
                combined = np.vstack([head_img, bottom_row])

                # Resize to target dimensions
                if combined.shape[:2] != (target_height, target_width):
                    combined = cv2.resize(combined, (target_width, target_height))

                # Convert BGR to RGB (matches existing emb format)
                combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                if combined_rgb.dtype != np.uint8:
                    combined_rgb = combined_rgb.astype(np.uint8)
                if not combined_rgb.flags["C_CONTIGUOUS"]:
                    combined_rgb = np.ascontiguousarray(combined_rgb)

                writer.write(combined_rgb)

            writer.release()
            return True

    except Exception as e:
        logger.error(f"Error creating video from {hdf5_path}: {e}")
        return False


def extract_qpos(hdf5_path: str, output_path: str) -> bool:
    """Extract qpos from HDF5 and save as .pt tensor (T, 14)."""
    try:
        with h5py.File(hdf5_path, "r") as f:
            if HDF5_QPOS_KEY in f:
                qpos = f[HDF5_QPOS_KEY][()]
            elif HDF5_ACTION_KEY in f:
                qpos = f[HDF5_ACTION_KEY][()]
            else:
                logger.error(f"No qpos/action found in {hdf5_path}")
                return False

            qpos_tensor = torch.from_numpy(qpos).float()
            if qpos_tensor.dim() != 2 or qpos_tensor.shape[1] != 14:
                logger.error(f"Unexpected qpos shape {qpos_tensor.shape} in {hdf5_path}")
                return False

            torch.save(qpos_tensor, output_path)
            return True

    except Exception as e:
        logger.error(f"Error extracting qpos from {hdf5_path}: {e}")
        return False


def create_meta_file(instructions: list, output_path: str) -> bool:
    """Create meta file with prefixed instructions, one per line."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for inst in instructions:
                f.write(f"{META_PREFIX}{inst}\n")
        return True
    except Exception as e:
        logger.error(f"Error creating meta file {output_path}: {e}")
        return False


def save_phase_info(phase_info: dict, subtasks: list, output_path: str) -> bool:
    """Save phase_info + subtasks as JSON for subtask prediction."""
    try:
        data = {
            "phase_info": phase_info,
            "subtasks": subtasks,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving phase_info to {output_path}: {e}")
        return False


def process_single_episode(args: Tuple) -> bool:
    """Process a single episode: extract video, qpos, meta, phase_info."""
    (
        hdf5_path,
        instruction_path,
        output_dir,
        episode_id,
        fps,
        target_width,
        target_height,
        skip_existing,
    ) = args

    output_dir = Path(output_dir)

    # Check if all outputs already exist
    video_path = output_dir / "videos" / f"{episode_id}.mp4"
    qpos_path = output_dir / "qpos" / f"{episode_id}.pt"
    meta_path = output_dir / "metas" / f"{episode_id}.txt"
    phase_path = output_dir / "phase_info" / f"{episode_id}.json"

    if skip_existing and video_path.exists() and qpos_path.exists() and meta_path.exists() and phase_path.exists():
        return True

    # Create output subdirectories
    for subdir in ["videos", "qpos", "metas", "phase_info"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Create umt5_wan directory (T5 will be generated in a separate pass)
    (output_dir / "umt5_wan").mkdir(parents=True, exist_ok=True)

    # 1. Video
    if not skip_existing or not video_path.exists():
        if not create_tshape_video(hdf5_path, str(video_path), fps, target_width, target_height):
            logger.error(f"Failed to create video for episode {episode_id}")
            return False

    # 2. Qpos
    if not skip_existing or not qpos_path.exists():
        if not extract_qpos(hdf5_path, str(qpos_path)):
            logger.error(f"Failed to extract qpos for episode {episode_id}")
            return False

    # 3. Instructions → meta + phase_info
    instructions = []
    subtasks = []
    phase_info = {}

    if instruction_path and os.path.exists(instruction_path):
        with open(instruction_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instructions = data.get("instructions", [])
        subtasks = data.get("subtasks", [])
        phase_info = data.get("phase_info", {})

    if not skip_existing or not meta_path.exists():
        if not create_meta_file(instructions, str(meta_path)):
            logger.error(f"Failed to create meta for episode {episode_id}")
            return False

    # 5. Phase info + subtasks
    if not skip_existing or not phase_path.exists():
        if not save_phase_info(phase_info, subtasks, str(phase_path)):
            logger.error(f"Failed to save phase_info for episode {episode_id}")
            return False

    return True


def scan_source_dataset(source_root: str) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """
    Scan source directory and return:
    {split_name: {task_name: [(hdf5_path, instruction_path), ...]}}
    """
    source_path = Path(source_root)
    result = {}

    for task_dir in sorted(source_path.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name

        for demo_dir in sorted(task_dir.iterdir()):
            if not demo_dir.is_dir():
                continue

            # Map demo dir name to split name
            split_name = None
            for key, val in DEMO_TYPE_MAPPING.items():
                if key in demo_dir.name:
                    split_name = val
                    break
            if split_name is None:
                # Try to infer from directory name
                if "clean" in demo_dir.name.lower():
                    split_name = "clean"
                elif "randomized" in demo_dir.name.lower():
                    split_name = "randomized"
                else:
                    logger.warning(f"Unknown demo type: {demo_dir.name}, skipping")
                    continue

            if split_name not in result:
                result[split_name] = {}

            # Find episode directories
            episodes = []
            for episode_dir in sorted(demo_dir.iterdir()):
                if not episode_dir.is_dir():
                    continue
                # Find HDF5 file
                hdf5_files = list(episode_dir.glob("*.hdf5"))
                if not hdf5_files:
                    continue
                hdf5_path = str(hdf5_files[0])

                # Find instructions.json
                inst_path = str(episode_dir / "instructions.json")

                episodes.append((hdf5_path, inst_path))

            if episodes:
                result[split_name][task_name] = episodes

    return result


class T5EmbeddingProcessor:
    """Generate T5 (UMT5-XXL) embeddings for meta text files."""

    def __init__(self, wan_repo_path: str, t5_max_length: int = 512, device: str = "cuda:0"):
        self.wan_repo_path = wan_repo_path
        self.t5_max_length = t5_max_length
        self.device = device
        self._encoder = None

    def _init_encoder(self):
        if self._encoder is not None:
            return
        # Add WAN bak module path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        wan_module_path = os.path.abspath(os.path.join(script_dir, "bak"))
        if wan_module_path not in sys.path:
            sys.path.insert(0, wan_module_path)

        from wan.modules.t5 import T5EncoderModel

        # After CUDA_VISIBLE_DEVICES is set in subprocess, always use cuda:0
        device_obj = torch.device("cuda:0")
        torch.cuda.set_device(device_obj)
        self._encoder = T5EncoderModel(
            text_len=self.t5_max_length,
            dtype=torch.bfloat16,
            device=device_obj,
            checkpoint_path=os.path.join(self.wan_repo_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.wan_repo_path, "google/umt5-xxl"),
        )
        logger.info(f"T5 encoder initialized on cuda:0 (physical GPU via CUDA_VISIBLE_DEVICES)")

    def process_meta_file(self, meta_path: str, t5_output_path: str) -> bool:
        """Process a single meta file → T5 embeddings."""
        try:
            self._init_encoder()

            if os.path.exists(t5_output_path):
                return True

            with open(meta_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"Empty meta file: {meta_path}")
                return False

            # Each line is one instruction variant
            lines = content.split("\n")
            prompts = [line for line in lines if line.strip()]

            if not prompts:
                logger.warning(f"No valid prompts in {meta_path}")
                return False

            device_obj = torch.device("cuda:0")
            encoded_texts = self._encoder(prompts, device_obj)

            # Convert to list of CPU tensors
            encoded_list = []
            for enc in encoded_texts:
                if isinstance(enc, torch.Tensor):
                    encoded_list.append(enc.cpu())
                else:
                    encoded_list.append(torch.from_numpy(enc).cpu())

            os.makedirs(os.path.dirname(t5_output_path), exist_ok=True)
            torch.save(encoded_list, t5_output_path)
            return True

        except Exception as e:
            logger.error(f"Error generating T5 for {meta_path}: {e}")
            return False


def process_t5_batch(args):
    """Process T5 embeddings batch (for ProcessPoolExecutor)."""
    processor, meta_files = args
    device_num = processor.device.split(":")[1] if ":" in processor.device else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num

    results = []
    for meta_path, t5_path in meta_files:
        success = processor.process_meta_file(meta_path, t5_path)
        results.append((meta_path, success))
    return results


def convert_dataset(
    source_root: str,
    target_root: str,
    tasks: Optional[List[str]] = None,
    fps: int = 30,
    target_width: int = 320,
    target_height: int = 360,
    max_workers: int = 8,
    skip_existing: bool = True,
    enable_t5: bool = True,
    wan_repo_path: str = "/cache/wwx1484778/motus_weights/Wan2.2-TI2V-5B",
    t5_cuda_devices: List[str] = None,
):
    """
    Main conversion function.

    Args:
        source_root: Path to RoboTwin_processed_pi0
        target_root: Path to output RoboTwin_processed_motus
        tasks: Optional list of task names to convert (None = all)
        fps: Video FPS
        target_width: Output video width
        target_height: Output video height
        max_workers: Number of parallel workers for episode processing
        skip_existing: Skip already converted episodes
        enable_t5: Whether to generate T5 embeddings
        wan_repo_path: Path to WAN model (for T5 encoder)
        t5_cuda_devices: CUDA device IDs for T5 processing
    """
    if t5_cuda_devices is None:
        t5_cuda_devices = ["0"]

    logger.info(f"Scanning source dataset: {source_root}")
    dataset_structure = scan_source_dataset(source_root)

    if not dataset_structure:
        logger.error("No valid dataset structure found")
        return

    # Filter tasks if specified
    if tasks is not None:
        filtered = {}
        for split_name, split_tasks in dataset_structure.items():
            filtered[split_name] = {
                k: v for k, v in split_tasks.items() if k in tasks
            }
        dataset_structure = filtered

    # Report
    total_episodes = 0
    for split_name, split_tasks in dataset_structure.items():
        for task_name, episodes in split_tasks.items():
            total_episodes += len(episodes)
            logger.info(f"  {split_name}/{task_name}: {len(episodes)} episodes")

    logger.info(f"Total episodes to convert: {total_episodes}")

    if total_episodes == 0:
        logger.warning("No episodes found, nothing to do")
        return

    # Prepare all conversion tasks
    all_tasks = []
    episode_counter = {}  # (split, task) -> next episode_id

    for split_name, split_tasks in dataset_structure.items():
        for task_name, episodes in split_tasks.items():
            task_output_dir = Path(target_root) / split_name / task_name
            key = (split_name, task_name)
            episode_counter[key] = 0

            for hdf5_path, inst_path in episodes:
                ep_id = episode_counter[key]
                episode_counter[key] += 1

                all_tasks.append((
                    hdf5_path,
                    inst_path,
                    str(task_output_dir),
                    ep_id,
                    fps,
                    target_width,
                    target_height,
                    skip_existing,
                ))

    # Process episodes in parallel
    logger.info(f"Processing {len(all_tasks)} episodes with {max_workers} workers...")

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = list(tqdm(
            executor.map(process_single_episode, all_tasks),
            total=len(all_tasks),
            desc="Converting episodes",
        ))

    for result in futures:
        if result:
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"Episode conversion: {success_count} success, {fail_count} failed")

    # T5 embedding generation (single-process to avoid CUDA subprocess issues)
    if enable_t5 and fail_count < len(all_tasks):
        logger.info("=" * 50)
        logger.info("Starting T5 embedding generation...")
        logger.info("=" * 50)

        # Collect meta files
        meta_files = []
        target_path = Path(target_root)

        for split_dir in target_path.iterdir():
            if not split_dir.is_dir():
                continue
            for task_dir in split_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                metas_dir = task_dir / "metas"
                umt5_dir = task_dir / "umt5_wan"
                if not metas_dir.exists():
                    continue
                umt5_dir.mkdir(exist_ok=True)

                for meta_file in sorted(metas_dir.glob("*.txt")):
                    t5_file = umt5_dir / f"{meta_file.stem}.pt"
                    if skip_existing and t5_file.exists():
                        continue
                    meta_files.append((str(meta_file), str(t5_file)))

        logger.info(f"Found {len(meta_files)} meta files for T5 encoding")

        if meta_files:
            # Set CUDA device and initialize T5 encoder
            t5_device_id = t5_cuda_devices[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(t5_device_id)
            logger.info(f"Using GPU {t5_device_id} for T5 encoding")

            # Add WAN bak module path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            wan_module_path = os.path.abspath(os.path.join(script_dir, "bak"))
            if wan_module_path not in sys.path:
                sys.path.insert(0, wan_module_path)

            from wan.modules.t5 import T5EncoderModel

            torch.cuda.set_device("cuda:0")
            t5_device = torch.device("cuda:0")
            t5_encoder = T5EncoderModel(
                text_len=512,
                dtype=torch.bfloat16,
                device=t5_device,
                checkpoint_path=os.path.join(wan_repo_path, "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=os.path.join(wan_repo_path, "google/umt5-xxl"),
            )
            logger.info("T5 encoder initialized")

            t5_success = 0
            t5_fail = 0

            for meta_path, t5_path in tqdm(meta_files, desc="T5 encoding"):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    prompts = [line for line in content.split("\n") if line.strip()]
                    if not prompts:
                        logger.warning(f"No prompts in {meta_path}")
                        t5_fail += 1
                        continue

                    encoded_texts = t5_encoder(prompts, t5_device)
                    encoded_list = []
                    for enc in encoded_texts:
                        if isinstance(enc, torch.Tensor):
                            encoded_list.append(enc.cpu())
                        else:
                            encoded_list.append(torch.from_numpy(enc).cpu())

                    os.makedirs(os.path.dirname(t5_path), exist_ok=True)
                    torch.save(encoded_list, t5_path)
                    t5_success += 1

                except Exception as e:
                    logger.error(f"T5 error for {meta_path}: {e}")
                    t5_fail += 1

            logger.info(f"T5 encoding: {t5_success} success, {t5_fail} failed")

    logger.info("=" * 50)
    logger.info("Conversion complete!")
    logger.info(f"Output directory: {target_root}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoboTwin_processed_pi0 → RoboTwin_processed_motus format"
    )
    parser.add_argument(
        "--source_root",
        type=str,
        default="/root/.cache/RoboTwin_processed_pi0",
        help="Source directory (RoboTwin_processed_pi0)",
    )
    parser.add_argument(
        "--target_root",
        type=str,
        default="/root/.cache/RoboTwin_processed_motus",
        help="Target directory (RoboTwin_processed_motus)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific task names to convert (default: all tasks)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--target_width", type=int, default=320, help="Output video width")
    parser.add_argument("--target_height", type=int, default=360, help="Output video height")
    parser.add_argument("--max_workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip existing files")
    parser.add_argument("--no_t5", action="store_true", help="Skip T5 embedding generation")
    parser.add_argument(
        "--wan_repo_path",
        type=str,
        default="/cache/wwx1484778/motus_weights/Wan2.2-TI2V-5B",
        help="Path to WAN model (for T5 encoder)",
    )
    parser.add_argument(
        "--t5_cuda_devices",
        type=str,
        nargs="+",
        default=["0"],
        help="CUDA device IDs for T5 encoding (e.g., 0 1 2 3)",
    )

    args = parser.parse_args()

    convert_dataset(
        source_root=args.source_root,
        target_root=args.target_root,
        tasks=args.tasks,
        fps=args.fps,
        target_width=args.target_width,
        target_height=args.target_height,
        max_workers=args.max_workers,
        skip_existing=args.skip_existing,
        enable_t5=not args.no_t5,
        wan_repo_path=args.wan_repo_path,
        t5_cuda_devices=args.t5_cuda_devices,
    )


if __name__ == "__main__":
    main()
