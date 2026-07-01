# Dobot Dataset Loader for Motus
# Supports Dobot dual-arm robot data (LeRobot v2.1 converted format)

import os
import random
import numpy as np
import torch
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import warnings

# VLM processing imports
from utils.vlm_utils import preprocess_vlm_messages
from transformers import AutoProcessor

# Import image processing utilities
from data.utils.image_utils import (
    tensor_to_pil, apply_image_augmentation,
    load_video_frames, get_video_frame_count, load_first_frame
)

logger = logging.getLogger(__name__)


class DobotDataset(data.Dataset):
    """
    Dataset for Dobot robot data.

    Data structure:
    /cache/.../Dobot/dobot_pour_water_full/dobot_pour_water_full/episode_000000/
    ├── videos/episode_000000.mp4
    ├── qpos/episode_000000.pt
    ├── metas/episode_000000.txt
    └── umt5_wan/trajectory.pt
    """

    def __init__(
        self,
        dataset_dir: str,
        global_downsample_rate: int = 3,
        video_action_freq_ratio: int = 2,
        num_video_frames: int = 8,
        video_size: Tuple[int, int] = (384, 320),
        max_episodes: Optional[int] = None,
        val: bool = False,
        image_aug: bool = False,
        vlm_checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize Dobot dataset.

        Args:
            dataset_dir: Root directory containing task subdirectories
            global_downsample_rate: Global downsampling rate (e.g., 3 for 30Hz->10Hz)
            video_action_freq_ratio: Frequency ratio between video and action
            num_video_frames: Number of video frames to predict
            video_size: Target video resolution (H, W)
            max_episodes: Maximum number of episodes to load (for debugging)
            val: Whether this is validation set
            image_aug: Whether to apply image augmentation
            vlm_checkpoint_path: Path to VLM model
        """
        self.dataset_dir = Path(dataset_dir)
        self.action_dim = 14  # Dobot dual-arm: 7 DOF per arm * 2
        self.state_dim = 14

        # Sampling parameters
        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames

        # Calculate action sequence length
        self.action_chunk_size = num_video_frames * video_action_freq_ratio

        # Standard parameters
        self.video_size = video_size
        self.max_episodes = max_episodes
        self.val = val
        self.image_aug = image_aug

        # Load dataset episodes
        self.episodes = self._load_episodes()

        # Initialize VLM processor
        self.vlm_processor = None
        if vlm_checkpoint_path is not None:
            try:
                self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                logger.info(f"VLM processor loaded from {vlm_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load VLM processor from {vlm_checkpoint_path}: {e}")

    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Scan dataset directory and build episode index."""
        logger.info("Scanning Dobot dataset...")

        all_episodes = []

        # Dobot structure: dataset_dir/{task_name}/{episode_id}/
        # e.g., /cache/.../Dobot/dobot_pour_water_full/episode_000000/
        task_dirs = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir()])

        for task_dir in task_dirs:
            task_name = task_dir.name

            for ep_dir in sorted(task_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                episode_id = ep_dir.name  # e.g., "episode_000000"

                # Build file paths
                videos_dir = ep_dir / "videos"
                qpos_dir = ep_dir / "qpos"
                metas_dir = ep_dir / "metas"
                t5_dir = ep_dir / "umt5_wan"

                video_file = videos_dir / f"{episode_id}.mp4"
                qpos_file = qpos_dir / f"{episode_id}.pt"
                meta_file = metas_dir / f"{episode_id}.txt"
                t5_file = t5_dir / "trajectory.pt"

                # Validate all files exist
                if not all([video_file.exists(), qpos_file.exists(), t5_file.exists()]):
                    logger.warning(f"Incomplete episode {task_name}/{episode_id}, skipping")
                    continue

                episode_data = {
                    'episode_name': episode_id,
                    'task_name': task_name,
                    'qpos_path': str(qpos_file),
                    'video_path': str(video_file),
                    'meta_path': str(meta_file) if meta_file.exists() else None,
                    'lang_path': str(t5_file),
                }

                all_episodes.append(episode_data)

        if not all_episodes:
            raise ValueError(f"No valid episodes found in {self.dataset_dir}")

        # Limit episodes if requested
        if self.max_episodes is not None:
            all_episodes = all_episodes[:self.max_episodes]

        logger.info(f"Dobot dataset: {len(all_episodes)} total episodes")

        return all_episodes

    def _load_robot_data(self, action_data: torch.Tensor, action_indices: List[int], initial_state_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load robot position data."""
        if initial_state_idx >= len(action_data):
            initial_state_idx = len(action_data) - 1
        initial_state = action_data[initial_state_idx].float()

        actions = []
        for idx in action_indices:
            if idx >= len(action_data):
                raise IndexError(f"Action index {idx} out of bounds for action data length {len(action_data)}")
            actions.append(action_data[idx])

        action_sequence = torch.stack(actions).float()
        return initial_state, action_sequence

    def _load_language_embedding(self, lang_path: str) -> Tuple[torch.Tensor, int]:
        """Load pre-encoded language embedding and return the selected index."""
        try:
            embedding_data = torch.load(lang_path, map_location='cpu')

            if isinstance(embedding_data, list):
                selected_idx = random.randint(0, len(embedding_data) - 1)
                embeddings = embedding_data[selected_idx]
            else:
                embeddings = embedding_data
                selected_idx = 0

            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)

            return embeddings, selected_idx
        except Exception as e:
            logger.error(f"Error loading language embedding from {lang_path}: {e}")
            raise

    def _load_text_instruction(self, text_path: Optional[str] = None) -> str:
        """Load text instruction from meta file."""
        try:
            if text_path is not None and os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            raise FileNotFoundError(f"Text instruction file not found: {text_path}")
        except Exception as e:
            logger.error(f"Error loading text instruction: {e}")
            raise

    def _calculate_sampling_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        """Calculate sampling indices for video frames and actions."""
        physical_chunk_size = self.action_chunk_size * self.global_downsample_rate

        max_condition_idx = total_frames - physical_chunk_size - 1
        if max_condition_idx < 0:
            condition_frame_idx = 0
        else:
            condition_frame_idx = random.randint(0, max_condition_idx)

        action_indices = []
        for i in range(self.action_chunk_size):
            action_idx = condition_frame_idx + (i + 1) * self.global_downsample_rate
            action_indices.append(min(action_idx, total_frames - 1))

        video_indices = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            if action_step < len(action_indices):
                video_indices.append(action_indices[action_step])
            else:
                video_indices.append(action_indices[-1])

        return condition_frame_idx, video_indices, action_indices

    def __len__(self):
        return len(self.episodes) * 10000

    def __getitem__(self, idx):
        """Get a training sample (idx is ignored, random sampling)."""
        if not self.episodes:
            return None
        episode = random.choice(self.episodes)

        try:
            # Load action data (qpos)
            action_data = torch.load(episode['qpos_path'], map_location='cpu').float()

            # Get video frame count
            total_frames = get_video_frame_count(episode['video_path'])
            if total_frames < 2:
                return None

            # Calculate sampling indices
            condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(total_frames)

            # Load condition frame and video frames
            first_frame, original_frame = load_first_frame(episode['video_path'], condition_frame_idx, self.video_size)
            video_frames = load_video_frames(episode['video_path'], video_indices, self.video_size)

            # Load robot data
            initial_state, action_sequence = self._load_robot_data(action_data, action_indices, condition_frame_idx)

            # Load language embedding
            language_embedding, instruction_idx = self._load_language_embedding(episode['lang_path'])

            # Load text instruction
            text_instruction = self._load_text_instruction(episode.get('meta_path'))

            # VLM processing
            vlm_inputs = None
            if self.vlm_processor is not None:
                first_frame_pil = tensor_to_pil(original_frame)
                vlm_inputs = preprocess_vlm_messages(text_instruction, first_frame_pil, self.vlm_processor)

            # No normalization — raw qpos
            return {
                'first_frame': first_frame,
                'video_frames': video_frames,
                'initial_state': initial_state,
                'action_sequence': action_sequence,
                'language_embedding': language_embedding,
                'vlm_inputs': vlm_inputs,
            }

        except Exception as e:
            logger.error(f"Error loading episode {idx} ({episode['episode_name']}): {e}")
            return None
