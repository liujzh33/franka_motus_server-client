# G1 Real World Dataset Loader for Motus
# Supports Unitree G1 humanoid robot data

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
    load_video_frames, get_video_frame_count
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)

# G1 action dimension (from actions['psi0']['qpos'])
G1_ACTION_DIM = 36


class G1Dataset(data.Dataset):
    """
    Dataset for G1 Real World humanoid robot data.

    Data structure (converted by g1_converter.py):
    {dataset_dir}/{task_name}/{episode_id}/
    ├── videos/{episode_id}.mp4
    ├── qpos/{episode_id}.pt
    ├── metas/{episode_id}.txt
    └── umt5_wan/trajectory.pt

    G1 action is 36-dim from actions['psi0']['qpos']:
    - hands: 14 (left 7 + right 7)
    - arms: 14 (left 7 + right 7)
    - torso: 3 (rpy) + 1 (height) + 4 (vx/vy/vyaw/target_yaw) = 8
    - Total: 14 + 14 + 8 = 36
    """

    def __init__(
        self,
        dataset_dir: str,
        task_name: str = "g1_water_bottle",
        global_downsample_rate: int = 3,
        video_action_freq_ratio: int = 5,
        num_video_frames: int = 3,
        video_size: Tuple[int, int] = (320, 384),
        max_episodes: Optional[int] = None,
        val: bool = False,
        image_aug: bool = False,
        vlm_checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize G1 dataset.

        Args:
            dataset_dir: Root directory containing {task_name}/{episode_id}/ subdirectories
            task_name: Task name subdirectory (e.g., "g1_water_bottle")
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
        self.task_name = task_name
        self.action_dim = G1_ACTION_DIM
        self.state_dim = G1_ACTION_DIM

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

        logger.info(f"G1 dataset initialized:")
        logger.info(f"  Task name: {task_name}, action_dim: {self.action_dim}")
        logger.info(f"  Dataset dir: {dataset_dir}")
        logger.info(f"  Global downsample rate: {global_downsample_rate}")
        logger.info(f"  Video:Action frequency ratio: {video_action_freq_ratio}")
        logger.info(f"  Action chunk size: {self.action_chunk_size}")
        logger.info(f"  Video frames to predict: {num_video_frames}")

        # Initialize VLM processor
        self.vlm_processor = None
        if vlm_checkpoint_path is not None:
            try:
                self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                logger.info(f"VLM processor loaded from {vlm_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load VLM processor from {vlm_checkpoint_path}: {e}")

        # Initialize data structures
        self.episodes = []
        self.total_episodes = 0

        # Load dataset episodes
        self._load_episodes()

    def _load_episodes(self):
        """Scan dataset directory and build episode index."""
        logger.info("Scanning G1 dataset...")

        task_dir = self.dataset_dir / self.task_name
        if not task_dir.exists():
            raise ValueError(f"Task directory not found: {task_dir}")

        all_episodes = []

        # G1 structure: dataset_dir/{task_name}/{episode_id}/
        for ep_dir in sorted(task_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            episode_id = ep_dir.name  # e.g., "episode_0030"

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
                logger.warning(f"Incomplete episode {episode_id}, skipping")
                continue

            episode_data = {
                'episode_name': episode_id,
                'task_name': self.task_name,
                'qpos_path': str(qpos_file),
                'video_path': str(video_file),
                'meta_path': str(meta_file) if meta_file.exists() else None,
                'lang_path': str(t5_file),
            }

            all_episodes.append(episode_data)

        if not all_episodes:
            raise ValueError(f"No valid episodes found in {task_dir}")

        # Limit episodes if requested
        if self.max_episodes is not None:
            all_episodes = all_episodes[:self.max_episodes]

        self.episodes = all_episodes
        self.total_episodes = len(all_episodes)

        logger.info(f"G1 dataset: {self.total_episodes} total episodes in task '{self.task_name}'")

    def _load_robot_data(self, qpos_path: str, action_indices: List[int], initial_state_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load robot position data."""
        qpos_data = torch.load(qpos_path, map_location='cpu')  # [T, feature_dim]

        if initial_state_idx >= len(qpos_data):
            initial_state_idx = len(qpos_data) - 1
        initial_state = qpos_data[initial_state_idx].float()

        actions = []
        for idx in action_indices:
            if idx >= len(qpos_data):
                actions.append(qpos_data[-1])
            else:
                actions.append(qpos_data[idx])

        action_sequence = torch.stack(actions).float()
        return initial_state, action_sequence

    def _load_language_embedding(self, lang_path: str) -> Tuple[torch.Tensor, int]:
        """Load pre-encoded language embedding."""
        try:
            embedding_data = torch.load(lang_path, map_location='cpu')

            # G1 T5 is a list with 1 element: [seq_len, 4096]
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

    def _load_text_instruction(self, meta_path: str, instruction_idx: int = None) -> str:
        """Load text instruction from meta file."""
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return content
        except Exception as e:
            logger.error(f"Failed to load text instruction from {meta_path}: {e}")
            return ""

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

    def __len__(self) -> int:
        """Return approximate dataset length."""
        return self.total_episodes * 10

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a training sample."""
        max_attempts = 8
        for _ in range(max_attempts):
            if not self.episodes:
                continue

            episode_data = random.choice(self.episodes)

            try:
                total_frames = get_video_frame_count(episode_data['video_path'])
                if total_frames < 2:
                    continue

                condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(total_frames)

                first_frame = load_video_frames(episode_data['video_path'], [condition_frame_idx], self.video_size)
                video_frames = load_video_frames(episode_data['video_path'], video_indices, self.video_size)
                initial_state, action_sequence = self._load_robot_data(episode_data['qpos_path'], action_indices, condition_frame_idx)
                language_embedding, instruction_idx = self._load_language_embedding(episode_data['lang_path'])

                # Load text instruction
                text_instruction = ""
                if episode_data['meta_path'] is not None and os.path.exists(episode_data['meta_path']):
                    text_instruction = self._load_text_instruction(episode_data['meta_path'], instruction_idx)

                # VLM processing
                vlm_inputs = None
                if self.vlm_processor is not None:
                    first_frame_pil = tensor_to_pil(first_frame.squeeze(0))
                    vlm_inputs = preprocess_vlm_messages(text_instruction, first_frame_pil, self.vlm_processor)

                return {
                    'first_frame': first_frame.squeeze(0),
                    'video_frames': video_frames,
                    'initial_state': initial_state,
                    'action_sequence': action_sequence,
                    'language_embedding': language_embedding,
                    'vlm_inputs': vlm_inputs,
                }

            except Exception as e:
                logger.warning(f"Retry due to sample error ({episode_data.get('episode_name', '?')}): {e}")
                continue

        return None