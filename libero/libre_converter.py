#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIBERO Text Embedding Converter for Cosmos

This script extracts task descriptions from LIBERO HDF5 files or metainfo
and generates Cosmos reason1 (Qwen2.5-VL-7B) text embeddings.

Usage:
    python libre_converter.py --dataset_dir /path/to/libero --output_dir /path/to/libero/reason1_cosmos
"""

import os
import sys
import json
import argparse
import logging
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiberoCosmosTextEncoder:
    """LIBERO text encoder using Cosmos reason1"""

    def __init__(
        self,
        cosmos_checkpoint_path: str = None,
        max_length: int = 512,
        device: str = "cuda:0"
    ):
        self.max_length = max_length
        self.device = device
        self.cosmos_checkpoint_path = cosmos_checkpoint_path
        self._text_encoder = None

    def _init_encoder(self):
        """Initialize Cosmos text encoder (lazy initialization)"""
        if self._text_encoder is None:
            try:
                # Ensure we use Three_MOT cosmos path
                cosmos_path = "/home/ma-user/work/wwx1484778/Three_MOT/cosmos-predict2.5"
                if cosmos_path not in sys.path:
                    sys.path.insert(0, cosmos_path)

                from cosmos_predict2._src.predict2.text_encoders.text_encoder import (
                    TextEncoder, TextEncoderConfig
                )

                # Default cosmos checkpoint path
                if self.cosmos_checkpoint_path is None:
                    self.cosmos_checkpoint_path = (
                        "/cache/wwx1484778/cosmos/Predict2.5-2B/robot/"
                        "mutiview-agibot/f740321e-2cd6-4370-bbfe-545f4eca2065_ema_bf16.pt"
                    )

                logger.info(f"Initializing Cosmos text encoder on {self.device}")

                # Create config that matches trained Cosmos models
                config = TextEncoderConfig(
                    compute_online=True,
                    embedding_concat_strategy="full_concat",  # FULL_CONCAT: 28 layers concat = 100352D
                    n_layers_per_group=5,
                )

                with torch.cuda.device(self.device):
                    self._text_encoder = TextEncoder(config, device=self.device)

                logger.info("Cosmos reason1 text encoder initialized")

            except Exception as e:
                logger.error(f"Failed to initialize Cosmos text encoder: {e}")
                import traceback
                traceback.print_exc()
                raise

    def encode_task_description(self, task_description: str) -> torch.Tensor:
        """
        Encode a task description using Cosmos text encoder.

        Args:
            task_description: Text description of the task

        Returns:
            Text embedding tensor of shape [seq_len, 100352]
        """
        # Initialize encoder if needed
        self._init_encoder()

        # Process text with Cosmos text encoder
        with torch.no_grad():
            data_batch = {"ai_caption": [task_description]}
            text_embeddings = self._text_encoder.compute_text_embeddings_online(
                data_batch=data_batch,
                input_caption_key="ai_caption",
            )

        return text_embeddings


def extract_task_names_from_hdf5_files(dataset_dir: str) -> dict:
    """
    Extract task names from HDF5 files in the dataset directory.

    Args:
        dataset_dir: Path to LIBERO dataset directory

    Returns:
        Dictionary mapping task names to their descriptions
    """
    dataset_path = Path(dataset_dir)
    task_names = {}

    # Get all HDF5 files
    hdf5_files = list(dataset_path.glob("*.hdf5"))

    for hdf5_file in hdf5_files:
        # Extract task name from filename (remove .hdf5 and _demo suffixes)
        filename = hdf5_file.name
        task_name = filename.replace(".hdf5", "").replace("_demo", "")
        # Convert to human-readable format
        task_description = task_name.replace("_", " ")

        task_names[task_name] = task_description
        logger.info(f"Found task: {task_name} -> {task_description}")

    return task_names


def extract_task_names_from_metainfo(dataset_dir: str) -> dict:
    """
    Extract task names from libero_goal_metainfo.json file.

    Args:
        dataset_dir: Path to LIBERO dataset directory

    Returns:
        Dictionary mapping task names to their descriptions
    """
    metainfo_path = Path(dataset_dir) / "libero_goal_metainfo.json"
    task_names = {}

    if not metainfo_path.exists():
        logger.warning(f"Metainfo file not found: {metainfo_path}")
        return task_names

    with open(metainfo_path, 'r') as f:
        metainfo = json.load(f)

    # metinfo structure: {task_name: {demo_0: {...}, ...}}
    for task_name in metainfo.keys():
        # Convert underscores to spaces for description
        task_description = task_name.replace("_", " ")
        task_names[task_name] = task_description
        logger.info(f"Found task from metainfo: {task_name} -> {task_description}")

    return task_names


def generate_and_save_embeddings(
    task_names: dict,
    output_dir: str,
    encoder: LiberoCosmosTextEncoder,
    skip_existing: bool = True
):
    """
    Generate Cosmos text embeddings for all tasks and save them.

    Args:
        task_names: Dictionary of {task_name: task_description}
        output_dir: Directory to save embeddings
        encoder: Cosmos text encoder instance
        skip_existing: Skip already processed tasks
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_tasks = len(task_names)
    processed = 0
    skipped = 0

    for task_name, task_description in task_names.items():
        output_file = output_path / f"{task_name}.pt"

        # Skip if already exists
        if skip_existing and output_file.exists():
            logger.info(f"Skipping existing: {output_file}")
            skipped += 1
            continue

        logger.info(f"Processing task: {task_name} - '{task_description}'")

        try:
            # Generate embedding
            embedding = encoder.encode_task_description(task_description)

            # Save embedding
            import torch
            torch.save(embedding.cpu(), output_file)
            logger.info(f"  Saved to: {output_file}")
            processed += 1

        except Exception as e:
            logger.error(f"Failed to generate embedding for {task_name}: {e}")
            continue

    logger.info(f"\nProcessing complete:")
    logger.info(f"  Total tasks: {total_tasks}")
    logger.info(f"  Processed: {processed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Failed: {total_tasks - processed - skipped}")


def main():
    parser = argparse.ArgumentParser(description="Generate Cosmos text embeddings for LIBERO tasks")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/cache/wwx1484778/LIBERO-Cosmos-Policy/success_only/libero_goal_regen/",
        help="Path to LIBERO dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to output directory for embeddings (default: dataset_dir/reason1_cosmos)"
    )
    parser.add_argument(
        "--cosmos_checkpoint",
        type=str,
        default=None,
        help="Path to Cosmos checkpoint (default: use standard path)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for text encoder"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenizer"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip tasks that already have embeddings"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embeddings"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.dataset_dir) / "reason1_cosmos")

    # Handle skip_existing flag
    if args.force:
        args.skip_existing = False

    logger.info("=" * 60)
    logger.info("LIBERO Cosmos Text Embedding Generator")
    logger.info("=" * 60)
    logger.info(f"Dataset directory: {args.dataset_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Max sequence length: {args.max_length}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("=" * 60)

    # Initialize text encoder
    logger.info("Initializing Cosmos text encoder...")
    encoder = LiberoCosmosTextEncoder(
        cosmos_checkpoint_path=args.cosmos_checkpoint,
        max_length=args.max_length,
        device=args.device
    )

    # Extract task names from metainfo file (preferred)
    logger.info("\nExtracting task names from metainfo...")
    task_names = extract_task_names_from_metainfo(args.dataset_dir)

    # Fallback to HDF5 filenames if metainfo is empty
    if not task_names:
        logger.info("No tasks found in metainfo, extracting from HDF5 files...")
        task_names = extract_task_names_from_hdf5_files(args.dataset_dir)

    if not task_names:
        logger.error("No tasks found in dataset!")
        sys.exit(1)

    # Generate and save embeddings
    logger.info("\nGenerating embeddings...")
    generate_and_save_embeddings(
        task_names=task_names,
        output_dir=args.output_dir,
        encoder=encoder,
        skip_existing=args.skip_existing
    )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()