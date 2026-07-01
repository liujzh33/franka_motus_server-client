#!/usr/bin/env python3
"""
Test script to compare VLM inputs between server and eval.
This script compares:
1. The VLM inputs built by the server (build_vlm_inputs)
2. The VLM inputs built by the dataset (preprocess_vlm_messages)

Key differences to check:
1. Content order in messages: [image, text] vs [text, image]
2. add_generation_prompt: True vs False
3. Image processing: process_vision_info vs direct image
"""

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor

# Import server's build_vlm_inputs
from inference.real_world.Motus.server_vlm_mask_g1 import build_vlm_inputs, resize_image_with_padding

# Import dataset's preprocess_vlm_messages
from utils.vlm_utils import preprocess_vlm_messages

def create_test_image():
    """Create a test image."""
    img = np.random.randint(0, 255, (320, 384, 3), dtype=np.uint8)
    return Image.fromarray(img)

def compare_vlm_inputs():
    """Compare VLM inputs from server vs dataset preprocessing."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instruction = "Pick up the water bottle"
    test_image = create_test_image()

    # Try to load VLM processor
    vlm_path = "/home/ma-user/work/wwx1484778/Our/Motus/checkpoints/qwen3_vl_2b"
    try:
        processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load VLM processor: {e}")
        return

    # Resize image for server (same as server does)
    image_hw = (320, 384)
    resized_pil = resize_image_with_padding(test_image, image_hw)

    print("=" * 60)
    print("Comparing VLM Input Construction")
    print("=" * 60)

    # Server approach
    print("\n[Server] build_vlm_inputs:")
    server_vlm_inputs = build_vlm_inputs(processor, instruction, resized_pil, device)
    print(f"  Keys: {server_vlm_inputs.keys()}")
    print(f"  input_ids shape: {server_vlm_inputs['input_ids'].shape}")
    print(f"  pixel_values shape: {server_vlm_inputs['pixel_values'].shape}")
    if 'image_grid_thw' in server_vlm_inputs:
        print(f"  image_grid_thw: {server_vlm_inputs['image_grid_thw']}")

    # Dataset approach
    print("\n[Dataset] preprocess_vlm_messages:")
    dataset_vlm_inputs = preprocess_vlm_messages(instruction, test_image, processor)
    print(f"  Keys: {dataset_vlm_inputs.keys()}")
    print(f"  input_ids shape: {dataset_vlm_inputs['input_ids'].shape}")
    print(f"  pixel_values shape: {dataset_vlm_inputs['pixel_values'].shape}")
    if 'image_grid_thw' in dataset_vlm_inputs:
        print(f"  image_grid_thw: {dataset_vlm_inputs['image_grid_thw']}")

    # Check input_ids differences
    server_ids = server_vlm_inputs['input_ids'].to("cpu")
    dataset_ids = dataset_vlm_inputs['input_ids'].to("cpu")

    print("\n" + "=" * 60)
    print("Input ID Comparison")
    print("=" * 60)
    print(f"Server input_ids[:20]: {server_ids[0, :20].tolist()}")
    print(f"Dataset input_ids[:20]: {dataset_ids[0, :20].tolist()}")

    # Check if they match
    if torch.equal(server_ids, dataset_ids):
        print("\n[OK] input_ids are IDENTICAL")
    else:
        diff_mask = (server_ids != dataset_ids)
        diff_count = diff_mask.sum().item()
        print(f"\n[DIFF] input_ids differ in {diff_count} positions")
        print(f"First few differing positions:")
        diff_indices = torch.where(diff_mask)[1][:10]
        for idx in diff_indices:
            print(f"  pos {idx}: server={server_ids[0, idx].item()}, dataset={dataset_ids[0, idx].item()}")

    # Check pixel_values
    server_pv = server_vlm_inputs['pixel_values'].to("cpu")
    dataset_pv = dataset_vlm_inputs['pixel_values'].to("cpu")

    print("\n" + "=" * 60)
    print("Pixel Values Comparison")
    print("=" * 60)
    print(f"Server pixel_values shape: {server_pv.shape}")
    print(f"Dataset pixel_values shape: {dataset_pv.shape}")

    if server_pv.shape != dataset_pv.shape:
        print(f"\n[DIFF] pixel_values have different shapes!")
    elif torch.allclose(server_pv, dataset_pv, atol=1e-6):
        print("\n[OK] pixel_values are IDENTICAL")
    else:
        diff = (server_pv - dataset_pv).abs()
        print(f"\n[DIFF] pixel_values differ, max diff: {diff.max().item():.6f}")


if __name__ == "__main__":
    compare_vlm_inputs()
