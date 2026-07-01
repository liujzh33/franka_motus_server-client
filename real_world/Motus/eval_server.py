#!/usr/bin/env python3
"""
Eval script for Motus server - compute MSE and open-loop plots.

Usage:
    python eval_server.py \
        --url http://localhost:6789 \
        --episode /cache/.../episode_000000 \
        --frame 50 \
        --instruction "cook vegetable" \
        --output_dir ./eval_results
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
import torch
import matplotlib.pyplot as plt

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[3]
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='Eval Motus server and generate open-loop plots')
    parser.add_argument('--url', type=str, required=True, help='Server URL, e.g., http://localhost:6789')
    parser.add_argument('--episode', type=str, required=True, help='Episode directory path')
    parser.add_argument('--frame', type=int, required=True, help='Frame index for state and image')
    parser.add_argument('--instruction', type=str, required=True, help='Task instruction text')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory for plots')
    parser.add_argument('--t5_embeddings_name', type=str, default='trajectory.pt', help='T5 embeddings filename')
    parser.add_argument('--qpos_name', type=str, default=None, help='Qpos filename (default: auto-detect)')
    parser.add_argument('--video_name', type=str, default=None, help='Video filename (default: auto-detect)')
    parser.add_argument('--meta_name', type=str, default=None, help='Meta filename for instruction override')
    parser.add_argument('--state_dim', type=int, default=None, help='State dimension (default: from qpos)')
    parser.add_argument('--action_chunk_size', type=int, default=16, help='Action chunk size')
    parser.add_argument('--global_downsample_rate', type=int, default=3, help='Downsample rate')
    return parser.parse_args()


def auto_detect_files(episode_dir: Path, args):
    """Auto-detect file paths from episode directory."""
    episode_name = episode_dir.name

    # T5 embeddings
    t5_dir = episode_dir / 'umt5_wan'
    if t5_dir.exists():
        t5_path = t5_dir / args.t5_embeddings_name
    else:
        t5_path = episode_dir / args.t5_embeddings_name

    # Qpos - look for qpos subdir or direct file
    if args.qpos_name:
        qpos_path = episode_dir / args.qpos_name
    else:
        qpos_dir = episode_dir / 'qpos'
        if qpos_dir.exists():
            qpos_path = qpos_dir / f"{episode_name}.pt"
        else:
            # Try common patterns
            for pattern in [f"{episode_name}.pt", 'qpos.pt', 'data.pt']:
                if (episode_dir / pattern).exists():
                    qpos_path = episode_dir / pattern
                    break
            else:
                raise FileNotFoundError(f"Cannot find qpos file in {episode_dir}")

    # Video
    if args.video_name:
        video_path = episode_dir / args.video_name
    else:
        videos_dir = episode_dir / 'videos'
        if videos_dir.exists():
            video_path = videos_dir / f"{episode_name}.mp4"
        else:
            for pattern in [f"{episode_name}.mp4", 'video.mp4', 'video.mp4.jpg']:
                if (episode_dir / pattern).exists():
                    video_path = episode_dir / pattern
                    break
            else:
                raise FileNotFoundError(f"Cannot find video file in {episode_dir}")

    # State
    state_path = episode_dir / 'state.json'

    return {
        't5_path': str(t5_path),
        'qpos_path': str(qpos_path),
        'video_path': str(video_path),
        'state_path': str(state_path) if state_path.exists() else None,
    }


def get_state_from_qpos(qpos, frame_idx, state_dim=None):
    """Extract state from qpos at given frame index."""
    if state_dim is None:
        state_dim = qpos.shape[1]
    return qpos[frame_idx].tolist()[:state_dim]


def extract_video_frame(video_path: str, frame_idx: int, output_path: str):
    """Extract a single frame from video using ffmpeg."""
    import subprocess
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'select=eq(n\\,{frame_idx})', '-vframes', '1',
        output_path, '-y'
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def call_server_inference(url: str, instruction: str, image_path: str, state: list, t5_path: str, timeout: int = 120):
    """Call server inference endpoint."""
    # Read image
    with open(image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = {
        'instruction': instruction,
        'image': f'data:image/jpeg;base64,{image_b64}',
        'state': state,
        't5_embeddings_path': t5_path,
    }

    resp = requests.post(f'{url}/inference', json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise Exception(f"Server error {resp.status_code}: {resp.text}")
    return resp.json()


def compute_mse(gt_actions: np.ndarray, pred_actions: np.ndarray):
    """Compute MSE metrics."""
    mse_per_dim = np.mean((gt_actions - pred_actions) ** 2, axis=0)
    total_mse = np.mean(mse_per_dim)
    return {
        'mse_per_dim': mse_per_dim.tolist(),
        'total_mse': float(total_mse),
        'rmse': float(np.sqrt(total_mse)),
    }


def get_gt_actions(qpos_path: str, frame: int, action_chunk_size: int, global_downsample_rate: int):
    """Load GT actions from qpos file."""
    qpos = torch.load(qpos_path)
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.numpy()

    action_indices = [
        min(frame + (i + 1) * global_downsample_rate, qpos.shape[0] - 1)
        for i in range(action_chunk_size)
    ]
    return qpos[action_indices]


def plot_openloop(gt_actions: np.ndarray, pred_actions: np.ndarray, action_dim: int,
                  title: str, output_path: str, format: str = 'vertical'):
    """Plot open-loop comparison for all action dimensions."""
    mse_per_dim = np.mean((gt_actions - pred_actions) ** 2, axis=0)
    t = np.arange(gt_actions.shape[0])

    if format == 'vertical':
        # Vertical layout: rows x 2 cols
        rows = (action_dim + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
        axes = axes.flatten() if action_dim > 2 else [axes] if action_dim == 1 else axes.flatten()
    else:
        # Horizontal layout: 2 x rows cols (like G1 36-dim)
        cols = 6
        rows = (action_dim + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if action_dim <= cols:
            axes = axes.reshape(1, -1) if action_dim > 1 else [axes]
        else:
            axes = axes.flatten()

    for d in range(action_dim):
        ax = axes[d] if action_dim > 1 else axes
        if action_dim == 1:
            ax = axes
        else:
            ax = axes[d] if format == 'vertical' else axes[d // cols, d % cols]

        ax.plot(t, gt_actions[:, d], 'b-', linewidth=2, label='GT')
        ax.plot(t, pred_actions[:, d], 'r--', linewidth=2, label='Pred')
        ax.set_title(f'Dim {d} (MSE: {mse_per_dim[d]:.6f})', fontsize=10)
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    # Hide unused axes
    for d in range(action_dim, len(axes)):
        axes[d].set_visible(False)

    plt.suptitle(f'{title}\nAvg MSE: {np.mean(mse_per_dim):.6f}', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    args = parse_args()

    episode_dir = Path(args.episode)
    if not episode_dir.exists():
        print(f"Error: Episode directory not found: {episode_dir}")
        sys.exit(1)

    # Auto-detect files
    files = auto_detect_files(episode_dir, args)
    print(f"Files detected:")
    for k, v in files.items():
        print(f"  {k}: {v}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frame image
    frame_image_path = output_dir / f"frame_{args.frame}.jpg"
    extract_video_frame(files['video_path'], args.frame, str(frame_image_path))

    # Load qpos and get state
    qpos = torch.load(files['qpos_path'])
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.numpy()

    action_dim = qpos.shape[1]
    state = get_state_from_qpos(qpos, args.frame, args.state_dim)

    # Save state for debugging
    state_path = output_dir / f"state_frame_{args.frame}.json"
    with open(state_path, 'w') as f:
        json.dump(state, f)

    # Call server
    print(f"\nCalling server at {args.url}...")
    result = call_server_inference(
        args.url, args.instruction, str(frame_image_path), state, files['t5_path']
    )
    pred_actions = np.array(result['predicted_actions'])
    print(f"Predicted actions shape: {pred_actions.shape}")

    # Get GT actions
    gt_actions = get_gt_actions(
        files['qpos_path'], args.frame, args.action_chunk_size, args.global_downsample_rate
    )
    print(f"GT actions shape: {gt_actions.shape}")

    # Compute MSE
    metrics = compute_mse(gt_actions, pred_actions)
    print(f"\nMetrics:")
    print(f"  Total MSE: {metrics['total_mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MSE per dim: {metrics['mse_per_dim']}")

    # Save metrics
    metrics_path = output_dir / f"metrics_frame_{args.frame}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Determine plot format based on action dim
    plot_format = 'horizontal' if action_dim >= 20 else 'vertical'

    # Plot openloop
    plot_path = output_dir / f"openloop_frame_{args.frame}.png"
    plot_openloop(
        gt_actions, pred_actions, action_dim,
        title=f"{episode_dir.name} - Frame {args.frame}",
        output_path=str(plot_path),
        format=plot_format
    )

    print(f"\nResults saved to {output_dir}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Plot: {plot_path}")


if __name__ == '__main__':
    main()
