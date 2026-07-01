#!/bin/bash
# Dim16 Franka training: 7-dim Franka data unified to 16-dim [left6+pad+grip+right_pad(8)]
# No pretrain weights loaded (action_dim mismatch 7->16)

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nproc_per_node=4 \
    --master_port=28102 \
    train/train_wan_vlm_mask_franka.py \
    --deepspeed configs/zero1.json \
    --config configs/robotwin_wan_vlm_mask_stage2_franka_dim16.yaml
