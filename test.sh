#!/bin/bash

echo "启动任务1 (GPU: 1)..."
CUDA_VISIBLE_DEVICES=1 nohup torchrun --nproc_per_node=1 --master_port=28400 train/train_wan_vlm.py --config configs/robotwin_wan_vlm.yaml > /dev/null 2>&1 &
echo "任务1 PID: $!"

sleep 5

echo "启动任务2 (GPU: 0)..."
CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=29100 train/train.py --config configs/robotwin.yaml > /dev/null 2>&1 &
echo "任务2 PID: $!"

echo ""
echo "运行以下命令检查任务："
echo "  nvidia-smi      # 看GPU占用"
echo "  ps aux | grep train   # 看进程"