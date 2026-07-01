#!/bin/bash

echo "=========================================="
echo "同时启动训练任务 (nohup 后台运行)"
echo "=========================================="

# 创建日志目录
mkdir -p /cache/wwx1484778/motus/logs

# 启动任务1: GPU 0
nohup bash wan_vlm_mask2_di1.sh > /cache/wwx1484778/motus/logs/task1.log 2>&1 &
PID1=$!
echo "[任务1] PID: $PID1 (GPU: 0) 日志: /cache/wwx1484778/motus/logs/task1.log"

sleep 2

# 启动任务2: GPU 1
nohup bash wan_vlm_mask_di2.sh > /cache/wwx1484778/motus/logs/task2.log 2>&1 &
PID2=$!
echo "[任务2] PID: $PID2 (GPU: 1) 日志: /cache/wwx1484778/motus/logs/task2.log"

echo "=========================================="
echo "等待两个任务完成..."
echo "=========================================="

# 等待两个任务都完成
wait $PID1 $PID2

echo "=========================================="
echo "所有训练任务已完成！"
echo "=========================================="