#!/bin/bash
# Check GPU memory and wait until condition is met, then launch training

MIN_MEM_FREE=40000  # Minimum free memory in MiB per GPU
CHECK_INTERVAL=60   # Check interval in seconds

echo "Checking GPU memory on GPUs 0-3..."
while true; do
    ALL_CLEAR=true
    for i in 0 1 2 3; do
        MEM_USED=$(nvidia-smi --id=$i --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -z "$MEM_USED" ]; then
            echo "GPU $i: not available, waiting..."
            ALL_CLEAR=false
            break
        fi
        if [ "$MEM_USED" -gt $MIN_MEM_FREE ]; then
            echo "GPU $i: ${MEM_USED} MiB used (exceeds ${MIN_MEM_FREE}), waiting..."
            ALL_CLEAR=false
            break
        else
            echo "GPU $i: ${MEM_USED} MiB used (OK)"
        fi
    done

    if [ "$ALL_CLEAR" = true ]; then
        echo "All GPUs have < ${MIN_MEM_FREE} MiB used, launching training..."
        break
    fi

    echo "Waiting ${CHECK_INTERVAL} seconds before rechecking..."
    sleep $CHECK_INTERVAL
done

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=28102 \
    train/train_wan_vlm_mask.py \
    --deepspeed configs/zero1.json \
    --config configs/robotwin_wan_vlm_mask_dobot_p.yaml
