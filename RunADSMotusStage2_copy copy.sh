#!/bin/bash

# DI平台训练脚本 - Motus Stage2 (WAN + Action Expert + Qwen3-VL Direct MoT + SubtaskTextDecoder)
# 数据: obs://yw-2030-gy/external/wwx1484778/RoboTwin_processed_motus

env
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate qwen3

echo "--------------Sh Start------------------"

USE_JUICEFS=0

export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000


function safe_mkdir() {
    if [ ! -d $1 ]; then
    mkdir $1
    fi
}
function safe_multi_mkdir() {
    if [ ! -d $1 ]; then
    mkdir -p $1
    fi
}
export S3_ENDPOINT="https://obs.cn-southwest-2.myhuaweicloud.com"
export S3_USE_HTTPS="0"
export ACCESS_KEY_ID="HPUACJ8EWHNONWBP1PGQ"
export SECRET_ACCESS_KEY="rx3ITEWElWS8dZnPHmFMmMiiGkQ2pk5FguQ0d1QN"


echo "----------------- copy code -----------------"
rm -rf /cache/wx1513998/Motus
mkdir -p /cache/wx1513998/Motus
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-2030-gy/external/wwx1513998/code/Motus_initial_ping_zi_tu_pian', '/cache/wx1513998/Motus/')"
echo "--------------- copy code done ---------------"


echo "------------------- set envs -------------------"
envs_name=motus
if [ ! -d "/root/miniconda3/envs/$envs_name" ]; then
    python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-2030-gy/external/wwx1484778/envs/motus.tar.gz','/cache/envs/$envs_name.tar.gz')"
    mkdir -p /root/miniconda3/envs/$envs_name
    tar -xf /cache/envs/$envs_name.tar.gz -C /root/miniconda3/envs
fi
conda activate $envs_name
echo "---------------- set envs done ----------------"


echo "---------------- prepare data ----------------"
DATASETDIR=obs://yw-2030-gy/external/wwx1484778/RoboTwin_processed_motus
if [ ! -d '/cache/wx1513998/RoboTwin_processed_motus' ]; then
  safe_multi_mkdir /cache/wx1513998/RoboTwin_processed_motus
  python -c "import moxing as mox; mox.file.copy_parallel('$DATASETDIR', '/cache/wx1513998/RoboTwin_processed_motus')"
fi
echo "-------------- prepare data done --------------"


echo "---------------- prepare model weights ----------------"
# Download pretrained weights (WAN 5B, Qwen3-VL-2B, pretrain checkpoint)
cd /cache/wx1513998/Motus
conda activate motus

WAN_WEIGHTS_OBS="obs://yw-2030-gy/external/wwx1484778/Wan2.2-TI2V-5B"
VLM_WEIGHTS_OBS="obs://yw-2030-gy/external/wwx1484778/Qwen3-VL-2B-Instruct"
PRETRAIN_CKPT_OBS="obs://yw-2030-gy/external/wwx1484778/train_log/checkpoints_0501_15w_pretrain/pretrain_multi_source_15w/motus_wan_vlm_multi_source_pretrain_bs8_lr5e-05/checkpoint_step_150000/pytorch_model"

if [ ! -d '/cache/wx1513998/motus_weights/Wan2.2-TI2V-5B' ]; then
  mkdir -p /cache/wx1513998/motus_weights/Wan2.2-TI2V-5B
  python -c "import moxing as mox; mox.file.copy_parallel('$WAN_WEIGHTS_OBS', '/cache/wx1513998/motus_weights/Wan2.2-TI2V-5B')"
  echo "WAN weights downloaded"
fi

if [ ! -d '/cache/wx1513998/motus_weights/Qwen3-VL-2B-Instruct' ]; then
  mkdir -p /cache/wx1513998/motus_weights/Qwen3-VL-2B-Instruct
  python -c "import moxing as mox; mox.file.copy_parallel('$VLM_WEIGHTS_OBS', '/cache/wx1513998/motus_weights/Qwen3-VL-2B-Instruct')"
  echo "VLM weights downloaded"
fi

PRETRAIN_CKPT_LOCAL="/cache/wx1513998/motus_weights/train_log/checkpoints_0501_15w_pretrain/pretrain_multi_source_15w/motus_wan_vlm_multi_source_pretrain_bs8_lr5e-05/checkpoint_step_150000/pytorch_model"

if [ ! -d "$PRETRAIN_CKPT_LOCAL" ]; then
  mkdir -p "$PRETRAIN_CKPT_LOCAL"
  python -c "import moxing as mox; mox.file.copy_parallel('$PRETRAIN_CKPT_OBS', '$PRETRAIN_CKPT_LOCAL')"
  echo "Pretrain checkpoint downloaded"
fi
echo "-------------- prepare model weights done --------------"


echo "----------------- Train Start -----------------"

echo "Cleaning up any lingering GPU processes..."
pkill -9 -f "python.*train" 2>/dev/null || true
pkill -9 -f "torchrun" 2>/dev/null || true

# Kill ALL GPU processes to free VRAM
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
    pid=$(echo "$pid" | tr -d ' ')
    if [ -n "$pid" ]; then
        echo "Killing GPU process: PID $pid"
        kill -9 "$pid" 2>/dev/null || true
    fi
done

# Verify GPU is clean
REMAINING=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "WARNING: $REMAINING GPU processes still running after cleanup!"
    nvidia-smi
else
    echo "GPU cleanup done - all GPUs free"
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

# NCCL settings for multi-node stability
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_IFNAME=bond0

echo "===== Distributed Env ====="
echo "HOSTNAME=$(hostname)"
echo "WORLD_SIZE=${WORLD_SIZE:-}"
echo "VC_TASK_INDEX=${VC_TASK_INDEX:-}"
echo "MASTER_ADDR=${MASTER_ADDR:-}"
echo "MASTER_PORT=${MASTER_PORT:-}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "==========================="

cd /cache/wx1513998/Motus

# ===== Finetune checkpoint path is already set in config, no sed needed =====
# The config YAML already has the correct finetune.checkpoint_path pointing to PRETRAIN_CKPT_LOCAL
# Do NOT use sed to replace checkpoint_path - it would also overwrite wan and vlm paths!
echo "Using finetune checkpoint from config: $PRETRAIN_CKPT_LOCAL"

# Set OBS upload target
export UPLOAD_OBS_BASE="obs://yw-2030-gy/external/wwx1484778/cache-DI/$(date '+%Y%m%d_%H%M%S')"
echo "Checkpoint upload OBS base: $UPLOAD_OBS_BASE/checkpoints"

# ===== Update config: dataset_dir points to local cache =====
# The YAML config has dataset_dir: "/root/.cache/RoboTwin_processed_motus"
# but on DI we download to /cache/wx1513998/RoboTwin_processed_motus
# Create symlink so config path works
if [ ! -L '/root/.cache/RoboTwin_processed_motus' ] && [ ! -d '/root/.cache/RoboTwin_processed_motus' ]; then
    mkdir -p /root/.cache
    ln -s /cache/wx1513998/RoboTwin_processed_motus /root/.cache/RoboTwin_processed_motus
    echo "Symlinked /root/.cache/RoboTwin_processed_motus -> /cache/wx1513998/RoboTwin_processed_motus"
fi

# ===== Launch training =====
torchrun \
    --nnodes=${WORLD_SIZE:-1} \
    --nproc_per_node=8 \
    --node_rank=${VC_TASK_INDEX:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-28102} \
    train/train_wan_vlm_mask.py \
    --deepspeed configs/zero1.json \
    --config configs/robotwin_wan_vlm_mask_stage2.yaml \
    --report_to tensorboard \
|| true

echo "------------------ Train End ------------------"


echo "------- Upload weights and logs to OBS -------"

if [ "${VC_TASK_INDEX:-0}" = "0" ]; then
    echo "------- Final Upload: weights and logs to OBS -------"
    target_obs="$UPLOAD_OBS_BASE"

    # Find the actual checkpoint directory (contains run name subdirectory)
    CKPT_BASE="/cache/wx1513998/motus/checkpoints_wan_vlm_mask_0519_8w_pretrain_Robotwin"
    TB_BASE="/cache/wx1513998/motus/tensorboard_wan_vlm_mask_0519_8w_pretrain_Robotwin"

    if [ -d "$CKPT_BASE" ]; then
        python -c "import moxing as mox; mox.file.copy_parallel('$CKPT_BASE', '$target_obs/checkpoints')"
        echo "Upload checkpoints done: $target_obs/checkpoints"
    else
        echo "WARNING: checkpoint dir not found at $CKPT_BASE, searching..."
        # Try to find any checkpoint directory
        FOUND_CKPT=$(find /cache/wx1513998/motus -maxdepth 1 -name "checkpoints_*" -type d 2>/dev/null | head -1)
        if [ -n "$FOUND_CKPT" ]; then
            echo "Found checkpoint dir: $FOUND_CKPT"
            python -c "import moxing as mox; mox.file.copy_parallel('$FOUND_CKPT', '$target_obs/checkpoints')"
            echo "Upload checkpoints done: $target_obs/checkpoints"
        else
            echo "WARNING: No checkpoint directory found, skip upload"
        fi
    fi

    if [ -d "$TB_BASE" ]; then
        python -c "import moxing as mox; mox.file.copy_parallel('$TB_BASE', '$target_obs/tensorboard_logs')"
        echo "Upload tensorboard logs done: $target_obs/tensorboard_logs"
    else
        echo "WARNING: tensorboard log dir not found, searching..."
        FOUND_TB=$(find /cache/wx1513998/motus -maxdepth 1 -name "tensorboard_*" -type d 2>/dev/null | head -1)
        if [ -n "$FOUND_TB" ]; then
            echo "Found tensorboard dir: $FOUND_TB"
            python -c "import moxing as mox; mox.file.copy_parallel('$FOUND_TB', '$target_obs/tensorboard_logs')"
            echo "Upload tensorboard logs done: $target_obs/tensorboard_logs"
        else
            echo "WARNING: No tensorboard directory found, skip upload"
        fi
    fi
fi
