#!/bin/bash

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
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-2030-gy/external/wwx1513998/code/Motus_threeclip_20w_batch8_1_data_clean_only/', '/cache/wx1513998/Motus/')"
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
DATASETDIR=obs://yw-2030-gy/external/wwx1513998/dataset/RoboTwin_processed_motus
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

# ===== Finetune checkpoint path is already set in config =====

# Set OBS upload target (use JOB_ID so all nodes share the same directory)
export RUN_ID=${JOB_ID:-$(date '+%Y%m%d_%H%M%S')}
export UPLOAD_OBS_BASE="obs://yw-2030-gy/external/wwx1484778/cache-DI/$RUN_ID"
echo "Checkpoint upload OBS base: $UPLOAD_OBS_BASE/checkpoints (RUN_ID=$RUN_ID)"

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
    --report_to tensorboard

TRAIN_EXIT_CODE=$?

echo "Training exit code: $TRAIN_EXIT_CODE"

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


# ============================================================================
# ================== Auto Evaluation after Training ==========================
# ============================================================================
# Multi-node parallel evaluation:
#   - Only node 0 has complete checkpoints and uploads to OBS
#   - All nodes synchronize: wait for node 0 to finish uploading via OBS flag
#   - Each node discovers checkpoint list from OBS
#   - Checkpoints distributed via round-robin (index % NUM_NODES)
#   - Each node downloads only its assigned checkpoints from OBS
#   - Each node runs full RoboTwin eval setup independently
#   - Eval logs uploaded to OBS per checkpoint per node
# ============================================================================

if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
    echo ""
    echo "================================================================="
    echo "======= Training succeeded, starting auto evaluation ======="
    echo "================================================================="

    NODE_ID=${VC_TASK_INDEX:-0}
    NUM_NODES=${WORLD_SIZE:-1}
    CKPT_OBS_BASE="$UPLOAD_OBS_BASE/checkpoints"

    # ---------------------------------------------------------------
    # 1. Synchronize: wait for node 0 to finish uploading to OBS
    #    Node 0: upload is already done above, proceed immediately
    #    Other nodes: poll OBS until checkpoints appear
    #    Uses python time.sleep instead of bash sleep (DI platform restriction)
    # ---------------------------------------------------------------
    echo "Node $NODE_ID: Waiting for checkpoints to be available on OBS..."
    conda activate qwen3

    ALL_CKPT_NAMES=()
    OBS_CKPT_LIST=$(python -c "
import moxing as mox
import time
import sys

ckpt_obs_base = '$CKPT_OBS_BASE'

while True:
    try:
        top_dirs = mox.file.list_directory(ckpt_obs_base)
        results = []
        for td in top_dirs:
            run_path = ckpt_obs_base + '/' + td
            if mox.file.is_directory(run_path):
                sub_dirs = mox.file.list_directory(run_path)
                for sd in sorted(sub_dirs):
                    if sd.startswith('checkpoint_step_'):
                        pt_path = run_path + '/' + sd + '/pytorch_model'
                        if mox.file.is_directory(pt_path):
                            results.append(td + '/' + sd)
        if results:
            for r in results:
                print(r)
            sys.exit(0)
    except Exception as e:
        pass
    print('Node waiting: checkpoints not yet on OBS, retrying in 30s...', file=sys.stderr)
    time.sleep(30)
" 2>&1)

    if [ -n "$OBS_CKPT_LIST" ]; then
        while IFS= read -r line; do
            # Skip stderr log lines
            case "$line" in Node\ waiting*) continue ;; esac
            ALL_CKPT_NAMES+=("$line")
        done <<< "$OBS_CKPT_LIST"
    fi

    NUM_CKPTS=${#ALL_CKPT_NAMES[@]}
    echo "Total checkpoints found on OBS: $NUM_CKPTS"
    for i in $(seq 0 $((NUM_CKPTS - 1))); do
        echo "  Checkpoint[$i]: ${ALL_CKPT_NAMES[$i]}"
    done

    if [ "$NUM_CKPTS" -eq 0 ]; then
        echo "ERROR: No checkpoints found on OBS, skip evaluation"
        exit $TRAIN_EXIT_CODE
    fi

    # ---------------------------------------------------------------
    # 2. Distribute checkpoints to nodes via round-robin
    #    Node 0 gets index 0, 4, 8, ...  (for 4 nodes)
    #    Node 1 gets index 1, 5, 9, ...
    #    Node 2 gets index 2, 6, 10, ...
    #    Node 3 gets index 3, 7, 11, ...
    # ---------------------------------------------------------------
    MY_CKPT_NAMES=()
    for i in $(seq 0 $((NUM_CKPTS - 1))); do
        if [ $((i % NUM_NODES)) -eq "$NODE_ID" ]; then
            MY_CKPT_NAMES+=("${ALL_CKPT_NAMES[$i]}")
        fi
    done

    echo ""
    echo "=== Node $NODE_ID distribution (round-robin mod $NUM_NODES) ==="
    echo "Assigned ${#MY_CKPT_NAMES[@]} / $NUM_CKPTS checkpoints:"
    for ckpt_name in "${MY_CKPT_NAMES[@]}"; do
        echo "  -> $ckpt_name"
    done
    echo "========================================================"

    if [ ${#MY_CKPT_NAMES[@]} -eq 0 ]; then
        echo "Node $NODE_ID: No checkpoints assigned, nothing to evaluate"
        exit $TRAIN_EXIT_CODE
    fi

    # ---------------------------------------------------------------
    # 3. Setup RoboTwin evaluation environment (one-time, all nodes)
    # ---------------------------------------------------------------
    echo ""
    echo "------- Setting up RoboTwin evaluation environment -------"

    # Install Xvfb for offscreen rendering
    sudo apt update
    sudo apt install xvfb -y
    Xvfb :99 -screen 0 1024x768x24 &
    export DISPLAY=:99
    sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

    # Download RoboTwin code
    echo "------- Download RoboTwin code -------"
    RoboTwinCodePath="obs://yw-2030-gy/external/wwx1513998/RoboTwin/"
    rm -rf /cache/wwx1513998/RoboTwin
    mkdir -p /cache/wwx1513998/RoboTwin
    conda activate qwen3
    python -c "import moxing as mox; mox.file.copy_parallel('$RoboTwinCodePath', '/cache/wwx1513998/RoboTwin/')"
    cd /cache/wwx1513998/RoboTwin
    echo "------- RoboTwin code downloaded -------"

    # Setup RoboTwin conda environment
    echo "------- Setup RoboTwin conda env -------"
    envs_name_eval=RoboTwin
    conda activate qwen3
    if [ ! -d "/root/miniconda3/envs/$envs_name_eval" ]; then
        python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-2030-gy/external/wwx1484778/envs/$envs_name_eval.tar.gz','/cache/envs/$envs_name_eval.tar.gz')"
        mkdir -p /root/miniconda3/envs/$envs_name_eval
        tar -xf /cache/envs/$envs_name_eval.tar.gz -C /root/miniconda3/envs
    fi
    conda activate $envs_name_eval

    echo "/cache/wwx1513998/RoboTwin/envs/curobo/src" > /root/miniconda3/envs/RoboTwin/lib/python3.10/site-packages/__editable__.nvidia_curobo-0.0.0.pth
    python script/update_embodiment_config_path.py
    echo "------- RoboTwin env setup done -------"

    # Quick render test
    echo "------- Render Test -------"
    bash rendernew.sh
    python script/test_render.py
    echo "------- Render Test Done -------"

    # Ensure pretrained weights are available
    WAN_LOCAL="/cache/wx1513998/motus_weights/Wan2.2-TI2V-5B"
    VLM_LOCAL="/cache/wx1513998/motus_weights/Qwen3-VL-2B-Instruct"

    if [ ! -d "$WAN_LOCAL" ]; then
        echo "Downloading WAN weights..."
        conda activate qwen3
        mkdir -p "$WAN_LOCAL"
        python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-2030-gy/external/wwx1484778/Wan2.2-TI2V-5B', '$WAN_LOCAL')"
    fi
    if [ ! -d "$VLM_LOCAL" ]; then
        echo "Downloading VLM weights..."
        conda activate qwen3
        mkdir -p "$VLM_LOCAL"
        python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-2030-gy/external/wwx1484778/Qwen3-VL-2B-Instruct', '$VLM_LOCAL')"
    fi

    conda activate $envs_name_eval
    echo "------- RoboTwin eval environment ready -------"

    # ---------------------------------------------------------------
    # 4. Evaluate each assigned checkpoint sequentially
    #    For each checkpoint: download from OBS -> evaluate -> upload logs
    # ---------------------------------------------------------------
    echo ""
    echo "========================================================="
    echo "Node $NODE_ID: Starting evaluation of ${#MY_CKPT_NAMES[@]} checkpoints"
    echo "========================================================="

    for CKPT_REL_NAME in "${MY_CKPT_NAMES[@]}"; do
        # CKPT_REL_NAME like: motus_wan_vlm_robotwin_bs10_lr5e-05/checkpoint_step_40000
        STEP_NAME=$(basename "$CKPT_REL_NAME")  # e.g. checkpoint_step_40000

        echo ""
        echo "---------------------------------------------------------"
        echo "Node $NODE_ID: Evaluating $STEP_NAME"
        echo "OBS path: $CKPT_OBS_BASE/$CKPT_REL_NAME/pytorch_model"
        echo "---------------------------------------------------------"

        # Download this checkpoint from OBS to local
        CKPT_OBS_PATH="$CKPT_OBS_BASE/$CKPT_REL_NAME/pytorch_model"
        CKPT_LOCAL_PATH="/cache/wx1513998/eval_checkpoints/$STEP_NAME/pytorch_model"

        echo "Downloading checkpoint from OBS..."
        conda activate qwen3
        rm -rf "$CKPT_LOCAL_PATH"
        mkdir -p "$CKPT_LOCAL_PATH"
        python -c "import moxing as mox; mox.file.copy_parallel('$CKPT_OBS_PATH', '$CKPT_LOCAL_PATH')"
        echo "Checkpoint downloaded to $CKPT_LOCAL_PATH"

        # Verify download
        if [ ! -d "$CKPT_LOCAL_PATH" ]; then
            echo "ERROR: Failed to download checkpoint for $STEP_NAME, skipping..."
            continue
        fi

        # Write paths_config.yml for this checkpoint
        CONFIG_DIR="/cache/wwx1513998/RoboTwin/policy/MotusWanVlmDirectMask"
        cat > "$CONFIG_DIR/paths_config.yml" << EOF
robotwin_root: "/cache/wwx1513998/RoboTwin"
conda_env: "RoboTwin"
checkpoint_path: "$CKPT_LOCAL_PATH"
wan_path: "$WAN_LOCAL"
vlm_path: "$VLM_LOCAL"
gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
task_config: "demo_randomized"
seed: 42
tasks_file: "tasks_all_new.txt"
EOF

        echo "paths_config.yml written for $STEP_NAME"

        # Make sure Xvfb is running
        if ! pgrep -x Xvfb > /dev/null; then
            Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
            export DISPLAY=:99
            echo "Xvfb restarted"
        fi

        # Run evaluation
        cd /cache/wwx1513998/RoboTwin
        conda activate $envs_name_eval
        bash policy/MotusWanVlmDirectMask/auto_eval_new.sh
        EVAL_EXIT_CODE=$?

        echo "Evaluation of $STEP_NAME exit code: $EVAL_EXIT_CODE"

        # -----------------------------------------------------------
        # 5. Upload evaluation logs for this checkpoint to OBS
        # -----------------------------------------------------------
        echo "------- Uploading eval logs for $STEP_NAME -------"

        LOG_DIR=$(ls -dt /cache/wwx1484778/RoboTwin/MotusWanVlmDirectMask/logs_wan_vlm_auto_* 2>/dev/null | head -1)

        if [ -n "$LOG_DIR" ] && [ -d "$LOG_DIR" ]; then
            LOG_BASENAME=$(basename "$LOG_DIR")
            EVAL_OBS_PATH="$UPLOAD_OBS_BASE/eval_logs/node${NODE_ID}/${STEP_NAME}/$LOG_BASENAME"
            echo "Uploading $LOG_DIR -> $EVAL_OBS_PATH"

            conda activate qwen3
            python -c "import moxing as mox; mox.file.copy_parallel('$LOG_DIR', '$EVAL_OBS_PATH')"
            if [ $? -eq 0 ]; then
                echo "Upload logs done: $EVAL_OBS_PATH"
            else
                echo "Moxing upload failed, trying OBS SDK..."
                /root/miniconda3/envs/motus/bin/python -c "
from obs import ObsClient; import os
obsClient = ObsClient(access_key_id='HPUACJ8EWHNONWBP1PGQ', secret_access_key='rx3ITEWElWS8dZnPHmFMmMiiGkQ2pk5FguQ0d1QN', server='obs.cn-southwest-2.myhuaweicloud.com')
bucket='yw-2030-gy'; local_dir='$LOG_DIR'; obs_prefix='external/wwx1484778/cache-DI/$(basename $UPLOAD_OBS_BASE)/eval_logs/node${NODE_ID}/${STEP_NAME}/$LOG_BASENAME'
count=0
for root, dirs, files in os.walk(local_dir):
    for f in files:
        lp = os.path.join(root, f)
        rp = os.path.relpath(lp, local_dir)
        obsClient.uploadFile(bucket, obs_prefix+'/'+rp, lp)
        count+=1
obsClient.close()
print(f'Upload done (obs SDK): {count} files -> obs://{bucket}/{obs_prefix}')
"
            fi

            # Also upload eval_result if it exists
            EVAL_RESULT_DIR="/cache/wwx1484778/RoboTwin/eval_result"
            if [ -d "$EVAL_RESULT_DIR" ]; then
                RESULT_OBS_PATH="$UPLOAD_OBS_BASE/eval_results/node${NODE_ID}/${STEP_NAME}"
                echo "Uploading eval results -> $RESULT_OBS_PATH"
                python -c "import moxing as mox; mox.file.copy_parallel('$EVAL_RESULT_DIR', '$RESULT_OBS_PATH')" 2>/dev/null
            fi
        else
            echo "WARNING: No log directory found for $STEP_NAME"
        fi

        # Free disk: remove local checkpoint copy after evaluation
        echo "Cleaning up local checkpoint copy for $STEP_NAME..."
        rm -rf "/cache/wx1513998/eval_checkpoints/$STEP_NAME"

        conda activate $envs_name_eval
        echo "------- $STEP_NAME evaluation and upload done -------"
    done

    # Cleanup Xvfb
    pkill Xvfb 2>/dev/null || echo "Xvfb already stopped"

    echo ""
    echo "========================================================="
    echo "Node $NODE_ID: All ${#MY_CKPT_NAMES[@]} checkpoint evaluations completed!"
    echo "========================================================="
else
    echo "Training failed (exit code: $TRAIN_EXIT_CODE), skip evaluation"
fi

exit $TRAIN_EXIT_CODE
