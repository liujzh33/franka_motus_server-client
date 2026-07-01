#!/bin/bash
# Eval script: timestamp 20260605_163645, checkpoint best_action_l2
# Model: motus_wan_vlm_direct_mask.py
# Uses 8 GPUs, checkpoint_path -> pytorch_model/ dir (contains mp_rank_00_model_states.pt)

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

sudo apt update
sudo apt install xvfb -y
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

conda activate qwen3

echo "--------------Sh Start------------------"

USE_JUICEFS=0

export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

export S3_ENDPOINT="https://obs.cn-southwest-2.myhuaweicloud.com"
export S3_USE_HTTPS="0"
export ACCESS_KEY_ID="HPUACJ8EWHNONWBP1PGQ"
export SECRET_ACCESS_KEY="rx3ITEWElWS8dZnPHmFMmMiiGkQ2pk5FguQ0d1QN"


echo "----------------- copy RoboTwin code -----------------"
RoboTwinCodePath="obs://yw-2030-gy/external/wwx1513998/RoboTwin/"
rm -rf /cache/wwx1513998/RoboTwin
mkdir -p /cache/wwx1513998/RoboTwin
python -c "import moxing as mox; mox.file.copy_parallel('$RoboTwinCodePath', '/cache/wwx1513998/RoboTwin/')"
cd /cache/wwx1513998/RoboTwin
echo "--------------- copy RoboTwin code done ---------------"


echo "------------------- set envs -------------------"
envs_name=RoboTwin
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-2030-gy/external/wwx1484778/envs/$envs_name.tar.gz','/cache/envs/$envs_name.tar.gz')"
mkdir -p /root/miniconda3/envs/$envs_name
tar -xf /cache/envs/$envs_name.tar.gz -C /root/miniconda3/envs
conda activate $envs_name

echo "/cache/wwx1513998/RoboTwin/envs/curobo/src" > /root/miniconda3/envs/RoboTwin/lib/python3.10/site-packages/__editable__.nvidia_curobo-0.0.0.pth
python script/update_embodiment_config_path.py
echo "----------------- set envs done -----------------"


echo "------------------ Render Test ------------------"
bash rendernew.sh
python script/test_render.py
echo "---------------- Render Test Done ----------------"


echo "----------------- Download pretrained weights -----------------"
conda activate qwen3
rm -rf /cache/wwx1513998/pretrained_models
mkdir -p /cache/wwx1513998/pretrained_models/Wan2.2-TI2V-5B
mkdir -p /cache/wwx1513998/pretrained_models/Qwen3-VL-2B-Instruct

WanPath="obs://yw-2030-gy/external/wwx1484778/Wan2.2-TI2V-5B"
VLMPath="obs://yw-2030-gy/external/wwx1484778/Qwen3-VL-2B-Instruct"
python -c "import moxing as mox; mox.file.copy_parallel('$WanPath', '/cache/wwx1513998/pretrained_models/Wan2.2-TI2V-5B')"
python -c "import moxing as mox; mox.file.copy_parallel('$VLMPath', '/cache/wwx1513998/pretrained_models/Qwen3-VL-2B-Instruct')"

# Download checkpoint: 20260605_163645, best_action_l2
CKPT_OBS="obs://yw-2030-gy/external/wwx1484778/cache-DI/20260605_163645/checkpoints/robotwin_wan_vlm_mask_stage2/motus_wan_vlm_robotwin_bs10_lr5e-05/best_action_l2/pytorch_model"
CKPT_LOCAL="/cache/wwx1513998/pretrained_models/ckpt_best_action_l2_163645/pytorch_model"
mkdir -p "$CKPT_LOCAL"
python -c "import moxing as mox; mox.file.copy_parallel('$CKPT_OBS', '$CKPT_LOCAL')"

conda activate $envs_name
echo "--------------- Download ckpts done ---------------"


echo "------------------ Setup paths_config.yml ------------------"
cat > /cache/wwx1513998/RoboTwin/policy/MotusWanVlmDirectMask/paths_config.yml << EOF
robotwin_root: "/cache/wwx1513998/RoboTwin"
conda_env: "RoboTwin"
checkpoint_path: "$CKPT_LOCAL"
wan_path: "/cache/wwx1513998/pretrained_models/Wan2.2-TI2V-5B"
vlm_path: "/cache/wwx1513998/pretrained_models/Qwen3-VL-2B-Instruct"
gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
task_config: "demo_randomized"
seed: 42
tasks_file: "tasks_all_new.txt"
EOF
echo "--------------- paths_config.yml done ---------------"


echo "-------------------- Test Start --------------------"
cd /cache/wwx1513998/RoboTwin
bash policy/MotusWanVlmDirectMask/auto_eval_new.sh
echo "--------------------- Test End ---------------------"


echo "----------------- Upload logs to OBS -----------------"
LOG_DIR=$(ls -dt /cache/wwx1484778/RoboTwin/MotusWanVlmDirectMask/logs_wan_vlm_auto_* 2>/dev/null | head -1)
if [ -n "$LOG_DIR" ]; then
    LOG_BASENAME=$(basename "$LOG_DIR")
    OBS_LOG_PATH="obs://yw-2030-gy/external/wwx1513998/eval_logs/20260605_163645_best_action_l2/$LOG_BASENAME"
    echo "Uploading logs from $LOG_DIR to $OBS_LOG_PATH"
    conda activate qwen3
    python -c "import moxing as mox; mox.file.copy_parallel('$LOG_DIR', '$OBS_LOG_PATH')"
    echo "Logs uploaded to $OBS_LOG_PATH"
else
    echo "Warning: No log directory found to upload"
fi
echo "--------------- Upload logs done ---------------"

pkill Xvfb || echo "Xvfb already stopped"
python -c "print('Well Done. Eval 20260605_163645_best_action_l2 completed.')"
