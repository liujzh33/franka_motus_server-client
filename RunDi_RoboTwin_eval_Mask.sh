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
RoboTwinCodePath="obs://yw-2030-gy/external/wx1469573/code2/RoboTwin/"
rm -rf /cache/wx1469573/RoboTwin
mkdir -p /cache/wx1469573/RoboTwin
python -c "import moxing as mox; mox.file.copy_parallel('$RoboTwinCodePath', '/cache/wx1469573/RoboTwin/')"
cd /cache/wx1469573/RoboTwin
echo "--------------- copy code done ---------------"



echo "------------------- set envs -------------------"
envs_name=RoboTwin
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-2030-gy/external/wwx1484778/envs/$envs_name.tar.gz','/cache/envs/$envs_name.tar.gz')"
mkdir -p /root/miniconda3/envs/$envs_name
tar -xf /cache/envs/$envs_name.tar.gz -C /root/miniconda3/envs
conda activate $envs_name

echo "/cache/wx1469573/RoboTwin/envs/curobo/src" > /root/miniconda3/envs/RoboTwin/lib/python3.10/site-packages/__editable__.nvidia_curobo-0.0.0.pth
python script/update_embodiment_config_path.py
echo "----------------- set envs done -----------------"



echo "------------------ Render Test ------------------"
bash rendernew.sh
python script/test_render.py
echo "---------------- Render Test Done ----------------"


echo "----------------- Download ckpts -----------------"
CKPT_pt="obs://yw-2030-gy/external/wx1469573/cache-DI/20260601_030133/checkpoints/best_action_l2_92000/node_4/pytorch_model/mp_rank_00_model_states.pt"
CKPT_cf="obs://yw-2030-gy/external/wx1469573/cache-DI/20260601_030133/checkpoints/best_action_l2_92000/node_4/config.json"
WanPath="obs://yw-2030-gy/external/wx1469573/code2/Motus/pretrained_models/Wan2.2-TI2V-5B/"
VLMPath="obs://yw-2030-gy/external/wx1469573/code2/Motus/pretrained_models/Qwen3-VL-2B-Instruct/"
conda activate qwen3
rm -rf /cache/wx1469573/pretrained_models
mkdir -p /cache/wx1469573/pretrained_models
python -c "import moxing as mox; mox.file.copy_parallel('$WanPath', '/cache/wx1469573/pretrained_models/Wan2.2-TI2V-5B')"
python -c "import moxing as mox; mox.file.copy_parallel('$VLMPath', '/cache/wx1469573/pretrained_models/Qwen3-VL-2B-Instruct')"
python -c "import moxing as mox; mox.file.copy('$CKPT_pt', '/cache/wx1469573/pretrained_models/ckpt/mp_rank_00_model_states.pt')"
python -c "import moxing as mox; mox.file.copy('$CKPT_cf', '/cache/wx1469573/pretrained_models/ckpt/config.json')"
conda activate $envs_name
echo "--------------- Download ckpts done ---------------"


echo "-------------------- Test Start --------------------"
bash policy/MotusWanVlmDirectMask/auto_eval_new.sh
echo "--------------------- Test End ---------------------"

pkill Xvfb || echo "Xvfb already stopped"
python -c "print('Well Done.')"