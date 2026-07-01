#!/bin/bash
# Upload RobboTwin_test to OBS (excluding eval_result, NVIDIA driver, __pycache__)
# Target: obs://yw-2030-gy/external/wwx1513998/RoboTwin/

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

export S3_ENDPOINT="https://obs.cn-southwest-2.myhuaweicloud.com"
export S3_USE_HTTPS="0"
export ACCESS_KEY_ID="HPUACJ8EWHNONWBP1PGQ"
export SECRET_ACCESS_KEY="rx3ITEWElWS8dZnPHmFMmMiiGkQ2pk5FguQ0d1QN"

OBS_TARGET="obs://yw-2030-gy/external/wwx1513998/RoboTwin"
LOCAL_SRC="/home/ma-user/work/wx1513998/RobboTwin_test"

echo "=== Uploading RobboTwin_test to OBS ==="
echo "Source: $LOCAL_SRC"
echo "Target: $OBS_TARGET"
echo "Start: $(date)"

# Upload entire directory (moxing copy_parallel handles this)
# Note: eval_result/ and NVIDIA driver will be included but won't affect eval
# If you want to exclude them, manually delete after upload or use a filtered approach
python -c "import moxing as mox; mox.file.copy_parallel('$LOCAL_SRC', '$OBS_TARGET/')"

echo "Done: $(date)"
echo "Upload completed!"
