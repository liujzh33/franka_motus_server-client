CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=${NNODES:-1} \
    --nproc_per_node=8 \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-29100} \
    train/train_wan_vlm_mask_initvlm.py \
    --deepspeed configs/zero1.json \
    --config configs/robotwin_wan_vlm_mask_stage1_initvlm.yaml
