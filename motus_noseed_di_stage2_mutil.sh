torchrun \
    --nnodes=${NNODES:-1} \
    --nproc_per_node=8 \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-28100} \
    train/train_noseed.py \
    --deepspeed configs/zero1.json \
    --config configs/robotwin_noseed_stage2.yaml
