CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=28102 train/train_wan_vlm_mask.py --deepspeed configs/zero1.json --config configs/robotwin_wan_vlm_mask_stage2.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=28102 train/train_wan_vlm_mask.py --deepspeed configs/zero1.json --config configs/robotwin_wan_vlm_mask_stage2_15w.yaml

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=28102 train/train.py --deepspeed configs/zero1.json --config configs/robotwin_stage2.yaml