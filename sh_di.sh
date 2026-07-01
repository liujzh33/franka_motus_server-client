CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=28100 train/train_novlm.py --config configs/robotwin_novlm.yaml
