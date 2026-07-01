CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=28200 train/train_frozenwan.py --config configs/robotwin_frozenwan.yaml
