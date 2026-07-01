CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=28200 train/train_frozenwan.py --config configs/robotwin_frozenwan.yaml
