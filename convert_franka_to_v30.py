#!/usr/bin/env python3
"""
将 franka 数据从 LeRobot v2.1 转换为 v3.0 格式（离线，不连 HuggingFace Hub）。
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

DATASETS = [
    ("place_objects_into_the_box_rc_0524",
     "/home/ma-user/work/wx1513998/franka_data"),
    ("stack_bowls_rc",
     "/home/ma-user/work/wx1513998/franka_data"),
]

for repo_id, root in DATASETS:
    print(f"\n===== Converting {repo_id} =====")
    convert_dataset(
        repo_id=repo_id,
        root=root,
        push_to_hub=False,
        force_conversion=True,
    )
    print(f"===== Done {repo_id} =====")

print("\nAll datasets converted to v3.0.")
