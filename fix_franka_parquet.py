"""
修复 v3.0 parquet: 移除视频列（string 类型），修正 state/actions/timestamp 数据类型为 float32。
视频路径已在 episodes parquet 中，data parquet 不需要视频列。
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import json

DATASETS = [
    "/home/ma-user/work/wx1513998/franka_data/place_objects_into_the_box_rc_0524",
    "/home/ma-user/work/wx1513998/franka_data/stack_bowls_rc",
]

VIDEO_COLS = ["right_image", "global_image", "wrist_image"]
# 需要转换为 float32 的列 (parquet 中是 double, info.json 中是 float32)
FLOAT32_COLS = ["state", "actions"]
# timestamp 在 info.json 中是 float32 shape [1]，parquet 中是 double
TIMESTAMP_COL = "timestamp"

for dataset_root in DATASETS:
    print(f"\n===== Fixing {dataset_root} =====")
    info_path = Path(dataset_root) / "meta" / "info.json"
    info = json.load(open(info_path))

    # 找到所有 data parquet 文件
    data_dir = Path(dataset_root) / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    print(f"  Found {len(parquet_files)} parquet files")

    for pf in parquet_files:
        print(f"  Processing {pf.name}...")
        table = pq.read_table(pf)
        original_cols = table.column_names

        # 移除视频列
        cols_to_keep = [c for c in original_cols if c not in VIDEO_COLS]
        table = table.select(cols_to_keep)

        # 修正数据类型: state, actions (list<double> -> list<float>)
        for col in FLOAT32_COLS:
            if col in table.column_names:
                arr = table.column(col)
                # 转换 list<double> -> list<float>
                flat = arr.combine_chunks().flatten()
                flat_f32 = flat.cast(pa.float32())
                # 重建 list array
                offsets = pa.array(range(len(flat_f32) + 1), type=pa.int32())
                # Actually, use ListArray.from_arrays
                list_arr = pa.ListArray.from_arrays(
                    offsets,
                    flat_f32
                )
                # Wait, that's wrong for variable-length lists. Let me use a different approach.
                # Just cast the inner type
                pass  # Will handle below

        # 简单方法: 用 pandas 转换
        df = table.to_pandas()
        for col in FLOAT32_COLS:
            if col in df.columns:
                # 转换为 float32 list
                df[col] = df[col].apply(lambda x: [float(v) for v in x] if x is not None else x)
        if TIMESTAMP_COL in df.columns:
            df[TIMESTAMP_COL] = df[TIMESTAMP_COL].astype('float32')

        # 重新构建 table with correct schema
        schema_fields = []
        for col in df.columns:
            if col in FLOAT32_COLS:
                schema_fields.append(pa.field(col, pa.list_(pa.float32())))
            elif col == TIMESTAMP_COL:
                schema_fields.append(pa.field(col, pa.float32()))
            elif col in ['frame_index', 'episode_index', 'index', 'task_index']:
                schema_fields.append(pa.field(col, pa.int64()))
            else:
                # 保留原类型
                schema_fields.append(pa.field(col, table.schema.field(col).type))

        new_schema = pa.schema(schema_fields)
        new_table = pa.Table.from_pandas(df, schema=new_schema)

        # 保存
        pq.write_table(new_table, pf, compression='snappy')
        print(f"    Fixed: {original_cols} -> {new_table.column_names}")
        print(f"    Schema: {new_table.schema}")

    # 更新 info.json: 移除视频 features（视频路径在 episodes parquet 中）
    # 实际上保留视频 features，因为 LeRobot 需要知道视频 key
    # 但需要标记它们为 video 类型（已经是了）
    print(f"  Done fixing {dataset_root}")

print("\nAll parquet files fixed.")
