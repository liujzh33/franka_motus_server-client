启动 server：

  CUDA_VISIBLE_DEVICES=4 python /home/ma-user/work/wwx1484778/Our/Motus/inference/real_world/Motus/server_vlm_mask.py \
    --model_config /home/ma-user/work/wwx1484778/Our/Motus/configs/robotwin_wan_vlm_mask_dobot_c.yaml \
    --ckpt_dir /cache/wwx1484778/motus/checkpoints_wan_vlm_mask_dobot_c_0513/robotwin_wan_vlm_mask_dobot_c/motus_wan_vlm_dobot_bs8_lr5e-05/checkpoint_step_20000/pytorch_model \
    --wan_path /cache/wwx1484778/motus_weights/Wan2.2-TI2V-5B \
    --vlm_path /cache/wwx1484778/motus_weights/Qwen3-VL-2B-Instruct \
    --default_instruction_file /home/ma-user/work/wwx1484778/Our/Motus/inference/real_world/Motus/cook_instruction.txt \
    --default_t5_embeddings_path /cache/wwx1484778/Dobot/dobot_cook_vegetable_full/dobot_cook_vegetable_full/episode_000000/umt5_wan/trajectory.pt \
    --dataset_name dobot_cook_vegetable \
    --port 6789

  启动 client 测试：

  # 健康检查
  python /home/ma-user/work/wwx1484778/Our/Motus/inference/real_world/Motus/client.py \
    --url http://localhost:6789 \
    --test connectivity

  # 推理测试
  python /home/ma-user/work/wwx1484778/Our/Motus/inference/real_world/Motus/client.py \
    --url http://localhost:6789 \
    --test inference \
    --image /cache/wwx1484778/Dobot/dobot_first_frame.jpg
    --state_csv "0,0,0,0,0,0,0,0,0,0,0,0,0,0"

  # mock 测试
  python /home/ma-user/work/wwx1484778/Our/Motus/inference/real_world/Motus/client.py \
    --url http://localhost:6789 \
    --test mock

说明：
  1. 如果 server 启动时已经通过 --default_instruction_file 和 --default_t5_embeddings_path 预加载了 instruction 和 embedding，
     client 推理时可以不再传 --instruction 和 --t5_embeddings_path。
  2. 当前接口优先使用 proprio_data；client 传 --state_csv/--state_json 时会自动转成 proprio_data 发给 server。

