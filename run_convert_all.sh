#!/bin/bash
set -e

SOURCE_ROOT="/root/.cache/RoboTwin_processed_pi0"
TARGET_ROOT="/root/.cache/RoboTwin_processed_motus"
SCRIPT_DIR="/home/ma-user/work/wx1513998/Motus"

TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle
    shake_bottle_horizontally
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)

TOTAL=${#TASKS[@]}

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    NUM=$((i + 1))
    echo "============================================"
    echo "[$NUM/$TOTAL] Processing task: $TASK"
    echo "============================================"

    echo "[Step 1] Converting $TASK (no T5)..."
    cd "$SCRIPT_DIR" && python convert_robotwin_pi0_to_motus.py \
        --source_root "$SOURCE_ROOT" \
        --target_root "$TARGET_ROOT" \
        --tasks "$TASK" \
        --max_workers 4 \
        --no_t5

    echo "[Step 2] T5 encoding $TASK on GPU 0,1..."
    cd "$SCRIPT_DIR" && python t5_encode_multigpu.py \
        --target_root "$TARGET_ROOT" \
        --gpus 0,1 \
        --tasks "$TASK" \
        --batch_size 2

    echo "[$NUM/$TOTAL] Task $TASK done!"
    echo ""
done

echo "============================================"
echo "All $TOTAL tasks completed!"
echo "============================================"
