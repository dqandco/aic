#!/usr/bin/env bash
#
# Fine-tune the ACT policy on expert demonstration data.
#
# Prerequisites:
#   1. Collect expert demonstrations (run with ground_truth:=true):
#
#      pixi run ros2 run aic_model aic_model --ros-args \
#        -p use_sim_time:=true \
#        -p policy:=my_policy.ExpertCollector
#
#      Or record with LeRobot for a dataset compatible with lerobot-train:
#
#      pixi run lerobot-record \
#        --robot.type=aic_controller --robot.id=aic \
#        --teleop.type=aic_keyboard_ee --teleop.id=aic \
#        --robot.teleop_target_mode=cartesian \
#        --robot.teleop_frame_id=base_link \
#        --dataset.repo_id=${HF_USER}/aic_expert_demos \
#        --dataset.single_task="insert cable into port" \
#        --dataset.push_to_hub=false \
#        --dataset.private=true \
#        --play_sounds=false \
#        --display_data=true
#
#   2. Run this script to fine-tune:
#
#      cd ~/ws_aic/src/aic
#      bash docker/my_policy/scripts/finetune_act.sh
#
# Environment variables (override defaults):
#   HF_USER         – HuggingFace username (required)
#   DATASET_REPO_ID – dataset repo (default: ${HF_USER}/aic_expert_demos)
#   PRETRAINED_REPO – pretrained model (default: grkw/aic_act_policy)
#   OUTPUT_DIR      – training output dir (default: outputs/train/act_finetuned)
#   NUM_STEPS       – training steps (default: 50000)
#   BATCH_SIZE      – batch size (default: 8)
#   LR              – learning rate (default: 1e-5, lower for fine-tuning)
#   DEVICE          – torch device (default: cuda)
#   WANDB           – enable wandb logging (default: false)

set -euo pipefail

: "${HF_USER:?Set HF_USER to your HuggingFace username}"
: "${DATASET_REPO_ID:=${HF_USER}/aic_expert_demos}"
: "${PRETRAINED_REPO:=grkw/aic_act_policy}"
: "${OUTPUT_DIR:=outputs/train/act_finetuned}"
: "${NUM_STEPS:=50000}"
: "${BATCH_SIZE:=8}"
: "${LR:=1e-5}"
: "${DEVICE:=cuda}"
: "${WANDB:=false}"

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo ~/ws_aic/src/aic)"

echo "============================================"
echo " ACT Fine-Tuning"
echo "============================================"
echo " Dataset:     ${DATASET_REPO_ID}"
echo " Pretrained:  ${PRETRAINED_REPO}"
echo " Output:      ${OUTPUT_DIR}"
echo " Steps:       ${NUM_STEPS}"
echo " Batch size:  ${BATCH_SIZE}"
echo " LR:          ${LR}"
echo " Device:      ${DEVICE}"
echo " W&B:         ${WANDB}"
echo "============================================"

pixi run lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --policy.type=act \
  --policy.pretrained_model_name_or_path="${PRETRAINED_REPO}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_finetune \
  --policy.device="${DEVICE}" \
  --training.num_steps="${NUM_STEPS}" \
  --training.batch_size="${BATCH_SIZE}" \
  --training.lr="${LR}" \
  --wandb.enable="${WANDB}" \
  --policy.repo_id="${HF_USER}/aic_act_finetuned"

echo ""
echo "Training complete. Model saved to: ${OUTPUT_DIR}"
echo "To use in HybridACTInsertion, update the repo_id in HybridACTInsertion.py"
echo "  from: grkw/aic_act_policy"
echo "  to:   ${HF_USER}/aic_act_finetuned"
