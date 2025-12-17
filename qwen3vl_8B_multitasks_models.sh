#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

MODEL_DIR="/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di93pux/multimodal/merged_models/merge_8B_only_qkvo/19"

TASKS=(mmbench_en_dev  mmstar )
DROPS=(0)

PORT=12988

# 收集 MODEL_DIR 下所有“子目录”（当作模型目录）
MODELS=()
for d in "${MODEL_DIR}"/*; do
  if [ -d "$d" ]; then
    MODELS+=("$d")
  fi
done

# 开始遍历：模型 × 任务 × drop
for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"
  echo "================ MODEL: ${MODEL_NAME} (${MODEL_PATH}) ================"

  for TASK in "${TASKS[@]}"; do
    for DROP in "${DROPS[@]}"; do
      echo "================ Running model=${MODEL_NAME} task=${TASK} drop=${DROP} ================"

      accelerate launch \
        --num_processes=1 \
        --main_process_port="${PORT}" \
        -m lmms_eval \
        --model qwen3_vl \
        --model_args pretrained="${MODEL_PATH}",attn_implementation=sdpa,interleave_visuals=False \
        --tasks "${TASK}" \
        --batch_size 1 \
        --drop "${DROP}"

      echo "================ Finished model=${MODEL_NAME} task=${TASK} drop=${DROP} ================"
    done
  done
done

echo "✅ All models, tasks, and drops finished"
