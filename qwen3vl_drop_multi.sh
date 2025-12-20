#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# ================== 配置 ==================

MODEL_ROOT="/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di93pux/multimodal/merged_models/merge_8B_only_qkvo/mms"

TASKS=(mmstar)
DROPS=({9..15})

PORT=12988

# ================== 收集所有模型目录 ==================

MODELS=()
for d in "${MODEL_ROOT}"/*; do
  if [ -d "$d" ]; then
    MODELS+=("$d")
  fi
done

echo "✅ Found ${#MODELS[@]} models:"
for m in "${MODELS[@]}"; do
  echo "  - $(basename "$m")"
done

# ================== 开始跑 ==================

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"
  echo
  echo "================ MODEL: ${MODEL_NAME} ================="

  for TASK in "${TASKS[@]}"; do
    for DROP in "${DROPS[@]}"; do
      echo "---- Running model=${MODEL_NAME} task=${TASK} drop=${DROP} ----"

      accelerate launch \
        --num_processes=8 \
        --main_process_port=${PORT} \
        -m lmms_eval \
        --model qwen3_vl \
        --model_args pretrained=${MODEL_PATH},attn_implementation=sdpa,interleave_visuals=False \
        --tasks ${TASK} \
        --batch_size 1 \
        --drop ${DROP}

      echo "---- Finished model=${MODEL_NAME} task=${TASK} drop=${DROP} ----"
    done
  done
done

echo "✅ All models, tasks, and drops finished"

/usr/bin/shutdown
