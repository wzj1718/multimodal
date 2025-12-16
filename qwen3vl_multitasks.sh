export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1


#   -   - pope_adv
#   - pope_pop
#   - pope_random
#   -   
#   -   gqa    seedbench_2_plus  realworldqa

TASKS=(seedbench_2_plus realworldqa pope_random pope_pop pope_adv gqa)
DROPS=({1..36})

for TASK in "${TASKS[@]}"; do
  for DROP in "${DROPS[@]}"; do
    echo "================ Running task=${TASK} drop=${DROP} ================"

    accelerate launch \
      --num_processes=8 \
      --main_process_port=12988 \
      -m lmms_eval \
      --model qwen3_vl \
      --model_args pretrained=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct,attn_implementation=sdpa,interleave_visuals=False \
      --tasks ${TASK} \
      --batch_size 1 \
      --drop ${DROP}

    echo "================ Finished task=${TASK} drop=${DROP} ================"
  done
done

echo "✅ All tasks and drops finished"

/usr/bin/shutdown
