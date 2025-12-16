# export HF_HOME="~/.cache/huggingface"
# pip3 install transformers==4.57.1 (Qwen3VL models)
# pip3 install ".[qwen]" (for Qwen's dependencies)

# Exmaple with Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 
# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model qwen3_vl \
#     --model_args=pretrained=/datas/huggingface/Qwen3-VL-8B-Instruct/,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
#     --tasks "mmstar" \
#     --batch_size 1
# /datas/huggingface/Qwen3-VL-4B-Instruct/

#   - mmbench_en_dev
#   - mmbench_en_test  mmstar
#   - mmbench_cn_dev  gqa  mmmu_val  mme

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
DROPS=({1..36})
for DROP in "${DROPS[@]}"; do
    echo "================ Running drop=${DROP} ================"

    accelerate launch \
        --num_processes=8 \
        --main_process_port=12988 \
        -m lmms_eval \
        --model qwen3_vl \
        --model_args pretrained=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct,attn_implementation=sdpa,interleave_visuals=False \
        --tasks gqa \
        --batch_size 1 \
        --drop ${DROP}

    echo "================ Finished drop=${DROP} ================"
done

echo "✅ All drops finished"


# export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1
# DROP=1
# echo "================ Running drop=${DROP} ================"
# accelerate launch \
#     --num_processes=8 \
#     --main_process_port=12588 \
#     -m lmms_eval \
#     --model qwen3_vl \
#     --model_args pretrained=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct,attn_implementation=sdpa,interleave_visuals=False \
#     --tasks mmstar \
#     --batch_size 1 \
#     --drop ${DROP} 

# accelerate launch \
#     --num_processes=8 \
#     --main_process_port=12347 \
#     -m lmms_eval \
#     --model qwen3_vl \
#     --model_args pretrained=/datas/huggingface/Qwen3-VL-4B-Instruct/,attn_implementation=flash_attention_2,interleave_visuals=False \
#     --tasks mmmu_val \
#     --batch_size 1 \
#     --drop ${DROP} 

