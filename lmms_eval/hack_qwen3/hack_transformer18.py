# This file is from https://github.com/haotian-liu/LLaVA/

from typing import Optional, Tuple
import warnings

import torch

import transformers
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import *
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import _flash_attention_forward, apply_multimodal_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache

from typing import List, Optional, Tuple, Union
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

from typing import Optional, Union
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import TransformersKwargs
from typing_extensions import Unpack
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb,eager_attention_forward

from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS

from typing import Optional, Union
import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
)
from transformers.utils import is_torchdynamo_compiling

def forwardllm_new(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,

    # ✅ 新增：给你 drop 用
    image_token_positions: Optional[torch.LongTensor] = None,
    image_token_lengths: Optional[torch.LongTensor] = None,

    **kwargs,
) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:

    # ✅ 关键：避免 generate() 额外塞 return_dict=True 导致重复传参
    kwargs.pop("return_dict", None)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,

        # ✅ 关键：把 drop 需要的两个张量一路传下去
        image_token_positions=image_token_positions,
        image_token_lengths=image_token_lengths,

        return_dict=True,
        **kwargs,
    )

    hidden_states = outputs[0]  # or outputs.last_hidden_state

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.text_config.vocab_size,
        )

    # ✅ 保持和原版一致：只返回它原来就返回的字段
    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=getattr(outputs, "past_key_values", None),
        rope_deltas=getattr(outputs, "rope_deltas", None),
    )

import torch
from typing import Optional, Union
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast

def forwardllm2_new_qwen3(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,

    # ✅ 新增：给你 token dropping 用
    image_token_positions: Optional[torch.LongTensor] = None,
    image_token_lengths: Optional[torch.LongTensor] = None,

    **kwargs,
) -> Union[tuple, Qwen3VLModelOutputWithPast]:

    # ✅ 保险：避免 generate()/上游重复塞 return_dict
    kwargs.pop("return_dict", None)

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if position_ids is None:
        attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = (not is_torchdynamo_compiling()) and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)

    # ✅ 关键：把 image_token_* 透传到 language_model
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,

        image_token_positions=image_token_positions,
        image_token_lengths=image_token_lengths,

        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )


import torch
from typing import Optional
from transformers.cache_utils import Cache

def forwarddec_new_qwen3(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,

    # ✅ 新增：显式接住
    image_token_positions: Optional[torch.LongTensor] = None,
    image_token_lengths: Optional[torch.LongTensor] = None,

    **kwargs,
) -> torch.Tensor:
    # ✅ 保险：避免 kwargs 里混入重复字段（按需）
    # kwargs.pop("return_dict", None)  # DecoderLayer一般不会遇到，但留着也无害

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention（把 image_token_* 往下传）
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,

        image_token_positions=image_token_positions,
        image_token_lengths=image_token_lengths,

        **kwargs,
    )

    # ✅ 如果你在 attention 里做了 token dropping，可能导致 seq_len 变化
    # 需要把 residual 同样裁掉再做残差相加（对齐你 2.5 的逻辑）
    if residual.shape != hidden_states.shape:
        # 这里严格复刻你 2.5 的写法：只用第 0 个 span
        # image_token_positions / lengths 形状一般是 [B, Nimg]，取第一个图 span
        if image_token_positions is None or image_token_lengths is None:
            raise ValueError("Seq length changed but image_token_positions/lengths is None")

        # 支持 tensor/list 两种
        if isinstance(image_token_positions, torch.Tensor):
            pos0 = int(image_token_positions[0, 0].item()) if image_token_positions.ndim == 2 else int(image_token_positions[0].item())
        else:
            pos0 = int(image_token_positions[0])

        if isinstance(image_token_lengths, torch.Tensor):
            len0 = int(image_token_lengths[0, 0].item()) if image_token_lengths.ndim == 2 else int(image_token_lengths[0].item())
        else:
            len0 = int(image_token_lengths[0])

        # residual: [B, L, D]
        # 你原逻辑是：保留 [:pos+1] 以及 [pos+len-1:]（等价于“中间删掉 len-2 个 token”）
        residual = torch.cat(
            [residual[:, :pos0 + 1], residual[:, pos0 + len0 - 1:]],
            dim=1,
        )

    hidden_states = residual + hidden_states

    # MLP block
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


from typing import Optional, Callable
import torch
from transformers.cache_utils import Cache

def make_qwen3vl_attention_forward_with_drop(start_drop_layer: int):

    def forward_with_drop(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,

        # ===== 新增：vision drop 所需 =====
        image_token_positions: Optional[torch.LongTensor] = None,
        image_token_lengths: Optional[torch.LongTensor] = None,

        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        bsz, q_len = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        # ---- QKV ----
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = (
            self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )

        # ---- RoPE ----
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # ---- KV cache ----
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # =====================================================
        # =============== Vision Token Dropping ===============
        # =====================================================
        do_drop = (
            image_token_positions is not None
            and image_token_lengths is not None
            and image_token_positions.numel() > 0
            and image_token_lengths.numel() > 0
            and self.layer_idx >= start_drop_layer
            and q_len > 1            # decode 单步时不动
        )

        if do_drop:
            # 与 Qwen2.5 完全一致：只取第一个 image span
            pos = int(image_token_positions[0, 0])
            length = int(image_token_lengths[0, 0])

            left = max(pos + 1, 0)
            right = min(pos + length - 1, key_states.shape[-2])

            # key/value: [B, heads, S, d]
            key_states = torch.cat(
                [key_states[:, :, :left, :], key_states[:, :, right:, :]],
                dim=2,
            )
            value_states = torch.cat(
                [value_states[:, :, :left, :], value_states[:, :, right:, :]],
                dim=2,
            )

            # ---- attention_mask 对齐裁剪 ----
            if attention_mask is not None:
                if attention_mask.dim() == 2:          # [B, S]
                    attention_mask = torch.cat(
                        [attention_mask[:, :left], attention_mask[:, right:]],
                        dim=1,
                    )
                elif attention_mask.dim() == 4:        # [B, 1, Q, K]
                    attention_mask = torch.cat(
                        [attention_mask[..., :left], attention_mask[..., right:]],
                        dim=-1,
                    )

        # ---- Attention ----
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward_with_drop



def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523349593"
        )
    # 用四个关键组件的forward方法替换原始实现
    # 替换Flash Attention层
    start_drop_layer = 36-18
    print("start_drop_layer",start_drop_layer)
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = make_qwen3vl_attention_forward_with_drop(start_drop_layer)
    # 替换整个模型
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = forwardllm_new
    
    # 替换模型主体
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.forward = forwardllm2_new_qwen3

    # 替换Decoder层
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextDecoderLayer.forward = forwarddec_new_qwen3
    # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = forward
    # transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = forward_dec
    # # transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = forward_LM
    # transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = forward

# def replace_llama_attn_with_flash_attnv2():
#     cuda_major, cuda_minor = torch.cuda.get_device_capability()
#     if cuda_major < 8:
#         warnings.warn(
#             "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
#             "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523349593"
#         )
#     transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = forward
#     transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = forwardllm
#     transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = forwardllm2
#     transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLDecoderLayer.forward = forwarddec

# scp -r -P 22112 zhuyuchen530@10.15.89.41:/public/home/zhuyuchen530/LLM_Rep/EAGLE/train_mem.py 2022233234/EAGLE/