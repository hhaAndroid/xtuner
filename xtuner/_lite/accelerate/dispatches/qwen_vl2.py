# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings
from typing import Optional, Tuple, List

import torch
from mmengine import MessageHub
from transformers.cache_utils import Cache
from transformers.models.qwen2_vl.modeling_qwen2_vl import (apply_multimodal_rotary_pos_emb, repeat_kv)
from transformers.modeling_outputs import BaseModelOutputWithPast
import torch.nn.functional as F

from ._attention import flash_attn_wo_mask, varlen_flash_attn

SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func
    _flash_supports_window_size = 'window_size' in list(
        inspect.signature(flash_attn_func).parameters)
    SUPPORT_FLASH2 = True
except ImportError:
    pass


def qwen2_vl_varlen_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # will become mandatory in v4.46
):
    is_training = self.training

    assert is_training == (past_key_value is None)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                'The cache structure has changed since version v4.36. '
                f'If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, '
                'please make sure to initialize the attention class '
                'with a layer index.')
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                       self.layer_idx)

    cos, sin = position_embeddings

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value
        # `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (getattr(self.config, 'sliding_window', None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    'past key must have a shape of (`batch_size, num_heads, '
                    'self.config.sliding_window-1, head_dim`), got'
                    f' {past_key.shape}')

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat(
                    [attention_mask,
                     torch.ones_like(attention_mask[:, -1:])],
                    dim=-1)

        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads for sequence parallel
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for
    # training stability reasons, therefore the input hidden states gets
    # silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # ----------------- flash attention forward ------------------------#

    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        causal = self.is_causal and q_len != 1

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, 'sliding_window', None) is not None
        and kv_seq_len > self.config.sliding_window
        and self.layer_idx < self.config.max_window_layers
        and self.config.use_sliding_window)

    window_size = (self.config.sliding_window,
                   self.config.sliding_window) if use_sliding_windows else (-1,
                                                                            -1)

    assert SUPPORT_FLASH2
    attn_context = MessageHub.get_instance('packed_sequence')
    cumulative_lengths = attn_context.get_info('cumulative_lengths')
    if cumulative_lengths is not None and SUPPORT_FLASH2 and bsz == 1:
        max_seqlen = attn_context.get_info('max_seqlen')
        attn_output = varlen_flash_attn(
            query_states,
            key_states,
            value_states,
            cumulative_lengths,
            max_seqlen,
            causal=causal,
            dropout_p=dropout_rate,
            window_size=window_size,
            training=self.training)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=causal,
            dropout_p=dropout_rate,
            window_size=window_size,
            training=self.training)

    # ---------------- flash attention forward end ------------------- #

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def qwen2_vlmodel_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            warnings.warn(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            # ====================================================================
            # pytorch 的 checkpoint() 模块无法处理 **kwargs，需要暂时修改
            layer_outputs = decoder_layer(
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
            # ====================================================================

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def qwen2_vision_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # 在开启 checkpoint 情况下，blk 不支持 **kwargs，需要暂时修改
    for blk in self.blocks:
        hidden_states = blk(hidden_states, cu_seqlens, rotary_pos_emb)

    return self.merger(hidden_states)
