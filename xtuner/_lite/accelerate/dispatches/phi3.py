# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple
import os
import torch
from mmengine.logging import MessageHub

from ._attention import SUPPORT_FLASH2, flash_attn_wo_mask, varlen_flash_attn
from xtuner._lite.yunchang import llama3_varlen_attention_sp_ulysses_ring
from xtuner._lite.parallel import get_ring_group, get_ring_world_size, get_sp_world_size, get_ulysess_group, \
    split_for_sequence_parallel, get_sp_group
from torch.distributed.nn.functional import all_gather


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _phi3_varlen_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    attn_context = MessageHub.get_instance('packed_sequence')

    position_ids = attn_context.get_info('position_ids')
    assert position_ids.size(1) == q_len, f'{position_ids.size(1)} {q_len}'

    qkv = self.qkv_proj(hidden_states)
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos: query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim:]

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.'
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    if get_sp_world_size() > 1:
        sp_group = get_sp_group()
        global_position_ids = attn_context.get_info('global_position_ids')
        if global_position_ids is None:
            global_position_ids = all_gather(position_ids, group=sp_group)
            global_position_ids = torch.cat(global_position_ids, dim=1)
            # TODO HHA： 比较特殊，否则 SP 情况下计算结果和不开 SP 不一致
        cos, sin = self.rotary_emb(value_states, global_position_ids)
        cos = split_for_sequence_parallel(cos, dim=1, sp_group=sp_group)
        sin = split_for_sequence_parallel(sin, dim=1, sp_group=sp_group)
    else:
        cos, sin = self.rotary_emb(value_states, position_ids)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    use_sliding_windows = False

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
                getattr(self.config, 'sliding_window', None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f'past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got'
                    f' {past_key.shape}'
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_dropout = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32.

    if query_states.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.qkv_proj.weight.dtype

        print(
            f'The input hidden states seems to be silently casted in float32, this might be related to'
            f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.'
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    assert SUPPORT_FLASH2
    cumulative_lengths = attn_context.get_info('cumulative_lengths')
    if cumulative_lengths is not None and SUPPORT_FLASH2 and bsz == 1:
        # 仅仅用于测试该分支下 sp ulyess 是否正确
        force_to_new_sp = os.environ.get('FORCE_TO_NEW_SP')
        if get_ring_world_size() > 1 or force_to_new_sp:
            # 只有开启了 ring 情况下才运行，如果只是普通 sp，则依然运行原先逻辑
            assert cumulative_lengths[-1] % get_sp_world_size() == 0, f'==={cumulative_lengths[-1]}===='
            q_unpad, k_unpad, v_unpad = query_states.flatten(0, 1), key_states.flatten(
                0, 1), value_states.flatten(0, 1)
            attn_output = llama3_varlen_attention_sp_ulysses_ring(
                q_unpad,
                k_unpad,
                v_unpad,
                cumulative_lengths,
                ulysses_pg=get_ulysess_group(),
                ring_pg=get_ring_group(),
                causal=True,
                dropout_p=attn_dropout,
                # 如果想更省显存，可以设置为 1。-1 表示不切分
                heads_k_stride=-1
            )
            attn_output = attn_output.unsqueeze(0)
        else:
            max_seqlen = attn_context.get_info('max_seqlen')
            attn_output = varlen_flash_attn(query_states, key_states, value_states,
                                cumulative_lengths, max_seqlen, dropout_p=attn_dropout)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=attn_dropout,
            training=self.training)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def ph3_varlen_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
    Optional[Tuple[torch.Tensor]]]:
    return _phi3_varlen_attn_forward(self, hidden_states, attention_mask,
                                     position_ids, past_key_value,
                                     output_attentions, use_cache)
