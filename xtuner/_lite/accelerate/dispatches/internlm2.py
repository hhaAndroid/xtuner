# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple
import inspect

import os
import torch
from einops import rearrange
from mmengine import MessageHub
from transformers.cache_utils import StaticCache

from ._attention import SUPPORT_FLASH2, flash_attn_wo_mask, varlen_flash_attn
from xtuner._lite.yunchang import llama3_varlen_attention_sp_ulysses_ring
from xtuner._lite.parallel.new_setup import get_ring_group, get_ring_world_size,get_sp_world_size, get_ulysess_group

class InternLM2RotaryEmbedding(torch.nn.Module):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=1000000,
                 device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (
            base**(torch.arange(0, dim, 2).float().to(device) / dim))

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if (seq_len > self.max_seq_len_cached
                or self.cos_cached.device != x.device
                or self.cos_cached.dtype != x.dtype):
            self.max_seq_len_cached = seq_len
            assert self.inv_freq.dtype == torch.float32
            t = torch.arange(
                self.max_seq_len_cached,
                device=x.device,
                dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(t.device))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().to(x.dtype)
            self.sin_cached = emb.sin().to(x.dtype)
        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_old(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def repeat_kv_bshd(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """The hidden states go from (batch, seqlen, num_key_value_heads, head_dim)
    to (batch, seqlen, num_attention_heads, head_dim)"""
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :,
                                  None, :].expand(batch, slen,
                                                  num_key_value_heads, n_rep,
                                                  head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep,
                                 head_dim)


def _internlm2_varlen_attn_forward(
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
    # Modified from https://huggingface.co/internlm/internlm-7b/blob/939a68c0dc1bd5f35b63c87d44af05ce33379061/modeling_internlm.py#L161  # noqa:E501
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            '`static` cache implementation is not compatible with '
            '`attn_implementation==flash_attention_2` make sure to use `sdpa` '
            'in the mean time, and open an issue at '
            'https://github.com/huggingface/transformers')

    bsz, q_len, _ = hidden_states.size()
    attn_context = MessageHub.get_instance('packed_sequence')

    position_ids = attn_context.get_info('position_ids')
    assert position_ids.size(1) == q_len, f'{position_ids.size(1)} {q_len}'

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., :self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    signature = inspect.signature(self.rotary_emb.forward)
    if 'seq_len' in signature.parameters:
        # old
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max() + 1)
        query_states, key_states = apply_rotary_pos_emb_old(query_states, key_states, cos, sin, position_ids)
    else:
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models;
        # cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons therefore the input hidden states gets silently
    # casted in float32. Hence, we need cast them back in the correct dtype
    # just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not
    # cast the LayerNorms in fp32. (InternLM2RMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.wqkv.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # repeat kv for sequence parallel
    key_states = repeat_kv_bshd(key_states, self.num_key_value_groups)
    value_states = repeat_kv_bshd(value_states, self.num_key_value_groups)

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
                # 如果想更省显存，可以设置为 1。-1 表示不切分
                heads_k_stride=-1
            )
            attn_output = attn_output.unsqueeze(0)
        else:
            max_seqlen = attn_context.get_info('max_seqlen')
            attn_output = varlen_flash_attn(query_states, key_states, value_states,
                                            cumulative_lengths, max_seqlen)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=True,
            training=self.training)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    # Due to the implementation of the PyTorch version of flash attention,
    # even when the output_attentions flag is set to True, it is not possible
    # to return the attn_weights.
    return attn_output, None, past_key_value


def internlm2_varlen_attn_forward(
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

    return _internlm2_varlen_attn_forward(self, hidden_states, attention_mask,
                                          position_ids, past_key_value,
                                          output_attentions, use_cache)
