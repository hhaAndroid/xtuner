import os
import torch.nn as nn

from .internlm2 import InternLM2Config, InternLM2ForCausalLM

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.geglu import LigerGEGLUMLP
    from liger_kernel.transformers.layer_norm import LigerLayerNorm
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import (
        LigerSwiGLUMLP, LigerSiLUMulFunction
    )
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLCausalLMOutputWithPast
    )
except ImportError:
    pass

from typing import List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from mmengine.utils.version_utils import digit_version


def register_remote_code():
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register('internlm2', InternLM2Config, exist_ok=True)
    AutoModelForCausalLM.register(
        InternLM2Config, InternLM2ForCausalLM, exist_ok=True)


def lce_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
):
    r"""
    Copy paste Qwen2VL's forward but replace torch cross entropy with liger fused linear cross entropy

    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    ```"""

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # 官方这部分代码有错误,只能替换
    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # ....

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    loss = None
    logits = None

    if self.training and (labels is not None):
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
        shift_labels = shift_labels.view(-1)

        lce = LigerFusedLinearCrossEntropyLoss()
        loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
    else:
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )


def apply_liger_kernel_to_qwen2_vl(
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    layer_norm: bool = True,
    swiglu: bool = True
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2-VL models.
    NOTE: Qwen2-VL is not available in transformers<=4.44.2

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen2_vl import modeling_qwen2_vl

    # TODO: Support Qwen2-VL's multimodal RoPE implementation

    if rms_norm:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L439
        modeling_qwen2_vl.Qwen2RMSNorm = LigerRMSNorm
    if layer_norm:
        modeling_qwen2_vl.LayerNorm = LigerLayerNorm
    if cross_entropy:
        modeling_qwen2_vl.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = lce_forward
    if swiglu:
        modeling_qwen2_vl.Qwen2MLP = LigerSwiGLUMLP


class LigerSwiGLUMLPForInternlm2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.w2(
            LigerSiLUMulFunction.apply(self.w1(x), self.w3(x))
        )


def apply_liger_kernel_to_llava_clip_internlm2(
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        layer_norm: bool = True,
        swiglu: bool = True
) -> None:
    # modeling_clip
    nn.LayerNorm = LigerLayerNorm

    from .internlm2 import modeling_internlm2
    if digit_version(torch.__version__) >= digit_version('2.4.0'):
        modeling_internlm2.InternLM2MLP = LigerSwiGLUMLPForInternlm2
    modeling_internlm2.InternLM2RMSNorm = LigerRMSNorm

    from ..accelerate.dispatches import internlm2
    internlm2.apply_rotary_pos_emb = liger_rotary_pos_emb
    os.environ["USE_LIGER_KERNEL"] = "1"
