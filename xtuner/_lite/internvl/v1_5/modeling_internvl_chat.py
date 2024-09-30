# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union
import os
import torch.utils.checkpoint
import transformers
from torch import nn
import math
import time
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .modeling_phi3 import Phi3ForCausalLM
from xtuner._lite.parallel import (LengthGroupedSampler, ParallelSampler,
                                   get_dp_mesh, get_dp_world_size,
                                   get_sp_group, get_sp_mesh,
                                   get_sp_world_size,
                                   reduce_sequence_parallel_loss,
                                   setup_parallel, get_sp_group, split_for_sequence_parallel)
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from mmengine.logging import MessageHub
import copy

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def rescale_sp_loss(loss_per_sp_rank,
                    labels_per_sp_rank,
                    sp_group: dist.ProcessGroup = None,
                    ignore_index=-100):
    if sp_group is None:
        sp_group = get_sp_group()

    if (sp_group is None) or (dist.get_world_size(sp_group) == 1):
        return loss_per_sp_rank

    shift_labels = labels_per_sp_rank
    active_tokens = (shift_labels != ignore_index).long().sum()
    global_active_tokens = copy.deepcopy(active_tokens)
    dist.all_reduce(global_active_tokens, group=sp_group)
    loss_weight = active_tokens / global_active_tokens * dist.get_world_size(
        group=sp_group)

    if active_tokens == 0:
        # convert nan to 0 just for logging
        loss_per_sp_rank = torch.nan_to_num(loss_per_sp_rank)

    return loss_per_sp_rank * loss_weight


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Phi3DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.36.2', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self._count = 0

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sp_size = get_sp_world_size()
        if sp_size > 1:
            sp_group = get_sp_group()
            sp_rank = dist.get_rank(sp_group)

            no_split_input_ids = os.environ.get('NO_SPLIT_INPUT_IDS')
            split_input_ids = not no_split_input_ids
            if split_input_ids:
                pad_id = 0
                orig_len_input_ids = input_ids.shape[1]
                image_flags = image_flags.squeeze(-1)
                assert input_ids.shape[0] == 1, 'batch size must be 1 for sequence parallel'
                # input_ids 均匀切分
                if orig_len_input_ids % sp_size != 0:  # 确保能均匀切
                    max_inputs_len = math.ceil(orig_len_input_ids / sp_size) * sp_size
                    _temp = input_ids.new_full((1, max_inputs_len - orig_len_input_ids), pad_id)
                    input_ids_new = torch.cat([input_ids, _temp], dim=-1)
                else:
                    input_ids_new = input_ids
                input_ids_list = torch.split(input_ids_new, input_ids_new.shape[1] // sp_size, dim=-1)
                input_ids_rank_pre = input_ids_list[sp_rank].contiguous()
                input_embeds_rank_pre = self.language_model.get_input_embeddings()(input_ids_rank_pre).clone()

                # torch.cuda.synchronize()
                # start_time = time.perf_counter()
                input_embeds = all_gather(input_embeds_rank_pre, group=sp_group)
                # torch.cuda.synchronize()
                # elapsed = time.perf_counter() - start_time
                # print(elapsed,'xxxx',flush=True)
                input_embeds = torch.cat(input_embeds, dim=1)
                input_embeds = input_embeds[:, :orig_len_input_ids]
            else:
                input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

            no_split_pixel_values = os.environ.get('NO_SPLIT_PIXEL_VALUES')
            split_pixel_values = not no_split_pixel_values
            # print(split_input_ids, split_pixel_values, os.environ.get('USE_CUSTOM_LOSS'), flush=True)
            if split_pixel_values:
                # pixel_values 均匀切分
                orig_img_batch = pixel_values.shape[0]
                if orig_img_batch % sp_size != 0:  # 确保能均匀切
                    max_inputs_len = math.ceil(orig_img_batch / sp_size) * sp_size
                    pad_img_batch = max_inputs_len - orig_img_batch
                    pad_pixel_values_ = pixel_values.new_zeros(pad_img_batch, 3,
                                                               pixel_values.shape[2],
                                                               pixel_values.shape[3])
                    pixel_values = torch.cat([pixel_values, pad_pixel_values_], dim=0)
                pixel_values = torch.split(pixel_values, len(pixel_values) // sp_size, dim=0)
                pixel_values = pixel_values[sp_rank].contiguous()

                vit_embeds = self.extract_feature(pixel_values)

                # torch.cuda.synchronize()
                # start_time = time.perf_counter()
                vit_embeds = all_gather(vit_embeds, group=sp_group)
                # torch.cuda.synchronize()
                # elapsed = time.perf_counter() - start_time
                # print(elapsed,'qqqqqxx',flush=True)

                vit_embeds = torch.cat(vit_embeds, dim=0)[:orig_img_batch]
            else:
                vit_embeds = self.extract_feature(pixel_values)
            vit_embeds = vit_embeds[image_flags == 1]
        else:
            image_flags = image_flags.squeeze(-1)
            input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

            vit_embeds = self.extract_feature(pixel_values)
            vit_embeds = vit_embeds[image_flags == 1]

        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if dist.get_rank() == 0 and self._count % 10 == 0:
            print(
                f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
        self._count += 1

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        if sp_size > 1:
            # 此处开始进行切分处理
            # 只需要处理 inputs_embeds 和 position_ids，其余用不到
            attn_context = MessageHub.get_instance('packed_sequence')
            position_ids = attn_context.get_info('position_ids')
            # phi3 attention 计算时候有特殊用途
            attn_context.update_info('global_position_ids', position_ids)

            assert position_ids.size(1) == input_embeds.shape[1] == labels.shape[1], \
                f'{position_ids.size(1)} {input_embeds.shape[1]} {labels.shape[1]}'

            sp_group = get_sp_group()
            # `dim` is 1 as the shape of tensor is (bs, seq_len)
            position_ids = split_for_sequence_parallel(
                position_ids, dim=1, sp_group=sp_group)
            input_embeds = split_for_sequence_parallel(
                input_embeds, dim=1, sp_group=sp_group)

            use_custom_loss = os.environ.get('USE_CUSTOM_LOSS')
            if not use_custom_loss:
                # 如果使用了定制化 loss 那么延迟切分
                labels = split_for_sequence_parallel(
                    labels, dim=1, sp_group=sp_group)

            attention_mask = None  # 不需要
            attn_context.update_info('position_ids', position_ids)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            batch_size = shift_labels.shape[0]
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            use_custom_loss = os.environ.get('USE_CUSTOM_LOSS')
            if not use_custom_loss:
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                if sp_size > 1:
                    # sp 间均衡
                    loss = rescale_sp_loss(loss, shift_labels, sp_group=sp_group)
            else:
                ctx = MessageHub.get_instance('packed_sequence')
                cumulative_lengths = ctx.get_info('cumulative_lengths')
                if cumulative_lengths is not None:
                    # packing 模式
                    num_tokens = cumulative_lengths[1:] - cumulative_lengths[:-1]
                    num_tokens = num_tokens.squeeze(dim=0).cpu().tolist()
                    if num_tokens[-1] == 0:  # 可能有 pading
                        num_tokens = num_tokens[:-1]
                    if sp_size > 1:
                        # 序列维度切分,有点特殊，只对 labels 进行偏移,然后再切分
                        logits_rank_pre = logits.view(-1, self.language_model.config.vocab_size).contiguous()

                        labels = labels.view(-1).contiguous()
                        shift_labels = labels[1:]  # 关键: shift 然后在补 -100，解决完整序列被切断带来的影响
                        shift_labels = torch.cat([shift_labels, shift_labels.new_ones(1) * -100], dim=0)
                        shift_labels_rank_pre = split_for_sequence_parallel(shift_labels, dim=0, sp_group=sp_group)

                        loss_fc = nn.CrossEntropyLoss(reduction='none')
                        loss_rank_pre = loss_fc(logits_rank_pre, shift_labels_rank_pre)

                        loss = all_gather(loss_rank_pre, group=sp_group)
                        loss = torch.cat(loss)

                        loss_list = loss.split(num_tokens)
                        labels_list = shift_labels.split(num_tokens)
                        loss_list = [loss.sum() / ((label != -100).sum().float() + 1e-12) for loss, label in
                                     zip(loss_list, labels_list)]
                        non_zero_loss_len = sum(1 for x in loss_list if x != 0)
                        if non_zero_loss_len == 0:
                            loss = torch.stack(loss_list).sum()
                        else:
                            loss = torch.stack(loss_list).sum() / non_zero_loss_len
                    else:
                        num_tokens[-1] -= 1  # shift label
                        loss_fc = nn.CrossEntropyLoss(reduction='none')
                        all_loss = loss_fc(shift_logits, shift_labels)
                        loss_list = all_loss.split(num_tokens)
                        labels_list = shift_labels.split(num_tokens)
                        loss_list = [loss.sum() / ((label != -100).sum().float() + 1e-12) for loss, label in
                                     zip(loss_list, labels_list)]
                        non_zero_loss_len = sum(1 for x in loss_list if x != 0)
                        if non_zero_loss_len == 0:
                            loss = torch.stack(loss_list).sum()
                        else:
                            loss = torch.stack(loss_list).sum() / non_zero_loss_len
                else:
                    # 非 packing 模式
                    loss_fct = CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss.reshape(batch_size, -1).sum(dim=1) / (
                            (labels[..., 1:] != -100).sum(dim=1).float() + 1e-12)
                    non_zero_loss_len = sum(1 for x in loss if x != 0)
                    if non_zero_loss_len == 0:
                        loss = loss.mean()
                    else:
                        loss = loss.sum() / non_zero_loss_len

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
