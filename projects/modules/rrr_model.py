# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training

from xtuner.registry import BUILDER
from xtuner.model.utils import (LoadWoInit, find_all_linear_names,
                                get_peft_model_state_dict, guess_load_checkpoint,
                                make_inputs_require_grad, traverse_dict)
from xtuner.model.modules import ProjectorConfig, ProjectorModel, dispatch_modules
import torch
import os.path as osp
from typing import List, Optional

import torch
from mmengine import print_log
from mmengine.utils.misc import get_object_from_string
from peft import PeftType
from torch import nn
from transformers import PreTrainedModel
from .constants import IMAGE_TOKEN_INDEX, REGION_FEAT_TOKEN_INDEX, SEG_TOKEN_INDEX, IGNORE_INDEX
from .visual_sampler import GeoRegionSampler
from segment_anything import build_sam_vit_h
from .utils import polygon_to_bitmap


class RRRModel(BaseModel):
    def __init__(self,
                 llm,
                 visual_encoder,
                 freeze_llm=True,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 # CLIP ViT parameters
                 input_size=672,
                 sliding_window_size=336,
                 sliding_window_stride=336,
                 backbone_output_stride=14,
                 use_visual_sampler=False,  # 预训练时候为 False,微调时候为 True
                 use_sam=False,
                 sam_pretrained_pth=None,
                 ):
        super().__init__()
        self.freeze_llm = freeze_llm

        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
            self.visual_encoder = self._build_from_cfg_or_module(
                visual_encoder)

        # sliding_window
        self.input_size = input_size
        self.backbone_output_stride = backbone_output_stride
        self.backbone_output_channel = self.visual_encoder.config.hidden_size
        self.sliding_window_stride = sliding_window_stride
        self.sliding_window_size = sliding_window_size
        self.h_grids = max(input_size - sliding_window_size + sliding_window_stride - 1, 0) // sliding_window_stride + 1
        self.w_grids = max(input_size - sliding_window_size + sliding_window_stride - 1, 0) // sliding_window_stride + 1
        self.window_pos_embed = nn.Parameter(
            torch.randn(1, (input_size // backbone_output_stride) ** 2, self.visual_encoder.config.hidden_size)).to(
            self.visual_encoder.dtype)

        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        projector_config = ProjectorConfig(
            visual_hidden_size=self.visual_encoder.config.hidden_size * 4,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth)
        self.projector = ProjectorModel(projector_config).to(
            self.visual_encoder.dtype)

        # visual sampler
        if use_visual_sampler:
            # 先对每个区域采样 512 个点，然后对 512 个进行 fps 计算得到 128 个点，然后对 128 个点中每个点进行 24 个邻居采样
            # 然后将 24 个邻居信息聚合到当前点上，最后得到 128 个点的特征，对这个128个点特征进行拉平，然后维度变换
            # 然后级联 2 次计算
            self.sampler = GeoRegionSampler(self.visual_encoder.config.hidden_size * 4,
                                            self.llm.config.hidden_size,
                                            256,  # 512
                                            num_sub_point=[64, 16],  # 128 32
                                            # 24, 24
                                            num_neighbor=[16, 16]).to(self.visual_encoder.dtype)
        else:
            self.sampler = None

        self.use_sam = use_sam
        if use_sam:
            self.sam_model = build_sam_vit_h(sam_pretrained_pth)

            for param in self.sam_model.parameters():
                param.requires_grad = False

            self.sam_model.mask_decoder.train()
            for param in self.sam_model.mask_decoder.parameters():
                param.requires_grad = True

            # 投影层
            text_fc = [
                nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.llm.config.hidden_size, 256),
                nn.Dropout(0.0),
            ]
            self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
            self.text_hidden_fcs.train()
            for param in self.text_hidden_fcs.parameters():
                param.requires_grad = True

        self.llm.requires_grad_(False)
        self.visual_encoder.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                self.visual_encoder.enable_input_require_grads()
            else:
                self.visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_visual_encoder_lora:  # off
            self._prepare_visual_encoder_for_lora(
                visual_encoder_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.visual_select_layer = visual_select_layer

        self._is_init = True

    @torch.no_grad()
    def sliding_window_vit_forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        # 这个地方的 dtype 只能用 visual_encoder 不能是 pixel_values
        # 因为 deepspeed 会自动将模型转为 bf16，但是 pixel_values 依然是 fp32 的，如果采用 fp32 后面模型会报错
        output_features = torch.zeros(
            (batch_size, self.input_size // self.backbone_output_stride, self.input_size // self.backbone_output_stride,
             self.backbone_output_channel), dtype=self.visual_encoder.dtype, device=pixel_values.device
        )
        counters = torch.zeros(
            (batch_size, self.input_size // self.backbone_output_stride, self.input_size // self.backbone_output_stride,
             1), dtype=self.visual_encoder.dtype, device=pixel_values.device
        )

        for h_idx in range(self.h_grids):
            for w_idx in range(self.w_grids):
                y1 = h_idx * self.sliding_window_stride
                x1 = w_idx * self.sliding_window_stride
                y2 = min(y1 + self.sliding_window_size, self.input_size)
                x2 = min(x1 + self.sliding_window_size, self.input_size)
                y1 = max(y2 - self.sliding_window_size, 0)
                x1 = max(x2 - self.sliding_window_size, 0)
                cur_pixel_values = pixel_values[..., y1:y2, x1:x2]

                cur_visual_outputs = self.visual_encoder(cur_pixel_values, output_hidden_states=True)
                # 无需考虑 cls_outputs
                last_hidden_state = cur_visual_outputs.hidden_states[self.visual_select_layer][:, 1:]

                output_features[:, y1 // self.backbone_output_stride:y2 // self.backbone_output_stride,
                x1 // self.backbone_output_stride:x2 // self.backbone_output_stride] += last_hidden_state.view(
                    batch_size, self.sliding_window_size // self.backbone_output_stride,
                                self.sliding_window_size // self.backbone_output_stride, -1)
                counters[:, y1 // self.backbone_output_stride:y2 // self.backbone_output_stride,
                x1 // self.backbone_output_stride:x2 // self.backbone_output_stride] += 1

        output_features /= counters
        encoded_pixel_features = output_features.view(batch_size, -1, self.backbone_output_channel)
        return encoded_pixel_features

    def prepare_for_eval(self, data):
        visual_outputs = self.sliding_window_vit_forward(data['pixel_values'])
        visual_outputs = visual_outputs + self.window_pos_embed
        bs, pn, hs = visual_outputs.shape
        # token merge
        visual_outputs = visual_outputs.view(bs, int(pn / 4), int(hs * 4))
        pixel_values = self.projector(visual_outputs)
        data['pixel_values'] = pixel_values
        if self.sampler:
            # 计算 Spatial-aware visual sampler, 模块输入是 visual_outputs 而不是 pixel_values
            # bbox 是原图尺度即可，内部会进行归一化处理
            region_mask = []
            for b in data['gt_bboxes_masks']:
                o_mask = []
                for _b in b:
                    if len(_b) == 4:
                        # bbox
                        coor_mask = torch.zeros((self.input_size, self.input_size), device=pixel_values.device)
                        coor_mask[_b[0]:_b[2], _b[1]:_b[3]] = 1
                    else:
                        # mask
                        coor_mask = polygon_to_bitmap(_b.reshape(-1, 2), self.input_size, self.input_size)
                        coor_mask = torch.tensor(coor_mask, device=pixel_values.device)
                    assert len(coor_mask.nonzero()) != 0
                    o_mask.append(coor_mask)
                region_mask.append(o_mask)
            region_feats = self.sampler(visual_outputs, region_mask, return_dtype=visual_outputs.dtype)  # b, 4096
            data['region_feats'] = region_feats
        data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        return data

    def forward(self, data, data_samples=None, mode='loss'):
        if 'pixel_values' in data:
            visual_outputs = self.sliding_window_vit_forward(data['pixel_values'])
            visual_outputs = visual_outputs + self.window_pos_embed
            bs, pn, hs = visual_outputs.shape
            # token merge
            visual_outputs = visual_outputs.view(bs, int(pn / 4), int(hs * 4))
            pixel_values = self.projector(visual_outputs)
            data['pixel_values'] = pixel_values

            if self.sampler:
                # 计算 Spatial-aware visual sampler, 模块输入是 visual_outputs 而不是 pixel_values
                # bbox 是原图尺度即可，内部会进行归一化处理
                region_mask = []
                for b in data['gt_bboxes_masks']:
                    o_mask = []
                    for _b in b:
                        if len(_b) == 4:
                            # bbox
                            coor_mask = torch.zeros((self.input_size, self.input_size), device=pixel_values.device)
                            coor_mask[_b[0]:_b[2], _b[1]:_b[3]] = 1
                        else:
                            # mask
                            coor_mask = polygon_to_bitmap(_b.reshape(-1, 2), self.input_size, self.input_size)
                            coor_mask = torch.tensor(coor_mask, device=pixel_values.device)
                        assert len(coor_mask.nonzero()) != 0
                        o_mask.append(coor_mask)
                    region_mask.append(o_mask)

                region_feats = self.sampler(visual_outputs, region_mask, return_dtype=visual_outputs.dtype)  # b, 4096
                data['region_feats'] = region_feats
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    # TODO: 暂时只支持 pretrain 阶段
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        # Step 4. Visual Sampler
        if self.sampler:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'sampler.' in k})
        # Step 5. window_pos_embed
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'window_pos_embed' in k})
        return to_return

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.visual_encoder.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)


def prepare_inputs_labels_for_multimodal(
        llm: PreTrainedModel,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        region_feats=None,
        **kwargs):
    if pixel_values is None:
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': None,
            'labels': labels
        }

    assert position_ids is None

    if region_feats is None:
        region_feats = [None] * len(pixel_values)

    if attention_mask is None:  # 训练时候必然有，单张图片评估时候可能没有
        attention_mask = [None] * len(pixel_values)

    new_inputs_embeds = []
    new_labels = []
    new_attention_mask = []
    for batch_idx, (cur_input_ids, region_feat, pixel_value, cur_attn_mask) in enumerate(
            zip(input_ids, region_feats, pixel_values, attention_mask)):
        if labels is not None:
            cur_labels = labels[batch_idx]
        if llm is not None:
            cur_inputs_embeds = llm.get_input_embeddings()(cur_input_ids)
        else:
            # for debug
            cur_inputs_embeds = torch.randn((cur_input_ids.shape[0], 4096)).to(cur_labels.device)

        if region_feat is not None:
            # 由于 region_feat 占位符是一个区域一个 token，而 region_feat 也是一个区域一个特征，因此可以直接替换
            # 但是 image_feat 会对应多个 token，因此写法不一样
            # 如果不加下面这行会报错，原因是 inplace 操作不允许
            # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
            cur_inputs_embeds = cur_inputs_embeds.clone()
            cur_inputs_embeds[cur_input_ids == REGION_FEAT_TOKEN_INDEX] = region_feat

        img_token_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        cur_inputs_embeds = torch.cat(
            [cur_inputs_embeds[:img_token_index[0]], pixel_value, cur_inputs_embeds[img_token_index[0]:]], dim=0)
        if labels is not None:
            cur_labels = torch.cat([cur_labels[:img_token_index[0]],
                                    torch.full((pixel_value.shape[0],),
                                               IGNORE_INDEX,
                                               device=cur_labels.device,
                                               dtype=cur_labels.dtype), cur_labels[img_token_index[0]:]], dim=0)
        if cur_attn_mask is not None:
            cur_attn_mask = torch.cat([cur_attn_mask[:img_token_index[0]],
                                       torch.ones((pixel_value.shape[0],), device=cur_attn_mask.device).bool(),
                                       cur_attn_mask[img_token_index[0]:]], dim=0)
        new_inputs_embeds.append(cur_inputs_embeds)
        if labels is not None:
            new_labels.append(cur_labels)
        if cur_attn_mask is not None:
            new_attention_mask.append(cur_attn_mask)

    new_inputs_embeds = torch.stack(new_inputs_embeds)
    if labels is not None:
        new_labels = torch.stack(new_labels)
    else:
        new_labels = None
    if len(new_attention_mask) > 0:
        new_attention_mask = torch.stack(new_attention_mask)
    else:
        new_attention_mask = None

    return {
        'input_ids': None,
        'position_ids': position_ids,
        'attention_mask': new_attention_mask,
        'past_key_values': past_key_values,
        'inputs_embeds': new_inputs_embeds,
        'labels': new_labels
    }
