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
import torch.nn.functional as F


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        scale=1000,  # 100000.0,
        eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


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
                 # 仅仅开启了 sam 训练才有效
                 ce_loss_weight=1.0,
                 dice_loss_weight=0.5,
                 bce_loss_weight=2.0
                 ):
        super().__init__()

        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight

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
        self._need_seg_token = False
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

            self._seg_token_mask = None

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
                # 可以接受 fp32 输入，model 是 bf16 格式，输出是 bf16，因为内部有 target_dtype = self.patch_embedding.weight.dtype 代码
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

    def postprocess_for_eval(self, llm_out, mask_data_dict):
        # 只有开启了 sam 才会调用
        # TODO 暂时没有验证 bs>1 情况下正确性
        # 第 0 个是第一次输入推理的隐含层输出，不需要, 因为不可能是我们关心的
        output_hidden_states = llm_out.hidden_states[1:]
        output_hidden_states = [o[-1] for o in output_hidden_states]  # 每一个shape 都是 (b,1,4096)
        output_hidden_states = torch.cat(output_hidden_states, dim=1)  # b,N,4096

        output_ids = llm_out.sequences  # (B,N+1)

        # 不需要前2个是因为, 本身第一个是开始符，第 2 个第一次推理结果，我们前面已经丢弃了。
        seg_token_mask = output_ids[:, 2:] == SEG_TOKEN_INDEX

        assert len(self.text_hidden_fcs) == 1
        hidden_states = self.text_hidden_fcs[0](output_hidden_states)
        pred_embeddings = hidden_states[seg_token_mask]

        if pred_embeddings.shape[0] == 0:
            # 所有预测都没有找到 seg token，暂时只验证 bs=1 情况
            return [None] * output_hidden_states.shape[0]  # bs

        # 上述操作会丢失 bs 信息，需要还原回来
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1, device=seg_token_counts.device).long(), seg_token_offset], dim=0
        )

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        image_embeddings = self.get_sam_visual_embs(mask_data_dict['sam_pixel_values'])

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.sam_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )

            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            orig_dtype = low_res_masks.dtype
            pred_mask = self.sam_model.postprocess_masks(
                low_res_masks.float(),
                input_size=mask_data_dict['padding_size'][i],
                original_size=mask_data_dict['orig_size'][i],
            )
            pred_masks.append(pred_mask[:, 0].to(orig_dtype))
        return pred_masks

    def forward(self, data, data_samples=None, mode='loss'):
        mask_data_dict = {}
        if 'pixel_values' in data:
            if self.training:
                # 训练时候才有这个逻辑，推理时候不能如此处理
                self._need_seg_token = SEG_TOKEN_INDEX in data['input_ids']
                if self._need_seg_token:
                    assert self.use_sam, 'sam model is not enabled, but seg token is found in input_ids'
                    seg_token_mask = data['input_ids'][:, 1:] == SEG_TOKEN_INDEX  # B,N
                    # 后面填充假数据，使其和 input_ids 一样长
                    seg_token_mask = torch.cat(
                        [
                            seg_token_mask,
                            torch.zeros((seg_token_mask.shape[0], 1), device=seg_token_mask.device).bool(),
                        ],
                        dim=1,
                    )
                    self._seg_token_mask = seg_token_mask
                    mask_data_dict['sam_pixel_values'] = data['sam_pixel_values']
                    mask_data_dict['orig_size'] = data['orig_size']
                    mask_data_dict['padding_size'] = data['padding_size']
                    mask_data_dict['sam_mask'] = data['sam_mask']

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
                            coor_mask = _b.to(pixel_values.device)
                        assert len(coor_mask.nonzero()) != 0
                        o_mask.append(coor_mask)
                    region_mask.append(o_mask)

                region_feats = self.sampler(visual_outputs, region_mask, return_dtype=visual_outputs.dtype)  # b, 4096
                data['region_feats'] = region_feats
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples, mask_data_dict)
        elif mode == 'predict':  # 不会运行，无需关注
            return self.predict(data, data_samples)
        elif mode == 'tensor':  # 不会运行，无需关注
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def get_sam_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                # 不允许 fp32 和 bf16 运算
                image_embeddings = self.sam_model.image_encoder(
                    pixel_values[i].unsqueeze(0).to(self.sam_model.image_encoder.pos_embed.dtype)
                )  # stride 16
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None, mask_data_dict=None):
        if self._need_seg_token and self.training:  # sam 训练
            outputs = self.llm(**data, output_hidden_states=True)
            loss_dict = {'loss_llm': outputs.loss * self.ce_loss_weight}
        else:
            outputs = self.llm(**data)
            loss_dict = {'loss': outputs.loss * self.ce_loss_weight}

        if self._need_seg_token and self.training:  # sam 训练
            image_embeddings = self.get_sam_visual_embs(mask_data_dict['sam_pixel_values'])

            # 考虑 sam loss
            output_hidden_states = outputs.hidden_states
            assert len(self.text_hidden_fcs) == 1
            hidden_states = self.text_hidden_fcs[0](output_hidden_states[-1])
            # 因为 image id 在前面，因此我们只需要截取后部分就可以，不会有啥影响，因为 seg token 不可能在前面
            pred_embeddings = hidden_states[:, -self._seg_token_mask.shape[1]:, :][self._seg_token_mask]

            # 上述操作会丢失 bs 信息，需要还原回来
            seg_token_counts = self._seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1, device=seg_token_offset.device).long(), seg_token_offset], dim=0
            )
            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):  # bs 维度
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.sam_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, _ = self.sam_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                # 不支持 bf16
                orig_dtype = low_res_masks.dtype
                pred_mask = self.sam_model.postprocess_masks(
                    low_res_masks.float(),
                    input_size=mask_data_dict['padding_size'][i],
                    original_size=mask_data_dict['orig_size'][i],
                )
                pred_masks.append(pred_mask[:, 0].to(orig_dtype))

            mask_bce_loss = 0
            mask_dice_loss = 0
            num_masks = 0
            gt_masks = mask_data_dict['sam_mask']

            for batch_idx in range(len(pred_masks)):
                pred_mask = pred_masks[batch_idx]

                gt_mask = gt_masks[batch_idx].to(device=pred_mask.device, dtype=pred_mask.dtype)

                assert (
                        gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
                mask_bce_loss += (
                        sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                        * gt_mask.shape[0]
                )
                mask_dice_loss += (
                        dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                        * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            loss_dict['loss_sam_bce'] = mask_bce_loss
            loss_dict['loss_sam_dice'] = mask_dice_loss

        return loss_dict

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

        if self.use_sam:
            # 只保存可训练参数就行
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if '.mask_decoder' in k})
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'text_hidden_fcs' in k})
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
