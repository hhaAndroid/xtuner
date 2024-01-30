from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel)
import torch
import torch.nn as nn
import os

import numpy as np
import torch

os.environ['HF_MODULES_CACHE'] = '../'

from transformers import (AutoTokenizer, CLIPImageProcessor)

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset import ConcatDataset
from projects.modules import RRRDataset, ADD_TOKENS_DECODER
from mmengine.visualization import Visualizer
from xtuner.dataset.samplers import LengthGroupedSampler
from projects.modules import withbbox_default_collate_fn
from mmengine.runner.runner import Runner
import re
from xtuner.model.modules import ProjectorConfig, ProjectorModel, dispatch_modules

data_root = '/home/PJLAB/huanghaian/dataset/coco/'

prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = int(2048 - (336 / 14) ** 2)  # TODO

visual_encoder_name_or_path = '/home/PJLAB/huanghaian/models--openai--clip-vit-large-patch14-336/snapshots' \
                              '/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
visual_encoder = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path=visual_encoder_name_or_path)

input_size = 532
backbone_output_stride = 14
backbone_output_channel = visual_encoder.config.hidden_size
sliding_window_stride = 196
sliding_window_size = 336
h_grids = max(input_size - sliding_window_size + sliding_window_stride - 1, 0) // sliding_window_stride + 1
w_grids = max(input_size - sliding_window_size + sliding_window_stride - 1, 0) // sliding_window_stride + 1
window_pos_embed = nn.Parameter(torch.randn(1, (input_size // backbone_output_stride) ** 2,
                                            visual_encoder.config.hidden_size))  # 1,5193,2048

visual_encoder.requires_grad_(False)


def sliding_window_vit_forward(pixel_values):
    batch_size = pixel_values.shape[0]
    output_features = torch.zeros(
        (batch_size, input_size // backbone_output_stride, input_size // backbone_output_stride,
         backbone_output_channel), dtype=pixel_values.dtype, device=pixel_values.device
    )
    counters = torch.zeros(
        (batch_size, input_size // backbone_output_stride, input_size // backbone_output_stride,
         1), dtype=pixel_values.dtype, device=pixel_values.device
    )

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * sliding_window_stride
            x1 = w_idx * sliding_window_stride
            y2 = min(y1 + sliding_window_size, input_size)
            x2 = min(x1 + sliding_window_size, input_size)
            y1 = max(y2 - sliding_window_size, 0)
            x1 = max(x2 - sliding_window_size, 0)
            cur_pixel_values = pixel_values[..., y1:y2, x1:x2]

            cur_visual_outputs = visual_encoder(cur_pixel_values, output_hidden_states=True)
            last_hidden_state = cur_visual_outputs.hidden_states[-2][:, 1:]

            output_features[:, y1 // backbone_output_stride:y2 // backbone_output_stride,
            x1 // backbone_output_stride:x2 // backbone_output_stride] += last_hidden_state.view(
                batch_size, sliding_window_size // backbone_output_stride,
                            sliding_window_size // backbone_output_stride, -1)
            counters[:, y1 // backbone_output_stride:y2 // backbone_output_stride,
            x1 // backbone_output_stride:x2 // backbone_output_stride] += 1

    output_features /= counters
    encoded_pixel_features = output_features.view(batch_size, -1, backbone_output_channel)
    return encoded_pixel_features


llm_name_or_path = 'internlm/internlm2-chat-7b'

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    added_tokens_decoder=ADD_TOKENS_DECODER,
    trust_remote_code=True,
    cache_dir='../internlm2-chat-7b',
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    do_center_crop=False,
    do_resize=False,
    trust_remote_code=True)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        dict(
            type=RRRDataset,
            data_root=data_root,
            ann_file='annotations/instances_train2017_rrrvlm_ovd.json',
            data_prefix=dict(img='train2017/'),
            tokenizer=tokenizer,
            image_processor=image_processor,
            dataset_map_fn=llava_map_fn,
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template),
            max_length=max_length
        ),
        # dict(
        #     type=RRRDataset,
        #     data_root=data_root,
        #     ann_file='annotations/instances_train2017_rrrvlm_region.json',
        #     data_prefix=dict(img='train2017/'),
        #     tokenizer=tokenizer,
        #     image_processor=image_processor,
        #     dataset_map_fn=llava_map_fn,
        #     template_map_fn=dict(
        #         type=template_map_fn_factory, template=prompt_template),
        #     max_length=max_length
        # )
    ]
)

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='length',
        per_device_batch_size=4 * 1),
    collate_fn=dict(type=withbbox_default_collate_fn))

train_dataloader = Runner.build_dataloader(train_dataloader)

projector_config = ProjectorConfig(
    visual_hidden_size=visual_encoder.config.hidden_size * 2,
    llm_hidden_size=4096,
    depth=2)
projector = ProjectorModel(projector_config).to(visual_encoder.dtype)

from projects.modules.rrr_model import prepare_inputs_labels_for_multimodal
from projects.modules import GeoRegionSampler

sampler = GeoRegionSampler(2048,
                           4096,
                           512,
                           num_sub_point=[128, 32],
                           num_neighbor=[24, 24])

for i, data_sampler in enumerate(train_dataloader):
    data = data_sampler['data']
    with torch.no_grad():
        visual_outputs = sliding_window_vit_forward(data['pixel_values'])
        visual_outputs += window_pos_embed
        bs, pn, hs = visual_outputs.shape
        # token merge
        visual_outputs = visual_outputs.view(bs, int(pn / 2), int(hs * 2))
        print(visual_outputs.shape)  # b, 722, 2048

        pixel_values = projector(visual_outputs)
        data['pixel_values'] = pixel_values  # b,722, 4096

        # 计算 Spatial-aware visual sampler, 模块输入是 visual_outputs 而不是 pixel_values
        # bbox 是原图尺度即可，内部会进行归一化处理
        region_feats = sampler(visual_outputs, data['bbox'])

        data = prepare_inputs_labels_for_multimodal(llm=None, **data)
