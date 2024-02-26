import os

import numpy as np
import torch

os.environ['HF_MODULES_CACHE'] = '../'

from transformers import (AutoTokenizer, CLIPImageProcessor)

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset import ConcatDataset
from projects.modules import RRRDataset, ADD_TOKENS_DECODER, withbbox_default_collate_fn
from mmengine.visualization import Visualizer
from xtuner.dataset.samplers import LengthGroupedSampler
from mmengine.runner.runner import Runner

data_root = '/home/PJLAB/huanghaian/dataset/coco/'

prompt_template = PROMPT_TEMPLATE.internlm2_chat
# image token 占 (336 / 14) ** 2, region token 占 1
max_length = int(2048 - (336 / 14) ** 2 - 1)

llm_name_or_path = 'internlm/internlm2-chat-7b'
visual_encoder_name_or_path = '/home/PJLAB/huanghaian/models--openai--clip-vit-large-patch14-336/snapshots' \
                              '/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'

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
            # 如果是 bbox 后缀则 use_mask 必须设置为 False
            ann_file='annotations/instances_val2017_rrrvlm_ovd1_mask.json',
            data_prefix=dict(img='val2017/'),
            use_mask=True,
            bbox_to_mask_prob=0.5,
            tokenizer=tokenizer,
            image_processor=image_processor,
            dataset_map_fn=llava_map_fn,
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template),
            max_length=max_length
        ),
        dict(
            type=RRRDataset,
            data_root=data_root,
            ann_file='annotations/instances_val2017_rrrvlm_region1_mask.json',
            data_prefix=dict(img='val2017/'),
            use_mask=True,
            bbox_to_mask_prob=0.5,
            tokenizer=tokenizer,
            image_processor=image_processor,
            dataset_map_fn=llava_map_fn,
            template_map_fn=dict(
                type=template_map_fn_factory, template=prompt_template),
            max_length=max_length
        )
    ]
)

class_type = train_dataset.pop('type')
rrr_dataset = class_type(**train_dataset)
print(len(rrr_dataset))

image_mean = [
    0.48145466,
    0.4578275,
    0.40821073
],
image_std = [
    0.26862954,
    0.26130258,
    0.27577711
]
mean = torch.tensor(image_mean).view(3, 1, 1)
std = torch.tensor(image_std).view(3, 1, 1)

vis = Visualizer()


def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    scales = 0.5 + (areas - min_area) // (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


debug_dataset = True

if debug_dataset:
    for data in rrr_dataset:
        pixel_values = data['pixel_values']
        pixel_values = pixel_values * std + mean
        pixel_values = pixel_values * 255
        pixel_values = torch.permute(pixel_values, (1, 2, 0))

        vis.set_image(pixel_values.numpy())

        conversation = data['conversation'][0]['input']
        print(conversation)
        print(data['conversation'][0]['output'])

        bboxes = data['bbox']
        name = data['name']

        image2colors = []
        for _ in range(len(bboxes)):
            colors = np.random.random((1, 3)) * 0.7 + 0.3
            colors = (colors * 255).astype(int).tolist()[0]
            image2colors.append(tuple(colors))

        bboxes = np.array(bboxes).reshape(-1, 4)
        positions = bboxes[:, :2] + 3
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        vis.draw_bboxes(bboxes, edge_colors=image2colors, line_widths=3)
        vis.draw_texts(
            name,
            positions,
            colors='g',
            font_sizes=[int(13 * s) for s in scales],
            bboxes=[{
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            }] * len(scales))

        if 'mask' in data:
            mask = data['mask']
            for i, m in enumerate(mask):
                vis.draw_polygons(m.reshape(-1, 2), edge_colors='w', face_colors=image2colors[i])

        vis.show()
else:
    train_dataloader = dict(
        batch_size=4,
        num_workers=0,
        dataset=rrr_dataset,
        sampler=dict(
            type=LengthGroupedSampler,
            length_property='length',
            per_device_batch_size=4 * 1),
        collate_fn=dict(type=withbbox_default_collate_fn))

    train_dataloader = Runner.build_dataloader(train_dataloader)
    for i, load in enumerate(train_dataloader):
        print(load)
        break
