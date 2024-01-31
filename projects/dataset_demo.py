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
from xtuner.dataset.collate_fns import default_collate_fn
from mmengine.runner.runner import Runner
import re

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
            ann_file='annotations/instances_train2017_rrrvlm_ovd.json',
            data_prefix=dict(img='train2017/'),
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
            ann_file='annotations/instances_train2017_rrrvlm_region.json',
            data_prefix=dict(img='train2017/'),
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

for data in rrr_dataset:
    pixel_values = data['pixel_values']
    pixel_values = pixel_values * std + mean
    pixel_values = pixel_values * 255
    pixel_values = torch.permute(pixel_values, (1, 2, 0))

    vis.set_image(pixel_values.numpy())

    conversation = data['conversation'][0]['input']
    print(data['conversation'][0]['output'])

    matches = re.findall(r'\[([^]]+)\]', conversation)[0]
    cleaned_text = matches.replace("[", "").replace("]", "").replace("'", "")
    numbers = cleaned_text.split(", ")
    numbers = [int(num) for num in numbers]
    vis.draw_bboxes(np.array([numbers]), edge_colors='r', line_widths=4)
    vis.show()

# train_dataloader = dict(
#     batch_size=4,
#     num_workers=0,
#     dataset=rrr_dataset,
#     sampler=dict(
#         type=LengthGroupedSampler,
#         length_property='length',
#         per_device_batch_size=4 * 1),
#     collate_fn=dict(type=default_collate_fn))
#
# train_dataloader = Runner.build_dataloader(train_dataloader)
# for i, load in enumerate(train_dataloader):
#     print(load)
#     break
