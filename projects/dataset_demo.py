import os

os.environ['HF_MODULES_CACHE'] = '../'

from transformers import (AutoTokenizer, CLIPImageProcessor)

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset import ConcatDataset
from projects.modules import RRRDataset, ADD_TOKENS_DECODER

data_root = '/home/PJLAB/huanghaian/dataset/coco/'

prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = int(2048 - (336 / 14) ** 2)  # TODO

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

for data in rrr_dataset:
    print(data)
