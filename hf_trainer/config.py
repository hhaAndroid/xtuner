# Copyright (c) OpenMMLab. All rights reserved.
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel)

from xtuner.dataset import LLaVADataset
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
from model import HFLLaVAModel
from concat_new_dataset import LENConcatDataset
import torch

training_args = dict(
    optim='adamw_torch',
    remove_unused_columns=False,
    bf16=True,
    do_train=True,
    group_by_length=True,
    learning_rate=2e-5,
    weight_decay=0,
    warmup_ratio=0.03,
    lr_scheduler_type='cosine',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    dataloader_num_workers=4,
    save_steps=1000,
    logging_steps=1,
    save_total_limit=1,
    num_train_epochs=1,
    save_only_model=True,
    deepspeed='zero_stage2_config.json'
)

# Model
llm_name_or_path = '/mnt/hwfile/xtuner/gaojianfei/Phi-3-mini-4k-instruct/models--microsoft--Phi-3-mini-4k' \
                   '-instruct/snapshots/3a811845d89f3c1b3f41b341d0f9f05104769f35'
visual_encoder_name_or_path = '/mnt/hwfile/xtuner/linzhihao/model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
# Specify the pretrained pth
pretrained_pth = '/mnt/petrelfs/huanghaian/code/xtuner/work_dirs/llava_phi3_mini_4k_instruct_clip_vit_large_p14_336_e1_gpu8_pretrain/iter_2181.pth'  # noqa: E501

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

model = dict(
    type=HFLLaVAModel,
    freeze_llm=False,
    freeze_visual_encoder=False,
    pretrained_pth=pretrained_pth,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        torch_dtype=torch.bfloat16,
        pretrained_model_name_or_path=visual_encoder_name_or_path))

# Data
data_root = '/mnt/hwfile/xtuner/linzhihao/dataset/internvl_sft/'

sharegpt4v_caption_data_path = data_root + 'sharegpt4v_instruct_gpt4-vision_cap100k.jsonl'  # noqa: E501
sharegpt4v_caption_image_folder = data_root + 'data'

llava_data_path = data_root + 'llava_instruct_150k_zh.jsonl'
llava_image_folder = data_root + 'data/coco'

sharegpt4v_data_path = data_root + 'sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl'  # noqa: E501
sharegpt4v_image_folder = data_root + 'data'

dvqa_data_path = data_root + 'dvqa_train_200k.jsonl'
dvqa_image_folder = data_root + 'data/dvqa'

chartqa_data_path = data_root + 'chartqa_train_18k.jsonl'
chartqa_image_folder = data_root + 'data/chartqa'

ai2d_data_path = data_root + 'ai2d_train_12k.jsonl'
ai2d_image_folder = data_root + 'data/ai2d'

docvqa_data_path = data_root + 'docvqa_train_10k.jsonl'
docvqa_image_folder = data_root + 'data/docvqa'

geoqa_data_path = data_root + 'geoqa+.jsonl'
geoqa_image_folder = data_root + 'data/geoqa+'

synthdog_data_path = data_root + 'synthdog_en.jsonl'
synthdog_image_folder = data_root + 'data/synthdog-en'

prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = int(4096 - (336 / 14)**2)

cache_root = '/mnt/hwfile/xtuner/huanghaian/phi3_internvl_v12/cache/'

sharegpt4v_caption_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder=cache_root + 'sharegpt4v_caption_dataset',
    data_path=sharegpt4v_caption_data_path,
    image_folder=sharegpt4v_caption_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

llava_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder=cache_root + 'llava_dataset',
    data_path=llava_data_path,
    image_folder=llava_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

sharegpt4v_dataset = dict(  # 有纯文本数据，其他数据集没有
    type=LLaVADataset,
    offline_processed_text_folder=cache_root + 'sharegpt4v_dataset',
    data_path=sharegpt4v_data_path,
    image_folder=sharegpt4v_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

dvqa_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder=cache_root + 'dvqa_dataset',
    data_path=dvqa_data_path,
    image_folder=dvqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

chartqa_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder=cache_root + 'chartqa_dataset',
    data_path=chartqa_data_path,
    image_folder=chartqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

ai2d_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder=cache_root + 'ai2d_dataset',
    data_path=ai2d_data_path,
    image_folder=ai2d_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

docvqa_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder=cache_root + 'docvqa_dataset',
    data_path=docvqa_data_path,
    image_folder=docvqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

geoqa_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder=cache_root + 'geoqa_dataset',
    data_path=geoqa_data_path,
    image_folder=geoqa_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

synthdog_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder=cache_root + 'synthdog_dataset',
    data_path=synthdog_data_path,
    image_folder=synthdog_image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

train_dataset = dict(
    type=LENConcatDataset,
    datasets=[
        sharegpt4v_caption_dataset, llava_dataset, sharegpt4v_dataset,
        dvqa_dataset, chartqa_dataset, ai2d_dataset, docvqa_dataset,
        geoqa_dataset, synthdog_dataset
    ])
