# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel)

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from peft import LoraConfig
from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler

from projects.modules import ADD_TOKENS_DECODER, RRRModel, RRREvaluateChatHook, RRRDataset, withbbox_default_collate_fn

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = 'internlm/internlm2-chat-7b'
visual_encoder_name_or_path = 'openai/clip-vit-large-patch14-336'
# llm_name_or_path = 'model/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f'
# visual_encoder_name_or_path = 'model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
pretrained_pth = '/mnt/petrelfs/huanghaian/code/mm/xtuner/work_dirs/rrr_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain/iter_2181.pth'  # noqa: E501

# Data
data_root = 'data/coco/'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = int(2048 - (336 / 14) ** 2 - 1)

# Scheduler & Optimizer
batch_size = 16  # per_device
accumulative_counts = 1
dataloader_num_workers = 0
max_epochs = 1
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''
# https://llava-vl.github.io/static/images/view.jpg
evaluation_images = '000000190753.jpg'
IMAGE_SIZE = 672
input_1 = ["<image>\nIn the conversation below, you simply answer the category name based on what you see in the "
           "imagery inside a region. If you don't find the category name in the provided list of categories, "
           "you should output other. I will input one or multiple regions and provide a list of categories. Please "
           "reply strictly in the order of the input. Categories: tv,oven,frisbee,toothbrush,truck,toilet,banana,"
           "umbrella,fork,dog,stop sign,suitcase,cell phone,baseball glove,toaster,bowl,surfboard,handbag,bench,"
           "parking meter,boat,car,donut,hair drier,bird,book,sandwich,cup,laptop,fire hydrant,horse,sheep,person,"
           "tie,cake,teddy bear,bus,apple,bear,kite,bed,scissors,cat,clock,remote,tennis racket,spoon,bicycle,bottle,"
           "mouse,cow,skis,knife,couch,elephant,chair,snowboard,pizza,wine glass,baseball bat,traffic light,"
           "potted plant,motorcycle,broccoli,giraffe,backpack,sink,sports ball,hot dog,airplane,microwave,"
           "refrigerator,vase,orange,train,carrot,skateboard,keyboard,zebra,dining table. Region: "
           "1.<region_feat>;2.<region_feat>;3.<region_feat>;4.<region_feat>;5.<region_feat>.",
           [[255, 350, 349, 477], [61, 360, 144, 408], [401, 411, 479, 507], [235, 352, 281, 411], [175, 342, 231, 516]]]
input_2 = ["<image>\nIn the conversation below, you simply answer the category name based on what you see in the "
           "imagery inside a region. I will input multiple regions. Please reply strictly in the order of the input. "
           "Region: 1.<region_feat>;2.<region_feat>;3.<region_feat>;4.<region_feat>.",
           [[401, 411, 479, 507], [255, 350, 349, 477], [244, 378, 462, 540], [455, 352, 552, 445]]]
# bench
evaluation_inputs = [input_1, input_2]

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    added_tokens_decoder=ADD_TOKENS_DECODER,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    do_center_crop=False,
    do_resize=False,
    trust_remote_code=True)

model = dict(
    type=RRRModel,
    freeze_llm=True,
    use_visual_sampler=True,
    pretrained_pth=pretrained_pth,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    llm_lora=dict(
        type=LoraConfig,
        r=512,
        lora_alpha=256,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        dict(
            type=RRRDataset,
            data_root=data_root,
            ann_file='annotations/instances_train2017_rrrvlm_ovd1.json',
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
            ann_file='annotations/instances_train2017_rrrvlm_region1.json',
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

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=withbbox_default_collate_fn))

# to inference demo
inference_dataset = dict(
    type=RRRDataset,
    data_root=data_root,
    ann_file='annotations/instances_val2017_rrrvlm_ovd1.json',
    data_prefix=dict(img='val2017/'),
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        T_max=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=RRREvaluateChatHook,
        tokenizer=tokenizer,
        image_processor=image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed environment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
