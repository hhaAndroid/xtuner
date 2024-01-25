# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel)

from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine import DatasetInfoHook, EvaluateChatHook
from xtuner.model import LLaVAModel
from xtuner.utils import PROMPT_TEMPLATE
from peft import LoraConfig

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = 'model/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f'
visual_encoder_name_or_path = 'model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

# pretrain model
# model = dict(
#     type=LLaVAModel,
#     freeze_llm=True,
#     freeze_visual_encoder=True,
#     llm=dict(
#         type=AutoModelForCausalLM.from_pretrained,
#         pretrained_model_name_or_path=llm_name_or_path,
#         trust_remote_code=True,
#         torch_dtype=torch.float32,  # torch.float16
#         # 如果不注释运行会报错
#         # quantization_config=dict(
#         #     type=BitsAndBytesConfig,
#         #     load_in_4bit=True,
#         #     load_in_8bit=False,
#         #     llm_int8_threshold=6.0,
#         #     llm_int8_has_fp16_weight=False,
#         #     bnb_4bit_compute_dtype=torch.float16,
#         #     bnb_4bit_use_double_quant=True,
#         #     bnb_4bit_quant_type='nf4')
#     ),
#     visual_encoder=dict(
#         type=CLIPVisionModel.from_pretrained,
#         pretrained_model_name_or_path=visual_encoder_name_or_path))

# qlora finetune
model = dict(
    type=LLaVAModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    pretrained_pth=None,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        # quantization_config=dict(
        #     type=BitsAndBytesConfig,
        #     load_in_4bit=True,
        #     load_in_8bit=False,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type='nf4')
    ),
    llm_lora=dict(
        type=LoraConfig,
        r=512,
        lora_alpha=256,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path),
    visual_encoder_lora=dict(
        type=LoraConfig, r=64, lora_alpha=16, lora_dropout=0.05, bias='none'))

class_type = model.pop('type')
model_clazz = class_type(**model).cuda()

print(model_clazz)

import torch

data = {'data': {'input_ids': torch.tensor([[1, 92543, 1008, 364, -200, 364, 3993, 505, 410, 4065,
                                             8305, 12386, 435, 410, 2321, 345, 92542, 364, 92543, 525,
                                             11353, 364, 918, 4065, 8305, 505, 12386, 395, 15605, 454,
                                             395, 18277, 435, 410, 2321, 281, 92542, 364, 92543, 1008,
                                             364, 4027, 410, 4065, 8305, 11410, 607, 11849, 345, 92542,
                                             364, 92543, 525, 11353, 364, 772, 410, 2321, 328, 410,
                                             4065, 8305, 505, 11410, 435, 4223, 446, 395, 19441, 6160,
                                             281, 92542, 364, 92543, 1008, 364, 4500, 505, 410, 4065,
                                             8305, 725, 7636, 435, 410, 6704, 345, 92542, 364, 92543,
                                             525, 11353, 364, 918, 4065, 8305, 505, 3422, 395, 3740,
                                             454, 36013, 10041, 528, 650, 410, 6405, 435, 410, 6704,
                                             281, 92542, 364]]).cuda(), 'attention_mask': torch.tensor(
    [[True, True, True, True, True, True, True, True, True, True, True, True,
      True, True, True, True, True, True, True, True, True, True, True, True,
      True, True, True, True, True, True, True, True, True, True, True, True,
      True, True, True, True, True, True, True, True, True, True, True, True,
      True, True, True, True, True, True, True, True, True, True, True, True,
      True, True, True, True, True, True, True, True, True, True, True, True,
      True, True, True, True, True, True, True, True, True, True, True, True,
      True, True, True, True, True, True, True, True, True, True, True, True,
      True, True, True, True, True, True, True, True, True, True, True, True,
      True, True, True, True, True]]).cuda(),
                 'labels': torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                          -100, -100, 918, 4065, 8305, 505, 12386, 395, 15605, 454,
                                          395, 18277, 435, 410, 2321, 281, 92542, -100, -100, -100,
                                          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                          -100, -100, -100, -100, -100, 772, 410, 2321, 328, 410,
                                          4065, 8305, 505, 11410, 435, 4223, 446, 395, 19441, 6160,
                                          281, 92542, -100, -100, -100, -100, -100, -100, -100, -100,
                                          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                                          -100, -100, -100, 918, 4065, 8305, 505, 3422, 395, 3740,
                                          454, 36013, 10041, 528, 650, 410, 6405, 435, 410, 6704,
                                          281, 92542, -100]]).cuda()}, 'data_samples': None}

data['data']['pixel_values'] = torch.ones([1, 3, 336, 336]).cuda()

with torch.no_grad():
    loss = model_clazz(data['data'])
    print(loss)

