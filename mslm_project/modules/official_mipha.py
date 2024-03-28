# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training

from xtuner.registry import BUILDER
from xtuner.model.modules import ProjectorConfig, ProjectorModel, dispatch_modules
from xtuner.model.utils import (LoadWoInit, find_all_linear_names,
                                get_peft_model_state_dict, guess_load_checkpoint,
                                make_inputs_require_grad,
                                prepare_inputs_labels_for_multimodal, traverse_dict)
from transformers import GenerationConfig
from xtuner.tools.utils import get_stop_criteria
import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, SiglipImageProcessor
from .others import MiphaPhiConfig, MiphaPhiForCausalLM


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="cuda",
                          device="cuda"):
    kwargs = {"device_map": device_map}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    print("load model from model_path: ", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    print("load Mipha-Phi MSLM!!!")
    config = MiphaPhiConfig.from_pretrained(model_path, trust_remote_code=True)
    model = MiphaPhiForCausalLM.from_pretrained(
        model_path,
        config=config,
        use_safetensors=True,
         **kwargs).to("cuda")

    image_processor = SiglipImageProcessor.from_pretrained(model_path)

    context_len = 2048
    model.to(device="cuda")
    print(kwargs)
    # print(model)
    return tokenizer, model, image_processor, context_len

from .others import KeywordsStoppingCriteria

class OfficialMipha(BaseModel):

    def __init__(self, model_path):
        super().__init__()
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)

    def forward(self,
                *args, **kwargs):
        pass

    def load_custom_weights(self, pretrained_pth):
        pass

    def preparing_eval(self, eval_dataset, max_new_tokens=100):
        pass

    def generate(self, data, data_samples=None):
        stop_str = "<|endoftext|>"
        # data 是单张图片的数据
        data.pop('id', None)
        input_ids = data['input_ids'].unsqueeze(0).cuda()
        pixel_values = data['pixel_values'].unsqueeze(0).half().cuda()

        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=pixel_values,
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=self.tokenizer.eos_token_id,  # Pad token
                stopping_criteria=[stopping_criteria]
            )

        predict = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        predict = predict.strip()
        if predict.endswith(stop_str):
            predict = predict[:-len(stop_str)]
        predict = predict.strip()
        return predict









