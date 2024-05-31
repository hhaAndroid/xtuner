# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch.nn as nn
from mmengine import MessageHub, print_log
from mmengine.dist import get_rank


def add_plora_to_linear(module, lora_r=256, lora_alpha=256, lora_dropout=0.05):
    device = module.weight.device
    dtype = module.weight.dtype
    Plora_A = nn.Linear(
        module.in_features, lora_r, bias=False, device=device, dtype=dtype)
    Plora_B = nn.Linear(
        lora_r, module.out_features, bias=False, device=device, dtype=dtype)
    nn.init.kaiming_uniform_(Plora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(Plora_B.weight)

    lora_dropout = nn.Dropout(p=lora_dropout)
    lora_scaling = lora_alpha / lora_r

    module.add_module('Plora_A', Plora_A)
    module.add_module('Plora_B', Plora_B)
    module.add_module('lora_dropout', lora_dropout)
    setattr(module, 'lora_scaling', lora_scaling)

    def forward_plora(self, x):
        res = self.forward_original(x)
        rank = get_rank()
        message_hub = MessageHub.get_instance('im_mask_info')
        im_mask = message_hub.get_info(f'im_mask_{rank}')
        # if rank in [0, 1]:
        #     print('*****', flush=True)
        #     print(rank, flush=True)
        #     print(im_mask.sum(-1), flush=True)
        #     print('*****', flush=True)
        if im_mask is not None and x.shape[1] == im_mask.shape[-1]:
            part_x = x[im_mask]
            res[im_mask] += self.Plora_B(self.Plora_A(self.lora_dropout(part_x))) * self.lora_scaling
        return res

    module.forward_original = module.forward
    module.forward = forward_plora.__get__(module, nn.Linear)


def add_plora(model, lora_r=256, lora_alpha=256, lora_dropout=0.05):
    for name, module in model.named_modules():
        if (isinstance(module, nn.Linear) and 'Plora' not in name
                and 'lm_head' not in name and 'output_layer' not in name):
            print_log(f'Add PLoRA to {name}', 'current')
            add_plora_to_linear(
                module,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout)
