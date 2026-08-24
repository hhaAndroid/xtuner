
# mmengine
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import tempfile
import os
from torch.distributed import init_process_group, destroy_process_group

from transformers import Qwen3_5MoeForConditionalGeneration, AutoProcessor
import torch

torch.set_printoptions(precision=8, sci_mode=False)

def forward_hf(path,inputs):
    hf_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_3", # flash_attention_2/3 一开就 Assertion `probability tensor contains either `inf，其余是可以的，默认跑的是 spda
        device_map="cuda",
        trust_remote_code=True
    )
    
    with torch.no_grad():
        output = hf_model(
            input_ids = inputs.input_ids,
            labels = inputs.input_ids.clone(),
            # attention_mask = inputs.attention_mask,
            use_cache = False
        )
    print('hf forward:',output.loss, output.aux_loss)


# torchrun --nproc_per_node=2 xtuner_qwen3_5_ddp.py
if __name__ == '__main__':
    init_process_group(backend='nccl')
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    
    print('local_rank:', int(os.environ["LOCAL_RANK"]))

    debug = False
    # debug = True
    if debug:
        import debugpy
        debugpy.connect(('10.103.23.59', 5680))

    path='/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'
    processor = AutoProcessor.from_pretrained(path,trust_remote_code=True)
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你觉得上海如何？请推荐几个景点？"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages1,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        max_length=2048,      # 你想固定 pad 到的长度
        # padding=True,
    )
    inputs = inputs.to('cuda')
    print(inputs.input_ids, inputs.input_ids.shape)
    forward_hf(path, inputs)

    destroy_process_group()
