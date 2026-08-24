import argparse
from transformers import AutoTokenizer
import time
from xtuner.v1.utils import Config
import torch
from torch.distributed import init_process_group
import os
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import init_device_mesh
from xtuner.v1.utils import (
    get_logger,
    log_format,
)
import sys
import re

# export XTUNER_TOKENIZE_WORKERS=1
# export PYTHONPATH="$(pwd)"
# export WORK_DIR="work_dirs_local/qwen35b/sft_all_local"
# export TOKENIZER_CACHE_DIR="./workspace/tokenizer_cache"
# export META_DATA_PATH="/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/meta_data/interns2_pre_base05_20260424a_agent.json"
# torchrun --nproc-per-node=8 debug_dataloader.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test dataloader')
    parser.add_argument('--cfg', type=str,
                        default='/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/qwen35/sft_qwen35vl_35b_config.py')
    args = parser.parse_args()

    trainer_cfg = Config.fromfile(args.cfg)['trainer']

    backend = "cpu:gloo,cuda:nccl"
    init_process_group(backend=backend)
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    
    rank = int(os.environ["LOCAL_RANK"])
    seed=42

    set_random_seed(seed)
    data_mesh = init_device_mesh(
            'cuda',
            (1, 1, 1),
            mesh_dim_names=("dp", "sp", "tp"),
        )

    tokenizer = AutoTokenizer.from_pretrained(trainer_cfg.tokenizer_path, trust_remote_code=True)

    dataloader = trainer_cfg.dataloader_cfg.build(
            tokenizer=tokenizer,
            dp_mesh=data_mesh["dp"],
            global_batch_size=1,
            micro_batch_size=1, # 假设梯度累计是 1
            seed=seed,
            total_step=8000,
        )
    
    rank = int(os.environ["LOCAL_RANK"])
    DEVICE= 'cuda'
    print(f"==================Rank {rank} len of dataloader: {len(dataloader)}=========================")
    for i, batch in enumerate(dataloader):
        assert len(batch) == 1
        seq_ctx = batch[0]['seq_ctx']
        input_ids = seq_ctx.input_ids

        input_str = tokenizer.decode(input_ids)

        print(f"=======Step {i}, Rank {rank}==========\n")
        # print(input_str)
        pattern = r'<think>[\s\S]*?<tool_call>[\s\S]*?</think>'
        matches = re.findall(pattern, input_str)
        if len(matches) > 0:
            print(input_str)
            raise ValueError('Found <think>\n<tool_call> in input_str')
        torch.distributed.barrier()

