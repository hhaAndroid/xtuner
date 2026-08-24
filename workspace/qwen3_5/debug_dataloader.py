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

# export XTUNER_TOKENIZE_WORKERS=1
# export PYTHONPATH="$(pwd)"
# export WORK_DIR="work_dirs_local/qwen35b/sft_all_local"
# export META_DATA_PATH="/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/meta_data/internss1_1_tiny_local_1.json"
# torchrun --nproc-per-node=8 debug_dataloader.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test dataloader')
    parser.add_argument('--cfg', type=str,
                        default='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/configs/sft_qwen35vl_35b_config.py')
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
            (8, 1, 1),
            mesh_dim_names=("dp", "sp", "tp"),
        )

    tokenizer = AutoTokenizer.from_pretrained(trainer_cfg.tokenizer_path, trust_remote_code=True)

    dataloader = trainer_cfg.dataloader_cfg.build(
            tokenizer=tokenizer,
            dp_mesh=data_mesh["dp"],
            global_batch_size=8,
            micro_batch_size=1, # 假设梯度累计是 1
            seed=seed,
            total_step=8000,
        )
    
    rank = int(os.environ["LOCAL_RANK"])
    DEVICE= 'cuda'
    print(f"==================Rank {rank} len of dataloader: {len(dataloader)}=========================")
    for i, batch in enumerate(dataloader):
        seq_ctx = batch[0]['seq_ctx']
        
        efficient_forward_tokens = torch.tensor(0, device=DEVICE, dtype=torch.long)
        total_forward_tokens = torch.tensor(0, device=DEVICE, dtype=torch.long)
        img_efficient_forward_tokens = torch.tensor(0, device=DEVICE, dtype=torch.long)
        img_total_forward_tokens = torch.tensor(0, device=DEVICE, dtype=torch.long)

        num_tokens = seq_ctx.cu_seq_lens_k[1:] - seq_ctx.cu_seq_lens_k[:-1]
        num_tokens_sum = num_tokens.long().sum()
        efficient_forward_tokens += (num_tokens.long() ** 2).sum()
        total_forward_tokens += (num_tokens.long().sum()) ** 2

        num_img_tokens = torch.tensor(seq_ctx.num_img_tokens) # list[int]
        num_img_tokens_sum = num_img_tokens.long().sum()
        img_efficient_forward_tokens += (num_img_tokens.long() ** 2).sum()
        img_total_forward_tokens += (num_img_tokens.long().sum()) ** 2

        efficient_attn_ratio = efficient_forward_tokens.float() / total_forward_tokens.float()
        img_efficient_attn_ratio = img_efficient_forward_tokens.float() / (img_total_forward_tokens.float()+1e-8)
        
        print('==============================================================')
        print(f"Step {i}, Rank {rank}, num_tokens_sum: {num_tokens_sum},  num_img_tokens_sum: {num_img_tokens_sum}, LLM Efficient Attn Ratio: {efficient_attn_ratio.item():.3f}, Image Efficient Attn Ratio: {img_efficient_attn_ratio.item():.3f}")
        torch.distributed.barrier()

