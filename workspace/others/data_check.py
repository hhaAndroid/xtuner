import argparse
import torch
import os

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'True')

import sys
from pathlib import Path
from torch.distributed import init_process_group
from torch.distributed.device_mesh import init_device_mesh
from mmengine.dist import get_rank, get_world_size
from mmengine.runner import set_random_seed

from xtuner.v1.utils import Config
from xtuner.v1.utils import (
    get_logger,
    log_format,
)
from transformers import AutoTokenizer

logger = get_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test dataloader')
    parser.add_argument('--cfg', type=str, default='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/examples/v1/cpt_qwen3vl_8b_config.py')
    args = parser.parse_args()

    backend = "cpu:gloo,cuda:nccl"
    init_process_group(backend=backend)
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    set_random_seed(42)

    dp_size = get_world_size()
    data_mesh = init_device_mesh(
            'cuda',
            (dp_size, 1),
            mesh_dim_names=("dp", "sp"),
        )
    trainer_cfg = Config.fromfile(args.cfg)['trainer']
    tokenizer = AutoTokenizer.from_pretrained(trainer_cfg.tokenizer_path, trust_remote_code=True)
    dataloader = trainer_cfg.dataloader_cfg.build(
        tokenizer=tokenizer,
        dp_mesh=data_mesh["dp"],
        global_batch_size=trainer_cfg.global_batch_size,
        micro_batch_size=1,
        seed=42
    )
    
    log_dir= Path('./work_dirs/logs')

    log_level = os.environ.get("XTUNER_LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(log_dir / f"rank{get_rank()}.log", format=log_format(), backtrace=True, catch=True)
    # Set log level to hide debug output
    logger.add(sys.stderr, format=log_format(rank=get_rank()), level=log_level)
    
    rank = get_rank()
    logger.info(f"Rank {rank} starts to iterate dataloader, len: {len(dataloader)}")
    for step, batch in enumerate(dataloader):
      logger.info(f"rank {rank} step {step}")

