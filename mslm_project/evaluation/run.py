import argparse
from mmengine.logging import MMLogger
from mmengine.config import Config
import os.path as osp
from mmengine.utils import mkdir_or_exist
import time
from mmengine.dist import broadcast
import torch
import datetime
import torch.distributed as dist
from mslm_project.evaluation.utils import get_rank_and_world_size
from xtuner.registry import BUILDER
from tqdm import tqdm
from mmengine.dist import (collect_results, get_rank)

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument('--work-dir', type=str, default='.', help='select the output directory')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    parser.add_argument('--nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=100,
        help='Maximum number of new tokens allowed in generated text')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)

    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))

    cfg = Config.fromfile(args.config)
    mkdir_or_exist(osp.abspath(args.work_dir))

    timestamp = torch.tensor(time.time(), dtype=torch.float64)
    # broadcast timestamp from 0 process to other processes
    broadcast(timestamp)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
    filename_no_ext = osp.splitext(osp.basename(cfg.filename))[0]
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    log_cfg = dict(log_level='INFO', log_file=log_file)
    log_cfg.setdefault('name', filename_no_ext)
    log_cfg.setdefault('file_mode', 'a')
    logger = MMLogger.get_instance(**log_cfg)

    logger.info(f'num worker:  {world_size}')

    # bulid model
    model = BUILDER.build(cfg.model)
    # 模型自身处理权重加载问题
    model.load_custom_weights(args.checkpoint)
    model = model.cuda().to(dtype=args.torch_dtype)

    eval_dataset = cfg.eval_dataset
    for _, dataset_cfg in enumerate(eval_dataset):
        logger.info(f'Running on dataset {dataset_cfg.name}')

        if rank == 0:
            dataset = BUILDER.build(dataset_cfg)
        if world_size > 1:
            dist.barrier()

        # 模型自己准备好评测前的事宜
        model.preparing_eval(dataset=dataset, max_new_tokens=args.max_new_tokens)

        results = []
        sheet_indices = list(range(rank, len(dataset), world_size))
        lt = len(sheet_indices)
        for i in tqdm(range(lt), desc=f'Rank {rank}'):
            data_sample = dataset[i]
            prediction = {}
            with torch.no_grad():
                # 模型生成，返回响应
                response = model.generate(data_sample)
            prediction['prediction'] = response
            prediction['index'] = data_sample['index']
            results.append(prediction)

        if world_size > 1:
            dist.barrier()

        results = collect_results(results, len(dataset))

        if get_rank() == 0:
            # 数据集进行后处理，可选的评估
            dataset.postprocess_results(results, args.work_dir, timestamp)

        del dataset
