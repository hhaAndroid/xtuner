import argparse
from mmengine.logging import MMLogger
from mmengine.config import Config
import os.path as osp
from mmengine.utils import mkdir_or_exist
import time
import torch
import datetime
from mslm_project.evaluation.utils import get_rank_and_world_size
from xtuner.registry import BUILDER
from tqdm import tqdm
import torch.distributed as dist
from mmengine.dist import broadcast, collect_results, get_rank
import math

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
    model.to(TORCH_DTYPE_MAP[args.torch_dtype])
    model.cuda()
    model.eval()

    eval_dataset = cfg.eval_dataset
    for _, dataset_cfg in enumerate(eval_dataset):
        if rank == 0:
            dataset = BUILDER.build(dataset_cfg)
            objects = [dataset]
        else:
            objects = [None]
        if world_size > 1:
            dist.broadcast_object_list(objects, src=0)
        dataset = objects[0]
        logger.info(f'======== Running on dataset:  {dataset.name}, total samples is {len(dataset)} ===========')

        model.preparing_eval(dataset, max_new_tokens=args.max_new_tokens)

        results = []
        n_samples = len(dataset)
        per_rank_samples = math.ceil(n_samples / world_size)

        per_rank_ids = range(per_rank_samples * rank,
                             min(n_samples, per_rank_samples * (rank + 1)))
        for i in tqdm(per_rank_ids, desc=f'Rank {rank}'):
            data_sample = dataset[i]
            prediction = {}
            prediction['id'] = data_sample['id']
            with torch.no_grad():
                response = model.generate(data_sample)
            prediction['prediction'] = response
            results.append(prediction)

        if world_size > 1:
            dist.barrier()

        results = collect_results(results, len(dataset))

        if get_rank() == 0:
            logger.info(f'======== Starting the evaluation on dataset:  {dataset.name} ===========')
            dataset.postprocess_results(results, args.work_dir, timestamp)

        del dataset
