import json

from mmengine.dist import infer_launcher, init_dist, get_rank
from mmengine.runner import set_random_seed
from mmengine.utils import mkdir_or_exist, scandir
from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.datasets.load import load_datasets
from datasets import Dataset
import os
import sys
from datetime import datetime
import torch

logger = get_logger()


def log_format(rank, debug=False):
    formatter = f'[XTuner][RANK {rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


class PretrainTextTokenizeFunction:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token

    def __call__(self, item):
        # text: sky pile 150b
        # content: wanjuan_1
        try:
            text = item['text'] + self.eos_token
        except:
            text = item['content'] + self.eos_token
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        training_data = {
            'input_ids': input_ids,
            # 'labels': input_ids, # save many disk space
            'num_tokens': len(input_ids),
        }
        return training_data


if __name__ == '__main__':
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(42)

    world_size = int(os.environ['WORLD_SIZE'])
    rank = get_rank()

    work_dir = 'work_dirs/llm'

    # dset_cache_dir = '/mnt/hwfile/xtuner/huanghaian/data/llm/SkyPile-150B/dataset_cache/'
    # datasets_roots = '/mnt/hwfile/xtuner/huanghaian/data/llm/SkyPile-150B/data/'
    # 41687436151 tokens = 42b tokens 157GB , 源文件 172GB 125 个文件
    # 32 卡 处理 25 分钟

    dset_cache_dir = '/mnt/hwfile/xtuner/huanghaian/data/llm/wanjuan_1/dataset_cache/'
    datasets_roots = '/mnt/hwfile/xtuner/huanghaian/data/llm/wanjuan_1/orig_jsonl/'
    # 原始文件 435GB，一共 105 个 jsonl, token 后 379 GB
    # 100381726640，100b, 32 卡处理 1 个小时

    datasets = []
    before_count = 0
    for path in scandir(datasets_roots):
        before_count += 1
        jsonl_path = datasets_roots + path
        file_size = os.path.getsize(jsonl_path)
        if file_size > 1000:
            datasets.append(jsonl_path)

    num_workers = 8
    max_length = 32768  # just for logging

    mkdir_or_exist(work_dir)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = os.path.join(work_dir, f'{timestamp}.rank{rank}.log')
    logger.add(sys.stderr, level='DEBUG', format=log_format(rank, True))
    logger.add(log_file, format=log_format(rank), backtrace=True, catch=True)

    logger.info(f'dataset filter, before: {before_count}, after: {len(datasets)}')

    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    tokenize_fn = PretrainTextTokenizeFunction(tokenizer)

    _datasets = load_datasets(
        paths=datasets,
        cache_dir=dset_cache_dir,
        sources=['local'],
        sample_ratios=[1],
        num_proc=num_workers,
        map_fns=[tokenize_fn],
        init_fns=[Dataset.from_list])

    if rank == 0:
        logger.info('======Begin counting the total number of tokens======')
        _path = os.path.join(dset_cache_dir, 'local_infos.json')
        world_cached_infos = json.load(open(_path))

        total_tokens = 0
        for key, value in world_cached_infos.items():
            total_tokens += value['num_tokens']
        logger.info(f'total tokens: {total_tokens}')
