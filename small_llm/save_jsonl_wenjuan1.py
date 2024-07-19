import gzip
import json
import math
import os

from mmengine.dist import get_rank, get_world_size
from mmengine.dist import infer_launcher, init_dist
from mmengine.fileio import get_local_path, list_dir_or_file
from tqdm import tqdm


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data

# srun -p llm_razor -N 2 --gres=gpu:8 --ntasks=16 --cpus-per-task=16 python save_jsonl_wenjuan1.py
if __name__ == '__main__':
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)

    root = 'wanjuan:s3://wanjuan/1.0/nlp/EN/WebText-en/'
    save_dir = '/mnt/hwfile/xtuner/huanghaian/data/llm/wanjuan_1/orig_jsonl/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json_files = list(list_dir_or_file(root))
    print(len(json_files))

    world_size = get_world_size()
    rank = get_rank()

    n_samples = len(json_files)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for i in tqdm(per_rank_ids, desc=f'Rank {rank}'):
        json_file = json_files[i]
        json_file = root + json_file
        with get_local_path(json_file) as local_path:
            output_file = os.path.basename(json_file).replace('.gz', '')
            output_file = os.path.join(save_dir, output_file)
            with gzip.open(local_path, 'rb') as gz_f:
                with open(output_file, 'wb') as out_f:
                    decompressed_data = gz_f.read()
                    out_f.write(decompressed_data)
            print(f'--- rank:{rank} ---done: {output_file}')
