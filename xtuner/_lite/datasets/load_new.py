# Copyright (c) OpenMMLab. All rights reserved.
import re
import hashlib
import inspect
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from mmengine import mkdir_or_exist
import numpy as np
import torch
from torch import distributed as dist
from tqdm import tqdm
import random

from datasets import Dataset, concatenate_datasets
from torch.utils.data import ConcatDataset

from xtuner._lite import get_logger

logger = get_logger()


def calculate_jsonl_sha256(path):
    with open(path, 'rb') as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def calculate_tokenize_fn_sha256(tokenize_fn):
    """Calculate SHA-256 hash for an instance method's source code."""
    # Get the source code of the method
    fn_source = inspect.getsource(tokenize_fn.__call__)
    return hashlib.sha256(fn_source.encode('utf-8')).hexdigest()


class JsonlDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path,
                 sample_ratio=1.0,
                 tokenize_fn=None,
                 cache_dir=None):
        super().__init__()

        assert sample_ratio <= 1
        self.tokenize_fn = tokenize_fn
        self.path = path

        if cache_dir:
            if os.path.exists(cache_dir):
                assert os.path.isdir(cache_dir)
            else:
                mkdir_or_exist(cache_dir)

            file_hash = calculate_jsonl_sha256(path)
            file_cache_dir = os.path.join(cache_dir, file_hash)

            if file_hash not in os.listdir(cache_dir):
                mkdir_or_exist(file_cache_dir)

            if 'offsets.npy' in os.listdir(file_cache_dir):
                _cached_file = os.path.join(file_cache_dir, 'offsets.npy')
                offsets = np.load(_cached_file)
            else:
                offsets = self.count_offsets(file_cache_dir)

            if self.tokenize_fn:
                tok_hash = calculate_tokenize_fn_sha256(tokenize_fn)
                tok_cache_dir = os.path.join(file_cache_dir, tok_hash)
                if tok_hash not in os.listdir(file_cache_dir):
                    mkdir_or_exist(tok_cache_dir)

                if 'num_tokens.npy' in os.listdir(tok_cache_dir):
                    _cached_file = os.path.join(tok_cache_dir,
                                                'num_tokens.npy')
                    num_tokens = np.load(_cached_file)
                else:
                    num_tokens = self.count_tokens(tok_cache_dir)
            else:
                num_tokens = None

            offsets = offsets
            num_tokens = num_tokens

        else:
            offsets = self.count_offsets()
            num_tokens = None

        num_samples = int(len(offsets) * sample_ratio)
        sampled = random.sample([i for i in range(len(offsets))], num_samples)

        self.offsets = offsets[sampled]
        if num_tokens is not None:
            num_tokens = num_tokens[sampled]

        self.num_tokens = num_tokens

    def count_offsets(self, cache_dir=None):
        offsets = []
        with open(self.path) as f:
            offsets.append(f.tell())
            line = f.readline()
            while line:
                offsets.append(f.tell())
                line = f.readline()

        offsets.pop(-1)
        offsets = np.array(offsets)

        if dist.get_rank() == 0 and cache_dir:
            save_path = os.path.join(cache_dir, 'offsets.npy')
            np.save(save_path, offsets)

        return offsets

    def count_tokens(self, cache_dir=None):

        dataset = []
        with open(self.path) as f:
            for line in f:
                dataset.append(json.loads(line))

        num_samples = len(dataset)

        if dist.is_available():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        num_per_rank = math.ceil(num_samples / world_size)

        start = rank * num_per_rank
        end = (rank + 1) * num_per_rank
        dataset_shard = dataset[start:end]

        desc = f'[Rank {rank}] {self.path}'
        with ThreadPoolExecutor(max_workers=8) as executor:
            tokenized = list(
                tqdm(
                    executor.map(self.tokenize_fn, dataset_shard),
                    desc=desc,
                    total=len(dataset_shard)))

        _num_tokens = [data['num_tokens'] for data in tokenized]
        _num_tokens = np.array(_num_tokens)

        if dist.is_available():
            num_tokens = [None] * world_size
            dist.all_gather_object(num_tokens, _num_tokens)
            num_tokens = np.concatenate(num_tokens, axis=0)
        else:
            num_tokens = _num_tokens

        if rank == 0 and cache_dir:
            save_path = os.path.join(cache_dir, 'num_tokens.npy')
            np.save(save_path, num_tokens)

        return num_tokens

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        with open(self.path) as f:
            f.seek(self.offsets[item])
            line = f.readline()

        raw_data = json.loads(line)

        if self.tokenize_fn:
            tokenized_data = self.tokenize_fn(raw_data)
            return tokenized_data
        else:
            return raw_data


def calculate_json_sha256(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    hash_object = hashlib.sha256(data)
    hash_hex = hash_object.hexdigest()
    return hash_hex


class JsonDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path,
                 sample_ratio=1.0,
                 tokenize_fn=None,
                 cache_dir=None):
        super().__init__()

        assert sample_ratio <= 1
        self.tokenize_fn = tokenize_fn
        self.path = path

        if cache_dir:
            if os.path.exists(cache_dir):
                assert os.path.isdir(cache_dir)
            else:
                mkdir_or_exist(cache_dir)

            file_hash = calculate_json_sha256(path)
            file_cache_dir = os.path.join(cache_dir, file_hash)

            if file_hash not in os.listdir(cache_dir):
                mkdir_or_exist(file_cache_dir)

            if self.tokenize_fn:
                tok_hash = calculate_tokenize_fn_sha256(tokenize_fn)
                tok_cache_dir = os.path.join(file_cache_dir, tok_hash)
                if tok_hash not in os.listdir(file_cache_dir):
                    mkdir_or_exist(tok_cache_dir)

                if 'num_tokens.npy' in os.listdir(tok_cache_dir):
                    _cached_file = os.path.join(tok_cache_dir,
                                                'num_tokens.npy')
                    num_tokens = np.load(_cached_file)
                else:
                    num_tokens = self.count_tokens(tok_cache_dir)
            else:
                num_tokens = None

        else:
            num_tokens = None

        with open(self.path) as f:
            dataset = json.load(f)

        num_samples = int(len(dataset) * sample_ratio)
        sampled = random.sample([i for i in range(len(dataset))], num_samples)
        self.sampled = sampled

        if num_tokens is not None:
            num_tokens = num_tokens[sampled]

        self.num_tokens = num_tokens

        self.dataset = None

    def count_tokens(self, cache_dir=None):

        dataset = []

        with open(self.path) as f:
            dataset = json.load(f)

        num_samples = len(dataset)

        if dist.is_available():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        num_per_rank = math.ceil(num_samples / world_size)

        start = rank * num_per_rank
        end = (rank + 1) * num_per_rank
        dataset_shard = dataset[start:end]

        desc = f'[Rank {rank}] {os.path.basename(self.path)}'
        with ThreadPoolExecutor(max_workers=8) as executor:
            tokenized = list(
                tqdm(
                    executor.map(self.tokenize_fn, dataset_shard),
                    desc=desc,
                    total=len(dataset_shard)))

        _num_tokens = [data['num_tokens'] for data in tokenized]
        _num_tokens = np.array(_num_tokens)

        if dist.is_available():
            num_tokens = [None] * world_size
            dist.all_gather_object(num_tokens, _num_tokens)
            num_tokens = np.concatenate(num_tokens, axis=0)
        else:
            num_tokens = _num_tokens

        if rank == 0 and cache_dir:
            save_path = os.path.join(cache_dir, 'num_tokens.npy')
            np.save(save_path, num_tokens)

        return num_tokens

    def __len__(self):
        return len(self.sampled)

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        if self.dataset is None:
            with open(self.path) as f:
                self.dataset = json.load(f)

        raw_data = self.dataset[self.sampled[item]]

        if self.tokenize_fn:
            tokenized_data = self.tokenize_fn(raw_data)
            return tokenized_data
        else:
            return raw_data


DATASET_CLS_MAP = {'.jsonl': JsonlDataset, '.json': JsonDataset}


def load_local_datasets(paths,
                        file_types,
                        file_pattern=None,
                        cache_dir=None,
                        sample_ratios=1.0,
                        map_fns=None):

    if isinstance(paths, str):
        paths = [paths]

    if isinstance(sample_ratios, (tuple, list)):

        if len(sample_ratios) == 1:
            sample_ratios = list(sample_ratios) * len(paths)

        if len(sample_ratios) != len(paths):
            raise RuntimeError(f'There are {len(paths)} paths, but only '
                               f'{len(sample_ratios)} sample ratios were set.')

    if map_fns is None:
        map_fns = [None] * len(paths)

    if isinstance(map_fns, (tuple, list)):

        if len(map_fns) == 1:
            map_fns = list(map_fns) * len(paths)

        if len(map_fns) != len(paths):
            raise RuntimeError(f'There are {len(paths)} paths, but only'
                               f'{len(map_fns)} map fns were set.')

    files = []
    file_sample_ratios = []
    file_map_fns = []

    for pid, path in enumerate(paths):
        if os.path.isdir(path):
            dir_files = []
            for root, dirs, _files in os.walk(path, followlinks=True):
                dirs.sort()
                for relative_path in sorted(_files):
                    suffix = os.path.splitext(relative_path)[-1]
                    absolute_path = os.path.join(root, relative_path)
                    if file_pattern is not None:
                        if bool(re.match(file_pattern, absolute_path)):
                            dir_files.append(absolute_path)
                    elif suffix in file_types:
                        dir_files.append(absolute_path)

            _num_dir_files = len(dir_files)
            if _num_dir_files == 0:
                raise RuntimeError(
                    f'There are no files with the suffix {file_types}'
                    f'in `{path}`.')

            logger.info(f'Found {len(dir_files)} files in {path}')
            files.extend(dir_files)
            file_sample_ratios.extend([sample_ratios[pid]] * _num_dir_files)
            file_map_fns.extend([map_fns[pid]] * _num_dir_files)

        elif os.path.isfile(path):
            files.append(path)
            file_sample_ratios.append(sample_ratios[pid])
            file_map_fns.append(map_fns[pid])

        else:
            raise RuntimeError(f'`{path}` not found.')

    num_files = len(files)

    datasets = []
    for i in range(num_files):
        _path = files[i]
        _ratio = file_sample_ratios[i]
        _map_fn = file_map_fns[i]
        _suffix = os.path.splitext(_path)[-1]

        dataset_cls = DATASET_CLS_MAP[_suffix]
        _dataset = dataset_cls(_path, _ratio, _map_fn, cache_dir)
        datasets.append(_dataset)

    return datasets


def load_datasets(paths,
                  sources='local',
                  sample_ratios=1.0,
                  file_types=DATASET_CLS_MAP.keys(),
                  file_pattern=None,
                  cache_dir=None,
                  map_fns=None):

    if isinstance(paths, str):
        paths = [paths]

    num_paths = len(paths)

    if isinstance(sample_ratios, (float, int)):
        sample_ratios = [sample_ratios] * num_paths

    if isinstance(sample_ratios, (tuple, list)):

        if len(sample_ratios) == 1:
            sample_ratios = list(sample_ratios) * num_paths

        if len(sample_ratios) != num_paths:
            raise RuntimeError(f'There are {num_paths} paths, but only '
                               f'{len(sample_ratios)} sample ratios were set.')

    if isinstance(sources, str):
        sources = [sources]

    if isinstance(sources, (tuple, list)):

        if len(sources) == 1:
            sources = list(sources) * num_paths

        if len(sources) != num_paths:
            raise RuntimeError(f'There are {num_paths} paths, but only '
                               f'{len(sources)} sources were set.')

    if not isinstance(map_fns, (tuple, list)):
        map_fns = [map_fns] * num_paths

    if isinstance(map_fns, (tuple, list)):

        if len(map_fns) == 1:
            map_fns = list(map_fns) * num_paths

        if len(map_fns) != num_paths:
            raise RuntimeError(f'There are {num_paths} paths, but only'
                               f'{len(map_fns)} map fns were set.')

    local_inds = [i for i, src in enumerate(sources) if src == 'local']
    local_paths = [paths[ind] for ind in local_inds]
    local_map_fns = [map_fns[ind] for ind in local_inds]
    local_sample_ratios = [sample_ratios[ind] for ind in local_inds]

    datasets = []
    if len(local_inds):
        local_datasets = load_local_datasets(local_paths, file_types,
                                             file_pattern, cache_dir,
                                             local_sample_ratios,
                                             local_map_fns)
        datasets.extend(local_datasets)

    return datasets


def load_ms_dataset():
    pass


class SoftPackDataset(torch.utils.data.Dataset):

    def __init__(self, datasets, target=2048, blend=False, sort=False):

        if blend:
            num_tokens = [
                np.concatenate([dset.num_tokens for dset in datasets])
            ]
            datasets = [ConcatDataset(datasets)]
        else:
            num_tokens = [dset.num_tokens for dset in datasets]
        self.datasets = datasets
        self.target = target

        pack_infos = []
        for i, dataset in enumerate(self.datasets):
            _infos = self.get_pack_infos(dataset, i, num_tokens[i])
            pack_infos.append(_infos)
        self.pack_infos = concatenate_datasets(pack_infos)

    def get_pack_infos(self, dataset, dataset_id, num_tokens):
        # _ori_lens = dataset['num_tokens']
        inds = [i for i in range(len(dataset))]
        random.shuffle(inds)

        item_buffer = []
        length_buffer = []
        max_length_one_pack = 0

        pack_infos = []
        for shfl_i in inds:
            if num_tokens[shfl_i] + sum(length_buffer) <= self.target:
                item_buffer.append(shfl_i)
                length_buffer.append(num_tokens[shfl_i])
                max_length_one_pack = max(max_length_one_pack,
                                          num_tokens[shfl_i])
            else:
                if len(item_buffer) > 0:
                    info = {
                        'dataset_id': dataset_id,
                        'indices': item_buffer,
                        'max_length': int(max_length_one_pack)
                    }
                    pack_infos.append(info)

                item_buffer = [shfl_i]
                length_buffer = [num_tokens[shfl_i]]
                max_length_one_pack = num_tokens[shfl_i]

        if len(item_buffer) > 0:
            info = {
                'dataset_id': dataset_id,
                'indices': item_buffer,
                'max_length': int(max_length_one_pack)
            }

            pack_infos.append(info)

        pack_infos = Dataset.from_list(pack_infos)

        return pack_infos

    def __len__(self):
        return len(self.pack_infos)

    def __getitem__(self, item):
        indices = self.pack_infos[item]['indices']
        dataset_id = self.pack_infos[item]['dataset_id']
        return [self.datasets[dataset_id][i] for i in indices]


from torch.nn.utils.rnn import pad_sequence

class SftCollator():

    def __init__(self, pad_token_id=0, ignore_id=-100, pack_batch=False):
        self.pack_batch = pack_batch
        self.pad_token_id = pad_token_id
        self.ignore_id = ignore_id

    def __call__(self, instances):

        _instances = []
        for ins in instances:
            if isinstance(ins, list):
                _instances.extend(ins)
            else:
                _instances.append(ins)

        instances = _instances

        input_ids = []
        labels = []
        num_tokens = []

        for data in instances:

            input_ids.append(torch.LongTensor(data['input_ids']))
            labels.append(torch.LongTensor(data['labels']))

            if isinstance(data['num_tokens'], int):
                num_tokens.append(data['num_tokens'])
            else:
                num_tokens.extend(data['num_tokens'])

        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        num_tokens = torch.IntTensor(num_tokens)

        if len(instances) > 1 and self.pack_batch:

            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            labels = torch.cat(labels, dim=0).unsqueeze(0)
            attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)

        elif len(instances) > 1 and not self.pack_batch:

            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=self.ignore_id)
            attention_mask = pad_sequence(
                attention_mask, batch_first=True, padding_value=0)
        else:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            attention_mask = torch.stack(attention_mask)

        if input_ids.shape != labels.shape:
            logger.error(f'[instances] {instances}')
            logger.error(f'[num_tokens] {num_tokens}')
            logger.error(f'[input_ids] {input_ids}')
            logger.error(f'[labels] {labels}')
            raise RuntimeError('The shape of input_ids and labels must be '
                               f'equal, but  found {input_ids.shape} and '
                               f'{labels.shape}.')
        # TODO support sp
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'num_tokens': num_tokens,
            'attention_mask': attention_mask.bool()
        }

        return data_dict

