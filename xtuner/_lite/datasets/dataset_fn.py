import os
from datetime import datetime
import torch.distributed as dist
from mmengine.utils import mkdir_or_exist, get_git_hash
import sys
from mmengine.utils.dl_utils import collect_env
from collections import OrderedDict
import shutil
from ..parallel.new_setup import get_dp_mesh, get_world_mesh
from .. import get_logger
from ..parallel import LengthGroupedSampler, ParallelSampler
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import numpy as np
import random
import json
import math
from concurrent.futures import ThreadPoolExecutor
import hashlib
from tqdm import tqdm
from PIL import Image
import copy
from collections.abc import Mapping
import torch

logger = get_logger()

_EXIF_ORIENT = 274  # exif 'Orientation' tag


def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def log_format(rank, debug=False):
    formatter = f'[XTuner][RANK {rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


def check_args(args):
    if args.resume_from and args.resume is False:
        args.resume = True
    if args.resume is True and args.resume_from is None:
        # find last checkpoint
        save_file = os.path.join(args.work_dir, 'last_checkpoint')
        if os.path.exists(save_file):
            with open(save_file) as f:
                args.resume_from = f.read().strip()
        else:
            logger.warning('Did not find last_checkpoint to be resumed. training from scratch.')
            args.resume = False
    if args.resume:
        assert not args.checkpoint_drop_optimizer, '`resume` and `checkpoint_drop_optimizer` cannot be set at the same time.'

    dp_size = get_dp_mesh().size()
    world_size = get_world_mesh().size()
    if args.global_batch_size < dp_size or args.global_batch_size % dp_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         f'should be divisible by the world_size{world_size}.')

    if (args.global_batch_size / dp_size) % args.mirco_batch_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         f'should be divisible by the world_size{world_size}*'
                         f'`mirco_batch_size`({args.mirco_batch_size})')

    if args.group_by_length and args.group_by_modality_length:
        print('if you set both `group_by_length` and `group_by_modality_length`,'
              ' the `group_by_modality_length` will be used.')

    if args.sp_size > 1 and args.mirco_batch_size > 1:
        raise NotImplementedError('Not support mirco_batch_size>1 when sp_size')


def set_logger_envs(args):
    rank = get_world_mesh().get_rank()
    world_size = get_world_mesh().size()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    objects = [timestamp]
    dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    args.work_dir = os.path.join(args.work_dir, timestamp)
    mkdir_or_exist(args.work_dir)

    log_file = os.path.join(args.work_dir, f'rank{rank}.log')

    # Change the log format printed in the terminal
    lvl = 'INFO'
    logger.add(sys.stderr, level=lvl, format=log_format(rank))
    # Change the format saved in the log file
    logger.add(log_file, format=log_format(rank), backtrace=True, catch=True)

    logger.info(args)
    if rank == 0:
        env = collect_env()
        import transformers

        import xtuner
        env['Transformers'] = transformers.__version__
        env['XTuner'] = f'{xtuner.__version__}+{get_git_hash(digits=6)}'
        runtime_env = OrderedDict()
        runtime_env.update(env)
        runtime_env['Seed'] = args.seed
        runtime_env['World Size'] = world_size

        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        logger.info('\n' + dash_line + '\nRuntime environment:' +
                    runtime_env_info + '\n' + dash_line + '\n')

        shutil.copy(__file__, args.work_dir)


class SoftPackDataset(Dataset):

    def __init__(self, datasets, pack_max_length=32768, concat_before_pack=True):
        if concat_before_pack:
            if dist.get_rank() == 0:
                logger.info(f'[Dataset] Concat before pack.')
            num_tokens = [
                np.concatenate([dset.num_tokens for dset in datasets])
            ]
            datasets = [ConcatDataset(datasets)]
        else:
            num_tokens: list = [dset.num_tokens for dset in datasets]
        self.datasets = datasets
        self.pack_max_length = pack_max_length

        pack_infos = []
        self.max_length_per_pack = []
        orig_lens = [len(dset) for dset in datasets]
        for i, dataset in enumerate(self.datasets):
            _infos = self.get_pack_infos(dataset, i, num_tokens[i])
            self.max_length_per_pack.extend([info['max_length_per_pack'] for info in _infos])
            pack_infos.extend(_infos)
        self.pack_infos = pack_infos
        assert len(self.pack_infos) == len(self.max_length_per_pack)

        if dist.get_rank() == 0:
            logger.info(f'[Dataset] (Original) {orig_lens} samples.')
            logger.info(f'[Dataset] (Packed) {len(self)} samples.')

    def get_pack_infos(self, dataset, dataset_id, num_tokens):
        inds = [i for i in range(len(dataset))]
        random.shuffle(inds)

        item_buffer = []
        length_buffer = []
        max_length_one_pack = 0

        pack_infos = []
        for shfl_i in inds:
            if num_tokens[shfl_i] > self.pack_max_length:
                raise ValueError(f'one sample len {num_tokens[shfl_i]} > pack_max_length {self.pack_max_length}. '
                                 'Please increase pack_max_length.')

            if num_tokens[shfl_i] + sum(length_buffer) <= self.pack_max_length:
                item_buffer.append(shfl_i)
                length_buffer.append(num_tokens[shfl_i])
                max_length_one_pack = max(max_length_one_pack,
                                          num_tokens[shfl_i])
            else:
                if len(item_buffer) > 0:
                    info = {
                        'indices': item_buffer,
                        'dataset_id': dataset_id,
                        'max_length_per_pack': int(max_length_one_pack)
                    }
                    pack_infos.append(info)

                item_buffer = [shfl_i]
                length_buffer = [num_tokens[shfl_i]]
                max_length_one_pack = num_tokens[shfl_i]

        if len(item_buffer) > 0:
            info = {
                'indices': item_buffer,
                'dataset_id': dataset_id,
                'max_length_per_pack': int(max_length_one_pack)
            }

            pack_infos.append(info)

        return pack_infos

    def __len__(self):
        return len(self.pack_infos)

    def __getitem__(self, item):
        indices = self.pack_infos[item]['indices']
        dataset_id = self.pack_infos[item]['dataset_id']
        return [self.datasets[dataset_id][i] for i in indices]


def load_json_or_jsonl(json_path):
    if json_path.endswith('.json'):
        with open(json_path) as f:
            data = json.load(f)
    elif json_path.endswith('.jsonl'):
        with open(json_path) as f:
            data = f.readlines()
    else:
        raise ValueError(f'Unsupported file format: {json_path}, '
                         f'only support .json and .jsonl.')
    return data


def calculate_jsonl_sha256(path):
    with open(path, 'rb') as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def calculate_json_sha256(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    hash_object = hashlib.sha256(data)
    hash_hex = hash_object.hexdigest()
    return hash_hex


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class BaseOrigDataset(Dataset):
    def __init__(self,
                 data_name,
                 data,
                 chat_template,
                 tokenizer,
                 max_length,
                 image_token_str='<image>',
                 group_by_length=False,
                 pack_data=False,
                 pack_data_cache_dir=None):
        self.data_name = data_name
        self.max_length = max_length
        self.group_by_length = group_by_length
        self.pack_data = pack_data
        self.pack_data_cache_dir = pack_data_cache_dir
        self.chat_template = chat_template
        self.image_token_str = image_token_str
        self.tokenizer = tokenizer

        self.root = data.get('media_root', '')
        logger.info(f"{dist.get_rank()} ======= Start to process dataset: {os.path.basename(data['annotation'])}")

        self.annotation = data['annotation']
        self._is_jsonl = self.annotation.endswith('.jsonl')
        self.raw_data = load_json_or_jsonl(self.annotation)
        repeat_time = data.get('repeat_time', 1)
        if repeat_time < 1:
            # If repeat_time is less than 1, select a portion of the data
            self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            # Repeat the list if repeat_time is greater than 1
            self.raw_data = self.raw_data * repeat_time

        self.group_length = []
        if self.group_by_length and not pack_data:
            self.group_length = self.calc_group_len()

        # -------------------pack---------------------------------------
        self.num_tokens = None
        self.pack_data_cache_dir = pack_data_cache_dir
        if pack_data:
            assert pack_data_cache_dir is not None, 'pack_data_cache_dir must be provided when pack_data is True'
            self.num_tokens = self.calc_packing_info()

    def __len__(self):
        return len(self.raw_data)

    def calc_group_len(self):
        raise NotImplementedError

    def calc_packing_info(self):
        if os.path.exists(self.pack_data_cache_dir):
            assert os.path.isdir(self.pack_data_cache_dir)
        else:
            mkdir_or_exist(self.pack_data_cache_dir)

        # TODO: more rubost way to calculate the hash
        if self._is_jsonl:
            file_hash = calculate_jsonl_sha256(self.annotation)
        else:
            file_hash = calculate_json_sha256(self.annotation)
        file_cache_dir = os.path.join(self.pack_data_cache_dir, file_hash)
        if not os.path.exists(file_cache_dir):
            mkdir_or_exist(file_cache_dir)

        if 'num_tokens.npy' in os.listdir(file_cache_dir):
            _cached_file = os.path.join(file_cache_dir, 'num_tokens.npy')
            num_tokens = np.load(_cached_file)
            logger.info(f"Load num_tokens from cache: {os.path.basename(self.annotation)}")
        else:
            num_tokens = self.count_tokens_for_pack(file_cache_dir)
        return num_tokens

    def count_tokens_for_pack(self, cache_dir=None):
        num_samples = len(self.raw_data)

        if dist.is_available():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        num_per_rank = math.ceil(num_samples / world_size)

        start = rank * num_per_rank
        end = (rank + 1) * num_per_rank
        dataset_shard = self.raw_data[start:end]

        desc = f'[Rank {rank}] {os.path.basename(self.annotation)}'
        with ThreadPoolExecutor(max_workers=8) as executor:
            tokenized = list(
                tqdm(
                    executor.map(self.pre_tokenize_fn_for_pack, dataset_shard),
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

    def pre_tokenize_fn_for_pack(self, data):
        raise NotImplementedError

    def process_text(self, conversations, media_type='image', image_grids=None):
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]

        assert len(conversations) % 2 == 0, f'Invalid conversation length: {len(conversations)}'

        input_ = ''
        out_conversation = []
        for msg in conversations:
            if msg['from'] == 'human':
                input_ += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input_,
                    'output': msg['value'].strip()
                })
                input_ = ''
            else:
                raise NotImplementedError(f'Unsupported message type: {msg}')

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input_ = single_turn_conversation.get('input', '')
            if input_ is None:
                input_ = ''
            input_ = self.chat_template['user'].format(user=input_)

            if i == 0:
                # 图片占位符只能在第一轮对话中出现
                input_ = self._process_media_format_first_round(input_, media_type, image_grids)
                input_ = self.chat_template['system'] + input_
                input_encode = self.tokenizer.encode(input_, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(input_, add_special_tokens=False)

            input_ids += input_encode
            labels += [-100] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            output_encode = self.chat_template['assistant'].format(assistant=output_text)
            output_encode = self.tokenizer.encode(output_encode, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            logger.info(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}')
        return {'input_ids': input_ids, 'labels': labels}

    def _process_media_format_first_round(self, input_, media_type, image_grids):
        raise NotImplementedError

    @property
    def modality_length(self):
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length


def build_dataset(args, datasets):
    assert len(datasets) > 0, 'No dataset found.'
    if args.dset_pack:
        train_dataset = SoftPackDataset(datasets,
                                        pack_max_length=args.pack_max_length,
                                        concat_before_pack=args.concat_before_pack)
    else:
        train_dataset = ConcatDataset(datasets)
        if dist.get_rank() == 0:
            logger.info(f'[Dataset] (Original) {len(train_dataset)} samples.')
    return train_dataset


def build_train_dataloader(args, train_dataset, collate_fn):
    dp_mesh = get_dp_mesh()
    if args.group_by_length:
        if args.dset_pack:
            length_property = 'max_length_per_pack'
        else:
            length_property = 'length'
        sampler = LengthGroupedSampler(train_dataset, dp_mesh,
                                       args.global_batch_size,
                                       seed=args.seed,
                                       length_property=length_property)
    elif args.group_by_modality_length:
        # 当开启 soft packing 时，暂时不支持模态区分
        if args.dset_pack:
            raise NotImplementedError
        else:
            sampler = LengthGroupedSampler(train_dataset, dp_mesh,
                                           args.global_batch_size,
                                           seed=args.seed,
                                           length_property='modality_length')
    else:
        sampler = ParallelSampler(
            train_dataset, dp_mesh, args.global_batch_size, seed=args.seed, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0)

    if dist.get_rank() == 0:
        logger.info(f'[Dataloader] {len(train_dataloader)} batches.')

    dist.barrier()
    return train_dataloader


def _prepare_input(data, device='cuda'):
    """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(non_blocking=True, **kwargs)
    return data

def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps
