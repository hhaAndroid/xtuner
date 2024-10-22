# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import os
import shutil
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import partial
import random
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate.utils import set_module_tensor_to_device
from torch.utils.data import Dataset as TorchDataset
from mmengine import load, mkdir_or_exist
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
from PIL import Image
import json
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_state_dict, set_state_dict)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor)
from xtuner._lite.parallel.new_setup import setup_parallel, \
    get_fsdp_mesh, get_dp_mesh, get_tp_mesh, get_world_mesh, get_sp_mesh, \
    profile_time_and_memory, get_torch_device_module
import numpy as np
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.accelerate import (LoadWoInit,
                                     dispatch_modules, packed_sequence)
from xtuner._lite.chat import CHAT_TEMPLATE_MAP
from xtuner._lite.datasets import (LlavaCollator, LlavaRawDataset,
                                   LlavaTokenizeFunction, SoftPackerForLlava)
from xtuner._lite.datasets.load import (LOAD_FN_MAP, load_datasets,
                                        load_from_cache)
from xtuner._lite.modelings import register_remote_code
from xtuner._lite.parallel import LengthGroupedSampler, ParallelSampler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor, distribute_tensor
from torch.distributed._composable import checkpoint
from torch.distributed._composable.fsdp import fully_shard
from collections.abc import Mapping


def check_qwen_vl_deps_install():
    """check qwen_vl_utils."""
    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        raise ImportError(
            'please install qwen_vl_utils by pip install qwen_vl_utils'  # noqa: E501
        )
    try:
        from transformers import Qwen2VLForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError(
            'please install latest transformers by '
            'pip install git+https://github.com/huggingface/transformers.git')


check_qwen_vl_deps_install()
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')
    model_args = parser.add_argument_group('model', 'Model Related Settings')
    model_args.add_argument(
        '--model', help='repo id or local path of the model')
    model_args.add_argument(
        '--freeze-llm',
        action='store_true',
        help="Not updating LLM's parameters")
    model_args.add_argument(
        '--freeze-vit',
        action='store_true',
        help="Not updating vit's parameters")
    model_args.add_argument(
        '--tp-size',
        default=1,
        type=int,
        help="tp size")
    model_args.add_argument(
        '--sp-size',
        default=1,
        type=int,
        help="sp size")
    model_args.add_argument(
        '--dtype',
        default='auto',
        choices=['fp16', 'bf16', 'auto'],
        help=("the dtype of the model forward. When set to 'auto', it will "
              'automatically determine whether bf16 is available, '
              'prioritizing the use of bf16.'))
    model_args.add_argument(
        '--selective-recompute',
        default=1.0,
        type=float,
        help=('the ratio of re-computation for transforemer layers. '
              'The maximum is 1; the larger the value, the less memory '
              'required for training. The default is 1, meaning all layers '
              'need to be re-computated.'))
    model_args.add_argument(
        '--shard-strategy',
        default='full',
        choices=['full', 'hybrid', 'no', 'zero2'],
        help=('The sharding strategy to be used for distributed training.'))
    data_args = parser.add_argument_group('data', 'Dataset Related Settings')
    data_args.add_argument(
        '--datasets',
        help=('repo id or local path or dir of the datasets. For repo ids, '
              'the `dset-sources` needs to be appropriately set to '
              '`modelscope` or `huggingface`. For local dir, all json and '
              'jsonl files will be loaded by default. The type of loaded '
              'files can be controlled by setting `dset-file-type`'))
    data_args.add_argument(
        '--dset-cache-dir',
        help=('the cache dir of the loaded datasets. When the `datasets` is '
              'set, the loaded datasets will be cached to this dir. If the '
              '`datasets` are not set, the cached dataset in this dir will be '
              'loaded.'))
    data_args.add_argument('--group-by-length', action='store_true')
    data_args.add_argument('--group-by-modality-length', action='store_true')
    data_args.add_argument(
        '--max-length',
        type=int,
        default=32768,
        help=('the maximum length of each piece of data, any excess will be '
              'truncated.'))
    data_args.add_argument(
        '--pack-max-length',
        type=int,
        default=32768,
        help='the maximum length of each pack of data')
    data_args.add_argument(
        '--max-keep-ckpts',
        type=int,
        default=1,
        help='the maximum number of checkpoints to keep.')
    data_args.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='how many subprocesses to use for data loading.')
    data_args.add_argument(
        '--num-proc',
        type=int,
        default=8,
        help='how many subprocesses to use for data mapping.')

    optim_args = parser.add_argument_group('optim', 'Optim Related Settings')
    optim_args.add_argument(
        '--mirco-batch-size',
        type=int,
        default=1,
        help='batch size for each forward + backward pass')
    optim_args.add_argument(
        '--global-batch-size',
        type=int,
        default=16,
        help='batch size for each parameter update')

    optim_args.add_argument(
        '--lr', default=4e-5, type=float, help='learning rate.')
    optim_args.add_argument(
        '--lr-min', default=0, type=float, help='min learning rate.')
    optim_args.add_argument(
        '--wd', default=0.01, type=float, help='weight decay.')
    optim_args.add_argument(
        '--max-grad-norm', default=1, type=float, help='gradient clipping')
    optim_args.add_argument(
        '-e', '--epochs', default=1, type=int, help='total training epochs.')
    optim_args.add_argument(
        '--warmup-ratio',
        default=0.03,
        type=float,
        help=('the proportion of training steps for learning rate warm-up in '
              'relation to the total training steps.'))
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='the dir to save logs and checkpoints')
    parser.add_argument(
        '--checkpoint-interval',
        default=0.25,
        type=float,
        help=('how many steps to save a checkpoint; it can be a floating '
              'point number less than 1, or an integer greater than or equal '
              "to 1. When it's a floating point, it will be multiplied by the "
              'total number of training steps.'))
    parser.add_argument(
        '--checkpoint-drop-optimizer',
        action='store_true',
        help=('only model parameters are saved when saving a checkpoint. '
              'This can significantly reduce the size of checkpoint files, '
              'but the saved checkpoints cannot be resumed.'))
    parser.add_argument(
        '--log-interval', default=1, type=int, help='log interval')
    parser.add_argument(
        '--resume', action='store_true', help='resume from the last checkpoint')
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed for the training')
    args = parser.parse_args()
    return args


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


def map_meta_modules(model, meta_model):
    modules = {name: mod for name, mod in model.named_modules()}
    meta_module_map = {
        mod: modules[name]
        for name, mod in meta_model.named_modules()
    }
    return meta_module_map


class MetaStateful(Stateful):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def state_dict(self):
        return self.kwargs

    def load_state_dict(self, state_dict) -> None:
        self.kwargs = state_dict

    def __getitem__(self, key):
        return self.kwargs[key]


@torch.no_grad
def lazy_init_megatron(module, rank0_map, dp_mesh, tp_mesh=None, pp_mesh=None):
    device = get_torch_device_module().current_device()

    if dp_mesh.get_rank() == 0:
        rank0_module = rank0_map[module]
        rank0_params = {
            name: param
            for name, param in rank0_module.named_parameters(recurse=False)
        }
        rank0_buffers = {
            name: buffer
            for name, buffer in rank0_module.named_buffers(recurse=False)
        }
    else:
        rank0_params = None
        rank0_buffers = None

    param_shapes = {
        name: param.full_tensor().shape
        if isinstance(param, DTensor) else param.shape
        for name, param in module.named_parameters(recurse=False)
    }

    module.to_empty(device=get_torch_device_module().current_device(), recurse=False)

    for name, param in module.named_parameters(recurse=False):
        dtype = param.dtype
        if dp_mesh.get_rank() == 0:
            rank0_param = rank0_params[name].to(device).to(dtype)
        else:
            full_shape = param_shapes[name]
            rank0_param = torch.zeros(full_shape, dtype=dtype, device=device)

        dist.broadcast(rank0_param, src=0)

        if isinstance(param, DTensor):
            mesh = param.device_mesh
            assert mesh == tp_mesh
            placements = param.placements
            rank0_param = distribute_tensor(rank0_param, mesh, placements)

        param.data.copy_(rank0_param)
        dist.barrier()

    # TP does not shard buffers
    for name, buffer in module.named_buffers(recurse=False):
        if dp_mesh.get_rank() == 0:
            rank0_buffer = rank0_buffers[name].to(device)
        else:
            rank0_buffer = torch.empty_like(buffer).to(device)

        dist.broadcast(rank0_buffer, src=0)
        buffer.data.copy_(rank0_buffer)


class LazyQwenVLDataset(TorchDataset):
    def __init__(self, data_name, data, model_name, max_length, group_by_length=False):
        self.data_name = data_name
        _model_cfg = AutoConfig.from_pretrained(model_name)
        self.image_token_id = _model_cfg.image_token_id  # <|image_pad|>
        self.video_token_id = _model_cfg.video_token_id  # <|video_pad|>
        self.image_token_str = '<|vision_start|><|image_pad|><|vision_end|>'
        self.video_token_str = '<|vision_start|><|video_pad|><|vision_end|>'
        self.chat_template = dict(system='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n',
                                  user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
                                  assistant='{assistant}<|im_end|>\n')
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.merge_length = self.processor.image_processor.merge_size ** 2
        self.tokenizer = self.processor.tokenizer

        self.max_length = max_length
        self.group_by_length = group_by_length
        self.root = data.get('media_root', '')

        logger.warning(f"{dist.get_rank()} ======= Start to process dataset: {data['annotation']}")
        assert data['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {data["annotation"]}'

        self.group_length = []
        repeat_time = data.get('repeat_time', 1)
        with open(data['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        if self.group_by_length:
            # 如果是多模态数据，则一定需要有 hw 属性
            # TODO: 支持视频
            for data_item in self.raw_data:
                if ('image' in data_item and data_item['image'] is not None) or (
                        'video' in data_item and data_item['video'] is not None):
                    assert 'image_hw' in data_item[
                        'image'], 'image must have `hw` attribute when group_by_length is True'

            # 由于动态分辨率特点，在开启 group_by_length 时候我们需要通过 image_hw 精确计算实际的 image token
            # 如果采用估算方式会差别较大
            raise NotImplementedError('group_by_length is not supported yet.')

    @property
    def modality_length(self):
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length

    def process_text(self, conversations, image_grid_thw, media_type='image'):
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
                if media_type == 'image':
                    assert '<image>' in input_, f'Image placeholder not found in the first conversation: {input_}'
                    index = 0
                    while '<image>' in input_:
                        input_ = input_.replace("<image>", self.image_token_str, 1)
                        input_ = input_.replace(
                            "<|image_pad|>", "<|placeholder|>" * (image_grid_thw[index].prod() // self.merge_length), 1
                        )
                        index += 1
                    input_ = input_.replace("<|placeholder|>", "<|image_pad|>")
                elif media_type == 'video':
                    raise NotImplementedError

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

    def multi_modal_get_item(self, data_item):
        # Ensure the first conversation contains an image placeholder
        if '<image>\n' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>\n', '')

        if '\n<image>' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('\n<image>', '')

        if '<image>' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>', '')

        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>' + data_item['conversations'][0]['value']

        image_path = data_item['image']
        if isinstance(image_path, list):
            image_path = image_path[0]
        image_path = os.path.join(self.root, image_path)
        image = Image.open(image_path).convert('RGB')
        media_inputs = self.processor.image_processor(images=image, videos=None, return_tensors='pt')
        media_grid_thw = media_inputs['image_grid_thw']

        ret = self.process_text(data_item['conversations'], media_grid_thw, media_type='image')

        out_data = {
            'input_ids': ret['input_ids'],
            'labels': ret['labels'],
            'pixel_values': media_inputs['pixel_values'],
            'image_grid_thw': media_grid_thw,
            'num_token': [len(ret['input_ids'])],
            'num_image_token': [media_grid_thw[0].prod() // self.merge_length + 2]
        }
        return out_data

    def multi_modal_multi_image_get_item(self, data_item):
        raise NotImplementedError

    def video_get_item(self, data_item):
        raise NotImplementedError

    def pure_text_get_item(self, data_item):
        image = Image.new('RGB', (224, 224), (255, 255, 255))
        media_inputs = self.processor.image_processor(images=image, videos=None, return_tensors='pt')
        media_grid_thw = media_inputs['image_grid_thw']

        ret = self.process_text(data_item['conversations'], media_grid_thw, media_type='text')

        out_data = {
            'input_ids': ret['input_ids'],
            'labels': ret['labels'],
            'pixel_values': media_inputs['pixel_values'],
            'image_grid_thw': media_grid_thw,
            'num_token': [len(ret['input_ids'])],
            'num_image_token': [0]
        }
        return out_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item and data_item['image'] is not None:
                    if type(data_item['image']) == list and len(data_item['image']) > 1:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                logger.info(f'Exception: {e} of {self.data_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_dataset(args):
    ds_collections = json.loads(open(args.dataset).read())
    _datasets = []
    for name, _data in ds_collections.items():
        _dataset = LazyQwenVLDataset(name, _data, args.model, args.max_length, group_by_length=args.group_by_length)
        _datasets.append(_dataset)

    assert len(_datasets) > 0, 'No dataset found.'
    train_dataset = ConcatDataset(_datasets)
    return train_dataset


def packing_collate(features, pack_batch=True, pad_id=0):
    input_ids = []
    labels = []
    pixel_values = []
    num_tokens = []
    num_img_tokens = []
    image_grid_thws = []

    for data in features:
        input_ids.append(torch.LongTensor(data['input_ids']))
        labels.append(torch.LongTensor(data['labels']))
        num_tokens.extend(data['num_tokens'])
        num_img_tokens.extend(data['num_img_tokens'])
        pixel_values.append(data['pixel_values'])
        image_grid_thws.append(data['image_grid_thw'])

    attention_mask = [ids.ne(pad_id) for ids in input_ids]
    num_tokens = torch.IntTensor(num_tokens)
    num_img_tokens = torch.IntTensor(num_img_tokens)

    if len(features) > 1 and pack_batch:
        # batch packing
        input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        labels = torch.cat(labels, dim=0).unsqueeze(0)
        attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)
        image_grid_thws = torch.cat(image_grid_thws, dim=0)
        pixel_values = torch.cat(pixel_values, dim=0)
    elif len(features) > 1 and not pack_batch:
        raise NotImplementedError
    else:
        # soft packing
        assert len(features) == 1
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        image_grid_thws = image_grid_thws[0]
        pixel_values = pixel_values[0]

    data_dict = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask.bool(),
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thws,
        'num_tokens': num_tokens,
        'num_img_tokens': num_img_tokens,
    }

    return data_dict


def model_sft(args):
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(args.seed)

    setup_parallel(tp_size=args.tp_size, sp_size=args.sp_size)
    set_random_seed(args.seed)

    dp_mesh = get_dp_mesh()
    tp_mesh = get_tp_mesh()
    sp_mesh = get_sp_mesh()
    fsdp_mesh = get_fsdp_mesh()  # dp_size * sp_size
    world_mesh = get_world_mesh()  # dp_size * sp_size * tp_size

    dp_size = dp_mesh.size()
    tp_size = tp_mesh.size()
    sp_size = sp_mesh.size()
    world_size = world_mesh.size()

    rank = world_mesh.get_rank()

    # 如果 args.resume 和 args.resume_from 同时都设置了，则以 args.resume_from 为准
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

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    objects = [timestamp]
    dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    args.work_dir = os.path.join(args.work_dir, timestamp)
    mkdir_or_exist(args.work_dir)

    log_file = os.path.join(args.work_dir, f'rank{rank}.log')

    # Change the log format printed in the terminal
    lvl = 'DEBUG' if args.debug else 'INFO'
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
        runtime_env['Distributed launcher'] = dist_launcher

        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        logger.info('\n' + dash_line + '\nRuntime environment:' +
                    runtime_env_info + '\n' + dash_line + '\n')

        shutil.copy(__file__, args.work_dir)

    # build dataset
    with profile_time_and_memory('[Dataset & Dataloader]'):
        train_dataset = build_dataset(args)
        logger.warning(f'{dist.get_rank()} ===== End of all dataset =====')

        if args.group_by_length:
            raise NotImplementedError
        elif args.group_by_modality_length:
            raise NotImplementedError
        else:
            sampler = ParallelSampler(
                train_dataset, dp_mesh, args.global_batch_size, seed=args.seed, shuffle=True)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.mirco_batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
            collate_fn=packing_collate,
            persistent_workers=args.num_workers > 0)

        if rank == 0:
            logger.info(f'[Dataset] (Original) {len(train_dataset)} samples.')
            logger.info(f'[Dataloader] {len(train_dataloader)} batches.')

        dist.barrier()

    if args.dtype == 'auto':
        args.dtype = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if args.dtype == 'fp16':
        dtype = torch.float16
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype)
        scaler = ShardedGradScaler()
    elif args.dtype == 'bf16':
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype)
            scaler = None
        else:
            raise RuntimeError('The device does not support `bf16`, '
                               'please set `dtype` to `fp16`.')
    else:
        raise RuntimeError('`dtype` only supports `fp16`，`bf16`, or `auto`, '
                           f'but found {args.dtype}.')

    with profile_time_and_memory('[Model]'):
        with torch.device('meta'):
            with LoadWoInit():
                meta_model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    attn_implementation='flash_attention_2',
                    torch_dtype=dtype)
                for module in meta_model.modules():
                    for p_name, param in module.named_parameters(recurse=False):
                        if param.requires_grad:
                            param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                            setattr(module, p_name, param_fp32)

        dispatch_modules(meta_model)

        if dist.get_rank() == 0:
            logger.info(meta_model)

        timeout = timedelta(
            minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=45)))
        group = dist.new_group(backend='gloo', timeout=timeout)
        if rank == 0:
            # Only load parameters on rank 0 to avoid each rank repeatedly loading
            # the same model into the CPU, wasting memory
            with torch.device('cpu'), profile_time_and_memory('[RANK_0 Load]'):
                with LoadWoInit():
                    rank0_model = AutoModelForCausalLM.from_pretrained(
                        args.model,
                        attn_implementation='flash_attention_2',
                        torch_dtype=dtype)
                    for module in rank0_model.modules():
                        for p_name, param in module.named_parameters(recurse=False):
                            if param.requires_grad:
                                param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                                setattr(module, p_name, param_fp32)
        else:
            rank0_model = None
        dist.monitored_barrier(group=group, timeout=timeout)

        if dp_mesh.get_rank() == 0:
            rank0_map = map_meta_modules(rank0_model, meta_model)
        else:
            rank0_map = None

        param_init_fn = partial(
            lazy_init_megatron,
            rank0_map=rank0_map,
            dp_mesh=dp_mesh,
            tp_mesh=tp_mesh,
        )

        if args.shard_strategy == 'full':
            reshard_after_forward = True
        else:
            reshard_after_forward = False

        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

        num_layers = len(meta_model.model.layers)
        num_recompute_layers = int(num_layers * 1.0)
        for i, block in enumerate(meta_model.model.layers):
            block.apply(param_init_fn)
            # # As an optimization, do not reshard after forward for the last
            # # transformer block since FSDP would prefetch it immediately
            # if i < num_layers - 1:
            #     _reshard = reshard_after_forward
            # else:
            #     _reshard = False

            fully_shard(
                block,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )

            if i < num_recompute_layers:
                checkpoint(block)

        meta_model.model.embed_tokens.apply(param_init_fn)
        meta_model.model.norm.apply(param_init_fn)

        model = fully_shard(
            meta_model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward)  # False is zero2, True is zero3
        model.train()

        if dist.get_rank() == 0:
            logger.info(model)

    requried_grad_params = [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = AdamW(
        requried_grad_params, lr=args.lr, weight_decay=args.wd, fused=True)

    max_memory = get_torch_device_module().max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024**3):.1f}GB')

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    # `iter` means once forward+backward
    # `step` means once optimizer step
    # `per_step_iters` means gradient accumulative counts
    per_step_iters = global_batch_size // mirco_batch_size // dp_size
    per_epoch_iters = len(train_dataloader)
    per_epoch_steps = math.ceil(per_epoch_iters / per_step_iters)

    total_epochs = args.epochs
    total_steps = per_epoch_steps * total_epochs

    if args.checkpoint_interval == -1:
        checkpoint_interval = total_steps
    elif args.checkpoint_interval < 1:
        checkpoint_interval = int(total_steps * args.checkpoint_interval)
    else:
        checkpoint_interval = int(args.checkpoint_interval)

    warmup_steps = int(args.warmup_ratio * total_steps)

    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr_min)

    start_step = 0

    if args.resume:
        logger.info(f'[Resume] Resume from {args.resume_from}')
        _options = StateDictOptions(
            cpu_offload=True, ignore_frozen_params=True)
        (shard_model_state_dict,
         shard_optimizer_state_dict) = get_state_dict(
            model, optimizer, options=_options)
        meta_stateful = MetaStateful(step=start_step, total_steps=total_steps)
        state_dict = {
            'model': shard_model_state_dict,
            'optimizer': shard_optimizer_state_dict,
            'meta_stateful': meta_stateful,
            'warmup_scheduler': warmup_scheduler,
            'cosine_scheduler': cosine_scheduler
        }
        # inplace state_dict
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=args.resume_from,
        )

        _options = StateDictOptions(
            cpu_offload=True, strict=False)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
            options=_options
        )

        start_step = meta_stateful['step']
        logger.info(f'[Resume] start_step to {start_step}')

    start_train_t = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024 ** 3):.1f}GB')
    save_hf_ckpt_names = []
    save_pt_ckpt_names = []
    max_keep_ckpts = args.max_keep_ckpts
    if max_keep_ckpts <= 0:
        # 全部都保存
        max_keep_ckpts = 100000000

    for step in range(start_step, total_steps):

        epoch = step // per_epoch_steps
        epoch_inner_step = step % per_epoch_steps
        if epoch_inner_step == 0 or step == start_step:
            # For the first step of each epoch, the data order needs to be
            # readjusted.
            # Or after resuming, for the first step, the dataloader needs to
            # be adjusted to the position before resume.
            # train_dataloader.sampler.set_epoch(epoch, inner_step)
            train_dataloader.sampler.set_epoch(epoch, epoch_inner_step)
            data_iterator = iter(train_dataloader)

        if step <= warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_lr()[0]

        torch.cuda.reset_peak_memory_stats()

        step_loss = 0
        step_data_time = 0
        step_start_t = time.time()
        step_consumed_tokens = 0
        step_consumed_img_tokens = 0
        for _ in range(per_step_iters):

            _data_start_t = time.time()
            data = next(data_iterator)
            step_data_time += time.time() - _data_start_t

            data = _prepare_input(data)
            num_tokens = data.pop('num_tokens')
            num_img_tokens = data.pop('num_img_tokens')

            packed_ctx = packed_sequence(num_tokens, enable=True)

            with packed_ctx:
                outputs = model(**data)
                avg_iter_loss = outputs.loss / per_step_iters

                if scaler:
                    scaler.scale(avg_iter_loss).backward()
                else:
                    avg_iter_loss.backward()

            step_loss += avg_iter_loss.item()
            step_consumed_tokens += num_tokens.sum()
            step_consumed_img_tokens += num_img_tokens.sum()

        grad_norm = model.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        step_text_tokens = step_consumed_tokens - step_consumed_img_tokens
        step_img_tokens = step_consumed_img_tokens
        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        tgs = int(step_consumed_tokens / step_time)
        max_memory = torch.cuda.max_memory_allocated()
        if is_interval(step, total_steps, args.log_interval):
            logger.info(
                f'[Train] (Epoch {epoch}) Step {step + 1}/{total_steps}  '  # noqa: E501
                f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                f'grad_norm: {grad_norm:.2f}  '
                f'max_memory: {(max_memory / 1024 ** 3):.1f}GB  '
                f'text_tokens: {step_text_tokens}  '
                f'image_tokens: {step_img_tokens}  '
                f'tgs: {tgs}  data_time: {step_data_time:.2f}s  '
                f'time: {step_time:.2f}s  '
                f'eta: {eta}')

        if is_interval(step, total_steps, checkpoint_interval):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            max_memory = torch.cuda.max_memory_allocated()
            logger.info('[Checkpoint] Before saving checkpoint, the peak GPU '
                        f'memory is {max_memory / 1024 ** 3:.1f}GB.')

            digits = len(str(abs(total_steps)))
            work_dir = args.work_dir

            ckpt_id = f'{(step + 1):0{digits}}-of-{total_steps:0{digits}}'
            ckpt_dir = os.path.join(work_dir, f'ckpt-{ckpt_id}')
            hf_dir = os.path.join(work_dir, f'hf-{ckpt_id}')
            _options = StateDictOptions(cpu_offload=True, full_state_dict=True)

            full_model_state_dict = get_model_state_dict(
                model, options=_options)
            if rank == 0:
                saved_llava = copy.deepcopy(rank0_model)
                saved_llava.to(dtype)
                for name, param in full_model_state_dict.items():
                    set_module_tensor_to_device(saved_llava, name, 'cpu',
                                                param)

                if args.llm_use_lora:
                    merged_llm = saved_llava.language_model.merge_and_unload()
                    saved_llava.language_model = merged_llm

                if args.vit_use_lora:
                    merged_vit = saved_llava.vision_tower.merge_and_unload()
                    saved_llava.vision_tower = merged_vit

                saved_llava.save_pretrained(hf_dir)
                del saved_llava

            dist.barrier()
            del full_model_state_dict

            if rank == 0:
                save_hf_ckpt_names.append(hf_dir)
                if len(save_hf_ckpt_names) > max_keep_ckpts:
                    # 移除最先加入的
                    remove_hf_ckpt_name = save_hf_ckpt_names.pop(0)
                    os.system(f'rm -rf {remove_hf_ckpt_name}')

            if args.checkpoint_drop_optimizer:
                logger.warning('[Checkpoint] The saved checkpoint cannot be '
                               'resumed. If you want to save a resumable '
                               'checkpoint, please remove '
                               '`--checkpoint-drop-optimizer` '
                               'from the command.')
            else:
                # FSDP cannot be saved via torch.save
                # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
                _options = StateDictOptions(
                    cpu_offload=True, ignore_frozen_params=True)
                (shard_model_state_dict,
                 shard_optimizer_state_dict) = get_state_dict(
                    model, optimizer, options=_options)
                meta_stateful = MetaStateful(step=step + 1, total_steps=total_steps)
                # warmup_scheduler/cosine_scheduler 并不是 Stateful 对象，但是是 work 的，
                # 怀疑是这个对象有啥特殊，存储后就变成了 Stateful 对象
                # 常数对象没有这个功能，需要自己封装
                state_dict = {
                    'model': shard_model_state_dict,
                    'optimizer': shard_optimizer_state_dict,
                    'meta_stateful': meta_stateful,
                    'warmup_scheduler': warmup_scheduler.state_dict(),
                    'cosine_scheduler': cosine_scheduler.state_dict()
                }
                dcp.save(state_dict, checkpoint_id=ckpt_dir)

                if rank == 0:
                    # 只能存储到 work_dir/../last_checkpoint 这，否则不好找
                    save_file = os.path.join(args.work_dir, '../', 'last_checkpoint')
                    with open(save_file, 'w') as f:
                        f.write(ckpt_dir)  # type: ignore

                    save_pt_ckpt_names.append(ckpt_dir)
                    if len(save_pt_ckpt_names) > max_keep_ckpts:
                        # 移除最先加入的
                        remove_pt_ckpt_name = save_pt_ckpt_names.pop(0)
                        os.system(f'rm -rf {remove_pt_ckpt_name}')

                max_memory = torch.cuda.max_memory_allocated()
                logger.info(
                    '[Checkpoint] During saving checkpoint, the peak GPU '
                    f'memory is {max_memory / 1024 ** 3:.1f}GB.')

    train_cost_time = time.time() - start_train_t
    m, s = divmod(train_cost_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    logger.info("[Train] Cost: %d day, %d:%d:%d" % (d, h, m, s))
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':
    args = parse_args()
    model_sft(args)
