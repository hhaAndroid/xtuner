# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import os
import shutil
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import partial
import json
from typing import Dict
import gc
from collections.abc import Mapping
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from copy import deepcopy
import torch
from PIL import Image
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate.utils import set_module_tensor_to_device
from torch.utils.data import Dataset
from mmengine import load, mkdir_or_exist
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    apply_activation_checkpointing
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_state_dict, set_state_dict)
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import _or_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModel)
from xtuner._lite.datasets.load import (load_from_cache)
from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.accelerate import (LoadWoInit,
                                     dispatch_modules, packed_sequence)
from xtuner._lite.accelerate.fsdp import (RECOMPUTE_MODULES,
                                          checkpoint_check_fn, dp_lazy_init,
                                          layer_auto_wrap_policy)
from xtuner._lite.parallel import LengthGroupedSampler, ParallelSampler
from torch.distributed.checkpoint.stateful import Stateful
from xtuner._lite.internvl.constants import IMG_CONTEXT_TOKEN
from xtuner._lite.internvl.dataset import (dynamic_preprocess, preprocess,
                                           preprocess_internlm, preprocess_mpt,
                                           preprocess_phi3, build_transform, preprocess_phi3_fast,
                                           TCSLoader, concat_pad_data_collator,
                                           packing_collate, dynamic_num_patch)
from xtuner._lite.internvl.modeling_intern_vit import InternVisionModel
from datasets import Dataset as HF_Dataset
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from xtuner._lite.datasets.text import SoftPackerForText

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config

    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Model Related Settings')

    # pretrain
    model_args.add_argument('--llm', help='repo id or local path of the model')
    model_args.add_argument(
        '--vit',
        default='openai/clip-vit-large-patch14-336',
        help='repo id or local path of the model')
    model_args.add_argument(
        '--projector',
        default=None,
        help='pretrained projector model')
    # sft
    model_args.add_argument(
        '--internvl', help='repo id or local path of the model')

    model_args.add_argument(
        '--chat-template',
        choices=['phi3-chat'],
        help='')
    model_args.add_argument(
        '--use-fast-tokenizer',
        action='store_true',
        help="")
    model_args.add_argument(
        '--use-orig',
        action='store_true',
        help="")
    model_args.add_argument(
        '--freeze-llm',
        action='store_true',
        help="Not updating LLM's parameters")
    model_args.add_argument(
        '--freeze-vit',
        action='store_true',
        help="Not updating vit's parameters")
    model_args.add_argument(
        '--dtype',
        default='auto',
        choices=['fp16', 'bf16', 'auto'],
        help=("the dtype of the model forward. When set to 'auto', it will "
              'automatically determine whether bf16 is available, '
              'prioritizing the use of bf16.'))
    model_args.add_argument(
        '--drop-path-rate',
        type=float,
        default=0.1,
        help='dataset meta path')
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
        '--meta-path',
        default=None,
        help='dataset meta path')
    data_args.add_argument(
        '--dset-pack-level',
        choices=['soft'],
        help=('the level of data packing. When `hard`, multiple data will be '
              'packed to `max_length`, potentially causing some data to be '
              'truncated, and the length of the packed data will always '
              'be `max_length`; When `soft`, it will pack multiple  data '
              'into nearly `max_length` without truncating the data.'))
    data_args.add_argument(
        '--dset-cache-dir',
        help=('the cache dir of the loaded datasets. When the `datasets` is '
              'set, the loaded datasets will be cached to this dir. If the '
              '`datasets` are not set, the cached dataset in this dir will be '
              'loaded.'))
    data_args.add_argument(
        '--dset-from-cache',
        action='store_true',
        help=('Load data directly from `dset-cache-dir`. This can save time '
              'on online tokenization, but if the tokenizer changed, '
              'recaching is needed.'))
    data_args.add_argument('--group-by-length', action='store_true')
    data_args.add_argument('--group-by-modality-length', action='store_true')
    data_args.add_argument(
        '--max-length',
        type=int,
        default=8192,
        help=('the maximum length of each piece of data, any excess will be '
              'truncated.'))
    data_args.add_argument(
        '--pack-max-length',
        type=int,
        default=8192,
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

    parser.add_argument('-c', '--config', default=None)
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
    parser.add_argument(
        '--debug', action='store_true', help='Set logger level to `DEBUG`')
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


def build_model(args, config, dtype=torch.bfloat16, tokenizer=None, device='cpu'):
    with torch.device(device):
        _cfg = copy.deepcopy(config)

        vision_model = None
        if args.vit is not None:
            logger.info('Loading pretrained vision model...')
            _cfg.vision_config.drop_path_rate = float(args.drop_path_rate)
            with LoadWoInit():
                vision_model = InternVisionModel.from_pretrained(args.vit, torch_dtype=dtype, config=_cfg.vision_config)

        llm = None
        if args.llm is not None:
            logger.info('Loading pretrained LLM model...')
            llm_config = AutoConfig.from_pretrained(args.llm, trust_remote_code=True)
            llm_config.use_cache = False
            if llm_config.model_type == 'internlm2':
                llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            else:
                llm_config._attn_implementation = 'flash_attention_2'
            with LoadWoInit():
                llm = AutoModelForCausalLM.from_pretrained(
                    args.llm, config=llm_config, torch_dtype=dtype, trust_remote_code=True)

            old_vocab_size = llm.config.vocab_size
            if old_vocab_size != len(tokenizer):
                logger.info('Resizing token embeddings...')
                llm.resize_token_embeddings(len(tokenizer))
                num_new_tokens = max(len(tokenizer) - old_vocab_size, 0)
                if num_new_tokens > 0:
                    output_embeddings = llm.get_output_embeddings().weight.data
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg

        # 暂时不加载权重
        model = AutoModel.from_config(config=_cfg,
                                      vision_model=vision_model,
                                      language_model=llm,
                                      torch_dtype=dtype,
                                      trust_remote_code=True)
        if device != 'meta':
            with LoadWoInit():
                if args.projector is not None:
                    logger.info('Loading pretrained MLP projector...')
                    state_dict = torch.load(args.projector, map_location='cpu')
                    message = model.mlp1.load_state_dict(state_dict)
                    logger.info(message)
        model.to(dtype)

        if args.freeze_llm:
            model.language_model = model.language_model.eval()
            model.language_model.requires_grad_(False)
        if args.freeze_vit:
            model.vision_model = model.vision_model.eval()
            model.vision_model.requires_grad_(False)

    return model


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


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            template_name,
            meta,
            tokenizer,
            tcs_loader,
            ds_name,
            num_image_token,
            image_size=224,
            is_train=True,
            pad2square=False,
            group_by_length=False,
            dynamic_image_size=False,
            use_thumbnail=False,
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            min_num_frame=4,  # for video data
            max_num_frame=12,  # for video data
            sampling_method='rand',  # for video data
            repeat_time=1,
            normalize_type='imagenet',
            random_seed=0,
            use_fast_tokenizer=False
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.use_fast_tokenizer = use_fast_tokenizer
        logger.warning(f"{dist.get_rank()} ======= Start to process dataset: {meta['annotation']}")
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        gc.collect()
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)
        gc.collect()

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            if self.use_fast_tokenizer:
                preprocess_function = preprocess_phi3_fast
            else:
                preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if "s3://" in image_path:  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    # 单图
    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>\n' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>\n', '')

        if '\n<image>' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('\n<image>', '')

        if '<image>' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>', '')

        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = data_item['image']
        if isinstance(image_path, list):
            image_path = image_path[0]

        image_path = os.path.join(self.root, image_path)
        if "s3://" in image_path:
            image = self.tcs_loader(image_path)
        else:
            image = Image.open(image_path).convert('RGB')

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_tokens=[len(ret['input_ids'][0])],
            num_img_tokens=[self.num_image_token * num_patches]
        )
        return ret

    # TODO
    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=self.max_dynamic_patch // num_image,
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, num_image=num_image)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    # TODO
    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens)

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, num_image=num_patches)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            num_tokens=[len(ret['input_ids'][0])],
            num_img_tokens=[0]
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
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
                print(e, self.ds_name, flush=True)
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(
        args,
        tokenizer,
        tcs_loader,
        model,
        group_by_length=False,
        dynamic_image_size=True,
        force_image_size=448,
        use_thumbnail=True,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        normalize_type='imagenet',
        use_fast_tokenizer=False
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(args.meta_path).read())

    _dataset_list = []
    _dataset_lengths = []
    for _, ds_name in enumerate(ds_collections.keys()):
        _data = ds_collections[ds_name]
        _repeat_time = _data.get('repeat_time', 1)
        _dataset_lengths.append(_repeat_time * _data['length'])
        _dataset_list.append([ds_name, ds_collections[ds_name]])

    # 按照长度对数据集进行从大到小排序
    combined = list(zip(_dataset_lengths, _dataset_list))
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
    _, _dataset_list = zip(*sorted_combined)
    _dataset_list = list(_dataset_list)

    if dist.is_available():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    # 切分到不同卡
    _dataset_list = _dataset_list[rank::world_size]
    logger.info(f'[{rank}] Assigned Files: {[_dataset[0] for _dataset in _dataset_list]}')
    timeout = timedelta(
        minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=45)))
    group = dist.new_group(backend='gloo', timeout=timeout)
    for i, _dataset in enumerate(_dataset_list):
        ds_name, ds_data = _dataset
        ds_idx = i * world_size + rank
        repeat_time = ds_data['repeat_time']
        if 'max_dynamic_patch' in ds_data:
            max_num = ds_data['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            args.chat_template, ds_data,
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=force_image_size,
            is_train=ds_data['data_augment'],
            pad2square=False,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            random_seed=ds_idx,
            use_fast_tokenizer=use_fast_tokenizer
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if False and args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if False and args.use_data_resampling:
        raise NotImplementedError
    else:
        if dist.is_available():
            buffers = [None] * world_size
            dist.all_gather_object(buffers, datasets, group=group)
            # 拉平
            datasets = [item for sublist in buffers for item in sublist]
            train_dataset = ConcatDataset(datasets)
        else:
            train_dataset = ConcatDataset(datasets)
    return train_dataset


class InternVLDatasetFunForPacking:
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            template_name,
            meta,
            tokenizer,
            tcs_loader,
            ds_name,
            num_image_token,
            image_size=224,
            is_train=True,
            pad2square=False,
            group_by_length=False,
            dynamic_image_size=False,
            use_thumbnail=False,
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            min_num_frame=4,  # for video data
            max_num_frame=12,  # for video data
            sampling_method='rand',  # for video data
            repeat_time=1,
            normalize_type='imagenet',
            random_seed=0,
            root=None,
            use_fast_tokenizer=True,
    ):
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.use_fast_tokenizer = use_fast_tokenizer
        logger.warning(f"{dist.get_rank()} ======= Start to process dataset: {meta['annotation']}")
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        self.root = root
        self.tcs_loader = tcs_loader
        self.dynamic_image_size = dynamic_image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            if self.use_fast_tokenizer:
                preprocess_function = preprocess_phi3_fast
            else:
                preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function

    def multi_modal_get_item(self, data_item):
        # Ensure the first conversation contains an image placeholder
        if '<image>\n' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>\n', '')

        if '\n<image>' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('\n<image>', '')

        if '<image>' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>', '')

        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        image_path = data_item['image']
        if isinstance(image_path, list):
            image_path = image_path[0]
        image_path = os.path.join(self.root, image_path)
        # Merge the image path
        if 'image_wh' not in data_item:
            warnings.warn(f'No image_wh in {self.ds_name}. this is not recommended')
            try:
                if "s3://" in image_path:
                    image = self.tcs_loader(image_path)
                else:
                    image = Image.open(image_path).convert('RGB')
                image_size = image.size
            except Exception as e:
                print(e, self.ds_name, flush=True)
                # 只是为了防止报错，实际上会剔除
                ret = dict(
                    input_ids=torch.tensor([0], dtype=torch.long),
                    labels=torch.tensor([0], dtype=torch.long),
                    num_tokens=[0],
                    image_path=image_path
                )
                return ret
        else:
            image_size = data_item['image_wh']
            image_size = image_size[0]

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            num_patches = dynamic_num_patch(image_size, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                            image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            num_patches = 1

        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=True, ds_name=self.ds_name)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            num_tokens=[len(ret['input_ids'][0])],
            image_path=image_path
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * 1], text_only=True,
                                  group_by_length=True, ds_name=self.ds_name)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            num_tokens=[len(ret['input_ids'][0])],
            image_path=None
        )
        return ret

    def __call__(self, item):
        data_item = json.loads(item)
        if 'image' in data_item and data_item['image'] is not None:
            if type(data_item['image']) == list and len(data_item['image']) > 1:
                raise NotImplementedError
            else:
                ret = self.multi_modal_get_item(data_item)
        elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
            raise NotImplementedError
        else:
            ret = self.pure_text_get_item(data_item)

        return ret


class SoftPackerForInternVL(SoftPackerForText):

    def __init__(self,
                 dataset,
                 max_length=2048,
                 pack_info=None,
                 tcs_loader=None,
                 dynamic_image_size=False,
                 is_train=False,
                 image_size=448,
                 pad2square=False,
                 normalize_type='imagenet',
                 min_dynamic_patch=1,
                 max_dynamic_patch=12,
                 use_thumbnail=True):
        super().__init__(dataset, max_length, pack_info)
        self._cached = False
        self.tcs_loader = tcs_loader
        self.dynamic_image_size = dynamic_image_size
        self.is_train = is_train
        self.image_size = image_size
        self.pad2square = pad2square
        self.normalize_type = normalize_type
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def __getitem__(self, item):
        transform = self.get_transform()

        packed_items = self.idx_per_pack[item]
        assert len(packed_items) > 0

        packed_input_ids = []
        packed_labels = []
        packed_num_tokens = []
        packed_num_img_tokens = []
        packed_pixel_values = []
        packed_image_flags = []

        for i in packed_items:
            data = self.dataset[i]

            # 错误数据删除
            if data['num_tokens'] == 0:
                continue

            # 图片如果读取错误，这条数据也不要了
            image_path = data.get('image_path', None)
            if image_path is not None:
                # 多模态数据
                assert isinstance(image_path, str), f"image_path should be str, but got {data['image_path']}"
                try:
                    if "s3://" in image_path:
                        image = self.tcs_loader(image_path)
                    else:
                        image = Image.open(image_path).convert('RGB')

                    if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                                    max_num=self.max_dynamic_patch,
                                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                    else:  # Otherwise, use the original image as a single patch
                        images = [image]
                    pixel_values = [transform(image) for image in images]
                    pixel_values = torch.stack(pixel_values)  # n,c,h,w
                    packed_pixel_values.append(pixel_values)
                    num_patches = pixel_values.size(0)
                    packed_image_flags.append(torch.tensor([1] * num_patches, dtype=torch.long))
                except Exception as e:
                    print(e, data, flush=True)
                    continue
            else:
                # 纯文本数据
                image = Image.new('RGB', (224, 224), (255, 255, 255))
                if self.dynamic_image_size:
                    # Dynamically preprocess the image to generate patches
                    images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                                image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                else:
                    images = [image]

                # Apply the transformation to each image patch and stack them into a tensor
                pixel_values = [transform(image) for image in images]
                pixel_values = torch.stack(pixel_values)
                num_patches = pixel_values.size(0)
                assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
                packed_pixel_values.append(pixel_values)
                packed_image_flags.append(torch.tensor([0] * num_patches, dtype=torch.long))

            packed_input_ids.extend(data['input_ids'])
            packed_labels.extend(data['labels'])

            _num_tokens = data['num_tokens']
            packed_num_tokens.append(_num_tokens)

            if data['num_img_tokens'] is not None and sum(data['num_img_tokens']) > 0:
                _num_img_tokens = data['num_img_tokens']
                packed_num_img_tokens.append(_num_img_tokens)

        pixel_values = torch.cat(packed_pixel_values, dim=0)
        image_flags = torch.cat(packed_image_flags, dim=0)

        if sum(packed_num_tokens) < self.max_length:
            # TODO: 是否能加速，存在疑问？
            num_pad_tokens = self.max_length - sum(packed_num_tokens)
            packed_input_ids.extend([DEFAULT_PAD_TOKEN_INDEX] * num_pad_tokens)
            packed_labels.extend([IGNORE_INDEX] * num_pad_tokens)
            packed_num_tokens.append(num_pad_tokens)
        else:
            packed_num_tokens.append(0)

        packed = {
            'input_ids': packed_input_ids,
            'labels': packed_labels,
            'pixel_values': pixel_values,
            'image_flags': image_flags,
            'num_tokens': packed_num_tokens,
            'num_img_tokens': packed_num_img_tokens
        }

        return packed


def multi_thread_map(map_fns, dataset, desc, num_proc=8):
    if not isinstance(map_fns, (tuple, list)):
        map_fns = [map_fns]

    def sequential_map(item):
        for fn in map_fns:
            item = fn(item)
        return item

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        results = list(
            tqdm(
                executor.map(sequential_map, dataset),
                desc=desc,
                total=len(dataset)))

    return results


def build_packing_datasets(
        args,
        tokenizer,
        tcs_loader,
        model,
        group_by_length=False,
        dynamic_image_size=True,
        force_image_size=448,
        use_thumbnail=True,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        normalize_type='imagenet',
        use_fast_tokenizer=True,
):
    ds_collections = json.loads(open(args.meta_path).read())

    if dist.is_available():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    if args.dset_from_cache:
        logger.info(f'[{rank}] Load datasets from cache: {args.dset_cache_dir}')
        logger.warning('Warning: Please ensure that the cached data and the model '
                       'correspond to each other, otherwise unexpected behavior may occur.')
        datasets = load_from_cache(args.dset_cache_dir)
        dist.barrier()
    else:
        # cache dataset
        datasets = []
        lengths = []
        _dataset_list = []
        _dataset_lengths = []
        for _, ds_name in enumerate(ds_collections.keys()):
            _data = ds_collections[ds_name]
            _repeat_time = _data.get('repeat_time', 1)
            _dataset_lengths.append(_repeat_time * _data['length'])
            _dataset_list.append([ds_name, ds_collections[ds_name]])

        _ds_names = [ds[0] for ds in _dataset_list]
        assert len(_ds_names) == len(set(_ds_names)), f"Dataset names should be unique, but got {_ds_names}"

        # 按照长度对数据集进行从大到小排序
        combined = list(zip(_dataset_lengths, _dataset_list))
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
        _, _dataset_list = zip(*sorted_combined)
        _dataset_list = list(_dataset_list)
        num_files = len(_dataset_list)

        # 切分到不同卡
        _dataset_list = _dataset_list[rank::world_size]
        logger.info(f'[{rank}] Assigned Files: {[_dataset[0] for _dataset in _dataset_list]}')
        timeout = timedelta(
            minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=45)))
        group = dist.new_group(backend='gloo', timeout=timeout)
        for i, _dataset in enumerate(_dataset_list):
            ds_name, ds_data = _dataset
            ds_idx = i * world_size + rank
            repeat_time = ds_data['repeat_time']
            if 'max_dynamic_patch' in ds_data:
                max_num = ds_data['max_dynamic_patch']
                logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
            else:
                max_num = max_dynamic_patch

            with open(ds_data['annotation'], 'r') as f:
                raw_data = f.readlines()
                if repeat_time < 1:
                    # If repeat_time is less than 1, select a portion of the data
                    raw_data = raw_data[:int(len(raw_data) * repeat_time)]
                if repeat_time > 1:
                    assert isinstance(repeat_time, int)
                    # Repeat the list if repeat_time is greater than 1
                    raw_data = raw_data * repeat_time
            map_fn = InternVLDatasetFunForPacking(
                args.chat_template, ds_data,
                tokenizer,
                tcs_loader,
                ds_name=ds_name,
                num_image_token=model.num_image_token,
                image_size=force_image_size,
                is_train=ds_data['data_augment'],
                pad2square=False,
                group_by_length=group_by_length,
                dynamic_image_size=dynamic_image_size,
                use_thumbnail=use_thumbnail,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_num,
                repeat_time=repeat_time,
                normalize_type=normalize_type,
                random_seed=ds_idx,
                root=ds_data['root'],
                use_fast_tokenizer=use_fast_tokenizer,
            )
            desc = f'[RANK {rank}] Map local file {ds_name}'
            dset = multi_thread_map(map_fn, raw_data, desc, 8)
            logger.debug(f'[File {ds_name}] Mapped Sample:\n{dset[0]}')
            dataset = HF_Dataset.from_list(dset)
            if args.dset_cache_dir:
                digits = len(str(abs(num_files)))
                cache_id = (f'{ds_name}_cache-local-{ds_idx:0{digits}}-of-'
                            f'{num_files:0{digits}}')
                sub_cache_dir = os.path.join(args.dset_cache_dir, cache_id)
                if os.path.exists(sub_cache_dir):
                    shutil.rmtree(sub_cache_dir)
                    logger.warning(f'Found {sub_cache_dir} exists. '
                                   'Clear it and re-cache.')
                dataset.save_to_disk(sub_cache_dir)
            logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
            datasets.append([ds_name, dataset])
            if False and args.use_data_resampling:
                lengths.append(math.sqrt(len(dataset)))
            else:
                lengths.append(len(dataset))

        if dist.is_available():
            buffers = [None] * world_size
            dist.all_gather_object(buffers, datasets, group=group)
            # 拉平
            datasets = [item for sublist in buffers for item in sublist]

    # soft packing
    logger.info(f'[{rank}] Start to soft pack datasets')
    ds_names = [ds[0] for ds in datasets]
    datasets = [ds[1] for ds in datasets]

    orig_dataset_len = sum([len(d) for d in datasets])

    _datasets = []
    pack_infos = SoftPackerForInternVL.get_pack_infos(datasets, args.pack_max_length)
    for i in range(len(datasets)):
        _infos = pack_infos[i]
        _dset = datasets[i]
        _ds_name = ds_names[i]
        ds_data = ds_collections[_ds_name]

        if 'max_dynamic_patch' in ds_data:
            max_num = ds_data['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch

        _packed_dset = SoftPackerForInternVL(_dset, args.pack_max_length, _infos,
                                             tcs_loader=tcs_loader,
                                             dynamic_image_size=dynamic_image_size,
                                             is_train=ds_data['data_augment'],
                                             image_size=force_image_size,
                                             pad2square=False,
                                             normalize_type=normalize_type,
                                             min_dynamic_patch=min_dynamic_patch,
                                             max_dynamic_patch=max_num,
                                             use_thumbnail=use_thumbnail)

        _datasets.append(_packed_dset)
    logger.info(f'[{rank}] Soft packing done')

    if False and args.use_data_resampling:
        raise NotImplementedError
    else:
        train_dataset = ConcatDataset(_datasets)

    if rank == 0:
        logger.info(f'[Dataset] (Original) {orig_dataset_len} samples.')
        logger.info(f'[Dataset] (Packed) {len(train_dataset)} samples.')
    return train_dataset


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


# @logger.catch
def internvl_train(args):
    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(args.seed)

    world_size = int(os.environ['WORLD_SIZE'])
    dp_size = world_size

    assert args.pack_max_length % args.max_length == 0

    if args.projector is not None:
        is_pretrain = False
        logger.info(f'============ SFT mode ============')
    else:
        is_pretrain = True
        raise NotImplementedError

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

    device_mesh = init_device_mesh(
        'cuda', (dp_size,), mesh_dim_names=('dp',))

    dp_mesh = device_mesh['dp']

    rank = dp_mesh.get_local_rank()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    objects = [timestamp]
    dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    args.work_dir = os.path.join(args.work_dir, timestamp)
    mkdir_or_exist(args.work_dir)

    log_file = os.path.join(args.work_dir, f'rank{rank}.log')

    # Change the log format printed in the terminal
    lvl = 'DEBUG' if args.debug else 'INFO'
    logger.add(sys.stderr, level=lvl, format=log_format(rank, args.debug))
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
    # -------------------    Environment  End  ------------------------------ #
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

    logger.info(f'Using Fast Tokenzier: {args.use_fast_tokenizer} for training.')
    tokenizer = AutoTokenizer.from_pretrained(args.internvl, use_fast=args.use_fast_tokenizer, trust_remote_code=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    internvl_config = AutoConfig.from_pretrained(args.internvl, trust_remote_code=True)
    internvl_config.llm_config.use_cache = False
    meta_internvl = build_model(args,
                                internvl_config,
                                dtype,
                                tokenizer=tokenizer,
                                device='meta')
    meta_internvl.img_context_token_id = img_context_token_id

    for module in meta_internvl.modules():
        for p_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                setattr(module, p_name, param_fp32)

    exclude_cls = None
    if args.use_orig:
        # 保持原始逻辑不变
        logger.info('Use original style.')
        exclude_cls = ['Phi3FlashAttention2']
    dispatch_modules(meta_internvl, exclude_cls)

    ###########################################################################
    #                     2. Dataset & Dataloader                             #
    ###########################################################################
    start_load_data_t = time.time()
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if args.dset_cache_dir and os.path.isdir(args.dset_cache_dir):
        if len(os.listdir(args.dset_cache_dir)):
            logger.warning(f'`{args.dset_cache_dir}` is not an empty '
                           'folder, which may lead to inaccurate '
                           'cache results.')
    if args.dset_pack_level:
        assert args.use_fast_tokenizer, 'Packing dataset only supports fast tokenizer.'
        if args.dset_from_cache:
            if not os.path.exists(args.dset_cache_dir):
                logger.warning(f'`{args.dset_cache_dir}` does not exist, re-cache the dataset.')
                args.dset_from_cache = False
        train_dataset = build_packing_datasets(args, tokenizer, tcs_loader, meta_internvl,
                                               args.group_by_length,
                                               force_image_size=internvl_config.force_image_size,
                                               dynamic_image_size=internvl_config.dynamic_image_size,
                                               use_thumbnail=internvl_config.use_thumbnail,
                                               min_dynamic_patch=internvl_config.min_dynamic_patch,
                                               max_dynamic_patch=internvl_config.max_dynamic_patch,
                                               use_fast_tokenizer=args.use_fast_tokenizer)
    else:
        train_dataset = build_datasets(args, tokenizer, tcs_loader, meta_internvl,
                                       args.group_by_length,
                                       force_image_size=internvl_config.force_image_size,
                                       dynamic_image_size=internvl_config.dynamic_image_size,
                                       use_thumbnail=internvl_config.use_thumbnail,
                                       min_dynamic_patch=internvl_config.min_dynamic_patch,
                                       max_dynamic_patch=internvl_config.max_dynamic_patch,
                                       use_fast_tokenizer=args.use_fast_tokenizer)
        if rank == 0:
            logger.info(f'[Dataset] (Original) {len(train_dataset)} samples.')
    logger.warning(f'{dist.get_rank()} ===== End of all dataset =====')

    if args.group_by_length:
        if args.dset_pack_level:
            length_property = 'max_length_per_pack'
        else:
            length_property = 'length'
        sampler = LengthGroupedSampler(train_dataset, dp_mesh,
                                       args.global_batch_size,
                                       seed=args.seed,
                                       length_property=length_property)
    elif args.group_by_modality_length:
        raise NotImplementedError
    else:
        sampler = ParallelSampler(
            train_dataset, dp_mesh, args.global_batch_size, seed=args.seed, shuffle=True)

    if args.use_orig:
        collate_fn = concat_pad_data_collator
    else:
        collate_fn = packing_collate
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0)

    if rank == 0:
        logger.info(f'[Dataloader] {len(train_dataloader)} batches.')
        _first_batch = [train_dataset[i] for i in range(args.mirco_batch_size)]
        # _first_batch = concat_pad_data_collator(_first_batch)
        _first_batch = packing_collate(_first_batch)
        _decoded = tokenizer.batch_decode(_first_batch['input_ids'])
        logger.debug(f'[Dataloader] Training Batch:\n{_first_batch}')
        logger.debug(f'[Dataloader] Training Batch(Decoded):\n{_decoded}')
    dist.barrier()

    load_data_cost_time = time.time() - start_load_data_t
    logger.info(f'[Dataset & Dataloader] Cost {load_data_cost_time:.2f}s')
    # -------------------    Dataset & Dataloader  End  --------------------- #

    ###########################################################################
    #                          3. FSDP                                        #
    ###########################################################################
    start_model_t = time.time()

    # Only load parameters on rank 0 to avoid each rank repeatedly loading the
    # same model into the CPU, wasting memory
    if rank == 0:
        logger.info(f'=====[Build Model]=======')
        internvl = build_model(args,
                               internvl_config,
                               dtype,
                               tokenizer=tokenizer,
                               device='cpu')
        internvl.img_context_token_id = img_context_token_id
        rank0_meta_llava = copy.deepcopy(meta_internvl)
        meta_llava_map = map_meta_modules(internvl, meta_internvl)
        logger.info(f'=====trainable parametere=======')
        for name, param in internvl.named_parameters():
            if param.requires_grad:
                logger.info(name)
    else:
        meta_llava_map = None

    dist.barrier()

    param_init_fn = partial(
        dp_lazy_init, module_map=meta_llava_map, dp_mesh=dp_mesh)

    policies = [layer_auto_wrap_policy]
    # if args.llm_use_lora or args.vit_use_lora:
    #     policies.append(all_required_grad_wrap_policy)

    if args.shard_strategy == 'full':
        fsdp_device_mesh = dp_mesh
        strategy = ShardingStrategy.FULL_SHARD
    elif args.shard_strategy == 'no':
        fsdp_device_mesh = dp_mesh
        strategy = ShardingStrategy.NO_SHARD
    elif args.shard_strategy == 'zero2':
        fsdp_device_mesh = dp_mesh
        strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args.shard_strategy == 'hybrid':
        fsdp_device_mesh = init_device_mesh('cuda', (dp_size // 8, 8))
        strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise ValueError

    torch.cuda.reset_peak_memory_stats()
    shard_llava = FSDP(
        meta_internvl,
        device_mesh=fsdp_device_mesh,
        sharding_strategy=strategy,
        auto_wrap_policy=partial(_or_policy, policies=policies),
        mixed_precision=MixedPrecision(
            param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        param_init_fn=param_init_fn,
        sync_module_states=True,
    )
    if rank == 0:
        logger.info(shard_llava)

    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Model] The peak GPU memory when building the FSDP model is '
                f'{max_memory / 1024 ** 3:.1f}GB.')

    if args.selective_recompute:
        check_fn = partial(
            checkpoint_check_fn,
            target=RECOMPUTE_MODULES,
            selective=args.selective_recompute)
        apply_activation_checkpointing(shard_llava, check_fn=check_fn)

    fsdp_cost_time = time.time() - start_model_t
    logger.info(f'[Model] Cost {fsdp_cost_time:.2f}s')
    # --------------------------    FSDP  End  ------------------------------ #

    ###########################################################################
    #                      4. Optimizer & Scheduler                           #
    ###########################################################################
    requried_grad_params = [
        param for param in shard_llava.parameters() if param.requires_grad
    ]
    optimizer = AdamW(
        requried_grad_params, lr=args.lr, weight_decay=args.wd, fused=True)

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

    # ----------------    Optimizer & Scheduler End   ----------------------- #
    if args.resume:
        logger.info(f'[Resume] Resume from {args.resume_from}')
        _options = StateDictOptions(
            cpu_offload=True, ignore_frozen_params=True)
        (shard_model_state_dict,
         shard_optimizer_state_dict) = get_state_dict(
            shard_llava, optimizer, options=_options)
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
            shard_llava,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
            options=_options
        )

        start_step = meta_stateful['step']
        logger.info(f'[Resume] start_step to {start_step}')

    ###########################################################################
    #                          5. Training                                    #
    ###########################################################################

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

            if args.use_orig:
                # 暂时设置为没有 batch packing
                packed_ctx = packed_sequence(None, enable=False)
            else:
                packed_ctx = packed_sequence(num_tokens, enable=True)

            with packed_ctx:
                outputs = shard_llava(**data)
                avg_iter_loss = outputs.loss / per_step_iters

                if scaler:
                    scaler.scale(avg_iter_loss).backward()
                else:
                    avg_iter_loss.backward()

            step_loss += avg_iter_loss.item()
            step_consumed_tokens += num_tokens.sum()
            step_consumed_img_tokens += num_img_tokens.sum()

        grad_norm = shard_llava.clip_grad_norm_(args.max_grad_norm)
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
                shard_llava, options=_options)
            if rank == 0:
                saved_llava = copy.deepcopy(rank0_meta_llava)
                saved_llava.to(dtype)
                for name, param in full_model_state_dict.items():
                    set_module_tensor_to_device(saved_llava, name, 'cpu',
                                                param)

                saved_llava.save_pretrained(hf_dir)
                tokenizer.save_pretrained(hf_dir)
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
                    shard_llava, optimizer, options=_options)
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
    logger.info(f'[Train] Cost {train_cost_time}s')
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':
    args = parse_args()
    internvl_train(args)
