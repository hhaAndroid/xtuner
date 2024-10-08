# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import os
import shutil
import sys
import time
import warnings
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import partial
import json
from typing import Dict
import gc
from collections.abc import Mapping
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate.utils import set_module_tensor_to_device
from torch.utils.data import Dataset
from mmengine import load, mkdir_or_exist
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
import torch.nn.functional as F
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
                                          dp_sp_lazy_init, layer_auto_wrap_policy)
from torch.distributed.checkpoint.stateful import Stateful
from xtuner._lite.internvl.constants import IMG_CONTEXT_TOKEN
from xtuner._lite.internvl.dataset import (dynamic_preprocess, preprocess,
                                           preprocess_internlm, preprocess_mpt,
                                           preprocess_phi3, build_transform, preprocess_phi3_fast,
                                           TCSLoader, concat_pad_data_collator_dpo,
                                           packing_collate, dynamic_num_patch)
from xtuner._lite.internvl.v1_5.modeling_intern_vit import InternVisionModel
from datasets import Dataset as HF_Dataset
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from xtuner._lite.datasets.text import SoftPackerForText
from xtuner._lite.internvl.constants import (IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN)
from xtuner._lite.parallel import (LengthGroupedSampler, ParallelSampler,
                                   get_dp_mesh, get_dp_world_size,
                                   get_sp_mesh,
                                   get_sp_world_size,
                                   setup_parallel)
import subprocess


def get_gpu_memory():
    try:
        output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        return output
    except Exception as e:
        print(f"Error: {e}")
        return e


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
    model_args.add_argument(
        '--internvl', help='repo id or local path of the model')
    # One of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'sppo_hard', 'nca_pair', 'robust']
    model_args.add_argument(
        '--loss-type',
        default='sigmoid',
        type=str,
        help='')
    model_args.add_argument(
        '--beta',
        default=0.1,
        type=float,
        help='')
    model_args.add_argument(
        '--label-smoothing',
        default=0.0,
        type=float,
        help='')
    model_args.add_argument('--sp-size', type=int, default=1, help='')
    model_args.add_argument(
        '--ring-size',
        default=1,
        type=int,
        help='The ring size. if it is 1, it is the same as sp ulysses')
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


def build_model(args, config, dtype=torch.bfloat16, device='cpu'):
    with torch.device(device):
        _cfg = copy.deepcopy(config)

        with LoadWoInit():
            if device != 'meta':
                model = AutoModel.from_pretrained(
                    args.internvl,
                    torch_dtype=torch.bfloat16,
                    config=config,
                    trust_remote_code=True)
            else:
                # 不需要加载权重，只要有结构进行
                model = AutoModel.from_config(config=_cfg,
                                              torch_dtype=dtype,
                                              trust_remote_code=True)

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


def tokenization_fn(tokenizer, prompt, chosen, rejected, num_image_token, text_only):
    if not text_only:
        image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
        prompt = prompt.replace('<image>', image_tokens, 1)
    else:
        assert '<image>' not in prompt, 'text_only is True, but <image> is in prompt'

    prompt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True)
    chosen = tokenizer.apply_chat_template(
        prompt + chosen,
        tokenize=False,
        add_generation_prompt=False)
    rejected = tokenizer.apply_chat_template(
        prompt + rejected,
        tokenize=False,
        add_generation_prompt=False)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    chosen_ids = tokenizer.encode(chosen, add_special_tokens=False)
    rejected_ids = tokenizer.encode(rejected, add_special_tokens=False)

    if len(prompt_ids) > tokenizer.model_max_length:
        warnings.warn(f'Prompt is too long: {len(prompt_ids)} > {tokenizer.model_max_length}')
    if len(chosen_ids) > tokenizer.model_max_length:
        chosen_ids = chosen_ids[:tokenizer.model_max_length]
        warnings.warn(f'Chosen is too long: {len(chosen_ids)} > {tokenizer.model_max_length}. Truncated')
    if len(rejected_ids) > tokenizer.model_max_length:
        rejected_ids = rejected_ids[:tokenizer.model_max_length]
        warnings.warn(f'Reject is too long: {len(rejected_ids)} > {tokenizer.model_max_length}. Truncated')

    prompt_len = min(len(prompt_ids), tokenizer.model_max_length)
    chosen_labels = [-100] * prompt_len + copy.deepcopy(
        chosen_ids[prompt_len:])
    rejected_labels = [-100] * prompt_len + copy.deepcopy(
        rejected_ids[prompt_len:])
    return chosen_ids, chosen_labels, rejected_ids, rejected_labels


class LazyDPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
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
        super(LazyDPODataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
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
            pass
        gc.collect()

    def __len__(self):
        return len(self.raw_data)

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

        prompt = data_item['prompt']
        chosen = data_item['chosen']
        rejected = data_item['rejected']

        if '<image>\n' in prompt:
            prompt = prompt.replace('<image>\n', '')
        if '<image>' in prompt:
            prompt = prompt.replace('<image>', '')
        if '\n<image>' in prompt:
            prompt = prompt.replace('\n<image>', '')
        prompt = '<image>\n' + prompt

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

        ret = tokenization_fn(self.tokenizer, prompt, chosen, rejected,
                              self.num_image_token * num_patches,
                              text_only=False)
        chosen_ids, chosen_labels, rejected_ids, rejected_labels = ret
        ret = dict(
            chosen_ids=chosen_ids,
            chosen_labels=chosen_labels,
            rejected_ids=rejected_ids,
            rejected_labels=rejected_labels,
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_tokens=[len(chosen_ids), len(rejected_ids)],
            num_img_tokens=[self.num_image_token * num_patches]
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

        prompt = data_item['prompt']
        chosen = data_item['chosen']
        rejected = data_item['rejected']

        ret = tokenization_fn(self.tokenizer, prompt, chosen, rejected,
                              self.num_image_token * num_patches,
                              text_only=True)
        chosen_ids, chosen_labels, rejected_ids, rejected_labels = ret
        ret = dict(
            chosen_ids=chosen_ids,
            chosen_labels=chosen_labels,
            rejected_ids=rejected_ids,
            rejected_labels=rejected_labels,
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            num_tokens=[len(chosen_ids), len(rejected_ids)],
            num_img_tokens=[0]
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item and data_item['image'] is not None:
                    ret = self.multi_modal_get_item(data_item)
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
        dynamic_image_size=True,
        force_image_size=448,
        use_thumbnail=True,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        normalize_type='imagenet'
):
    # 由于数据集通常不大，因此不需要啥 trick，直接读取就行
    datasets = []
    lengths = []
    ds_collections = json.loads(open(args.meta_path).read())

    for i, ds_name in enumerate(ds_collections.keys()):
        _dataset = ds_collections[ds_name]
        if 'max_dynamic_patch' in _dataset:
            max_num = _dataset['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch

        dataset = LazyDPODataset(
            _dataset,
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=force_image_size,
            is_train=_dataset['data_augment'],
            pad2square=False,
            group_by_length=False,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=1,
            normalize_type=normalize_type,
            random_seed=i,
            use_fast_tokenizer=False
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
        train_dataset = ConcatDataset(datasets)
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


class DPOWrapper:
    def __init__(self, model, model_ref, loss_type='sigmoid', beta=0.1, label_smoothing=0.0):
        self.model = model
        self.model_ref = model_ref
        self.loss_type = loss_type
        self.beta = beta
        self.label_smoothing = label_smoothing

    def _gather_masked_logits(self, logits, labels, mask):
        logits = torch.gather(
            logits.log_softmax(-1), dim=2,
            index=labels.unsqueeze(2)).squeeze(2)
        return logits * mask

    def get_logps(
            self,
            all_logits,  # bs, seqlen,vocab_size
            all_ref_logits,  # bs, seqlen,vocab_size
            labels,  # bs, seqlen
    ):
        labels = labels[:, 1:].clone()
        all_logits = all_logits[:, :-1, :]
        all_ref_logits = all_ref_logits[:, :-1, :]

        labels[labels == -100] = 0
        loss_mask = labels != 0
        all_logps = self._gather_masked_logits(all_logits, labels,
                                               loss_mask).sum(-1)
        all_ref_logps = self._gather_masked_logits(all_ref_logits, labels,
                                                   loss_mask).sum(-1)

        if self.loss_type == 'ipo':  # average_log_prob
            all_logps = all_logps / loss_mask.sum(-1)
            all_ref_logps = all_ref_logps / loss_mask.sum(-1)

        policy_chosen_logps = all_logps[::2]
        policy_rejected_logps = all_logps[1::2]
        reference_chosen_logps = all_ref_logps[::2]
        reference_rejected_logps = all_ref_logps[1::2]
        return (policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps)

    def __call__(self, data):
        labels = data.pop('labels')
        all_logits = self.model(**data).logits
        with torch.no_grad():
            all_ref_logits = self.model_ref(**data).logits

        (policy_chosen_logps, policy_rejected_logps,
         reference_chosen_logps,
         reference_rejected_logps) = self.get_logps(
            all_logits, all_ref_logits, labels)
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios
        if self.loss_type == 'sigmoid':
            loss = (-F.logsigmoid(self.beta * logits) *
                    (1 - self.label_smoothing) -
                    F.logsigmoid(-self.beta * logits) * self.label_smoothing)
        elif self.loss_type == 'robust':
            loss = (-F.logsigmoid(self.beta * logits) *
                    (1 - self.label_smoothing) +
                    F.logsigmoid(-self.beta * logits) *
                    self.label_smoothing) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == 'hinge':
            loss = torch.relu(1 - self.beta * logits)
        elif self.loss_type == 'ipo':
            # eqn (17) of the paper where beta is the regularization
            # parameter for the IPO loss, denoted by tau in the paper.  # noqa
            loss = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == 'kto_pair':
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps -
                         reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps -
                           reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = \
                policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected)
            # is estimated using the rejected (chosen) half.  # noqa
            loss = torch.cat(
                (
                    1 - F.sigmoid(self.beta *
                                  (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta *
                                  (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.loss_type == 'sppo_hard':
            # In the paper (https://arxiv.org/pdf/2405.00675),
            # SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation
            # is conducted outside of the trainer class.
            # The version described here is the hard probability version,
            # where P in Equation (4.7) of Algorithm 1 is set to 1 for
            # the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            loss = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == 'nca_pair':
            chosen_rewards = (policy_chosen_logps -
                              reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps -
                                reference_rejected_logps) * self.beta
            loss = (-F.logsigmoid(chosen_rewards) -
                    0.5 * F.logsigmoid(-chosen_rewards) -
                    0.5 * F.logsigmoid(-rejected_rewards))
        else:
            raise ValueError(
                f'Unknown loss type: {self.loss_type}. Should be one of '
                "['sigmoid', 'hinge', 'ipo', 'kto_pair', "
                "'sppo_hard', 'nca_pair', 'robust']")
        # for logging
        chosen_rewards = self.beta * (
                policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (
                policy_rejected_logps - reference_rejected_logps)
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()

        loss_dict = {
            'loss': loss,
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_acc': reward_acc,
            'reward_margin': (chosen_rewards - rejected_rewards).mean(),
        }
        return loss_dict


# @logger.catch
def internvl_train(args):
    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(args.seed)

    output = get_gpu_memory()

    world_size = int(os.environ['WORLD_SIZE'])
    sp_size = args.sp_size

    setup_parallel(sp_size=sp_size, ring_size=args.ring_size)
    dp_mesh = get_dp_mesh()
    dp_size = get_dp_world_size()

    assert args.pack_max_length % args.max_length == 0

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

    rank = dist.get_rank()
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
    # logger.info(f'GPU Memory: {output}')

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

    tokenizer = AutoTokenizer.from_pretrained(args.internvl, trust_remote_code=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    internvl_config = AutoConfig.from_pretrained(args.internvl, trust_remote_code=True)
    internvl_config.llm_config.use_cache = False
    meta_internvl = build_model(args,
                                internvl_config,
                                dtype,
                                device='meta')
    meta_internvl.img_context_token_id = img_context_token_id

    for module in meta_internvl.modules():
        for p_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                setattr(module, p_name, param_fp32)

    exclude_cls = []
    if args.use_orig:
        # 保持原始逻辑不变
        logger.info('Use original style.')
        exclude_cls = ['Phi3FlashAttention2']
    dispatch_modules(meta_internvl, exclude_cls)

    if not args.use_orig:
        if args.dset_pack_level == 'soft':
            logger.info(f'======= Using soft packing style. =========')
        else:
            logger.info(f'======= Using packing style. =========')
    if args.sp_size > 1:
        assert args.dset_pack_level == 'soft', 'Only support soft packing with sp_size > 1.'
        logger.info(
            f'======= Using SP mode. sp_ulysess:{args.sp_size // args.ring_size}, sp_ring:{args.ring_size}======')
    ###########################################################################
    #                     2. Dataset & Dataloader                             #
    ###########################################################################
    start_load_data_t = time.time()
    tcs_loader = None

    if args.dset_cache_dir and os.path.isdir(args.dset_cache_dir):
        if len(os.listdir(args.dset_cache_dir)):
            logger.warning(f'`{args.dset_cache_dir}` is not an empty '
                           'folder, which may lead to inaccurate '
                           'cache results.')
    if args.dset_pack_level:
        raise NotImplementedError
    else:
        train_dataset = build_datasets(args, tokenizer, tcs_loader, meta_internvl,
                                       force_image_size=internvl_config.force_image_size,
                                       dynamic_image_size=internvl_config.dynamic_image_size,
                                       use_thumbnail=internvl_config.use_thumbnail,
                                       min_dynamic_patch=internvl_config.min_dynamic_patch,
                                       max_dynamic_patch=internvl_config.max_dynamic_patch)
        if rank == 0:
            logger.info(f'[Dataset] (Original) {len(train_dataset)} samples.')
    logger.warning(f'{dist.get_rank()} ===== End of all dataset =====')

    sampler = ParallelSampler(train_dataset, dp_mesh, args.global_batch_size, seed=args.seed, shuffle=True)

    if args.use_orig:
        collate_fn = concat_pad_data_collator_dpo
    else:
        raise NotImplementedError
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.mirco_batch_size,  # 实际 batch 会翻倍
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0)

    if rank == 0:
        logger.info(f'[Dataloader] {len(train_dataloader)} batches.')
        _first_batch = [train_dataset[i] for i in range(args.mirco_batch_size)]
        _first_batch = collate_fn(_first_batch)
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
    timeout = timedelta(
        minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=45)))
    group = dist.new_group(backend='gloo', timeout=timeout)
    if rank == 0:
        logger.info(f'=====[Build Model]=======')
        internvl = build_model(args,
                               internvl_config,
                               dtype,
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

    dist.monitored_barrier(group=group, timeout=timeout)

    if get_sp_world_size() > 1:
        raise NotImplementedError
    else:
        param_init_fn = partial(
            dp_lazy_init, module_map=meta_llava_map, dp_mesh=dp_mesh)

    policies = [layer_auto_wrap_policy]
    # if args.llm_use_lora or args.vit_use_lora:
    #     policies.append(all_required_grad_wrap_policy)

    if args.shard_strategy == 'full':
        fsdp_device_mesh = init_device_mesh('cuda', (world_size,))
        strategy = ShardingStrategy.FULL_SHARD
    elif args.shard_strategy == 'no':
        fsdp_device_mesh = init_device_mesh('cuda', (world_size,))
        strategy = ShardingStrategy.NO_SHARD
    elif args.shard_strategy == 'zero2':
        fsdp_device_mesh = init_device_mesh('cuda', (world_size,))
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

    # ================================= ref model ====================================================
    meta_internvl_ref = copy.deepcopy(meta_internvl)
    meta_internvl_ref.eval()
    meta_internvl_ref.requires_grad_(False)

    if rank == 0:
        meta_llava_map_ref = map_meta_modules(internvl, meta_internvl_ref)
    else:
        meta_llava_map_ref = None

    if get_sp_world_size() > 1:
        raise NotImplementedError
    else:
        param_init_fn = partial(
            dp_lazy_init, module_map=meta_llava_map_ref, dp_mesh=dp_mesh)

    shard_llava_ref = FSDP(
        meta_internvl_ref,
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
    dpo_model = DPOWrapper(shard_llava, shard_llava_ref,
                           loss_type=args.loss_type,
                           beta=args.beta,
                           label_smoothing=args.label_smoothing)
    # ================================= ref model ====================================================

    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Model] The peak GPU memory when building the FSDP model is '
                f'{max_memory / 1024 ** 3:.1f}GB.')

    if args.selective_recompute:
        check_fn = partial(
            checkpoint_check_fn,
            target=RECOMPUTE_MODULES,
            selective=args.selective_recompute)
        apply_activation_checkpointing(shard_llava, check_fn=check_fn)
        apply_activation_checkpointing(shard_llava_ref, check_fn=check_fn)

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
    logger.info(f'[Optimizer] Global batch size: {global_batch_size}, Gradient accumulative counts: {per_step_iters}')
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
                # outputs = shard_llava(**data)
                outputs = dpo_model(data)
                avg_iter_loss = outputs['loss'] / per_step_iters

                if scaler:
                    scaler.scale(avg_iter_loss).backward()
                else:
                    avg_iter_loss.backward()

                reward_acc = outputs['reward_acc'].item()
                reward_margin = outputs['reward_margin'].item()
                chosen_rewards = outputs['chosen_rewards'].item()
                rejected_rewards = outputs['rejected_rewards'].item()

            step_loss += avg_iter_loss.item()
            if args.dset_pack_level == 'soft':
                # During a soft pack process, the data with a length that is
                # still smaller than the max length after packing, will be
                # padded to the max length. The last element of num tokens
                # represents the count of pad tokens.
                step_consumed_tokens += num_tokens[:-1].sum(
                ) / get_sp_world_size()
            else:
                # TODO 如果是均匀切，那确实如此，但是...
                step_consumed_tokens += num_tokens.sum() / get_sp_world_size()
            step_consumed_img_tokens += num_img_tokens.sum() / get_sp_world_size()

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
                f'reward_acc: {reward_acc:.3f}  reward_margin: {reward_margin:.3f}  '
                f'chosen_rewards: {chosen_rewards:.3f}  rejected_rewards: {rejected_rewards:.3f}  '
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
    m, s = divmod(train_cost_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    logger.info("[Train] Cost: %d day, %d:%d:%d" % (d, h, m, s))
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':
    args = parse_args()
    internvl_train(args)
