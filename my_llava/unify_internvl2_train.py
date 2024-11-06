# Copyright (c) OpenMMLab. All rights reserved.
import warnings

# display once
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*")

import argparse
import math
import os
import time
from copy import deepcopy
from datetime import timedelta
from functools import partial
import random
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from PIL import Image
import json
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from xtuner._lite.parallel.new_setup import setup_parallel, \
    get_fsdp_mesh, get_dp_mesh, get_tp_mesh, get_world_mesh, get_sp_mesh, \
    profile_time_and_memory, get_torch_device_module
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from xtuner._lite import get_logger
from xtuner._lite.accelerate import (LoadWoInit,
                                     dispatch_modules, packed_sequence)
from xtuner._lite.accelerate.fsdp import clip_grad_norm_
from torch.distributed._composable.fsdp import fully_shard
from transformers import (AutoConfig, AutoTokenizer, AutoModel)

from xtuner._lite.datasets.dataset_fn import check_args, \
    set_logger_envs, build_train_dataloader, build_dataset, BaseOrigDataset, \
    _apply_exif_orientation, _prepare_input, is_interval
from xtuner._lite.modelings.model_fn import map_meta_modules, lazy_init_megatron, save_ckpt, resume
from xtuner._lite.checkpoint import checkpoint
from xtuner._lite.internvl.dataset import (dynamic_preprocess, preprocess,
                                           preprocess_internlm, preprocess_mpt,
                                           preprocess_phi3, build_transform, preprocess_phi3_fast,
                                           dynamic_num_patch, read_frames_decord)
from xtuner._lite.internvl.v1_5.modeling_intern_vit import InternVisionModel

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Model Related Settings')
    model_args.add_argument(
        '--model', help='repo id or local path of the model')
    parser.add_argument(
        '--liger', action='store_true', help='use liger kernel')
    model_args.add_argument(
        '--freeze-llm',
        action='store_true',
        help="Not updating LLM's parameters")
    model_args.add_argument(
        '--freeze-vit',
        action='store_true',
        help="Not updating vit's parameters")
    model_args.add_argument(
        '--use-fast-tokenizer',
        action='store_true',
        help="")
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
    model_args.add_argument('--sp-size', type=int, default=1, help='')
    model_args.add_argument(
        '--tp-size',
        default=1,
        type=int,
        help="tp size")
    model_args.add_argument(
        '--ring-size',
        default=1,
        type=int,
        help='The ring size. if it is 1, it is the same as sp ulysses')
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
    data_args.add_argument('--dset-pack', action='store_true')
    data_args.add_argument('--concat-before-pack', action='store_true')
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


class LazyInternVL2Dataset(BaseOrigDataset):
    def __init__(self, data_name, data, model_name,
                 max_length=8192,
                 group_by_length=False,
                 pack_data=False, pack_data_cache_dir=None,
                 use_fast_tokenizer=False,
                 min_num_frames=8,  # video
                 max_num_frames=8,  # video
                 sampling_method='middle',  # video
                 local_num_frames=8):  # video

        _model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        architectures = _model_cfg.llm_config.architectures[0]
        assert architectures in ['InternLM2ForCausalLM', 'Phi3ForCausalLM', 'LlamaForCausalLM']

        if architectures == 'InternLM2ForCausalLM':
            self.template_name = 'internlm2-chat'
        elif architectures == 'Phi3ForCausalLM':
            self.template_name = 'phi3-chat'
        elif architectures == 'LlamaForCausalLM':
            self.template_name = 'Hermes-2'

        self.image_size = _model_cfg.force_image_size or _model_cfg.vision_config.image_size
        self.patch_size = _model_cfg.vision_config.patch_size
        self.max_dynamic_patch = _model_cfg.max_dynamic_patch
        self.min_dynamic_patch = _model_cfg.min_dynamic_patch
        self.dynamic_image_size = _model_cfg.dynamic_image_size
        self.use_thumbnail = _model_cfg.use_thumbnail
        self.downsample_ratio = _model_cfg.downsample_ratio
        self.num_image_token = int((self.image_size // self.patch_size) ** 2 * (self.downsample_ratio ** 2))

        self.use_fast_tokenizer = use_fast_tokenizer
        logger.info(f'Using Fast Tokenzier: {use_fast_tokenizer} for training.')
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  use_fast=use_fast_tokenizer,
                                                  trust_remote_code=True)

        # video
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.local_num_frames = local_num_frames
        self.sampling_method = sampling_method

        super().__init__(data_name, data, None,
                         tokenizer=tokenizer,
                         max_length=-1,
                         group_by_length=group_by_length,
                         pack_data=pack_data,
                         pack_data_cache_dir=pack_data_cache_dir)

    def calc_group_len(self):
        group_length = []
        print('Calculating the length of text data...')
        conv2length = {}

        for data_item in self.raw_data:
            is_media = False
            if self._is_jsonl:
                data_item = json.loads(data_item)

            if ('image' in data_item and data_item['image'] is not None) or (
                    'video' in data_item and data_item['video'] is not None):
                assert 'image_wh' in data_item, 'image must have `hw` attribute when group_by_length is True'
                is_media = True

            if 'length' in data_item:
                token_length = data_item['length']  # Use precomputed length if available
            else:
                # Compute token length using the tokenizer
                conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                str_length = len(conversations)
                if str_length not in conv2length:
                    token_length = self.tokenizer(
                        conversations, return_tensors='pt', padding=False, truncation=False,
                    ).input_ids.size(1)
                    if 'video' in data_item and data_item['video'] is not None:
                        # TODO: 暂时写死
                        token_length += self.num_image_token * self.max_num_frames
                    else:
                        conv2length[str_length] = token_length + self.num_image_token * (
                                self.max_dynamic_patch + self.use_thumbnail)
                else:
                    token_length = conv2length[str_length]

            if is_media:
                group_length.append(token_length)
            else:
                group_length.append(-token_length)
        print('Finished calculating the length of text data...')
        return group_length

    def pre_tokenize_fn_for_pack(self, data_item):
        if self._is_jsonl:
            data_item = json.loads(data_item)
        if 'image' in data_item and data_item['image'] is not None:
            if type(data_item['image']) == list and len(data_item['image']) > 1:
                num_tokens = self.multi_modal_multi_image_get_item(data_item, pack_data=True)
            else:
                num_tokens = self.multi_modal_get_item(data_item, pack_data=True)
        elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
            num_tokens = self.video_get_item(data_item, pack_data=True)
        else:
            num_tokens = self.pure_text_get_item(data_item, pack_data=True)
        return {'num_tokens': num_tokens}

    def _get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.use_fast_tokenizer:
            # 不管啥模板，都是统一流程
            preprocess_function = preprocess_phi3_fast
        else:
            if self.template_name == 'Hermes-2':
                preprocess_function = preprocess_mpt
            elif self.template_name == 'internlm2-chat':
                preprocess_function = preprocess_internlm
            elif self.template_name == 'phi3-chat':
                preprocess_function = preprocess_phi3
            else:
                preprocess_function = preprocess
        return preprocess_function

    def _get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=False, input_size=self.image_size,
                                    pad2square=False)
        return transform

    def multi_modal_get_item(self, data_item, pack_data=False):
        # Ensure the first conversation contains an image placeholder
        if '<image>\n' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>\n', '')

        if '\n<image>' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('\n<image>', '')

        if '<image>' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>', '')

        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        if pack_data:
            # 只需要 num_tokens 即可，其余不需要
            assert 'image_wh' in data_item, 'image must have `hw` attribute when group_by_length is True'
            image_size = data_item.get('image_wh', [0, 0])
            if isinstance(image_size[0], list):
                image_size = image_size[0]

            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                num_patches = dynamic_num_patch(image_size, min_num=self.min_dynamic_patch,
                                                max_num=self.max_dynamic_patch,
                                                image_size=self.image_size, use_thumbnail=self.use_thumbnail)
            else:  # Otherwise, use the original image as a single patch
                num_patches = 1

            if not self.dynamic_image_size:
                assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

            preprocess_function = self._get_preprocess_function()

            # Preprocess the conversations and generate the return dictionary
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                      self.tokenizer, [self.num_image_token * num_patches],
                                      group_by_length=True, ds_name=self.data_name)
            return len(ret['input_ids'][0])

        image_path = data_item['image']
        if isinstance(image_path, list):
            image_path = image_path[0]

        image_path = os.path.join(self.root, image_path)
        image = Image.open(image_path).convert('RGB')
        image = _apply_exif_orientation(image)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        transform = self._get_transform()
        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self._get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=True, ds_name=self.data_name)

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_tokens=[len(ret['input_ids'][0])],
            num_img_tokens=[self.num_image_token * num_patches],
            num_imgs=[1],
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item, pack_data=False):
        image_path = data_item['image']
        if '<image>' not in data_item['conversations'][0]['value']:
            temp_str = ''
            for i in range(len(image_path)):
                temp_str += f'Image-{i + 1}: <image>\n'
            data_item['conversations'][0]['value'] = temp_str + data_item['conversations'][0][
                'value']
        assert 'Image-1: ' in data_item['conversations'][0]['value'], 'Image-1: not found in the first conversation'

        assert len(image_path) == data_item['conversations'][0]['value'].count('<image>'), \
            f'Image number not match: {image_path} vs {data_item["conversations"][0]["value"]}'

        if pack_data:
            # 只需要 num_tokens 即可，其余不需要
            assert 'image_wh' in data_item, 'image must have `hw` attribute when group_by_length is True'
            image_size = data_item.get('image_wh', [[0, 0]])
            if not isinstance(image_size[0], list):
                image_size = [image_size]

            num_tiles = []
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                for size in image_size:
                    num_patches = dynamic_num_patch(size, min_num=self.min_dynamic_patch,
                                                    max_num=self.max_dynamic_patch // len(image_path),
                                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                    num_tiles.append(num_patches)
            else:  # Otherwise, use the original image as a single patch
                num_tiles.append(1)

            preprocess_function = self._get_preprocess_function()

            num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                      self.tokenizer, num_image_tokens, group_by_length=True,
                                      ds_name=self.data_name, num_image=len(image_size))

            return len(ret['input_ids'][0])

        num_tiles = []
        images = []
        for image_path_ in image_path:
            image_path_ = os.path.join(self.root, image_path_)
            image = Image.open(image_path_).convert('RGB')
            image = _apply_exif_orientation(image)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=self.max_dynamic_patch // len(image_path),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)

        transform = self._get_transform()
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self._get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=True,
                                  ds_name=self.data_name, num_image=len(image_path))

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_tokens=[len(ret['input_ids'][0])],
            num_img_tokens=num_image_tokens,
            num_imgs=[len(image_path)]
        )
        return ret

    def _get_num_frames_by_duration(self, duration):
        if self.local_num_frames != -1:
            num_segments = int(duration // self.local_num_frames)
            if num_segments == 0:
                num_frames = self.local_num_frames
            else:
                num_frames = self.local_num_frames * num_segments
        else:
            num_frames = max(1, int(duration))

        num_frames = min(self.max_num_frames, num_frames)
        num_frames = max(self.min_num_frames, num_frames)

        return num_frames

    def video_get_item(self, data_item, pack_data=False):
        # Build transformation function
        transform = self._get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        if pack_data:
            # 暂时写死，估计取 8 帧
            assert 'duration' in data_item, data_item
            # duration = data_item['duration']
            # num_frames = self._get_num_frames_by_duration(duration)
            assert self.max_num_frames == self.min_num_frames
            num_frames = self.min_num_frames
            image_list = [0] * num_frames
        else:
            image_list = read_frames_decord(
                video_path,
                num_frames=self.max_num_frames,
                min_num_frames=self.min_num_frames,
                sample=self.sampling_method,
                clip=data_item.get('clip', None))

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self._get_preprocess_function()

        if pack_data:
            num_image_tokens = [self.num_image_token] * len(image_list)
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                      self.tokenizer, num_image_tokens, group_by_length=True,
                                      ds_name=self.data_name, num_image=len(image_list))
            return len(ret['input_ids'][0])

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=True,
                                  ds_name=self.data_name, num_image=num_patches)
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_tokens=[len(ret['input_ids'][0])],
            num_img_tokens=num_image_tokens,
            num_imgs=[num_patches]
        )
        return ret

    def pure_text_get_item(self, data_item, pack_data=False):
        preprocess_function = self._get_preprocess_function()

        if pack_data:
            # Preprocess the conversations and generate the return dictionary
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                      self.tokenizer, [0],
                                      text_only=True,
                                      group_by_length=True, ds_name=self.data_name)

            return len(ret['input_ids'][0])

        # Build transformation function
        transform = self._get_transform()

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

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  text_only=True,
                                  group_by_length=True, ds_name=self.data_name)
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            num_tokens=[len(ret['input_ids'][0])],
            num_img_tokens=[0],
            num_imgs=[0],
        )
        return ret

    def __getitem__(self, i):
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = self.raw_data[i]
                if self._is_jsonl:
                    data_item = json.loads(data_item)
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
                print(f'Exception: {e} of {self.data_name}', flush=True)
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def packing_collate(features, pack_batch=True, pad_id=0):
    _features = []
    for ins in features:
        if isinstance(ins, list):
            _features.extend(ins)
        else:
            _features.append(ins)
    features = _features

    input_ids = []
    labels = []
    pixel_values = []
    num_tokens = []
    num_img_tokens = []
    num_imgs = []
    image_flags = []

    for data in features:
        input_ids.append(torch.LongTensor(data['input_ids']))
        labels.append(torch.LongTensor(data['labels']))
        image_flags.append(data['image_flags'])
        num_tokens.extend(data['num_tokens'])
        num_img_tokens.extend(data['num_img_tokens'])
        pixel_values.append(data['pixel_values'])
        num_imgs.append(data['num_imgs'])

    num_tokens = torch.IntTensor(num_tokens)
    num_img_tokens = torch.IntTensor(num_img_tokens)
    num_imgs = torch.IntTensor(num_imgs)

    if len(features) > 1 and pack_batch:
        # packing
        input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        labels = torch.cat(labels, dim=0).unsqueeze(0)
        pixel_values = torch.cat(pixel_values, dim=0)
        image_flags = torch.cat(image_flags, dim=0)
    elif len(features) > 1 and not pack_batch:
        raise NotImplementedError
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        pixel_values = torch.cat(pixel_values, dim=0)
        image_flags = image_flags[0]

    data_dict = {
        'input_ids': input_ids,
        'labels': labels,
        'pixel_values': pixel_values,
        'image_flags': image_flags,
        'num_tokens': num_tokens,
        'num_img_tokens': num_img_tokens,
        'num_imgs': num_imgs,
    }

    return data_dict


def build_llava_model(args, dtype=torch.float32, device='cpu'):
    with torch.device(device):
        with LoadWoInit():
            _cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            # 防止 fsdp 构建时候 dpr 报错
            vision_model = InternVisionModel(_cfg.vision_config)
            internvl = AutoModel.from_pretrained(
                args.model,
                vision_model=vision_model,
                use_flash_attn=True,
                trust_remote_code=True,
                torch_dtype=dtype)
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            internvl.img_context_token_id = img_context_token_id

        internvl.to(dtype)

        if args.freeze_llm:
            internvl.language_model.requires_grad_(False)
            internvl.language_model.eval()

        if args.freeze_vit:
            internvl.vision_model.requires_grad_(False)
            internvl.vision_model.eval()

    for module in internvl.modules():
        for p_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                setattr(module, p_name, param_fp32)
    return internvl


def build_fsdp_model(rank0_model, meta_model, dp_mesh, tp_mesh, dtype, args):
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

    # visual
    meta_model.vision_model.apply(param_init_fn)
    fully_shard(
        meta_model.vision_model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    for i, layers in enumerate(meta_model.vision_model.encoder.layers):
        checkpoint(layers)

    num_layers = len(meta_model.language_model.model.layers)
    num_recompute_layers = int(num_layers * 1.0)
    for i, block in enumerate(meta_model.language_model.model.layers):
        block.apply(param_init_fn)

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        if i < num_recompute_layers:
            checkpoint(block)

    meta_model.mlp1.apply(param_init_fn)

    try:
        meta_model.language_model.model.tok_embeddings.apply(param_init_fn)
    except AttributeError:
        meta_model.language_model.model.embed_tokens.apply(param_init_fn)

    meta_model.language_model.model.norm.apply(param_init_fn)
    try:
        meta_model.language_model.output.apply(param_init_fn)
    except AttributeError:
        meta_model.language_model.lm_head.apply(param_init_fn)

    model = fully_shard(
        meta_model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)  # False is zero2, True is zero3

    return model


def llava_train(args):
    if args.liger:
        raise NotImplementedError

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

    rank = world_mesh.get_rank()

    check_args(args)
    set_logger_envs(args)

    with profile_time_and_memory('[Dataset & Dataloader]'):
        ds_collections = json.loads(open(args.datasets).read())
        _datasets = []
        for name, _data in ds_collections.items():
            _dataset = LazyInternVL2Dataset(name, _data, args.model,
                                            max_length=args.max_length,
                                            group_by_length=args.group_by_length,
                                            pack_data=args.dset_pack,
                                            pack_data_cache_dir=args.dset_cache_dir,
                                            use_fast_tokenizer=args.use_fast_tokenizer)
            if dist.get_rank() == 0:
                logger.info(f'[Dataset] (Original) {name}: {len(_dataset)} samples.')
            _datasets.append(_dataset)
        train_dataset = build_dataset(args, _datasets)
        logger.warning(f'{dist.get_rank()} ===== End of all dataset =====')
        train_dataloader = build_train_dataloader(args, train_dataset, packing_collate)

    args.dtype = 'bf16'
    dtype = torch.bfloat16
    with profile_time_and_memory('[Model]'):
        meta_model = build_llava_model(args, dtype=dtype, device='meta')
        dispatch_modules(meta_model)
        if dist.get_rank() == 0:
            logger.info(meta_model)

        timeout = timedelta(
            minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=45)))
        group = dist.new_group(backend='gloo', timeout=timeout)
        if rank == 0:
            # 用于初始化 meta_model 权重和后续保存权重
            logger.info(f'=====[Build CPU Model]=======')
            rank0_model = build_llava_model(args, dtype=dtype, device='cpu')
        else:
            rank0_model = None
        dist.monitored_barrier(group=group, timeout=timeout)

        fsdp_model = build_fsdp_model(rank0_model, meta_model, fsdp_mesh, world_mesh, dtype, args)
        fsdp_model.train()
        if dist.get_rank() == 0:
            logger.info(fsdp_model)

    requried_grad_params = [
        param for param in fsdp_model.parameters() if param.requires_grad
    ]
    requried_grad_name = [name for name, param in fsdp_model.named_parameters() if param.requires_grad]
    if rank == 0:
        logger.info(f'[Optimizer] {requried_grad_name}')

    optimizer = AdamW(
        requried_grad_params, lr=args.lr, weight_decay=args.wd, fused=True)

    max_memory = get_torch_device_module().max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024 ** 3):.1f}GB')

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    # `iter` means once forward+backward
    # `step` means once optimizer step
    # `per_step_iters` means gradient accumulative counts
    per_step_iters = global_batch_size // mirco_batch_size // dp_size
    per_epoch_iters = len(train_dataloader)
    per_epoch_steps = math.ceil(per_epoch_iters / per_step_iters)
    logger.info(f'[Optimizer] Global batch size: {global_batch_size}, Gradient accumulative counts: {per_step_iters}')
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
        start_step = resume(args, fsdp_model, optimizer, warmup_scheduler, cosine_scheduler, start_step, total_steps)

    start_train_t = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()

    save_hf_ckpt_names = []
    save_pt_ckpt_names = []
    max_keep_ckpts = args.max_keep_ckpts
    if max_keep_ckpts <= 0:
        # 全部都保存
        max_keep_ckpts = 100000000

    if rank == 0:
        logger.info('[Train] Begin Train Loop. The current GPU memory is '
                    f'{(max_memory / 1024 ** 3):.1f}GB')
        logger.info('The FSDP adopts a lazy design, so the first iteration will be slow.')
        if args.liger:
            logger.info('====== use liger kernel =====')

    processor = None
    tokenizer = None

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
        step_consumed_imgs = 0
        for _ in range(per_step_iters):
            _data_start_t = time.time()
            data = next(data_iterator)
            step_data_time += time.time() - _data_start_t

            data = _prepare_input(data)
            num_tokens = data.pop('num_tokens')
            num_img_tokens = data.pop('num_img_tokens')
            num_imgs = data.pop('num_imgs')

            packed_ctx = packed_sequence(num_tokens, enable=True)

            with packed_ctx:
                outputs = fsdp_model(**data, use_cache=False)
                avg_iter_loss = outputs.loss / per_step_iters
                avg_iter_loss.backward()

            step_loss += avg_iter_loss.item()
            step_consumed_tokens += num_tokens.sum()
            step_consumed_img_tokens += num_img_tokens.sum()
            step_consumed_imgs += num_imgs.sum()

        grad_norm = clip_grad_norm_(requried_grad_params, fsdp_mesh, args.max_grad_norm)
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
                f'num_imgs: {step_consumed_imgs}  '
                f'tgs: {tgs}  data_time: {step_data_time:.2f}s  '
                f'time: {step_time:.2f}s  '
                f'eta: {eta}')

        if is_interval(step, total_steps, checkpoint_interval):
            save_ckpt(args, step, total_steps, fsdp_model, rank0_model, warmup_scheduler, cosine_scheduler,
                      optimizer, max_keep_ckpts, save_hf_ckpt_names, save_pt_ckpt_names, tokenizer, processor)

    train_cost_time = time.time() - start_train_t
    m, s = divmod(train_cost_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    logger.info("[Train] Cost: %d day, %d:%d:%d" % (d, h, m, s))
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':
    args = parse_args()
    llava_train(args)
