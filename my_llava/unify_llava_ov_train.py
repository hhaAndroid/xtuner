# Copyright (c) OpenMMLab. All rights reserved.
import warnings

# display once
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*")

import argparse
import math
import os
import time
from datetime import timedelta
from functools import partial
import random
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from PIL import Image
import json
import numpy as np
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
from transformers import (AutoConfig, AutoTokenizer, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor,
                          LlavaOnevisionImageProcessor)
from transformers.models.llava_onevision.modeling_llava_onevision import image_size_to_num_patches
from transformers.image_transforms import resize

from xtuner._lite.datasets.dataset_fn import check_args, \
    set_logger_envs, build_train_dataloader, build_dataset, BaseOrigDataset, \
    _apply_exif_orientation, _prepare_input, is_interval
from xtuner._lite.modelings.model_fn import map_meta_modules, lazy_init_megatron, save_ckpt, resume
from xtuner._lite.checkpoint import checkpoint
from xtuner._lite.internvl.dataset import read_frames_decord

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


class LazyLLaVAOVDataset(BaseOrigDataset):
    def __init__(self, data_name, data, model_name,
                 max_length=8192,
                 group_by_length=False,
                 pack_data=False,
                 pack_data_cache_dir=None,
                 min_num_frames=8,  # video
                 max_num_frames=8,  # video
                 sampling_method='middle',  # video
                 local_num_frames=8):  # video

        self.config = AutoConfig.from_pretrained(model_name)
        self.processor = LlavaOnevisionProcessor.from_pretrained(model_name)
        chat_template = dict(system="",
                             user='<|im_start|>user {user}<|im_end|><|im_start|>assistant\n',
                             assistant='{assistant}<|im_end|>\n')
        self.num_image_token = self.processor.num_image_token  # 729
        # video
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.local_num_frames = local_num_frames
        self.sampling_method = sampling_method

        super().__init__(data_name, data, chat_template,
                         tokenizer=self.processor.tokenizer,
                         max_length=max_length,
                         group_by_length=group_by_length,
                         pack_data=pack_data,
                         pack_data_cache_dir=pack_data_cache_dir)

    def calc_group_len(self):
        # 不需要特别准确，大概就行
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
                        # 视频没有动态分辨率
                        token_length += self.num_image_token * self.max_num_frames
                    elif "image" in data_item:
                        if type(data_item['image']) == list and len(data_item['image']) > 1:
                            # 多图没有动态分辨率
                            token_length += self.num_image_token * len(data_item['image'])
                        else:
                            # 单图动态分辨率
                            image_size = data_item.get('image_wh', [0, 0])
                            if isinstance(image_size[0], list):
                                image_size = image_size[0]
                            image_size_hw = image_size[::-1]
                            num_patch = image_size_to_num_patches(image_size_hw,
                                                                  self.config.image_grid_pinpoints,
                                                                  patch_size=self.config.vision_config.image_size)
                            token_length += self.num_image_token * num_patch
                    else:
                        raise ValueError('Unknown media type')
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

    def _process_media_format_first_round(self, input_, media_type, image_grids):
        # 图片占位符只能在第一轮对话中出现
        if media_type == 'image':
            assert '<image>' in input_, f'Image placeholder not found in the first conversation: {input_}'
            index = 0
            while '<image>' in input_:
                input_ = input_.replace(
                    "<|image_pad|>", "<|placeholder|>" * image_grids[index], 1
                )
                index += 1
            input_ = input_.replace("<|placeholder|>", "<image>")
        elif media_type == 'video':
            raise NotImplementedError
        return input_

    def multi_modal_get_item(self, data_item, pack_data=False):
        # Ensure the first conversation contains an image placeholder
        if '<image>\n' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>\n', '')

        if '\n<image>' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('\n<image>', '')

        if '<image>' in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>', '')

        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>' + data_item['conversations'][0]['value']

        if pack_data:
            # 只需要 num_tokens 即可，其余不需要
            assert 'image_wh' in data_item, 'image must have `hw` attribute when group_by_length is True'
            image_size = data_item.get('image_wh', [0, 0])
            if isinstance(image_size[0], list):
                image_size = image_size[0]
            num_image_tokens = self.processor._get_number_of_features(image_size[1], image_size[0], 384, 384)
            if self.processor.vision_feature_select_strategy == "default":
                num_image_tokens -= 1
            ret = self.process_text(data_item['conversations'], media_type='image', image_grids=[num_image_tokens])
            return len(ret['input_ids'])

        image_path = data_item['image']
        if isinstance(image_path, list):
            image_path = image_path[0]

        image_path = os.path.join(self.root, image_path)
        image = Image.open(image_path).convert('RGB')
        image = _apply_exif_orientation(image)

        image_inputs = self.processor.image_processor(images=image, return_tensors='pt')
        image_sizes = iter(image_inputs["image_sizes"])
        height, width = image_inputs["pixel_values"].shape[-2:]  # 应该始终是 384x384
        assert height == width == 384, f'Image size should be 384x384, but got {height}x{width}'

        image_size_list = next(image_sizes)
        orig_height, orig_width = image_size_list
        num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height, width)
        if self.processor.vision_feature_select_strategy == "default":
            num_image_tokens -= 1

        ret = self.process_text(data_item['conversations'], media_type='image', image_grids=[num_image_tokens])

        out_data = {
            'input_ids': ret['input_ids'],
            'labels': ret['labels'],
            'pixel_values': image_inputs["pixel_values"],
            "image_sizes": list(image_inputs["image_sizes"]),
            'num_tokens': [len(ret['input_ids'])],
            'num_img_tokens': [num_image_tokens],
            'num_imgs': [1]
        }
        return out_data

    def multi_modal_multi_image_get_item(self, data_item, pack_data=False):
        image_path = data_item['image']
        if '<image>' not in data_item['conversations'][0]['value']:
            temp_str = ''
            for i in range(len(image_path)):
                temp_str += f'<image>'
            data_item['conversations'][0]['value'] = temp_str + data_item['conversations'][0][
                'value']
        assert '<image>' in data_item['conversations'][0]['value'], '<image> not found in the first conversation'

        assert len(image_path) == data_item['conversations'][0]['value'].count('<image>'), \
            f'Image number not match: {image_path} vs {data_item["conversations"][0]["value"]}'

        if pack_data:
            # 只需要 num_tokens 即可，其余不需要
            # 没有动态分辨率
            num_image_tokens = self.num_image_token * len(image_path)
            if self.processor.vision_feature_select_strategy == "default":
                num_image_tokens -= 1
            num_image_tokens += len(image_path)  # image_newline
            ret = self.process_text(data_item['conversations'], media_type='image', image_grids=[num_image_tokens])
            return len(ret['input_ids'])

        all_image = []
        image_sizes = []
        for image_path_ in image_path:
            image_path_ = os.path.join(self.root, image_path_)
            image = Image.open(image_path_).convert('RGB')
            image = _apply_exif_orientation(image)
            hw = image.size[::-1]
            image_sizes.append([hw])
            resized_original_image = resize(np.array(image), (384, 384), resample=3)
            pixel_value = self.processor._preprocess(
                [resized_original_image],
                do_resize=True,
                size=(383, 384),
                resample=3,
                do_rescale=self.processor.do_rescale,
                rescale_factor=self.processor.rescale_factor,
                do_normalize=self.processor.do_normalize,
                image_mean=self.processor.image_mean,
                image_std=self.processor.image_std
            )
            pixel_values = np.array(pixel_value)
            all_image.append(pixel_values)
        pixel_values = torch.concat(all_image, dim=0)
        num_image_tokens = self.num_image_token * len(image_path)
        if self.processor.vision_feature_select_strategy == "default":
            num_image_tokens -= 1
        num_image_tokens += len(image_path)  # image_newline
        ret = self.process_text(data_item['conversations'], media_type='image', image_grids=[num_image_tokens])

        out_data = {
            'input_ids': ret['input_ids'],
            'labels': ret['labels'],
            'pixel_values': pixel_values,
            "image_sizes": image_sizes,
            'num_tokens': [len(ret['input_ids'])],
            'num_img_tokens': [num_image_tokens],
            'num_imgs': [len(image_path)]
        }
        return out_data

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
        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        if pack_data:
            # 暂时写死，估计取 8 帧
            # assert 'duration' in data_item, data_item
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
        special_tokens = '\n'.join(['<video>' for _ in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>', special_tokens)

        video_inputs = self.processor.video_processor(videos=image_list, return_tensors='pt')
        one_video = video_inputs["pixel_values_videos"][0].numpy()
        num_frames = one_video.shape[0]  # frame dim is always after batch dim
        patches_height_width = int(math.sqrt(self.processor.num_image_tokens))
        pooled_height_width = math.ceil(patches_height_width / 2)
        num_video_tokens = (num_frames * pooled_height_width * pooled_height_width) + 1  # +1 for newline token
        ret = self.process_text(data_item['conversations'], media_type='image', image_grids=[num_video_tokens])

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            pixel_values_videos=video_inputs["pixel_values_videos"],
            image_sizes_videos=video_inputs["image_sizes_videos"],
            num_tokens=[len(ret['input_ids'][0])],
            num_img_tokens=[num_video_tokens],
            num_imgs=[num_frames]
        )
        return ret

    def pure_text_get_item(self, data_item, pack_data=False):
        if pack_data:
            ret = self.process_text(data_item['conversations'], media_type='text')
            return len(ret['input_ids'][0])

        # Create a blank white image
        image = Image.new('RGB', (384, 384), (255, 255, 255))

        image_inputs = self.processor.image_processor(images=image, return_tensors='pt')
        height, width = image_inputs["pixel_values"].shape[-2:]  # 应该始终是 384x384
        assert height == width == 384, f'Image size should be 384x384, but got {height}x{width}'

        ret = self.process_text(data_item['conversations'], media_type='text')

        out_data = {
            'input_ids': ret['input_ids'],
            'labels': ret['labels'],
            'pixel_values': image_inputs["pixel_values"],
            "image_sizes": list(image_inputs["image_sizes"]),
            'num_tokens': [len(ret['input_ids'])],
            'num_img_tokens': [0],
            'num_imgs': [0]
        }
        return out_data

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


def packing_collate(features, pack_batch=True, pad_id=0, sp_size=1):
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
    image_sizes = []
    video_pixel_values = []
    image_sizes_videos = []

    for data in features:
        input_ids.append(torch.LongTensor(data['input_ids']))
        labels.append(torch.LongTensor(data['labels']))
        image_sizes.append(data['image_sizes'])
        num_tokens.extend(data['num_tokens'])
        num_img_tokens.extend(data['num_img_tokens'])
        pixel_values.append(data['pixel_values'])
        num_imgs.append(data['num_imgs'])

        if 'pixel_values_videos' in data:
            video_pixel_values.append(data['pixel_values_videos'])
            image_sizes_videos.append(data['image_sizes_videos'])

    num_tokens = torch.IntTensor(num_tokens)
    num_img_tokens = torch.IntTensor(num_img_tokens)
    num_imgs = torch.IntTensor(num_imgs)

    if len(features) > 1 and pack_batch:
        # packing
        input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        labels = torch.cat(labels, dim=0).unsqueeze(0)
        pixel_values = torch.cat(pixel_values, dim=0)
        image_sizes = torch.cat(image_sizes, dim=0)

        if len(video_pixel_values) > 0:
            video_pixel_values = torch.cat(video_pixel_values, dim=0)
            image_sizes_videos = torch.cat(image_sizes_videos, dim=0)

    elif len(features) > 1 and not pack_batch:
        raise NotImplementedError
    else:
        raise NotImplementedError

    if sp_size > 1:
        if input_ids.shape[1] % sp_size != 0:
            pad_len = sp_size - input_ids.shape[1] % sp_size
            input_ids = torch.cat([input_ids,
                                   torch.full((1, pad_len), pad_id, dtype=torch.long)], dim=1)
            labels = torch.cat([labels,
                                torch.full((1, pad_len), -100, dtype=torch.long)], dim=1)
            num_tokens = torch.cat([num_tokens,
                                    torch.full((1,), pad_len, dtype=torch.int)], dim=0)
        else:
            num_tokens = torch.cat([num_tokens,
                                    torch.full((1,), 0, dtype=torch.int)], dim=0)

    data_dict = {
        'input_ids': input_ids,
        'labels': labels,
        'pixel_values': pixel_values,
        'image_sizes': image_sizes,
        'num_tokens': num_tokens,
        'num_img_tokens': num_img_tokens,
        'num_imgs': num_imgs,
    }
    if len(video_pixel_values) > 0:
        data_dict['pixel_values_videos'] = video_pixel_values
        data_dict['image_sizes_videos'] = image_sizes_videos
    return data_dict


def build_llava_model(args, dtype=torch.float32, device='cpu'):
    with torch.device(device):
        with LoadWoInit():
            _cfg = AutoConfig.from_pretrained(args.model)
            llava_ov = LlavaOnevisionForConditionalGeneration.from_pretrained(
                args.model,
                use_flash_attention_2=True,
                torch_dtype=dtype)

        llava_ov.to(dtype)

        if args.freeze_llm:
            llava_ov.language_model.requires_grad_(False)
            llava_ov.language_model.eval()

        if args.freeze_vit:
            llava_ov.vision_tower.requires_grad_(False)
            llava_ov.vision_tower.eval()
            # TODO: 是否需要
            llava_ov.image_newline.requires_grad_(False)

    for module in llava_ov.modules():
        for p_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                setattr(module, p_name, param_fp32)
    return llava_ov


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
    meta_model.vision_tower.apply(param_init_fn)
    fully_shard(
        meta_model.vision_tower,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    for i, layers in enumerate(meta_model.vision_tower.vision_model.encoder.layers):
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

    meta_model.multi_modal_projector.apply(param_init_fn)
    meta_model.image_newline.apply(param_init_fn)
    meta_model.language_model.model.embed_tokens.apply(param_init_fn)
    meta_model.language_model.model.norm.apply(param_init_fn)
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

    setup_parallel(tp_size=args.tp_size, sp_size=args.sp_size, sp_ring_degree=args.ring_size)
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
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              use_fast=args.use_fast_tokenizer,
                                              trust_remote_code=True)
    try:
        pad_token_id = tokenizer.pad_token_id
    except Exception as e:
        logger.warning('Tokenizer does not have pad_token_id attribute. Use 0 instead.')
        pad_token_id = 0

    with profile_time_and_memory('[Dataset & Dataloader]'):
        ds_collections = json.loads(open(args.datasets).read())
        _datasets = []
        for name, _data in ds_collections.items():
            _dataset = LazyLLaVAOVDataset(name, _data, args.model,
                                          max_length=args.max_length,
                                          group_by_length=args.group_by_length,
                                          pack_data=args.dset_pack,
                                          pack_data_cache_dir=args.dset_cache_dir)
            if dist.get_rank() == 0:
                logger.info(f'[Dataset] (Original) {name}: {len(_dataset)} samples.')
            _datasets.append(_dataset)
        train_dataset = build_dataset(args, _datasets)
        logger.warning(f'{dist.get_rank()} ===== End of all dataset =====')
        packing_collate_partial = partial(packing_collate, pad_id=pad_token_id, sp_size=sp_size)
        train_dataloader = build_train_dataloader(args, train_dataset, packing_collate_partial)

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
        requried_grad_params, lr=args.lr, weight_decay=args.wd, fused=False)

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
        if args.dset_pack:
            logger.info(f'======= Using soft packing style. =========')
        if args.sp_size > 1:
            assert args.dset_pack, 'Only support soft packing with sp_size > 1.'
            logger.info(
                f'======= Using SP mode. sp_ulysess:{args.sp_size // args.ring_size}, sp_ring:{args.ring_size}======')

        logger.info('[Train] Begin Train Loop. The current GPU memory is '
                    f'{(max_memory / 1024 ** 3):.1f}GB')
        logger.info('The FSDP adopts a lazy design, so the first iteration will be slow.')
        if args.liger:
            logger.info('====== use liger kernel =====')

    processor = None

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
            if sp_size > 1:
                step_consumed_tokens += num_tokens[:-1].sum() / sp_size
            else:
                step_consumed_tokens += num_tokens.sum() / sp_size
            step_consumed_img_tokens += num_img_tokens.sum() / sp_size
            step_consumed_imgs += num_imgs.sum() / sp_size

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
