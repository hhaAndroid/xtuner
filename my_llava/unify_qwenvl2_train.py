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
from transformers import (AutoConfig, AutoProcessor)

from xtuner._lite.datasets.dataset_fn import check_args, \
    set_logger_envs, build_train_dataloader, build_dataset, BaseOrigDataset, \
    _apply_exif_orientation, _prepare_input, is_interval
from xtuner._lite.modelings.model_fn import map_meta_modules, lazy_init_megatron, save_ckpt, resume
from xtuner._lite.checkpoint import checkpoint


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
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers import Qwen2VLForConditionalGeneration

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


def smart_get_thw(image_size, image_processor):
    orig_width, orig_height = image_size

    resized_height, resized_width = smart_resize(
        orig_height,
        orig_width,
        factor=image_processor.patch_size * image_processor.merge_size,
        min_pixels=image_processor.min_pixels,
        max_pixels=image_processor.max_pixels,
    )
    grid_t = 1  # 单图
    grid_h, grid_w = resized_height // image_processor.patch_size, resized_width // image_processor.patch_size
    return [grid_t, grid_h, grid_w]


class LazyQwenVLDataset(BaseOrigDataset):
    def __init__(self, data_name, data, model_name,
                 max_length=32768,
                 group_by_length=False,
                 pack_data=False, pack_data_cache_dir=None):

        _model_cfg = AutoConfig.from_pretrained(model_name)
        self.image_token_id = _model_cfg.image_token_id  # <|image_pad|>
        self.vision_start_token_id = _model_cfg.vision_start_token_id  # <vision_start_token_id>
        self.video_token_id = _model_cfg.video_token_id  # <|video_pad|>
        self.image_token_str = '<|vision_start|><|image_pad|><|vision_end|>'
        self.video_token_str = '<|vision_start|><|video_pad|><|vision_end|>'
        self.chat_template = dict(system='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n',
                                  user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
                                  assistant='{assistant}<|im_end|>\n')
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.merge_length = self.processor.image_processor.merge_size ** 2

        # default min_pixels 3136=56x56=28x28x2x2=56x56 pix 一张图片会占 4 个 token
        # default max_pixels 12845056=28x28x128x128=3584x3584 一张图片会占 16384 个 token
        if 'min_pixels' in data:
            self.processor.image_processor.min_pixels = data['min_pixels']
        if 'max_pixels' in data:
            self.processor.image_processor.max_pixels = data['max_pixels']

        super().__init__(data_name, data, self.chat_template,
                         tokenizer=self.processor.tokenizer,
                         max_length=max_length,
                         image_token_str=self.image_token_str,
                         group_by_length=group_by_length,
                         pack_data=pack_data,
                         pack_data_cache_dir=pack_data_cache_dir)

    def calc_group_len(self):
        group_length = []
        print('Calculating the length of text data...')
        conv2length = {}
        # 由于动态分辨率特点，在开启 group_by_length 时候我们需要通过 image_hw 精确计算实际的 image token
        # 如果采用估算方式会差别较大。因此如果是多模态数据，则一定需要有 hw 属性
        # TODO: 支持视频
        for data_item in self.raw_data:
            is_media = False
            if self._is_jsonl:
                data_item = json.loads(data_item)
            if ('image' in data_item and data_item['image'] is not None) or (
                    'video' in data_item and data_item['video'] is not None):
                assert 'image_wh' in data_item, 'image must have `hw` attribute when group_by_length is True'
                is_media = True

            image_size = data_item.get('image_wh', [[0, 0]])
            if not isinstance(image_size[0], list):
                image_size = [image_size]

            num_image_tokens = 0
            for image_size_ in image_size:
                if image_size_[0] == 0 or image_size_[1] == 0:
                    pass
                else:
                    thw = smart_get_thw(image_size_, self.processor.image_processor)
                    num_image_tokens_ = thw[0] * thw[1] * thw[2]
                    num_image_tokens += num_image_tokens_
                    num_image_tokens += 2

            conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
            str_length = len(conversations)
            if str_length not in conv2length:
                token_length = self.tokenizer(
                    conversations, return_tensors='pt', padding=False, truncation=False,
                ).input_ids.size(1)
                token_length += num_image_tokens
                conv2length[str_length] = token_length
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
            image_size = data_item.get('image_wh', [[0, 0]])
            if not isinstance(image_size[0], list):
                image_size = [image_size]

            media_grid_thw = []
            for size in image_size:
                media_grid_thw.append(smart_get_thw(size, self.processor.image_processor))
            media_grid_thw = torch.tensor(media_grid_thw, dtype=torch.int).reshape(-1, 3)
            sum_media_grid_thw = media_grid_thw.prod(dim=1) // self.merge_length
            ret = self.process_text(data_item['conversations'], media_type='image', image_grids=sum_media_grid_thw)
            return len(ret['input_ids'])

        image_path = data_item['image']
        if isinstance(image_path, list):
            image_path = image_path[0]

        image_path = os.path.join(self.root, image_path)
        image = Image.open(image_path).convert('RGB')
        image = _apply_exif_orientation(image)
        media_inputs = self.processor.image_processor(images=image, videos=None, return_tensors='pt')
        media_grid_thw = media_inputs['image_grid_thw']

        sum_media_grid_thw = media_grid_thw.prod(dim=1) // self.merge_length
        ret = self.process_text(data_item['conversations'], media_type='image', image_grids=sum_media_grid_thw)
        position_id = self.calc_position_id(ret['input_ids'], media_grid_thw)

        out_data = {
            'input_ids': ret['input_ids'],
            'labels': ret['labels'],
            'position_id': position_id,  # (3,n)
            'pixel_values': media_inputs['pixel_values'],
            'image_grid_thw': media_grid_thw,
            'num_tokens': [len(ret['input_ids'])],
            'num_img_tokens': [sum_media_grid_thw[0] + 2]
        }
        return out_data

    def multi_modal_multi_image_get_item(self, data_item, pack_data=False):
        image_path = data_item['image']
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>' * len(image_path) + data_item['conversations'][0][
                'value']

        assert len(image_path) == data_item['conversations'][0]['value'].count('<image>'), \
            f'Image number not match: {image_path} vs {data_item["conversations"][0]["value"]}'

        if pack_data:
            # 只需要 num_tokens 即可，其余不需要
            assert 'image_wh' in data_item, 'image must have `hw` attribute when group_by_length is True'
            image_size = data_item.get('image_wh', [[0, 0]])
            if not isinstance(image_size[0], list):
                image_size = [image_size]

            media_grid_thw = []
            for size in image_size:
                media_grid_thw.append(smart_get_thw(size, self.processor.image_processor))
            media_grid_thw = torch.tensor(media_grid_thw, dtype=torch.int).reshape(-1, 3)
            sum_media_grid_thw = media_grid_thw.prod(dim=1) // self.merge_length
            ret = self.process_text(data_item['conversations'], media_type='image', image_grids=sum_media_grid_thw)
            return len(ret['input_ids'])

        all_image = []
        for image_path_ in image_path:
            image_path_ = os.path.join(self.root, image_path_)
            image = Image.open(image_path_).convert('RGB')
            image = _apply_exif_orientation(image)
            all_image.append(image)
        media_inputs = self.processor.image_processor(images=all_image, videos=None, return_tensors='pt')
        media_grid_thw = media_inputs['image_grid_thw']
        sum_media_grid_thw = media_grid_thw.prod(dim=1) // self.merge_length
        ret = self.process_text(data_item['conversations'], media_type='image', image_grids=sum_media_grid_thw)
        position_id = self.calc_position_id(ret['input_ids'], media_grid_thw)

        num_img_tokens = sum(sum_media_grid_thw[i].item() + 2 for i in range(len(all_image)))
        out_data = {
            'input_ids': ret['input_ids'],
            'labels': ret['labels'],
            'position_id': position_id,  # (3,n)
            'pixel_values': media_inputs['pixel_values'],
            'image_grid_thw': media_grid_thw,
            'num_tokens': [len(ret['input_ids'])],
            'num_img_tokens': [num_img_tokens]
        }
        return out_data

    def video_get_item(self, data_item, pack_data=False):
        raise NotImplementedError

    def pure_text_get_item(self, data_item, pack_data=False):
        image = Image.new('RGB', (224, 224), (255, 255, 255))
        media_inputs = self.processor.image_processor(images=image, videos=None, return_tensors='pt')
        media_grid_thw = media_inputs['image_grid_thw']

        ret = self.process_text(data_item['conversations'], media_type='text')

        if pack_data:
            return len(ret['input_ids'])

        out_data = {
            'input_ids': ret['input_ids'],
            'labels': ret['labels'],
            'position_id': torch.arange(len(ret['input_ids']))[None].expand(3, -1),  # (3,n)
            'pixel_values': media_inputs['pixel_values'],
            'image_grid_thw': media_grid_thw,
            'num_tokens': [len(ret['input_ids'])],
            'num_img_tokens': [0]
        }
        return out_data

    def _process_media_format_first_round(self, input_, media_type, image_grids):
        # 图片占位符只能在第一轮对话中出现
        if media_type == 'image':
            assert '<image>' in input_, f'Image placeholder not found in the first conversation: {input_}'
            index = 0
            while '<image>' in input_:
                input_ = input_.replace("<image>", self.image_token_str, 1)
                input_ = input_.replace(
                    "<|image_pad|>", "<|placeholder|>" * image_grids[index], 1
                )
                index += 1
            input_ = input_.replace("<|placeholder|>", "<|image_pad|>")
        elif media_type == 'video':
            raise NotImplementedError
        return input_

    def calc_position_id(self, input_ids, media_grid_thw):
        # TODO: check video
        input_ids_ = torch.tensor(input_ids, dtype=torch.long)
        vision_start_indices = torch.argwhere(input_ids_ == self.vision_start_token_id).squeeze(1)
        vision_tokens = input_ids_[vision_start_indices + 1]
        image_nums = (vision_tokens == self.image_token_id).sum()
        st = 0
        llm_pos_ids_list: list = []
        for i in range(image_nums):
            thw = media_grid_thw[i]
            ed = input_ids.index(self.image_token_id, st)
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t, h, w = thw
            h = h // self.processor.image_processor.merge_size
            w = w // self.processor.image_processor.merge_size
            t_index = torch.arange(t).view(-1, 1).expand(-1, h * w).flatten()
            h_index = torch.arange(h).view(1, -1, 1).expand(t, -1, w).flatten()
            w_index = torch.arange(w).view(1, 1, -1).expand(t, h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + t * h * w

        if st < len(input_ids):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_ids) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
        position_ids = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        return position_ids

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
    image_grid_thws = []
    position_ids = []

    for data in features:
        input_ids.append(torch.LongTensor(data['input_ids']))
        labels.append(torch.LongTensor(data['labels']))
        num_tokens.extend(data['num_tokens'])
        num_img_tokens.extend(data['num_img_tokens'])
        pixel_values.append(data['pixel_values'])
        image_grid_thws.append(data['image_grid_thw'])
        position_ids.append(data['position_id'])

    attention_mask = [ids.ne(pad_id) for ids in input_ids]
    num_tokens = torch.IntTensor(num_tokens)
    num_img_tokens = torch.IntTensor(num_img_tokens)

    if len(features) > 1 and pack_batch:
        # packing
        input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        labels = torch.cat(labels, dim=0).unsqueeze(0)
        attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)
        image_grid_thws = torch.cat(image_grid_thws, dim=0)
        pixel_values = torch.cat(pixel_values, dim=0)
        position_ids = torch.cat(position_ids, dim=1).unsqueeze(1)  # (3,1,n)
    elif len(features) > 1 and not pack_batch:
        raise NotImplementedError
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        image_grid_thws = torch.stack(image_grid_thws)
        pixel_values = torch.cat(pixel_values, dim=0)
        position_ids = torch.stack(position_ids, dim=1)  # (3,b,n)

    data_dict = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask.bool(),
        'pixel_values': pixel_values,
        'position_ids': position_ids,
        'image_grid_thw': image_grid_thws,
        'num_tokens': num_tokens,
        'num_img_tokens': num_img_tokens,
    }

    return data_dict


def build_llava_model(args, dtype=torch.float32, device='cpu'):
    with torch.device(device):
        with LoadWoInit():
            qwenvl = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model,
                attn_implementation='flash_attention_2',
                torch_dtype=dtype)

        qwenvl.to(dtype)

        if args.freeze_llm:
            qwenvl.model.requires_grad_(False)
            qwenvl.model.eval()

        if args.freeze_vit:
            qwenvl.visual.requires_grad_(False)
            qwenvl.visual.eval()

    for module in qwenvl.modules():
        for p_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                setattr(module, p_name, param_fp32)
    return qwenvl


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
    meta_model.visual.apply(param_init_fn)
    fully_shard(
        meta_model.visual,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    for i, block in enumerate(meta_model.visual.blocks):
        checkpoint(block)

    num_layers = len(meta_model.model.layers)
    num_recompute_layers = int(num_layers * 1.0)
    for i, block in enumerate(meta_model.model.layers):
        block.apply(param_init_fn)

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
    meta_model.lm_head.apply(param_init_fn)

    model = fully_shard(
        meta_model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)  # False is zero2, True is zero3

    return model


def llava_train(args):
    if args.liger:
        from xtuner._lite.modelings import apply_liger_kernel_to_qwen2_vl
        try:
            from liger_kernel.transformers.geglu import LigerGEGLUMLP
        except ImportError:
            raise ImportError('Please install liger_kernel to use liger.')
        apply_liger_kernel_to_qwen2_vl()

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
            _dataset = LazyQwenVLDataset(name, _data, args.model,
                                         max_length=args.max_length,
                                         group_by_length=args.group_by_length,
                                         pack_data=args.dset_pack,
                                         pack_data_cache_dir=args.dset_cache_dir)
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

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

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

            packed_ctx = packed_sequence(num_tokens, enable=True, skip_position_ids=True)

            with packed_ctx:
                outputs = fsdp_model(**data, use_cache=False)
                avg_iter_loss = outputs.loss / per_step_iters
                avg_iter_loss.backward()

            step_loss += avg_iter_loss.item()
            step_consumed_tokens += num_tokens.sum()
            step_consumed_img_tokens += num_img_tokens.sum()

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
