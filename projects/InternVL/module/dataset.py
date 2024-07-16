import json
import gc
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import os
from typing import Dict
from PIL import Image
import torch
from copy import deepcopy
import random
from .dataset_utils import build_transform, dynamic_preprocess, preprocess_internlm, preprocess_phi3, TCSLoader, \
    SoftPackDataset

from transformers.utils import logging

logger = logging.get_logger(__name__)

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config

    has_tcs_loader = True
except ImportError as E:
    has_tcs_loader = False


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, template_name, meta, tokenizer, tcs_loader, num_image_token,
                 image_size=448, data_augment=False, pad2square=False, group_by_length=True,
                 dynamic_image_size=True, use_thumbnail=True, min_dynamic_patch=1,
                 max_dynamic_patch=12, repeat_time=1, normalize_type='imagenet',
                 varlen_attn=False, is_multi_style=False):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        # multi image or video, not use dynamic_image_size
        self.is_multi_style = is_multi_style

        self.ds_name = meta['annotation']
        logger.warning(f"{dist.get_rank()} ======= Start to process dataset: {meta['annotation']}")
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.data_augment = data_augment
        self.pad2square = pad2square

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl') or meta['annotation'].endswith('json'), \
            f'annotation must be json/jsonl, but got {meta["annotation"]}'

        self.is_jsonl = meta['annotation'].endswith('jsonl')
        if self.is_jsonl:
            with open(meta['annotation'], 'r') as f:
                self.raw_data = f.readlines()
        else:
            with open(meta['annotation']) as f:
                self.raw_data = json.load(f)
        if repeat_time < 1:
            # choice top len(self.raw_data) * repeat_time samples
            self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        gc.collect()

        self.root = meta['root']  # image_root
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        if self.group_by_length or varlen_attn:
            self.conv2length = {}  # using dict to speedup the calculation of token length
            self.length = []
            for data_item in self.raw_data:
                if self.is_jsonl:
                    data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # use precomputed length if exists
                else:
                    if self.is_multi_style:
                        # In multi-image and video mode, due to the varying number of images,
                        # the text length alone cannot accurately estimate the content.
                        conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        if 'image' in data_item and data_item['image'] is not None:
                            assert isinstance(data_item['image'], list), 'image should be a list'
                            num_images = len(data_item['image'])
                            assert conversations.count(
                                "<image>") == num_images, 'image count should be equal to the number of <image> tokens ' \
                                                          f'{conversations},{num_images}'
                            # 2 is IMG_START_TOKEN/IMG_END_TOKEN
                            # These two tokens need to be added to each image.
                            token_length += num_images * (num_image_token + 2)
                    else:
                        if 'image' in data_item and data_item['image'] is not None:
                            if isinstance(data_item['image'], list):
                                assert len(data_item['image']) == 1, 'image should be one element ' \
                                                                     'when not multi image style'
                        # compute token length using tokenizer
                        conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                        str_length = len(conversations)
                        if str_length not in self.conv2length:
                            token_length = tokenizer(
                                conversations, return_tensors='pt', padding=False, truncation=False,
                            ).input_ids.size(1)
                            self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail) + 2  # 2 is IMG_START_TOKEN/IMG_END_TOKEN
                        else:
                            token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def multi_modal_get_item(self, data_item):
        conversation_0_value = data_item['conversations'][0]['value']
        if '<image>\n' in data_item['conversations'][0]['value']:
            conversation_0_value = conversation_0_value.replace('<image>\n', '')
        if '\n<image>' in conversation_0_value:
            conversation_0_value = conversation_0_value.replace('\n<image>', '')
        if '<image>' in conversation_0_value:
            conversation_0_value = conversation_0_value.replace('<image>', '')
        data_item['conversations'][0]['value'] = '<image>\n' + conversation_0_value

        image_path = data_item['image']
        if isinstance(image_path, list):
            image_path = image_path[0]

        image_path = os.path.join(self.root, image_path)
        if "s3://" in image_path:
            if self.tcs_loader is None:
                raise ValueError('petrel_client is not installed !!!')
            image = self.tcs_loader(image_path)
        else:
            image = Image.open(image_path).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:
            images = [image]
        transform = build_transform(data_augment=self.data_augment, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        if self.template_name in ['internlm2-chat', 'internlm2-chat-internvl2']:
            preprocess_function = preprocess_internlm
        elif self.template_name in ['phi3-chat', 'phi3-chat-internvl2']:
            preprocess_function = preprocess_phi3
        else:
            raise NotImplementedError()

        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, self.num_image_token * num_patches,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,  # n, 3, 448, 448
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)  # n
        )
        return ret

    def multi_modal_multi_images_get_item(self, data_item):
        image_paths = data_item['image']
        assert isinstance(image_paths, list), 'image should be a list'
        image_paths = [os.path.join(self.root, path) for path in image_paths]
        transform = build_transform(data_augment=self.data_augment, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        pixel_values = []
        for image_path in image_paths:
            if "s3://" in image_path:
                if self.tcs_loader is None:
                    raise ValueError('petrel_client is not installed !!!')
                image = self.tcs_loader(image_path)
            else:
                image = Image.open(image_path).convert('RGB')
            pixel_value = transform(image)
            pixel_values.append(pixel_value)
        pixel_values = torch.stack(pixel_values)

        if self.template_name in ['internlm2-chat', 'internlm2-chat-internvl2']:
            preprocess_function = preprocess_internlm
        elif self.template_name in ['phi3-chat', 'phi3-chat-internvl2']:
            preprocess_function = preprocess_phi3
        else:
            raise NotImplementedError()

        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, self.num_image_token * 1,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,  # n, 3, 448, 448
            image_flags=torch.tensor([1] * len(image_paths), dtype=torch.long)  # n
        )
        return ret

    def pure_text_get_item(self, data_item):
        image = Image.new('RGB', (224, 224), (255, 255, 255))
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        transform = build_transform(data_augment=self.data_augment, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        if self.template_name in ['internlm2-chat', 'internlm2-chat-internvl2']:
            preprocess_function = preprocess_internlm
        elif self.template_name in ['phi3-chat', 'phi3-chat-internvl2']:
            preprocess_function = preprocess_phi3
        else:
            raise NotImplementedError()

        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, self.num_image_token * num_patches, text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,  # 1, 3, 448, 448
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)  # 1
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        while True:
            try:
                if self.is_jsonl:
                    data_item = json.loads(self.raw_data[i])
                else:
                    data_item = self.raw_data[i]
                if 'image' in data_item and data_item['image'] is not None:
                    if self.is_multi_style:
                        ret = self.multi_modal_multi_images_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                print(e)
                print(f'error the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(data_args, tokenizer, model, group_by_length=True,
                   dynamic_image_size=True, use_thumbnail=True, min_dynamic_patch=1,
                   max_dynamic_patch=12, normalize_type='imagenet'):
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None
    datasets = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_name in ds_collections.keys():
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch

        # multi image or video
        is_multi_style = ds_collections[ds_name].get('is_multi_style', False)

        try:
            dataset = LazySupervisedDataset(
                data_args.conv_style, ds_collections[ds_name],
                tokenizer,
                tcs_loader,
                num_image_token=model.num_image_token,
                image_size=data_args.force_image_size,
                data_augment=ds_collections[ds_name]['data_augment'],
                pad2square=data_args.pad2square,
                group_by_length=group_by_length,
                dynamic_image_size=dynamic_image_size,
                use_thumbnail=use_thumbnail,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_num,
                repeat_time=repeat_time,
                normalize_type=normalize_type,
                varlen_attn=data_args.varlen_attn,
                is_multi_style=is_multi_style
            )
        except Exception as e:
            logger.warning(f'Error in loading dataset: {ds_name},==== {e}')
            print(f'Error in loading dataset: {ds_name},==== {e}', flush=True)
            exit()
        dataset.ds_name = ds_name
        repeat_time = 1 if repeat_time < 1 else repeat_time  # don't repeat if repeat_time is less than 1
        for i in range(repeat_time):
            logger.warning(f'{dist.get_rank()} ===Add dataset:{ds_name}_{i} with length: {len(dataset)}')
            datasets.append(dataset)
    train_dataset = ConcatDataset(datasets)
    logger.warning(f'{dist.get_rank()} ===== End of all dataset =====')
    if data_args.varlen_attn:
        logger.warning(f'{dist.get_rank()} ===== Start to process packing dataset=====')
        train_dataset = SoftPackDataset(train_dataset, data_args.max_seq_length_for_varlen)
        logger.warning(f'{dist.get_rank()} ===== End of process packing dataset=====')

    return train_dataset
