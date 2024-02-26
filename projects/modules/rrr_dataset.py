# Copyright (c) OpenMMLab. All rights reserved.
import json
import os

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
from .dataset_huggingface import process_hf_dataset_rrr
from xtuner.registry import BUILDER
import torchvision.transforms.functional as F
from xtuner.dataset.utils import expand2square
from xtuner.dataset.llava import LLaVADataset
import torch
from .constants import IMAGE_TOKEN_INDEX
import random


class PretrainLLaVADataset(LLaVADataset):
    def __init__(self, *args, img_size=(672, 672), **kwargs):
        self.img_size = img_size
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        input_ids = data_dict['input_ids']
        # -200 是 llava 定义的，由于没有改代码，因此只能这里修改掉
        input_ids = [IMAGE_TOKEN_INDEX if x == -200 else x for x in input_ids]
        data_dict['input_ids'] = input_ids

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
            old_w, old_h = F.get_image_size(image)
            scale_factor = min(self.img_size[0] / max(old_h, old_w),
                               self.img_size[0] / min(old_h, old_w))
            neww = int(old_w * float(scale_factor) + 0.5)
            newh = int(old_h * float(scale_factor) + 0.5)
            image = F.resize(image, size=(newh, neww), interpolation=F.InterpolationMode.BICUBIC)
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            assert image.shape == (3, self.img_size[0], self.img_size[1]), \
                f'the shape is {image.shape} not {(3, self.img_size[0], self.img_size[1])}'
            data_dict['pixel_values'] = image
        else:
            data_dict['pixel_values'] = torch.zeros(3, self.img_size[0], self.img_size[1])
        return data_dict


class RRRDataset(Dataset):
    def __init__(self,
                 data_root,
                 ann_file,
                 data_prefix,
                 tokenizer,
                 image_processor,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 img_size=(672, 672),
                 use_mask=False,
                 bbox_mask_prob=0.5,  # 只有 use_mask 为 True 才有效
                 input_ids_with_output=True  # 推理时候应该是 false
                 ):
        self.data_root = data_root
        self.img_root = os.path.join(data_root, data_prefix['img'])
        ann_file_path = os.path.join(data_root, ann_file)
        json_data = json.load(open(ann_file_path))

        self.use_mask = use_mask
        # =0 表示只有 bbox，=0.7 表示 70% 概率用 mask， =1 表示只有 mask
        self.bbox_mask_prob = bbox_mask_prob
        if self.use_mask:
            mask_path = ann_file_path.replace('.json', '.pth')
            mask_list_dict = torch.load(mask_path)
            assert len(mask_list_dict) == len(json_data), f'the length of mask_data: {len(mask_list_dict)} ' \
                                                          f'is not equal to json_data: {len(json_data)}'
            self.id_to_mask = {}
            for i, mask_dict in enumerate(mask_list_dict):
                self.id_to_mask[mask_dict['id']] = mask_dict['mask']

        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})

        self.text_data = process_hf_dataset_rrr(
            dataset=json_data,
            tokenizer=tokenizer,
            max_length=max_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            split='train',
            max_dataset_length=max_dataset_length,
            remove_unused_columns=False,
            input_ids_with_output=input_ids_with_output,
            pack_to_max_length=False,
            with_image_token=True)

        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = True
        self.img_size = img_size

    @property
    def length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            # print(image_file)
            image = Image.open(os.path.join(self.img_root,
                                            image_file)).convert('RGB')
            old_w, old_h = F.get_image_size(image)
            scale_factor = min(self.img_size[0] / max(old_h, old_w),
                               self.img_size[0] / min(old_h, old_w))
            neww = int(old_w * float(scale_factor) + 0.5)
            newh = int(old_h * float(scale_factor) + 0.5)
            image = F.resize(image, size=(newh, neww), interpolation=F.InterpolationMode.BICUBIC)
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            assert image.shape == (3, self.img_size[0], self.img_size[1]), \
                f'the shape is {image.shape} not {(3, self.img_size[0], self.img_size[1])}'
            data_dict['pixel_values'] = image

            if self.use_mask:
                mask = self.id_to_mask[data_dict['id']]
                if random.random() < self.bbox_mask_prob:
                    data_dict['mask'] = mask
                    # bbox 还是保留，方便可视化啥的
                    # 如果存在 mask 数据，则训练和推理只用 mask
                    # del data_dict['bbox']
        else:
            raise NotImplementedError()
        return data_dict
