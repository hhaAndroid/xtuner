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
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
import torch.nn.functional as Torch_F
import cv2


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
                 bbox_to_mask_prob=0.5,  # 只有 use_mask 为 True 才有效
                 use_sam=False,  # 如果需要，则需要返回 sam 图片数据
                 input_ids_with_output=True  # 推理时候应该是 false
                 ):
        self.data_root = data_root
        self.img_root = os.path.join(data_root, data_prefix['img'])
        ann_file_path = os.path.join(data_root, ann_file)
        json_data = json.load(open(ann_file_path))

        self.use_mask = use_mask
        # =0 表示只有 bbox，=0.7 表示 70% 概率用 mask， =1 表示只有 mask
        self.bbox_to_mask_prob = bbox_to_mask_prob
        if self.use_mask:
            mask_path = ann_file_path.replace('_bbox', '_mask').replace('.json', '.pth')
            mask_list_dict = torch.load(mask_path)
            assert len(mask_list_dict) == len(json_data), f'the length of mask_data: {len(mask_list_dict)} ' \
                                                          f'is not equal to json_data: {len(json_data)}'
            self.id_to_mask = {}
            self.id_to_sam_mask = {}
            for i, mask_dict in enumerate(mask_list_dict):
                self.id_to_mask[mask_dict['id']] = mask_dict['mask']
                self.id_to_sam_mask[mask_dict['id']] = mask_dict['sam_mask']

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

        self.use_sam = use_sam
        self.sam_pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.sam_pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.sam_transform = ResizeLongestSide(1024)

    def sam_preprocess(self, x: torch.Tensor, img_size=1024) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.sam_pixel_mean) / self.sam_pixel_std

        # Pad
        h, w = x.shape[-2:]  # 往后 padding
        padh = img_size - h
        padw = img_size - w
        x = Torch_F.pad(x, (0, padw, 0, padh))
        return x

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
            if self.use_sam:
                # sam loss 是在原图尺度计算的
                data_dict['orig_size'] = (old_h, old_w)  # 原始图片尺寸
                sam_image = self.sam_transform.apply_image(np.array(image))
                padding_h, padding_w = sam_image.shape[:2]  # 网络训练的输入尺寸,不包括 padding 部分
                data_dict['padding_size'] = (padding_h, padding_w)
                sam_image = self.sam_preprocess(torch.from_numpy(sam_image).permute(2, 0, 1).contiguous())
                # print(data_dict['orig_size'],data_dict['padding_size'],sam_image.shape)
                data_dict['sam_pixel_values'] = sam_image
                # 原图尺度的 polygon mask
                sam_mask = self.id_to_sam_mask[data_dict['id']]
                data_dict['sam_mask'] = self.polygons2masks(old_h, old_w, sam_mask)

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
                if random.random() < self.bbox_to_mask_prob:
                    data_dict['mask'] = self.polygons2masks(image.shape[1], image.shape[2], mask)
                    # bbox 还是保留，方便可视化啥的
                    # 如果存在 mask 数据，则训练和推理只用 mask
                    # del data_dict['bbox']
        else:
            raise NotImplementedError()
        return data_dict

    def polygon2mask(self,
                     h,
                     w,
                     polygons: np.ndarray,
                     color: int = 1) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        polygons = np.asarray(polygons, dtype=np.int32)
        cv2.fillPoly(mask, polygons.reshape(1, -1, 2), color=1)
        return mask

    def polygons2masks(self,
                       h,
                       w,
                       polygons,
                       color: int = 1):
        masks = []
        for si in range(len(polygons)):
            mask = self.polygon2mask(h, w, polygons[si], color)
            masks.append(mask)
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

