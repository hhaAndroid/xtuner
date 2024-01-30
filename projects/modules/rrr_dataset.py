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
                 img_size=(532, 532)):
        self.data_root = data_root
        self.img_root = os.path.join(data_root, data_prefix['img'])
        json_data = json.load(open(os.path.join(data_root, ann_file)))
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
            neww = int(old_w * float(scale_factor))
            newh = int(old_h * float(scale_factor))
            image = F.resize(image, size=(newh, neww), interpolation=F.InterpolationMode.BICUBIC)
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            assert image.shape == (3, self.img_size[0], self.img_size[1])
            data_dict['pixel_values'] = image
        else:
            raise NotImplementedError()
        return data_dict
