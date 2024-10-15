import os

import torch
import numpy as np
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from xtuner._lite.chat import ChatMessages
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from .format import OPENAI_FORMAT_MAP
from .text import SoftPackerForText


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class LlavaTokenizeFunction():

    def __init__(self,
                 tokenizer,
                 chat_template,
                 per_img_tokens,
                 image_dir=None,
                 raw_format='llava',
                 max_length=2048):

        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.image_dir = image_dir
        self.raw_format = raw_format
        self.per_img_tokens = per_img_tokens
        self.max_length = max_length

    def __call__(self, item):

        formatter = OPENAI_FORMAT_MAP[self.raw_format]
        msg = ChatMessages.from_dict(formatter(item))
        tokenized = msg.tokenize(self.tokenizer, self.chat_template)

        input_ids = tokenized['input_ids']
        labels = tokenized['labels']

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            tokenized['num_tokens'] = self.max_length
            tokenized['input_ids'] = input_ids
            tokenized['labels'] = labels

        if 'image_urls' in tokenized:
            image_urls = []
            for url in tokenized['image_urls']:
                if url is not None:
                    if self.image_dir:
                        image_urls.append(os.path.join(self.image_dir, url))
                    else:
                        image_urls.append(url)

            num_images = len(image_urls)
            num_img_tokens = [self.per_img_tokens for _ in image_urls]
            tokenized['num_tokens'] += sum(num_img_tokens) - num_images
            tokenized['num_img_tokens'] = sum(num_img_tokens)
            if len(image_urls) > 0:
                tokenized['image_urls'] = image_urls

        return tokenized


class LlavaTokenizedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, image_processor, pad_image_to_square=False):
        super().__init__()
        self.image_processor = image_processor
        self.dataset = dataset
        self.pad_image_to_square = pad_image_to_square

    def process_tokenized_data(self, tokenized_data):
        images = []
        for url in tokenized_data.get('image_urls', []):
            img = Image.open(url).convert('RGB')
            if self.pad_image_to_square:
                img = expand2square(
                        img,
                        tuple(
                            int(x * 255) for x in self.image_processor.image_mean))
            images.append(img)

        if len(images):
            outputs = self.image_processor(images, return_tensors='pt')
            pixel_values = outputs['pixel_values']
            num_img_tokens = [tokenized_data['num_img_tokens']]
        else:
            pixel_values = None
            num_img_tokens = [0]

        data = {
            'input_ids': tokenized_data['input_ids'],
            'labels': tokenized_data['labels'],
            'pixel_values': pixel_values,
            'num_tokens': [tokenized_data['num_tokens']],
            'num_img_tokens': num_img_tokens,
        }

        return data

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        tokenized_data = self.dataset[item]

        return self.process_tokenized_data(tokenized_data)


class LlavaRawDataset(LlavaTokenizedDataset):

    def __init__(self, dataset, image_processor, tokenize_fn, pad_image_to_square=False):
        # dataset is json raw file of items
        super().__init__(dataset, image_processor, pad_image_to_square)

        self.tokenize_fn = tokenize_fn
        self.conv2length_text = {}  # using dict to speedup the calculation of token length
        self.group_length = []
        print('Calculating the length of text data...')
        for data_item in dataset:
            conversations = '\n'.join(
                [temp['value'] for temp in data_item['conversations']])
            str_length = len(conversations)
            if str_length not in self.conv2length_text:
                token_length = self.tokenize_fn.tokenizer(
                    conversations,
                    return_tensors='pt',
                    padding=False,
                    truncation=False,
                ).input_ids.size(1)
                self.conv2length_text[str_length] = token_length
            else:
                token_length = self.conv2length_text[str_length]
            if 'image' in data_item and data_item['image'] is not None:
                token_length += self.tokenize_fn.per_img_tokens
            else:
                token_length = -token_length
            self.group_length.append(token_length)
        print('Finished calculating the length of text data...')

        del self.conv2length_text

    @property
    def modality_length(self):
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length

    def __getitem__(self, item):
        raw_data = self.dataset[item]
        tokenized_data = self.tokenize_fn(raw_data)
        return self.process_tokenized_data(tokenized_data)

    def __len__(self):
        return len(self.dataset)


class SoftPackerForLlava(SoftPackerForText):

    def __init__(self,
                 dataset,
                 image_processor,
                 max_length=2048,
                 pack_info=None,
                 pad_image_to_square=False):
        super().__init__(dataset, max_length, pack_info)
        self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square
        self._cached = False

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        if self._cached:
            self.load_cache()

        packed_items = self.idx_per_pack[item]
        assert len(packed_items) > 0

        packed_input_ids = []
        packed_labels = []
        packed_img_urls = []
        packed_num_tokens = []
        packed_num_img_tokens = []
        for i in packed_items:
            data = self.dataset[i]
            packed_input_ids.extend(data['input_ids'])
            packed_labels.extend(data['labels'])

            _num_tokens = data['num_tokens']
            packed_num_tokens.append(_num_tokens)

            # 之前将 i 写错为了 item 导致训练 loss 下降非常慢，grad norm 偏小很多
            # 原因是：整个 batch 内部只有1 张图片是对的，所以 loss 下降特别慢
            # 同时因为整个 batch 里面算 loss 的 text token 就一点点(每个样本几乎不超过 50 个)，所以 grad norm 偏小

            # image_urls 存在但是可能是 None,因为虽然我们在LlavaTokenizeFunction中处理了如果是纯文本就不会有 image_urls
            # 但是在启动缓存时候 Dataset.from_list 依然会强行设置 image_urls key
            if 'image_urls' in data and data['image_urls'] is not None:
                packed_img_urls.extend(data['image_urls'])

            if 'num_img_tokens' in data and data['num_img_tokens'] is not None:
                _num_img_tokens = data['num_img_tokens']
                packed_num_img_tokens.append(_num_img_tokens)

        images = []
        for url in packed_img_urls:
            img = Image.open(url).convert('RGB')
            if self.pad_image_to_square:
                img = expand2square(
                        img,
                        tuple(
                            int(x * 255) for x in self.image_processor.image_mean))
            images.append(img)

        if len(images):
            outputs = self.image_processor(images, return_tensors='pt')
            pixel_values = outputs['pixel_values']
        else:
            pixel_values = None

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
            'num_tokens': packed_num_tokens,
            'num_img_tokens': packed_num_img_tokens
        }

        return packed


class LlavaCollator():

    def __init__(self, pack_batch=False):
        self.pack_batch = pack_batch

    def __call__(self, instances):

        pad_index = DEFAULT_PAD_TOKEN_INDEX

        input_ids = []
        labels = []
        attention_mask = []
        pixel_values = []
        num_tokens = []
        num_img_tokens = []

        for data in instances:
            input_ids.append(torch.LongTensor(data['input_ids']))
            labels.append(torch.LongTensor(data['labels']))
            num_tokens.extend(data['num_tokens'])
            num_img_tokens.extend(data['num_img_tokens'])
            if data['pixel_values'] is not None:
                pixel_values.append(data['pixel_values'])
            # breakpoint()
        attention_mask = [torch.ones_like(ids) for ids in input_ids]

        num_tokens = torch.IntTensor(num_tokens)
        num_img_tokens = torch.IntTensor(num_img_tokens)

        if len(instances) > 1 and self.pack_batch:

            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            labels = torch.cat(labels, dim=0).unsqueeze(0)
            attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)

        elif len(instances) > 1 and not self.pack_batch:

            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=pad_index)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX)
            attention_mask = pad_sequence(
                attention_mask, batch_first=True, padding_value=0)
        else:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            attention_mask = torch.stack(attention_mask)

        if len(pixel_values) > 0:
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None

        # TODO support sp
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values,
            'num_tokens': num_tokens,
            'num_img_tokens': num_img_tokens,
            'attention_mask': attention_mask.bool()
        }

        return data_dict


##################### 新 dataset ###################################################################
from .load import load_json, load_jsonl


class LlavaDatasetForNonPack(torch.utils.data.Dataset):

    def __init__(self,
                 path,
                 sample_ratio=1.0,
                 tokenize_fn=None,
                 cache_dir=None):
        super().__init__()

        if path.endswith('.json'):
            self.dataset = load_json(path)
        else:
            self.dataset = load_jsonl(path)

        self.tokenize_fn = tokenize_fn
        self.conv2length_text = {}  # using dict to speedup the calculation of token length
        self.group_length = []
        print('Calculating the length of text data...')
        for data_item in self.dataset:
            conversations = '\n'.join(
                [temp['value'] for temp in data_item['conversations']])
            str_length = len(conversations)
            if str_length not in self.conv2length_text:
                token_length = self.tokenize_fn.tokenizer(
                    conversations,
                    return_tensors='pt',
                    padding=False,
                    truncation=False,
                ).input_ids.size(1)
                self.conv2length_text[str_length] = token_length
            else:
                token_length = self.conv2length_text[str_length]
            if 'image' in data_item and data_item['image'] is not None:
                token_length += self.tokenize_fn.per_img_tokens
            else:
                token_length = -token_length
            self.group_length.append(token_length)
        print('Finished calculating the length of text data...')

        del self.conv2length_text

    @property
    def modality_length(self):
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length

    def __getitem__(self, item):
        raw_data = self.dataset[item]
        return self.tokenize_fn(raw_data)

    def __len__(self):
        return len(self.dataset)


class NewLlavaTokenizeFunction():

    def __init__(self,
                 tokenizer,
                 chat_template,
                 per_img_tokens,
                 image_dir=None,
                 raw_format='llava',
                 max_length=2048,
                 image_processor=None,
                 pad_image_to_square=False):

        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.image_dir = image_dir
        self.raw_format = raw_format
        self.per_img_tokens = per_img_tokens
        self.max_length = max_length
        self.pad_image_to_square = pad_image_to_square
        self.image_processor = image_processor

    def __call__(self, item):

        formatter = OPENAI_FORMAT_MAP[self.raw_format]
        msg = ChatMessages.from_dict(formatter(item))
        tokenized = msg.tokenize(self.tokenizer, self.chat_template)

        input_ids = tokenized['input_ids']
        labels = tokenized['labels']

        if len(input_ids) > self.max_length:
            print(f'The length of input_ids is {len(input_ids)} larger than {self.max_length}. Truncate it.')
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            tokenized['num_tokens'] = self.max_length
            tokenized['input_ids'] = input_ids
            tokenized['labels'] = labels

        images = []
        if 'image_urls' in tokenized:
            for url in tokenized.get('image_urls', []):
                if url is not None:
                    if self.image_dir:
                        url = os.path.join(self.image_dir, url)
                    img = Image.open(url).convert('RGB')
                    if self.pad_image_to_square:
                        img = expand2square(
                            img,
                            tuple(
                                int(x * 255) for x in self.image_processor.image_mean))
                    images.append(img)

        if len(images):
            num_img_tokens = [self.per_img_tokens for _ in images]
            outputs = self.image_processor(images, return_tensors='pt')
            pixel_values = outputs['pixel_values']
            tokenized['pixel_values'] = pixel_values
            tokenized['num_img_tokens'] = [sum(num_img_tokens)]
            tokenized['num_tokens'] += sum(num_img_tokens) - len(images)
        else:
            tokenized['pixel_values'] = None
            tokenized['num_img_tokens'] = [0]

        return tokenized


class NewLlavaCollator():

    def __init__(self, pad_token_id=0, ignore_id=-100, pack_batch=False):
        self.pack_batch = pack_batch
        self.pad_token_id = pad_token_id
        self.ignore_id = ignore_id

    def __call__(self, instances):

        _instances = []
        for ins in instances:
            if isinstance(ins, list):
                _instances.extend(ins)
            else:
                _instances.append(ins)

        instances = _instances

        input_ids = []
        labels = []
        num_tokens = []
        num_img_tokens = []
        pixel_values = []

        for data in instances:

            input_ids.append(torch.LongTensor(data['input_ids']))
            labels.append(torch.LongTensor(data['labels']))

            if isinstance(data['num_tokens'], int):
                num_tokens.append(data['num_tokens'])
            else:
                num_tokens.extend(data['num_tokens'])

            num_img_tokens.extend(data['num_img_tokens'])
            if data['pixel_values'] is not None:
                pixel_values.append(data['pixel_values'])

        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        num_tokens = torch.IntTensor(num_tokens)
        num_img_tokens = torch.IntTensor(num_img_tokens)

        if len(instances) > 1 and self.pack_batch:

            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            labels = torch.cat(labels, dim=0).unsqueeze(0)
            attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)

        elif len(instances) > 1 and not self.pack_batch:

            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=self.ignore_id)
            attention_mask = pad_sequence(
                attention_mask, batch_first=True, padding_value=0)
        else:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            attention_mask = torch.stack(attention_mask)

        if input_ids.shape != labels.shape:
            raise RuntimeError('The shape of input_ids and labels must be '
                               f'equal, but  found {input_ids.shape} and '
                               f'{labels.shape}.')

        if len(pixel_values) > 0:
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None

        # TODO support sp
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values,
            'num_tokens': num_tokens,
            'num_img_tokens': num_img_tokens,
            'attention_mask': attention_mask.bool()
        }

        return data_dict
