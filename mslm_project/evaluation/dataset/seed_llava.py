import os
import os.path as osp
import re
import string

import numpy as np
import pandas as pd
import torch
from mmengine.dist import (master_only)
from rich.console import Console
from rich.table import Table
from torch.utils.data import Dataset

from xtuner.dataset.utils import decode_base64_to_image, expand2square
from xtuner.tools.utils import is_cn_string
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from xtuner.registry import BUILDER
from collections import defaultdict
import copy as cp


def build_choices(item):
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret


def can_infer_option(answer, choices):
    verbose = os.environ.get('VERBOSE', 0)
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3 and verbose:
                print(f'A might be a quantifier in the string: {answer}.')
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def prefetch_acc(data):
    tot = defaultdict(lambda: 0)
    match = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        tot['Overall'] += 1
        tot[cate] += 1
        choices = build_choices(item)
        matched = can_infer(item['prediction'], choices)
        if matched:
            match['Overall'] += 1
            match[cate] += 1
            if matched == item['answer']:
                hit['Overall'] += 1
                hit[cate] += 1
    res = defaultdict(list)
    for k in tot.keys():
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['match'].append(match[k])
        res['hit'].append(hit[k])
        res['match_rate'].append(match[k] / tot[k] * 100)
        if match[k] == 0:
            res['acc'].append(0)
        else:
            res['acc'].append(hit[k] / tot[k] * 100)
    return res


class LLaVASEEDDataset(Dataset):

    def __init__(self, data_file, prompt_template, image_processor, tokenizer, pad_image_to_square=True):
        self.data_file = data_file
        self.df = pd.read_csv(data_file, sep='\t')

        template = prompt_template
        self.template = template

        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)
        self.pad_image_to_square = pad_image_to_square
        self.name = os.path.splitext(os.path.basename(data_file))[0]
        self.results_xlsx_path = os.path.splitext(os.path.basename(data_file))[0] + '-results.xlsx'
        self.data = self.load_data_list()

    def get_image(self, image):
        while len(image) < 16:
            image = self.df[self.df['index'] == int(image)]['image'].values
            assert len(image) == 1
            image = image[0]
        image = decode_base64_to_image(image)
        return image

    def __len__(self):
        return len(self.df)

    def load_data_list(self):
        data_list = []
        for idx in range(len(self.df)):
            index = self.df.iloc[idx]['index']
            image = self.df.iloc[idx]['image']
            question = self.df.iloc[idx]['question']
            answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[
                0].keys() else None
            category = self.df.iloc[idx]['category']

            options = {
                cand: self.load_from_df(idx, cand)
                for cand in string.ascii_uppercase
                if self.load_from_df(idx, cand) is not None
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            data = {
                'img': image,
                'question': question,
                'answer': answer,
                'category': category,
                'index': index,
                'options': options_prompt,
                'options_dict': options,
                'id': idx
            }
            data_list.append(data)
        return data_list

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = {'id': data['id']}

        text = data['question'] + '\n' + data['options']
        text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if is_cn_string(text):
            text = text + '请直接回答选项字母。'
        else:
            text = text + ("Answer with the option's letter from the "
                           'given choices directly.')

        inputs = ''
        inputs += self.template['INSTRUCTION'].format(input=text, round=1)

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids)
        data_dict['input_ids'] = ids

        image = self.get_image(data['img']).convert('RGB')
        if self.pad_image_to_square:
            image = expand2square(
                image,
                tuple(
                    int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        data_dict['pixel_values'] = image

        return data_dict

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None

    @master_only
    def postprocess_results(self, result, work_dir, timestamp, show=True):

        # 拼接数据
        orig_index = [x['id'] for x in self.data]
        results = []
        for pred_dict in result:
            index = pred_dict['id']
            new_index = orig_index.index(index)
            filtered_rows = self.data[new_index]

            cur_result = {}
            cur_result['question'] = filtered_rows.get('question')
            cur_result.update(filtered_rows.get('options_dict'))
            cur_result['prediction'] = pred_dict['prediction']
            if filtered_rows.get('category') is not None:
                cur_result['category'] = filtered_rows.get('category')
            cur_result['index'] = filtered_rows.get('index')
            cur_result['answer'] = filtered_rows.get('answer')
            results.append(cur_result)

        results_df = pd.DataFrame(results)
        with pd.ExcelWriter(osp.join(work_dir, self.results_xlsx_path), engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        res = prefetch_acc(results_df)
        print(res)
        return res
