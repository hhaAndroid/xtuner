# 基于 vlmevalkit 里面的数据来构建数据集，方便一点

import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from mmengine.dist import (master_only)
from torch.utils.data import Dataset

from xtuner.dataset.utils import decode_base64_to_image, expand2square
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from xtuner.registry import BUILDER
from collections import defaultdict
from mmengine.logging import print_log


def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText


def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'


def Hallusion_rating(data):
    def calc_fAcc(data):
        res = defaultdict(list)
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            res[f"{line['l2-category']}_{line['set_id']}_{line['figure_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100

    def calc_qAcc(data):
        res = defaultdict(list)
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            res[f"{line['l2-category']}_{line['set_id']}_{line['question_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100

    def calc_aAcc(data):
        return np.mean(data['score']) * 100

    data['set_id'] = [x.split('_')[3] for x in data['index']]
    data['figure_id'] = [x.split('_')[4] for x in data['index']]
    data['question_id'] = [x.split('_')[5] for x in data['index']]

    res = dict(split=[], aAcc=[], fAcc=[], qAcc=[])
    res['split'].append('Overall')
    res['aAcc'].append(calc_aAcc(data))
    res['fAcc'].append(calc_fAcc(data))
    res['qAcc'].append(calc_qAcc(data))

    if 'category' in data:
        cates = list(set(data['category']))
        for c in cates:
            sub = data[data['category'] == c]
            res['split'].append(c)
            res['aAcc'].append(calc_aAcc(sub))
            res['fAcc'].append(calc_fAcc(sub))
            res['qAcc'].append(calc_qAcc(sub))

    if 'l2-category' in data:
        cates = list(set(data['l2-category']))
        for c in cates:
            sub = data[data['l2-category'] == c]
            res['split'].append(c)
            res['aAcc'].append(calc_aAcc(sub))
            res['fAcc'].append(calc_fAcc(sub))
            res['qAcc'].append(calc_qAcc(sub))
    return res


class HallusionLLaVADataset(Dataset):

    def __init__(self, data_file, prompt_template, image_processor, tokenizer, pad_image_to_square=True, use_system=False):
        self.use_system = use_system
        self.data_file = data_file
        self.df = pd.read_csv(data_file, sep='\t')

        skip_noimg = True
        if skip_noimg:
            self.df = self.df[~pd.isna(self.df['image'])]

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
            image_path = self.df.iloc[idx]['image_path']
            question = self.df.iloc[idx]['question']
            category = self.df.iloc[idx]['category']
            l2_category = self.df.iloc[idx]['l2-category']
            answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[
                0].keys() else None

            data = {
                'img': image,
                'image_path': image_path,
                'question': question,
                'answer': answer,
                'category': category,
                'index': index,
                'l2-category': l2_category,
                'id': idx
            }
            data_list.append(data)
        return data_list

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = {'id': data['id']}

        text = data['question']
        text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if self.use_system:
            inputs = self.template.get('SYSTEM', '{system}').format(system='')
        else:
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
            cur_result['prediction'] = pred_dict['prediction']
            cur_result['category'] = filtered_rows['category']
            cur_result['index'] = filtered_rows.get('index')
            cur_result['answer'] = filtered_rows.get('answer')
            cur_result['image_path'] = filtered_rows.get('image_path')
            cur_result['l2-category'] = filtered_rows.get('l2-category')
            results.append(cur_result)

        results_df = pd.DataFrame(results)
        with pd.ExcelWriter(osp.join(work_dir, self.results_xlsx_path), engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        data = results_df.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        ans_map = {k: YOrN_Extraction(v) for k, v in zip(data['index'], data['prediction'])}
        # 不使用 gpt
        data['extracted'] = [ans_map[x] for x in data['index']]
        data['score'] = (data['answer'] == data['extracted'])

        results_df = pd.DataFrame(data)
        with pd.ExcelWriter(osp.join(work_dir, self.results_xlsx_path), engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        score = Hallusion_rating(data)
        print_log('============================================', 'current')
        print_log(score, 'current')
        print_log('============================================', 'current')
        print_log(f'YOrN_eval successfully finished evaluating', 'current')
        return score


