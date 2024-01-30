# Copyright (c) OpenMMLab. All rights reserved.
import logging
from functools import partial

import copy
import numpy as np
from datasets import DatasetDict
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.utils.misc import get_object_from_string
from torch import distributed as dist

from xtuner.registry import BUILDER, MAP_FUNC
from .constants import IGNORE_INDEX
from xtuner.dataset.utils import Packer
from typing import Dict, Sequence
import torch
from .constants import DEFAULT_PAD_TOKEN_INDEX
from torch.nn.utils.rnn import pad_sequence


def process(dataset,
            tokenizer,
            max_length,
            dataset_map_fn=None,
            template_map_fn=None,
            max_dataset_length=None,
            split='train',
            remove_unused_columns=False,
            rename_maps=[],
            shuffle_before_pack=True,
            pack_to_max_length=True,
            input_ids_with_output=True,
            with_image_token=False,
            map_num_proc=32):
    """Post-process the dataset loaded from the Hugging Face Hub, or a local
    dataset.

    Args:
        dataset: The dataset to be post-processed.
        tokenizer: The tokenizer processes some raw text as input and outputs
            an Encoding.
        max_length: Max length of the sequence.
        dataset_map_fn: Map the original dataset format to the one defined
            by xTuner.
        template_map_fn: Add the prompt template to the dataset
        max_dataset_length: If the length of the dataset is too long, we can
            randomly extract `max_dataset_length` from it.
        split: Which split of the data to load.
            If `None`, will return a `dict` with all splits (typically
            `datasets.Split.TRAIN` and `datasets.Split.TEST`).
            If given, will return a single Dataset.
        remove_unused_columns: Whether to remove columns from the dataset
            that are not used during training.
        rename_maps: Rename the column name of the dataset.
        shuffle_before_pack: Whether to shuffle the dataset before
            packing them.
        pack_to_max_length: Whether to pack the dataset to the `max_length `.
            This usually improves gpu utilization and therefore reduces
            training time.
        input_ids_with_output: Whether to put the groundtruth output
            corresponding to the question into the dataset. Typically set
            it to True during training and False during testing.
        with_image_token: Whether to convert DEFAULT_IMAGE_TOKEN to
            IMAGE_TOKEN_INDEX. Typically set it to True during the training
            of VLM.
        map_num_proc: Max number of processes when mapping the dataset.
    """

    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]
    elif isinstance(dataset, dict) or isinstance(
            dataset, Config) or isinstance(dataset, ConfigDict):
        dataset = BUILDER.build(dataset)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]

    # sample `max_dataset_length` items from the original dataset to
    # save time consumed by map function
    if max_dataset_length is not None:
        max_dataset_length = min(max_dataset_length, len(dataset))
        indices = np.random.choice(
            len(dataset), max_dataset_length, replace=False)
        dataset = dataset.select(indices)

    # Extract the useful data for training from the original dataset.
    if dataset_map_fn is not None:
        if isinstance(dataset_map_fn, str):
            map_fn_obj = MAP_FUNC.get(
                dataset_map_fn) or get_object_from_string(dataset_map_fn)
            if map_fn_obj is not None:
                dataset_map_fn = map_fn_obj
            else:
                raise TypeError('dataset_map_fn must be a function or a '
                                "registered function's string in MAP_FUNC, "
                                f"but got a string of '{dataset_map_fn}'")

        dataset = dataset.map(dataset_map_fn, num_proc=map_num_proc)

    # Add prompt template, such as <|System|>: xxx <|User|>: xxx <|Bot|>: xxx
    if template_map_fn is not None:
        if isinstance(template_map_fn, dict) or isinstance(
                template_map_fn, Config) or isinstance(template_map_fn,
                                                       ConfigDict):
            template_map_fn = BUILDER.build(template_map_fn)
        dataset = dataset.map(template_map_fn, num_proc=map_num_proc)

    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)

    # remove unused columns
    if pack_to_max_length and (not remove_unused_columns):
        print_log(
            'We have to remove unused columns if '
            '`pack_to_max_length` is set to True.',
            logger='current',
            level=logging.WARNING)
        remove_unused_columns = True

    # remove invalid data
    dataset = dataset.filter(
        lambda example: len(example['conversation']) > 0,
        num_proc=map_num_proc)

    # tokenize
    if isinstance(tokenizer, dict) or isinstance(
            tokenizer, Config) or isinstance(tokenizer, ConfigDict):
        tokenizer = BUILDER.build(tokenizer)
    dataset = dataset.map(
        partial(
            encode_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            with_image_token=with_image_token,
            input_ids_with_output=input_ids_with_output),
        remove_columns=list(dataset.column_names)
        if remove_unused_columns else None,
        num_proc=map_num_proc)

    if input_ids_with_output:
        # remove data that does not have the valid labels.
        dataset = dataset.filter(
            lambda example: any(label >= 0 for label in example['labels']),
            num_proc=map_num_proc)

    # pack to max length
    if pack_to_max_length and split == 'train':
        if shuffle_before_pack:
            dataset = dataset.shuffle()
            dataset = dataset.flatten_indices(num_proc=map_num_proc)
        dataset = dataset.map(
            Packer(max_length), batched=True, num_proc=map_num_proc)

    # add 'length'
    setattr(dataset, 'length', [len(i['input_ids']) for i in dataset])

    return dataset


def encode_fn(example,
              tokenizer,
              max_length,
              input_ids_with_output=True,
              with_image_token=False):
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        bos_token_id = []
        eos_token_id = tokenizer.eos_token_id
        assert eos_token_id is not None, \
            'Please set eos_token for Qwen tokenizer!'
    elif tokenizer.__class__.__name__ == 'ChatGLMTokenizer':
        bos_token_id = [64790, 64792]
        eos_token_id = tokenizer.eos_token_id
    else:
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
    if isinstance(bos_token_id, int):
        bos_token_id = [bos_token_id]
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        # 输入不需要考虑 loss
        labels += [IGNORE_INDEX] * len(input_encode)

        if input_ids_with_output:
            # Add output (with loss)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                labels += copy.deepcopy(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}


def process_hf_dataset_rrr(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()):
        return process(*args, **kwargs)

    if dist.get_rank() == 0:
        dataset = process(*args, **kwargs)
        objects = [dataset]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def withbbox_default_collate_fn(
        instances: Sequence[Dict],
        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
        return_hf_format: bool = False) -> Dict[str, torch.Tensor]:

    gt_bboxes = [inst['bbox'] for inst in instances]

    input_ids = []
    labels = []
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    if has_image:
        pixel_values = []
    for example in instances:
        input_ids.append(torch.tensor(example['input_ids']))
        labels.append(torch.tensor(example['labels']))
        if has_image:
            pixel_values.append(example['pixel_values'])
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    data_dict = {
        'gt_bboxes': gt_bboxes,
        'input_ids': input_ids,
        'attention_mask': input_ids.ne(pad_index),
        'labels': labels
    }
    if has_image:
        pixel_values = torch.stack(pixel_values)
        data_dict['pixel_values'] = pixel_values

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
