import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner._lite import get_logger
from xtuner._lite.datasets import OPENAI_CONVERT_MAP

logger = get_logger()


class SftTokenizeFunction():

    def __init__(self, tokenizer, chat_template, raw_format='openai'):

        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.raw_format = raw_format

    def __call__(self, item):

        formatter = OPENAI_CONVERT_MAP[self.raw_format]
        msg = formatter(item)
        tokenized = msg.tokenize(self.tokenizer, self.chat_template)
        return tokenized


def pad_to_max_length(tensor, max_length, padding_value, dim=-1):
    length = tensor.shape[dim]
    pad_num = max_length - length
    if pad_num == 0:
        return tensor

    pad_shape = (*tensor.shape[:dim], pad_num,
                 *tensor.shape[dim + 1:]) if dim != -1 else (
        *tensor.shape[:dim], pad_num)
    pad = torch.full(
        pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat([tensor, pad], dim=dim)
    return tensor


class SftCollator():

    def __init__(self, pad_token_id=0, ignore_id=-100, pack_batch=False, max_length=None):
        self.pack_batch = pack_batch
        self.pad_token_id = pad_token_id
        self.ignore_id = ignore_id
        self.max_length = max_length

    def __call__(self, instances):

        batch_input_ids = []
        batch_labels = []
        batch_num_tokens = []

        for _instances in instances:
            if not isinstance(_instances, list):
                _instances = [_instances]

            input_ids = []
            labels = []
            num_tokens = []
            for _instance in _instances:
                _input_ids = _instance['input_ids']
                _labels = _instance['labels']
                _num_tokens = _instance['num_tokens']

                # TODO remove list
                if isinstance(_num_tokens, list):
                    assert len(_num_tokens) == 1
                    _num_tokens = _num_tokens[0]

                assert isinstance(_num_tokens, int)

                if len(_input_ids) > self.max_length:
                    _input_ids = _input_ids[:self.max_length]
                    _labels = _labels[:self.max_length]
                    _num_tokens = min(_num_tokens, self.max_length)

                input_ids.append(torch.LongTensor(_input_ids))
                labels.append(torch.LongTensor(_labels))
                num_tokens.append(_num_tokens)

            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            labels = torch.cat(labels, dim=0).unsqueeze(0)

            # padding 到 max_length
            pad_num = self.max_length - labels.shape[1]
            if pad_num > 0:
                input_ids = pad_to_max_length(input_ids, self.max_length, 0, dim=1)
                labels = pad_to_max_length(labels, self.max_length, -100, dim=1)
                num_tokens.append(pad_num)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_num_tokens.append(torch.IntTensor(num_tokens))

        input_ids = torch.cat(batch_input_ids, dim=0)
        labels = torch.cat(batch_labels, dim=0)

        assert input_ids.shape[1] == self.max_length

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'num_tokens': batch_num_tokens
        }
        return data_dict
