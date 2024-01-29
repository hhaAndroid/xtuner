import os

os.environ['HF_MODULES_CACHE'] = '../'

from transformers import AutoTokenizer

from xtuner.dataset import LLaVADataset
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset import ConcatDataset
from projects.modules import RRRDataset, ADD_TOKENS_DECODER

llm_name_or_path = 'internlm/internlm2-chat-7b'

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    added_tokens_decoder=ADD_TOKENS_DECODER,
    trust_remote_code=True,
    cache_dir='../internlm2-chat-7b',
    padding_side='right')

class_type = tokenizer.pop('type')
tokenizer = class_type(**tokenizer)

vocab_length = len(tokenizer)  # 获取词表长度
print("词表长度:", vocab_length)

vocab = tokenizer.get_vocab()  # 获取词表

new_v = {}
for k, v in vocab.items():
    if 'UNUSED' in k:
        new_v[k] = v
print(new_v, len(new_v))

print(tokenizer._convert_id_to_token(92535))
print(tokenizer._convert_id_to_token(92543))
