import os

os.environ['HF_MODULES_CACHE'] = './'

from transformers import AutoTokenizer

from xtuner.dataset import LLaVADataset
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.collate_fns import default_collate_fn

# data_path = 'blip_laion_cc_sbu_558k_s.json'
data_path = 'llava_v1_5_mix665k_s.json'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = int(2048 - (336 / 14) ** 2)
llm_name_or_path = 'internlm/internlm2-chat-7b'

# 如果不指定 cache_dir，会自动下载到 ~/.cache/huggingface/ 下，其中 tokenizer 相关的在 hub 中，但是代码相关的也会重新缓存一份到 ~/.cache/huggingface/modules 下，方便导入这个代码
# 此时如果再指定 cache_dir，则会在 cache_dir 里面又保存一份, 代码运行时候优先找 cache_dir 里面的代码，
# 此时，假设你想 debug remote code，你在 cache_dir 里面修改了代码，但是代码运行时候，还是会从 ~/.cache/huggingface/modules 下面加载代码，所以你改了也没用
# 想方便启动调试，你需要设置 os.environ['HF_MODULES_CACHE'] = './'，这样就会将 cache_dir 里面代码缓存一份到 ./transformers_modules 下，你调试这个文件就可以了
# 代码缓存到当前路径下，你再进行修改就生效了，可以调试了
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    cache_dir='./internlm2-chat-7b',
    padding_side='right')

# {'SYSTEM': '<|im_start|>system\n{system}<|im_end|>\n',
# 'INSTRUCTION': '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
# 'SUFFIX': '<|im_end|>', 'SUFFIX_AS_EOS': True, 'SEP': '\n', 'STOP_WORDS': ['<|im_end|>']}
llava_dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=None,
    tokenizer=tokenizer,
    image_processor=None,
    dataset_map_fn=llava_map_fn,  # 每个 item 都会经过一次，处理得到 conversation 字段
    template_map_fn=dict(  # 前面处完后，每一个 item 都会经过一次，增加 prompt
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False)

# 最重要的 encode_fn 要是可以外部传入那就更好了，否则比较麻烦。

class_type = llava_dataset.pop('type')
llava_dataset = class_type(**llava_dataset)
print(len(llava_dataset))  # 1000
# for data in llava_dataset:
#     # dataloader_out = default_collate_fn([data])
#     # print(dataloader_out)
#     data.pop('pixel_values')
#     print(data)
#     break

# 单轮对话原始数据
# {"id": "004539375", "image": "00453/004539375.jpg",
# "conversations": [{"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
# {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}]}

# 单轮对话解析后数据
# {'id': '004539375', 'image': '00453/004539375.jpg',
# 'conversation': [{'input': '<|im_start|>user\n<image>\nRender a clear and concise summary of the photo.<|im_end|>\n<|im_start|>assistant\n', 'need_eos_token': False,
# 'output': 'select luxury furniture 3 - inch gel memory foam mattress topper<|im_end|>', 'sep': '\n'}],
# 'input_ids': [1, 92543, 1008, 364, -200, 364, 6901, 395, 2961, 454, 3690, 1206, 12279, 446, 410, 6704, 281, 92542, 364, 92543, 525, 11353, 364, 1894, 19520, 14685, 262, 308, 612, 17261, 17982, 5097, 31287, 32379, 442, 7075, 92542, 364],
# 'labels':  [-100,-100, -100, -100, -100, -100,-100, -100, -100,-100, -100,-100, -100, -100, -100, -100, -100,-100, -100,-100, -100, -100, -100, 1894, 19520, 14685, 262, 308, 612, 17261, 17982, 5097, 31287, 32379, 442, 7075, 92542, -100]}

# 1 <s> 是开始符
# 92543 <|im_start|> 这个是文本开始符
# 92542 <|im_end|> 这个作为结束符
# 364 \n 由于模板时候设置了 'SEP': '\n'，会在最后追加这个 token。
# -200 是 <image> token，这个只是占位符，实际上不在词表里面

# 多轮对话

from xtuner.dataset.samplers import LengthGroupedSampler

sampler = LengthGroupedSampler(llava_dataset, per_device_batch_size=6, length_property='modality_length')

for batch in sampler:
    print(batch)
    break
