# INTERNVL2

transformers 版本必须大于等于 4.45.0

以最麻烦的 InternVL2-2B 为例，因为他的 tokenizer 是 internlm2

- 如果版本是 4.44,那么训练时候会出现 `_pickle.PicklingError: Can't pickle <class 'transformers_modules.InternVL2-2B.tokenization_internlm2.InternLM2Tokenizer'>: it's not the same object as transformers_modules.InternVL2-2B.tokenization_internlm2.InternLM2Tokenizer` 错误
- 如果版本是 4.45.0，那么保存时候会出现 `KeyError: 'architectures'` 错误

基本上就无解了，为了避免这个问题，一个简单的办法就是将 transformers 版本升级到 4.45.1，然后对下载的 model 配置进行修改

```python
# InternVL2-2B/configuration_internvl_chat.py
if 'architectures' in llm_config:
    if llm_config['architectures'][0] == 'LlamaForCausalLM':
        self.llm_config = LlamaConfig(**llm_config)
    elif llm_config['architectures'][0] == 'InternLM2ForCausalLM':
        self.llm_config = InternLM2Config(**llm_config)
    else:
        raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))
```
将第 50 行修改为如上代码。

## 数据集相关

- 允许多个 jsonl，同时也强烈推荐一个类型任务的数据集单独组成一个 jsonl
- 为了便于控制数据组织和配比，采用额外的 json 文件记录信息，参考 `data/internvl2_sft.json`

数据组织格式为：

```text
{
    "docvqa_train_56k": {
        "media_root": "xxx/data/docvqa/",
        "annotation": "xxx/docvqa_train_56k_wh.jsonl",
        "repeat_time": 5, # 重复次数，暂时设置为如果大于 1 则必须是整数
      },
    "openhermes2_5_cleaned": {
        "media_root": "",
        "annotation": "xxx/openhermes2_5_cleaned.jsonl",
        "repeat_time": 0.3 # 取前 0.3 的数据，暂时不是随机抽样
      },  
      "llava150k_zh": {
        "media_root": "xxx/data/coco/",
        "annotation": "xxx/llava_instruct_150k_zh_wh.jsonl"
      }
}
```

如果在开启 `--group-by-length` 或者 `--group-by-modality-length` 或者 `soft_packing` 情况下，强制要求多模态数据必须要提供 `image_wh` 属性，否则会报错。

典型的某个多模态 jsonl 文件格式如下：

```json
{"id": 0, "image": "train/documents/1.png", "conversations": [{"from": "human", "value": "<image>\nwhat is the date mentioned in this letter?\nAnswer the question using a single word or phrase."}, {"from": "gpt", "value": "1/8/93"}],"image_wh": [[1695, 2025]]}
{"id": 1, "image": "train/documents/2.png", "conversations": [{"from": "human", "value": "<image>\nwhat is the contact person name mentioned in letter?\nAnswer the question using a single word or phrase."}, {"from": "gpt", "value": "P. Carter"}], "image_wh": [[2695, 2025]]}
```

如果是纯文本数据，则需要 image_wh

