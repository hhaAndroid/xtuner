# QWENVL2

## 数据集相关

- 允许多个 jsonl，同时也强烈推荐一个类型任务的数据集单独组成一个 jsonl
- 为了便于控制数据组织和配比，采用额外的 json 文件记录信息，参考 `data/qwenvl2_sft.json`

数据组织格式为：

```text
{
    "docvqa_train_56k": {
        "media_root": "xxx/data/docvqa/",
        "annotation": "xxx/docvqa_train_56k_wh.jsonl",
        "repeat_time": 5, # 重复次数，暂时设置为如果大于 1 则必须是整数
        "max_pixels": 1073296
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

特别注意 max_pixels 参数，因为 qwenvl 采用的是原先动态分辨率设置，默认的 max_pixels 非常大，如果不进行自适应设置非常容易 OOM。
max_pixels 的设置方式为： 28 x 28 x num_token_h x num_token_w，假设想设置最大分辨率为 1036x1036(或者 2072x532)，那么 num_token_h x num_token_w= 37x37(或74x19)
通过对每个数据集设置不同的 max_pixels 或者 min_pixels，可以有效控制数据集的大小，避免 OOM。

如果在开启 `--group-by-length` 或者 `--group-by-modality-length` 或者 `soft_packing` 情况下，强制要求多模态数据必须要提供 `image_wh` 属性，否则会报错。

典型的某个多模态 jsonl 文件格式如下：

```json
{"id": 0, "image": "train/documents/1.png", "conversations": [{"from": "human", "value": "<image>\nwhat is the date mentioned in this letter?\nAnswer the question using a single word or phrase."}, {"from": "gpt", "value": "1/8/93"}],"image_wh": [[1695, 2025]]}
{"id": 1, "image": "train/documents/2.png", "conversations": [{"from": "human", "value": "<image>\nwhat is the contact person name mentioned in letter?\nAnswer the question using a single word or phrase."}, {"from": "gpt", "value": "P. Carter"}], "image_wh": [[2695, 2025]]}
```

如果是纯文本数据，则需要 image_wh

