# Copyright (c) OpenMMLab. All rights reserved.
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN_INDEX = 0
IMAGE_TOKEN_INDEX = 92537
REGION_FEAT_TOKEN_INDEX = 92536
SEG_TOKEN_INDEX = 92535

ADD_TOKENS_DECODER = {
    "0": {
        "content": "<unk>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "1": {
        "content": "<s>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "2": {
        "content": "</s>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "92543": {
        "content": "<|im_start|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "92542": {
        "content": "<|im_end|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "92541": {
        "content": "<|action_start|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "92540": {
        "content": "<|action_end|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "92539": {
        "content": "<|interpreter|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "92538": {
        "content": "<|plugin|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "92537": {
        "content": "<image>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "92536": {
        "content": "<region_feat>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "92535": {
        "content": "<p>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
    "92534": {
        "content": "</p>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    },
}

# 增加 MAX_REGION_NUM 个 <seg> token 用于模型预测
MAX_REGION_NUM = 30
for i in range(92533, 92533-MAX_REGION_NUM, -1):
    ADD_TOKENS_DECODER[str(i)] = {
        "content": f"<seg{92533-i}>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True
    }
