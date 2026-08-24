from typing import List, Literal
from transformers import AutoTokenizer
import json
import os
import random
import time
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig


def data_has_conversation_timestamp(obj) -> bool:
    """Whether the record contains conversation_timestamp or conversation_timestamps (e.g. on video or text parts)."""
    if isinstance(obj, dict):
        if "conversation_timestamps" in obj:
            return True
        return any(data_has_conversation_timestamp(v) for v in obj.values())
    if isinstance(obj, list):
        return any(data_has_conversation_timestamp(x) for x in obj)
    return False


def show():
    ceph_config='/mnt/shared-storage-user/huanghaian/petreloss.conf'
    oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})

    # tokenizer_path = "/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B"
    tokenizer_path = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns2_preview_official/official_interns2_preview_stable_20260401a_256k_0_1_20000_decay_20260304d_256k_0_0_cpt_tiny_20260404_0407_normal_256k_muon/20260407153035/hf-1338-smtp"
    # tokenizer_path = "/mnt/shared-storage-user/llmrazor-share/yehaochen/InternS2Preview"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    add_vision_id = True
    tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=tokenizer_path, 
                                            chat_template="qwen3.5-vl", 
                                            add_vision_id=add_vision_id,
                                            oss_loader_cfg=oss_loader_cfg).build(tokenizer)
    


    # jsonl_path = '/mnt/shared-storage-user/huanghaian/code/temp/xtuner/tests/resource/qwen35_tokenize_data.jsonl'
    # media_root=''

    # jsonl_path='/mnt/shared-storage-user/llmrazor-share/data/llava_instruct_150k_zh_wh_new.jsonl'
    # media_root='/mnt/shared-storage-user/llmrazor-share/data/'
    
    root_path='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/meta_data/qwen35_new_demo_data.json'
    with open(root_path, 'r') as f:
        data = json.load(f)
    
    key = 'demo15'
    jsonl_path = data[key]['annotation']
    media_root = data[key].get('media_root', '')
    print(f'jsonl_path: {jsonl_path}')
    
    random.seed(42)
    n = 10  # 从大 JSONL 中随机抽样的条数（蓄水池抽样，不整文件读入内存）
    all_data: List[dict] = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            if len(all_data) < n:
                all_data.append(record)
            else:
                j = random.randint(0, i)
                if j < n:
                    all_data[j] = record
    random.shuffle(all_data)
    
    sep = "\\" * 70
    color_prefix = "\033[31m"
    color_suffix = "\033[0m"
    current_string = ""

    current_type: Literal["positive", "negative"]
    token_type: Literal["positive", "negative"]

    def flush_tokens(current_tokens: List[int]) -> str:
        if not current_tokens:
            return ""
        text = tokenizer.decode(current_tokens, skip_special_tokens=False)
        if current_type == "positive":
            return f"{color_prefix}{text}{color_suffix}"
        return text
    
    for j, data in enumerate(all_data):
        # if j in [12, 13,14]: # 里面是假图，没法真的跑
        #     continue
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>当前是第 {j+1} 条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # 判断数据是否包含 thinking 内容。只要有 reasoning_content 就认为有。
        enable_thinking = any("reasoning_content" in msg for msg in data['messages'])
        has_conversation_ts = data_has_conversation_timestamp(data)
        print(f"含 conversation_timestamp / conversation_timestamps: {has_conversation_ts}")

        ret = tokenize_fn(data, media_root=media_root)
        token_ids = ret['input_ids']
        labels = ret['labels']

        decode_str = tokenizer.decode(token_ids, skip_special_tokens=False)

        decode_str = decode_str.replace('<|image_pad|>', '')
        decode_str = decode_str.replace('<|vision_start|><|vision_end|>', '<|vision_start|><|image_pad|><|vision_end|>')
        

        current_string = ""
        current_tokens: List[int] = []
        current_type = "negative"

        for i, label in zip(token_ids, labels):
            token_type = "positive" if label >= 0 else "negative"
            if token_type != current_type:
                current_string += flush_tokens(current_tokens)
                current_type = token_type
                current_tokens = []
            current_tokens.append(i)

        current_string += flush_tokens(current_tokens)
        current_string += f"\n{sep}\n"

        current_string = current_string.replace('<|image_pad|>', '')
        current_string = current_string.replace('<|vision_start|><|vision_end|>', '<|vision_start|><|image_pad|><|vision_end|>')
        current_string = current_string.replace('<|video_pad|>', '')
        current_string = current_string.replace('<|vision_start|><|vision_end|>', '<|vision_start|><|video_pad|><|vision_end|>')

        print(current_string)
        hf_text = tokenizer.apply_chat_template(data['messages'],   
                                               tools=data.get('tools'),       
                                               add_vision_id=add_vision_id,   
                                               tokenize=False,
                                               enable_thinking=enable_thinking,
                                               add_generation_prompt=False)
        # assert decode_str == hf_text, f"自定义实现与 Hugging Face 处理结果不一致！{j}={data}"


if __name__ == "__main__":
    show()
