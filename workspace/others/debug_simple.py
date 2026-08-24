from transformers import AutoTokenizer
import json

tokenizer_path = "/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-30B-A3B-Thinking/snapshots/7e9bbfa2c1b2059edd18160793fd421194da2c10"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# 简单 case
messages = [
    {"role": "system", "content": "这是单轮无think例子"},
    {"role": "user", "content": "这是第一个问题"},
    {"role": "assistant", "content": "我需要先调用一些工具才能知道"}
]

# 使用 HF 的 apply_chat_template
hf_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print("HF output:")
print(repr(hf_text))
print()

# 检查 special tokens 对应的 token IDs
for token_str in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]:
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    print(f"{token_str!r} -> {ids}")
