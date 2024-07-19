from transformers import Qwen2ForCausalLM, Qwen2Config
import json

config_path = 'qwen2_1_5b_config.json'
config_dict = json.load(open(config_path))
config_dict['attn_implementation'] = 'flash_attention_2'

config = Qwen2Config(**config_dict)
model = Qwen2ForCausalLM(config)
print(model)
