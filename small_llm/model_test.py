from transformers import Qwen2ForCausalLM, Qwen2Config
import json

config_path = 'qwen2_0_5b_config.json'  # 494.03 M
config_dict = json.load(open(config_path))
config_dict['attn_implementation'] = 'flash_attention_2'

config = Qwen2Config(**config_dict)
model = Qwen2ForCausalLM(config)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f} M")
