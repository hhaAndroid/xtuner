from typing import List, Literal, Dict
from transformers import AutoTokenizer,AutoProcessor
import json
import time


if __name__ == "__main__":
    tokenizer_path = "/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B"
    processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    add_vision_id=True

    jsonl_path = '/mnt/shared-storage-user/huanghaian/code/temp/xtuner/tests/resource/qwen35_tokenize_data.jsonl'
    all_data= []
    with open(jsonl_path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    
    for j, data in enumerate(all_data):
        if j !=3:
            continue
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>当前是第 {j} 条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # messages = data['messages']
        # messages=messages[:4]
        data={"id":8,"messages": [{"role": "system", "content": "这是单轮有think+toolcall例子"}, {"role": "user", "content": "北京今天的天气如何？"},{"role": "assistant", "content": "我需要先调用一些工具才能知道", "reasoning_content": "这是 reasoning_content 内容","tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": {"location": "Boston"}}}]}, {"role": "tool", "content": "35"}], "tools": [{"type": "function", "function": {"name": "get_current_temperature", "description": "Gets the temperature at a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for"}}, "required": ["location"]}}}, {"type": "function", "function": {"name": "get_current_wind_speed", "description": "Get the current wind speed in km/h at a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the wind speed for, in the format \"City, Country\""}}, "required": ["location"]}}}]}
        hf_text = processor.apply_chat_template(data['messages'],   
                                               tools=data.get('tools'),       
                                               add_vision_id=add_vision_id,   
                                               enable_thinking=True,
                                               tokenize=False,
                                               add_generation_prompt=True)
        print(hf_text)

