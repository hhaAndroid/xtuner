from transformers import AutoTokenizer
from xtuner.v1.data_proto.messages.qwen35_chat import render_content

data = {
    "id": 1,
    "messages": [{"role": "user", "content": [{"type": "image", "image": {"url": "resource/mscoco_dog_000000319154.jpg", "image_wh": [375, 500]}},{"type":"text", "text": "图片中的狗是什么颜色?"}, {"type": "image", "image": {"url": "resource/mscoco_twocat_000000039769.jpg", "image_wh": [640, 480]}},{"type": "text", "text": "图中有几只猫?"}]}]}

tokenizer_path = "/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B"
# tokenizer_path = "/mnt/shared-storage-user/llmrazor-share/yehaochen/InternS2Preview"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

hf_text = tokenizer.apply_chat_template(data['messages'],   
                                               tools=data.get('tools'),       
                                               add_vision_id=True,   
                                               tokenize=False,
                                               enable_thinking=False,
                                               add_generation_prompt=False)

print(hf_text)

xtuner_text = render_content(data['messages'][0]['content'], True, 0, 0, True)
print(xtuner_text)

