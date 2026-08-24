test_case = [{"content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\".", "role": "user"}]
prompts = [test_case[0]['content']] * 1


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = '/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'

print(f'{model_path=}')
tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                          trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True, 
                                             ).cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
inputs = tokenizer(prompts, return_tensors="pt")
for k,v in inputs.items():
    inputs[k] = v.cuda()
gen_kwargs = {"max_new_tokens": 16, "top_p": 1.0, "temperature": 1.0, "do_sample": False, "repetition_penalty": 1.0, "top_k": 1}
output = model.generate(**inputs, **gen_kwargs)