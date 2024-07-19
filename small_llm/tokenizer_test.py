from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tokenizer")

x = "Hey how are you doing today?"
print(tokenizer(x, return_tensors="pt"))
x = "你好呀，今天天气是真的好，你也这么觉得吧？"+tokenizer.eos_token
print(tokenizer(x, return_tensors="pt"))

# y = '是的'
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": x},
#     {"role": "assistant", "content": y},
# ]
#
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=False,
# )
# print(text)
# id = tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
# )
# print(id)
