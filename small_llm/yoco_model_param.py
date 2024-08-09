from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from torch.nn.utils.rnn import pad_sequence


def calc_param_count(config):
    # 默认是 1b 参数 1073.30 M
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True)
    print(model)
    model = model.bfloat16()

    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_M = total_params / 1e6
    print(f"Total parameters: {total_params_in_M:.2f} M")

    # 统计 embeddings 和 output 层参数量
    tok_embeddings = sum(p.numel() for p in model.model.tok_embeddings.parameters())
    output_params_in_M = tok_embeddings / 1e6
    print(f"Embed layer parameters: {output_params_in_M:.2f} M")
    output = sum(p.numel() for p in model.output.parameters())
    output_params_in_M = output / 1e6
    print(f"Output layer parameters: {output_params_in_M:.2f} M")


def model_forward(config):
    # 默认是 1b 参数 1073.30 M
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True)
    print(model)
    model = model.bfloat16()

    # 模拟一次 forward 过程
    model.cuda()
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_str = "你好啊，你觉得今天天气咋样"
    input_ids = tokenizer(input_str, return_tensors='pt')['input_ids']
    print(input_ids.shape)
    output = model(input_ids.to(model.device))
    print(output.logits.shape)

    # 考虑 attention mask
    example1 = tokenizer("你好啊，你觉得今天天气咋样", return_tensors='pt')['input_ids'][0]
    example2 = tokenizer("这是一个测试代码", return_tensors='pt')['input_ids'][0]
    ori_length = [len(example1), len(example2)]
    input_ids = pad_sequence(
        [example1, example2], batch_first=True, padding_value=9000).cuda()
    attention_mask = torch.zeros_like(input_ids).bool().cuda()
    for i, len1 in enumerate(ori_length):
        attention_mask[i, :len1] = True
    print(input_ids.shape)
    output = model(input_ids, attention_mask=attention_mask)
    print(output.logits.shape)

    # # 生成超过窗口大小的数据
    input_ids = torch.randint(0, 100, (1, config.sliding_window + 200)).to(model.device)
    print(input_ids.shape)
    output = model(input_ids.to(model.device))
    print(output.logits.shape)


def model_generate(config):
    # 默认是 1b 参数 1073.30 M
    config.use_cache = True
    model = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True)
    model = model.bfloat16()

    model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_str = ["你好啊，你觉得今天天气咋样"]
    inputs = tokenizer(input_str, return_tensors='pt')

    gen_kwargs = {"max_length": 16, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.0}
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    output = model.generate(**inputs, **gen_kwargs)
    output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(output)

    # 超过窗口大小
    gen_kwargs = {"max_length": config.sliding_window + 50, "top_p": 0.8, "temperature": 0.8, "do_sample": True,
                  "repetition_penalty": 1.0}
    output = model.generate(**inputs, **gen_kwargs)
    output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(output)


if __name__ == '__main__':
    model_path = '../yoco'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, attn_implementation="flash_attention_2")

    # 修改参数进行快速调试
    config.num_hidden_layers = 24
    config.hidden_size = 1024
    config.num_self_decoder_layers = int(config.num_hidden_layers * 0.5)
    config.tie_word_embeddings = True  # 小模型要开启，否则 embedding 占比太大了

    # 计算模型参数量
    calc_param_count(config)
    # # 模拟一次生成
    # model_forward(config)
    # # 模拟生成
    # model_generate(config)
