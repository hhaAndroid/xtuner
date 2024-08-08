from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def calc_param_count(model):
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


if __name__ == '__main__':
    model_path = '/home/PJLAB/huanghaian/yolo/jianfei/xtuner/yoco'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 修改参数进行快速调试
    config.num_hidden_layers = 4
    config.num_self_decoder_layers = int(config.num_hidden_layers*0.5)

    # 默认是 1b 参数 1073.30 M
    model = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True)
    print(model)

    # 计算模型参数量
    calc_param_count(model)

    # 模拟一次 forward 过程
    model.cuda()
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_str = "你好啊，你觉得今天天气咋样"
    input_ids = tokenizer(input_str, return_tensors='pt')['input_ids']
    output = model(input_ids.to(model.device))
    print(output)

