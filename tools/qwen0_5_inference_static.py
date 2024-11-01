import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, AutoConfig
from xtuner._lite.accelerate.dispatches import dispatch_modules
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from xtuner._lite.parallel import (split_for_sequence_parallel)
from xtuner._lite.parallel.setup import get_sp_group, setup_parallel

import torch.distributed as dist
from transformers.cache_utils import StaticCache
import argparse


def single_device(max_new_tokens):
    model_name = "/mnt/petrelfs/huanghaian/Qwen2.5-0.5B-Instruct"

    config = AutoConfig.from_pretrained(model_name)
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    dispatch_modules(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "请简要介绍下什么是代码。"
    messages = [
        {"role": "system", "content": "你是 Qwen，由阿里云创建。你是一个有帮助的助手。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = model_inputs.input_ids.to(model.device)

    past_key_values = StaticCache(config,
                                  batch_size=1,
                                  max_cache_len=input_ids.shape[1] + max_new_tokens,
                                  dtype=torch.bfloat16,
                                  device='cuda')
    output_ids = []
    for i in range(max_new_tokens):
        output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = output[0]
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        input_ids = next_token_id[None]
        output_ids.append(input_ids)

    generated_ids = torch.cat(output_ids, dim=-1)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


def multi_device(max_new_tokens):
    # 分布式
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(42)

    sp_size = dist.get_world_size()
    setup_parallel(sp_size, ring_size=sp_size)

    model_name = "/mnt/petrelfs/huanghaian/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "请简要介绍什么是代码。"
    messages = [
        {"role": "system", "content": "你是 Qwen，由阿里云创建。你是一个有帮助的助手。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    input_ids = model_inputs.input_ids.cuda()
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0).cuda()

    # TODO 切分,暂时只考虑必须被整除的情况
    assert input_ids.shape[1] % sp_size == 0
    sp_group = get_sp_group()
    input_ids = split_for_sequence_parallel(input_ids, dim=1, sp_group=sp_group)
    position_ids = split_for_sequence_parallel(position_ids, dim=1, sp_group=sp_group)

    config = AutoConfig.from_pretrained(model_name)
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).cuda()
    model.eval()
    dispatch_modules(model)

    world_size = dist.get_world_size(get_sp_group())
    rank = dist.get_rank(get_sp_group())

    past_key_values = StaticCache(config,
                                  batch_size=1,
                                  max_cache_len=input_ids.shape[1] + max_new_tokens,
                                  dtype=torch.bfloat16,
                                  device='cuda')
    output_ids = []
    for i in range(max_new_tokens):
        output = model(input_ids=input_ids, past_key_values=past_key_values, position_ids=position_ids, use_cache=True)
        logits = output[0]
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        input_ids = next_token_id[None]

        dist.broadcast(input_ids, src=world_size - 1)

        position_ids += 1
        position_ids = position_ids[:, -1][None]
        dist.broadcast(position_ids, src=world_size - 1)

        output_ids.append(input_ids)

    if rank == 0:
        generated_ids = torch.cat(output_ids, dim=-1)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-s', '--single', action='store_true', help='single device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    max_new_tokens = 512
    if args.single:
        # srun -p llm_razor --gres=gpu:1 --time 1:00:00 python a.py
        single_device(max_new_tokens)
    else:
        # srun -p llm_razor --gres=gpu:2 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=16 --time 1:00:00 python a.py
        multi_device(max_new_tokens)
