import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
from xtuner._lite.accelerate.dispatches import dispatch_modules
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from xtuner._lite.parallel import (split_for_sequence_parallel)
from xtuner._lite.parallel.setup import get_sp_group, setup_parallel

import torch.distributed as dist
from mmengine.dist import is_distributed
import argparse


class CustomQwen2ForCausalLM(Qwen2ForCausalLM):

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            num_logits_to_keep=None,
            **kwargs,
    ):
        if is_distributed():
            rank = dist.get_rank(get_sp_group())
            world_size = dist.get_world_size(get_sp_group())
        else:
            rank = 0
            world_size = 1

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # 核心在于这一段
        if position_ids is not None:
            if past_key_values is not None:
                if cache_position.shape[0] == 1:  # decode 阶段
                    # 要保证除了最后一个 rank 外的其余数据 cache_position 保持不变
                    # 最后一个 rank 的 cache_position 递增
                    position_ids = cache_position.new_full((1, 1), past_key_values.init_max_position)

                    if world_size > 1:
                        # position_ids 和 input_ids 必须要和最后一个 rank 保持一致
                        dist.broadcast(position_ids, src=world_size - 1)
                        dist.broadcast(input_ids, src=world_size - 1)

                    if rank == world_size - 1:
                        past_key_values.init_max_position += 1
                    else:
                        cache_position = cache_position.new_full((1,), past_key_values.init_max_position)
                else:  # prefill 阶段
                    past_key_values.init_max_position = position_ids.max().item() + 1

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        # print("rank: ", rank, "position_ids: ", position_ids, "cache_position: ", cache_position)

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


def single_device(max_new_tokens):
    model_name = "/mnt/petrelfs/huanghaian/Qwen2.5-0.5B-Instruct"

    model = CustomQwen2ForCausalLM.from_pretrained(
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
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0).cuda()

    generated_ids = model.generate(
        input_ids=input_ids,
        position_ids=position_ids,
        cache_implementation='static',
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


def multi_device(max_new_tokens):
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

    model = CustomQwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).cuda()
    model.eval()
    dispatch_modules(model)

    rank = dist.get_rank(get_sp_group())

    generated_ids = model.generate(
        input_ids=input_ids,
        position_ids=position_ids,
        cache_implementation='static',
        max_new_tokens=max_new_tokens
    )

    world_size = dist.get_world_size(sp_group)
    if rank == world_size - 1:
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
        ]
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
