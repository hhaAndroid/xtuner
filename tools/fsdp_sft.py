# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta

import torch
import torch.distributed.checkpoint as dcp
from mmengine import mkdir_or_exist
from mmengine.dist import init_dist
from torch.distributed._tensor import DTensor, Replicate, distribute_tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    apply_activation_checkpointing
from torch.distributed.checkpoint.state_dict import (get_state_dict,
                                                     set_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               RowwiseParallel,
                                               parallelize_module)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from xtuner._lite import AutoModelForCausalLM, AutoTokenizer, get_logger
from xtuner._lite.accelerate import packed_sequence_fwd_and_bwd
from xtuner._lite.chat import ChatTemplate
from xtuner._lite.datasets import FinetuneDataset
from xtuner._lite.parallel import ParallelSampler

layer_tp_plan = {
    'attention.wqkv': ColwiseParallel(),
    'attention.wo': RowwiseParallel(),
    'feed_forward.w1': ColwiseParallel(),
    'feed_forward.w2': RowwiseParallel(),
    'feed_forward.w3': ColwiseParallel(),
}

# from transformers import AutoModelForCausalLM

logger = get_logger()


def parallel_formatter(dp_rank, tp_rank, debug=False):

    formatter = f'[DP_RANK {dp_rank}][TP_RANK {tp_rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Group 1 description')
    model_args.add_argument('-m', '--model', help='config file name or path.')
    model_args.add_argument('-t', '--tokenizer', default=None)
    model_args.add_argument(
        '--selective-checkpointing', default=1.0, type=float)

    data_args = parser.add_argument_group('data', 'Group 1 description')
    data_args.add_argument('--dataset', help='')
    data_args.add_argument('--dataset-format', default='openai', help='')
    data_args.add_argument('--dataset-cache', help='')
    data_args.add_argument('--max-length', type=int, default=2048, help='')
    data_args.add_argument('--mirco-batch-size', type=int, default=1, help='')
    data_args.add_argument('--num-workers', type=int, default=8, help='')

    dist_args = parser.add_argument_group('dist', 'Group 1 description')
    dist_args.add_argument('--tp-size', type=int, default=1, help='')
    dist_args.add_argument('--sp-size', type=int, default=1, help='')

    optim_args = parser.add_argument_group('optimizer', 'Group 1 description')
    optim_args.add_argument(
        '--global-batch-size', type=int, default=16, help='')
    optim_args.add_argument(
        '--lr',
        '--learning-rate',
        default=4e-5,
        type=float,
        help='the dir to save logs and models')
    optim_args.add_argument('--weight-decay', default=0.01, type=float)
    optim_args.add_argument('--max-grad-norm', default=1, type=float)
    optim_args.add_argument('-e', '--epochs', default=1, type=int)
    optim_args.add_argument('--warmup-ratio', default=0.03, type=float)

    # engine_args = parser.add_argument_group('engine', 'Group 1 description')
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='the dir to save logs and models')
    parser.add_argument('--checkpoint-interval', default=10, type=int)
    parser.add_argument('--save-optimizer', default=10, type=int)
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=0, help='Random seed for the training')
    args = parser.parse_args()
    return args


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


@logger.catch
def sft(args):

    init_dist('slurm')

    world_size = int(os.environ['WORLD_SIZE'])
    dp_size = world_size // args.tp_size
    tp_size = args.tp_size

    device_mesh = init_device_mesh(
        'cuda', (dp_size, tp_size), mesh_dim_names=('dp', 'tp'))
    tp_mesh = device_mesh['tp']
    dp_mesh = device_mesh['dp']

    # For TP, input needs to be same across all TP ranks.
    # while for SP, input can be different across all ranks.
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader.
    dp_rank = dp_mesh.get_local_rank()
    tp_rank = tp_mesh.get_local_rank()

    mkdir_or_exist(args.work_dir)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    log_file = os.path.join(args.work_dir,
                            f'{timestamp}.dp{dp_rank}.tp{tp_rank}.log')
    formatter = parallel_formatter(dp_rank, tp_rank)
    # Change the log format printed in the terminal
    logger.add(sys.stderr, format=formatter)
    # Change the format saved in the log file
    logger.add(log_file, format=formatter, backtrace=True, catch=True)

    with torch.device('meta'):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.float32)
        model.config.use_cache = False

    if dp_rank == 0:
        with torch.device('cpu'):
            master_model = AutoModelForCausalLM.from_pretrained(
                args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)

        master_mods = {name: mod for name, mod in master_model.named_modules()}
        master_mod_map = {
            mod: master_mods[name]
            for name, mod in model.named_modules()
        }
    else:
        master_mod_map = None

    if args.tp_size > 1:
        for layer in model.model.layers:
            attention = layer.attention
            attention.num_heads = attention.num_heads // tp_mesh.size()
            attention.hidden_size = attention.hidden_size // tp_mesh.size()
            parallelize_module(
                module=layer,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan,
            )

        model = parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan={
                'model.tok_embeddings':
                RowwiseParallel(input_layouts=Replicate(), ),
                'output': ColwiseParallel(output_layouts=Replicate(), ),
            })

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer else args.model,
        trust_remote_code=True,
        padding_side='right')

    @torch.no_grad
    def lazy_param_init_fn(module):
        device = torch.cuda.current_device()
        module.to_empty(device=torch.cuda.current_device(), recurse=False)

        if dp_mesh.get_local_rank() == 0 and tp_mesh.get_local_rank() == 0:
            master_module = master_mod_map[module]
            master_params = {
                name: param
                for name, param in master_module.named_parameters(
                    recurse=False)
            }
            master_buffers = {
                name: buffer
                for name, buffer in master_module.named_buffers(recurse=False)
            }
        else:
            master_params = None
            master_buffers = None

        if dp_mesh.get_local_rank() == 0:

            for name, param in module.named_parameters(recurse=False):

                if isinstance(param, DTensor):

                    p_full = param.full_tensor()
                    if tp_mesh.get_local_rank() == 0:
                        p_copy = master_params[name]
                        p_copy = p_copy.to(device).to(torch.float32)
                    else:
                        p_copy = torch.empty_like(p_full)

                    mesh = param.device_mesh
                    placements = param.placements

                    p_dtensor = distribute_tensor(p_copy, mesh, placements)
                    param.data.copy_(p_dtensor)

                else:
                    if tp_mesh.get_local_rank() == 0:
                        # breakpoint()
                        p_copy = master_params[name]
                        p_copy = p_copy.to(device).to(torch.float32)
                    else:
                        p_copy = torch.empty_like(param)

                    tp_group = tp_mesh.get_group()
                    torch.distributed.broadcast(p_copy, 0, tp_group)
                    param.data.copy_(p_copy)

            for name, buffer in module.named_buffers(recurse=False):

                if isinstance(buffer, DTensor):

                    b_full = buffer.full_tensor()
                    if tp_mesh.get_local_rank() == 0:
                        b_copy = master_buffers[name]
                        b_copy = b_copy.to(device).to(torch.float32)
                    else:
                        b_copy = torch.empty_like(b_full)

                    mesh = buffer.device_mesh
                    placements = buffer.placements

                    b_dtensor = distribute_tensor(b_copy, mesh, placements)
                    buffer.data.copy_(b_dtensor)

                else:
                    if tp_mesh.get_local_rank() == 0:
                        b_copy = master_buffers[name]
                        b_copy = b_copy.to(device).to(torch.float32)
                    else:
                        b_copy = torch.empty_like(buffer)

                    tp_group = tp_mesh.get_group()
                    torch.distributed.broadcast(b_copy, 0, tp_group)
                    buffer.data.copy_(b_copy)

    shard_model = FSDP(
        model,
        device_mesh=dp_mesh,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        param_init_fn=lazy_param_init_fn,
        sync_module_states=True)

    if args.selective_checkpointing:

        def checkpoint_check_fn(submodule, target='InternLM2DecoderLayer'):
            ret = False
            if type(submodule).__name__ == target:
                if random.uniform(0, 1) < args.selective_checkpointing:
                    ret = True
            return ret

        apply_activation_checkpointing(
            shard_model, check_fn=checkpoint_check_fn)

    optimizer = AdamW(
        shard_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    chat_template = ChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>\n',
        stop_words=['<|im_end|>'])

    dataset = FinetuneDataset(
        tokenizer,
        chat_template,
        max_length=args.max_length,
        data_files=args.dataset,
        pack_to_max_length=True)

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        sampler=ParallelSampler(dataset, dp_mesh, shuffle=True),
        collate_fn=FinetuneDataset.dataloader_collate_fn,
        persistent_workers=True)

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    per_step_iters = global_batch_size // mirco_batch_size // dp_size
    per_epoch_iters = len(train_dataloader)
    per_epoch_steps = math.ceil(per_epoch_iters / per_step_iters)

    total_epochs = args.epochs
    total_steps = per_epoch_steps * total_epochs

    warmup_steps = int(args.warmup_ratio * total_steps)

    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=0)

    start_step = 0
    if args.resume:

        model_state_dict, optim_state_dict = get_state_dict(
            shard_model, optimizer)
        warump_state_dict = warmup_scheduler.state_dict()
        cosine_state_dict = cosine_scheduler.state_dict()

        state_dict = {
            'model': model_state_dict,
            'optimizer': optim_state_dict,
            'step': start_step,
            'total_steps': total_steps,
            'warmup_scheduler': warmup_scheduler.state_dict(),
            'cosine_scheduler': cosine_scheduler.state_dict()
        }
        reader = dcp.FileSystemReader(args.resume)
        dcp.load(state_dict, reader)

        if state_dict['total_steps'] != total_steps:
            raise RuntimeError

        set_state_dict(
            shard_model,
            optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict)

        warmup_scheduler.load_state_dict(warump_state_dict)
        cosine_scheduler.load_state_dict(cosine_state_dict)

        start_step = state_dict['step']

    for step in range(start_step, total_steps):

        epoch = step // per_epoch_iters
        if step + 1 % per_epoch_steps == 0 or step == start_step:
            # For the first step of each epoch, the data order needs to be
            # readjusted.
            # Or after resuming, for the first step, the dataloader needs to
            # be adjusted to the position before resume.
            inner_step = step % per_epoch_steps
            train_dataloader.sampler.set_epoch(epoch, inner_step)
            data_iterator = iter(train_dataloader)

        if step <= warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_lr()[0]

        torch.cuda.reset_peak_memory_stats()

        step_losses = []
        step_data_time = 0
        step_start_t = time.time()
        step_consumed_tokens = 0
        for i in range(per_step_iters):
            if step * per_step_iters + i + 1 == per_epoch_iters:
                break

            _data_start_t = time.time()
            data = next(data_iterator)
            step_data_time += time.time() - _data_start_t

            input_ids = data['input_ids'].cuda()

            labels = data['labels'].cuda()
            position_ids = data['position_ids'].cuda()
            unpack_sizes = data['chunk_sizes'].cuda()

            loss = packed_sequence_fwd_and_bwd(shard_model, input_ids,
                                               position_ids, labels,
                                               unpack_sizes)
            step_losses.append(loss)
            step_consumed_tokens += data['attention_mask'].sum()
        grad_norm = shard_model.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        tgs = int(step_consumed_tokens / step_time / args.tp_size)
        max_memory = torch.cuda.max_memory_allocated()
        if is_interval(step, total_steps, args.log_interval):
            step_loss = sum(step_losses) / len(step_losses)
            logger.info(f'(Epoch {epoch}) Step {step+1}/{total_steps}  '
                        f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                        f'grad_norm: {grad_norm:.2f}  '
                        f'max_memory: {(max_memory / 1024**3):.1f}GB  '
                        f'tgs: {tgs}  data_time: {step_data_time:.2f}s  '
                        f'time: {step_time:.2f}s  '
                        f'eta: {eta}')

        if is_interval(step, total_steps, args.checkpoint_interval):
            # FSDP cannot be saved via torch.load
            # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
            model_state_dict, optimizer_state_dict = get_state_dict(
                shard_model, optimizer)

            state_dict = {
                'model': model_state_dict,
                'optimizer': optimizer_state_dict,
                'step': step,
                'total_steps': total_steps,
                'warmup_scheduler': warmup_scheduler.state_dict(),
                'cosine_scheduler': cosine_scheduler.state_dict()
            }

            num_digits = len(str(abs(total_steps)))
            work_dir = args.work_dir
            ckpt_dir = os.path.join(work_dir, f'ckpt-{step:0{num_digits}}')
            writer = dcp.FileSystemWriter(ckpt_dir)
            mkdir_or_exist(ckpt_dir)
            dcp.save(state_dict, writer)


if __name__ == '__main__':

    args = parse_args()

    sft(args)
