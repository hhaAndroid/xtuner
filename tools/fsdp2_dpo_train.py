# Copyright (c) OpenMMLab. All rights reserved.
import gc

gc.disable()

import warnings
import copy
import json

# display once
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*")
import argparse
import math
import os
from concurrent.futures import wait
import time
from datetime import timedelta

import torch
import sys
from collections import OrderedDict
import shutil
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather, all_reduce
from mmengine.runner import set_random_seed
from mmengine.utils import mkdir_or_exist, get_git_hash
from mmengine.utils.dl_utils import collect_env
from mmengine.logging import MessageHub

from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
from contextlib import contextmanager
from torch.utils.data import DataLoader
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_state_dict, set_state_dict)
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from xtuner._lite.parallel import (MetaStateful, get_dp_mesh, get_fsdp_mesh, ParallelSampler,
                                   get_sp_mesh, get_tp_mesh, get_world_mesh, LengthGroupedSampler,
                                   setup_parallel, pad_for_sequence_parallel, split_for_sequence_parallel)
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from xtuner._lite import get_logger, get_torch_device_module, get_device

from xtuner._lite.parallel.megatron import megatron_parallelize
from xtuner._lite.accelerate import (LoadWoInit, profile_time_and_memory, unpack_sequence)
from xtuner._lite.parallel.fsdp import clip_grad_norm_
from transformers import (AutoTokenizer, AutoModelForCausalLM)
import torch.distributed.checkpoint as dcp
from transformers.utils.import_utils import is_flash_attn_2_available

from xtuner._lite.datasets import JsonlDataset
from xtuner._lite.datasets.pack_utils import RichSoftPackDataset
from xtuner._lite.datasets.utils import move_data_to_device
from xtuner._lite import get_repo_git_info
from xtuner._lite.accelerate import dispatch_hf_code
from xtuner._lite.modelings.dpo_utils import RunningMoments

try:
    from liger_kernel.chunked_loss import LigerFusedLinearPackDPOLoss
except ImportError:
    pass

logger = get_logger()
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def parse_args():
    parser = argparse.ArgumentParser(description='Train DPO LLM')

    model_args = parser.add_argument_group('model', 'Model Related Settings')
    model_args.add_argument(
        '--model', help='repo id or local path of the model')
    model_args.add_argument(
        '--loss-type',
        default='sigmoid',
        type=str,
        help='')
    model_args.add_argument(
        '--loss-weight',
        default='1.0',
        type=str,
        help='')
    model_args.add_argument(
        '--rpo-alpha',
        default=0.0,
        type=float,
        help='')
    model_args.add_argument(
        '--beta',
        default=0.1,
        type=float,
        help='')
    model_args.add_argument(
        '--label-smoothing',
        default=0.0,
        type=float,
        help='')
    parser.add_argument('--use-liger', action='store_true')
    parser.add_argument('--use-hsdp', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')
    model_args.add_argument(
        '--dtype',
        default='auto',
        choices=['fp16', 'bf16', 'auto'],
        help=("the dtype of the model forward. When set to 'auto', it will "
              'automatically determine whether bf16 is available, '
              'prioritizing the use of bf16.'))
    model_args.add_argument(
        '--selective-recompute',
        default=1.0,
        type=float,
        help=('the ratio of re-computation for transforemer layers. '
              'The maximum is 1; the larger the value, the less memory '
              'required for training. The default is 1, meaning all layers '
              'need to be re-computated.'))
    parser.add_argument(
        '--reshard-after-forward', action='store_true', help='')
    model_args.add_argument('--sp-size', type=int, default=1, help='')
    model_args.add_argument(
        '--tp-size',
        default=1,
        type=int,
        help="tp size")
    data_args = parser.add_argument_group('data', 'Dataset Related Settings')
    data_args.add_argument(
        '--datasets',
        nargs='*',
        help=('repo id or local path or dir of the datasets. For repo ids, '
              'the `dset-sources` needs to be appropriately set to '
              '`modelscope` or `huggingface`. For local dir, all json and '
              'jsonl files will be loaded by default. The type of loaded '
              'files can be controlled by setting `dset-file-type`'))
    data_args.add_argument(
        '--dset-cache-dir',
        help=('the cache dir of the loaded datasets. When the `datasets` is '
              'set, the loaded datasets will be cached to this dir. If the '
              '`datasets` are not set, the cached dataset in this dir will be '
              'loaded.'))
    data_args.add_argument('--concat-before-pack', action='store_true')
    data_args.add_argument('--group-by-length', action='store_true')
    data_args.add_argument(
        '--max-length',
        type=int,
        default=32768,
        help=('the maximum length of each piece of data, any excess will be '
              'truncated.'))
    data_args.add_argument(
        '--pack-max-length',
        type=int,
        default=32768,
        help='the maximum length of each pack of data')
    data_args.add_argument(
        '--max-keep-ckpts',
        type=int,
        default=1,
        help='the maximum number of checkpoints to keep.')
    data_args.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='how many subprocesses to use for data loading.')
    data_args.add_argument(
        '--pack-len-type',
        default='total_block',
        choices=['total_block', 'max_block'],
        help='')
    data_args.add_argument(
        '--pack-extra-buffer-size',
        type=int,
        default=1000,
        help='')
    optim_args = parser.add_argument_group('optim', 'Optim Related Settings')
    optim_args.add_argument(
        '--mirco-batch-size',
        type=int,
        default=1,
        help='batch size for each forward + backward pass')
    optim_args.add_argument(
        '--global-batch-size',
        type=int,
        default=16,
        help='batch size for each parameter update')
    optim_args.add_argument(
        '--lr', default=4e-5, type=float, help='learning rate.')
    optim_args.add_argument(
        '--lr-min', default=0, type=float, help='min learning rate.')
    optim_args.add_argument(
        '--wd', default=0.01, type=float, help='weight decay.')
    optim_args.add_argument(
        '--max-grad-norm', default=1, type=float, help='gradient clipping')
    optim_args.add_argument(
        '-e', '--epochs', default=1, type=int, help='total training epochs.')
    optim_args.add_argument(
        '--warmup-ratio',
        default=0.03,
        type=float,
        help=('the proportion of training steps for learning rate warm-up in '
              'relation to the total training steps.'))
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='the dir to save logs and checkpoints')
    parser.add_argument(
        '--checkpoint-interval',
        default=1000,
        type=int,
        help=('how many steps to save a checkpoint; it can be a floating '
              'point number less than 1, or an integer greater than or equal '
              "to 1. When it's a floating point, it will be multiplied by the "
              'total number of training steps.'))
    parser.add_argument(
        '--checkpoint-drop-optimizer',
        action='store_true',
        help=('only model parameters are saved when saving a checkpoint. '
              'This can significantly reduce the size of checkpoint files, '
              'but the saved checkpoints cannot be resumed.'))
    parser.add_argument('--gc-interval', default=200, type=int)
    parser.add_argument(
        '--hf-interval',
        default=-1,
        type=int,
        help=('how many steps to save a hf model; it can be a floating '
              'point number less than 1, or an integer greater than or equal '
              "to 1. When it's a floating point, it will be multiplied by the "
              'total number of training steps.'))
    parser.add_argument(
        '--log-interval', default=1, type=int, help='log interval')
    parser.add_argument(
        '--resume', action='store_true', help='resume from the last checkpoint')
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed for the training')
    parser.add_argument(
        '--debug', action='store_true', help='Set logger level to `DEBUG`')
    args = parser.parse_args()
    return args


@contextmanager
def packed_sequence(num_tokens, enable=True, sp_mesh=None):
    from mmengine import MessageHub
    ctx = MessageHub.get_instance('packed_sequence')

    device = get_device()
    if enable:
        num_tokens = num_tokens.to(device)
        device = num_tokens.device
        _zero_length = torch.zeros(1, device=device)
        _pad_length = torch.cat([_zero_length, num_tokens]).int()
        cumulative_lengths = torch.cumsum(_pad_length, 0).int()
        position_ids = [torch.arange(num.item()) for num in num_tokens]
        position_ids = torch.cat(position_ids, dim=0).to(device)
        position_ids = position_ids.unsqueeze(0)

        if sp_mesh:
            # `dim` is 1 as the shape of tensor is (bs, seq_len)
            position_ids = split_for_sequence_parallel(
                position_ids, dim=1, sp_mesh=sp_mesh)

        ctx.update_info('num_tokens', num_tokens)
        ctx.update_info('position_ids', position_ids)
        ctx.update_info('cumulative_lengths', cumulative_lengths)
        ctx.update_info('max_seqlen', num_tokens.max())
        ctx.update_info('sp_mesh', sp_mesh)

    else:
        ctx.update_info('num_tokens', None)
        ctx.update_info('position_ids', None)
        ctx.update_info('cumulative_lengths', None)
        ctx.update_info('max_seqlen', None)
        ctx.update_info('sp_mesh', None)
    yield

    ctx.update_info('num_tokens', None)
    ctx.update_info('position_ids', None)
    ctx.update_info('cumulative_lengths', None)
    ctx.update_info('max_seqlen', None)
    ctx.update_info('sp_mesh', None)


def record_tensorboard(tensorboard_kargs, queue):
    writer = SummaryWriter(**tensorboard_kargs)
    i = 0
    while True:
        if not queue.empty():
            tag, value, step = queue.get()
            if tag == 'over':
                writer.close()
                break
            writer.add_scalar(tag, value, step)
            i += 1
        else:
            time.sleep(0.01)


class SummaryWriterWrapper(SummaryWriter):
    def __init__(
            self,
            # tensorboard args
            log_dir=None,
            comment="",
            purge_step=None,
            max_queue=10,
            flush_secs=120,
            filename_suffix="",
            queue_size=3000,
            only_rank0=True,
    ):
        if only_rank0 and dist.get_rank() != 0:
            self.queue = None
            self.thread = None
        else:
            tensorboard_kargs = dict(
                log_dir=log_dir,
                comment=comment,
                purge_step=purge_step,
                max_queue=max_queue,
                flush_secs=flush_secs,
                filename_suffix=filename_suffix,
            )
            ctx = mp.get_context('spawn')
            self.queue = ctx.Queue(maxsize=queue_size)
            self.thread = ctx.Process(
                target=record_tensorboard, args=(tensorboard_kargs, self.queue)
            )
            self.thread.start()

    def qsize(self):
        if self.queue is not None:
            return self.queue.qsize()
        else:
            return 0

    def add_scalar(
            self,
            tag,
            scalar_value,
            global_step=None,
            walltime=None,
            new_style=False,
            double_precision=False,
            reduce_op=None,
    ):
        if reduce_op is not None:
            scalar_value = torch.tensor(scalar_value).cuda()
            dist.all_reduce(scalar_value, op=reduce_op)
            scalar_value = scalar_value.item()
        if self.thread is not None:
            self.queue.put((tag, scalar_value, global_step))

    def add_optimize_info(self, grad_norm, inf_nan_skip_batches, lr, steps):
        self.add_scalar("optimize/grad_norm", grad_norm, global_step=steps)
        self.add_scalar("optimize/lr", lr, global_step=steps)
        self.add_scalar(
            "optimize/inf_nan_skip_batches",
            inf_nan_skip_batches,
            global_step=steps,
        )

    def add_speed_info(self, tgs, e2e_tgs, step):
        self.add_scalar("speed/tgs", tgs, step, reduce_op=None)
        self.add_scalar("speed/e2e_tgs", e2e_tgs, step, reduce_op=None)
        self.add_scalar("speed/tb_qsize", self.qsize(), step, reduce_op=None)

    def close(self):
        if self.queue is not None:
            self.queue.put(('over', 0, 0))


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


def parse_dataset_info(input_string):
    import re

    # Define the regular expression pattern,
    # considering that prompt_type and prompt_options are optional.
    pattern = r'(?P<file>.+?)::(?P<ratio>[^[]+)(?:\[(?P<type>[^\]]+)\])?(?::(?P<prompt>.+))?'  # noqa: E501
    match = re.match(pattern, input_string)

    if match:
        file_path = match.group('file')
        sample_ratio = match.group('ratio') or None
        prompt_type = match.group('type') or None
        prompt_option = match.group('prompt') or None

        return file_path, float(sample_ratio), prompt_type, prompt_option
    else:
        raise ValueError('Input string format is incorrect')


def log_format(rank, debug=False):
    formatter = f'[XTuner][RANK {rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


def check_args(args):
    if args.use_liger:
        try:
            from liger_kernel.chunked_loss import LigerFusedLinearPackDPOLoss
        except ImportError:
            raise ImportError('Please install liger_kernel to use Liger by '
                              'running `pip install git+https://github.com/hhaAndroid/Liger-Kernel.git@hha`')
        os.environ['XTUNER_USE_LIGER'] = '1'

    if args.resume_from and args.resume is False:
        args.resume = True
    if args.resume is True and args.resume_from is None:
        # find last checkpoint
        ckpt_dirs = [d for d in os.listdir(args.work_dir) if
                     os.path.isdir(os.path.join(args.work_dir, d)) and d.startswith('ckpt-')]
        if len(ckpt_dirs) > 0:
            ckpt_dirs.sort(reverse=True)
            is_success = False
            for ckpt_dir in ckpt_dirs:
                if os.path.exists(os.path.join(args.work_dir, ckpt_dir, '.metadata')):
                    args.resume_from = os.path.join(args.work_dir, ckpt_dir)
                    is_success = True
                    break
                else:
                    os.system(f'rm -rf {os.path.join(args.work_dir, ckpt_dir)}')
            if is_success is False:
                logger.warning('Did not find last_checkpoint to be resumed. training from scratch.')
                args.resume = False
        else:
            logger.warning('Did not find last_checkpoint to be resumed. training from scratch.')
            args.resume = False

    if args.resume:
        assert not args.checkpoint_drop_optimizer, '`resume` and `checkpoint_drop_optimizer` cannot be set at the same time.'

    dp_size = get_dp_mesh().size()
    world_size = get_world_mesh().size()
    if args.global_batch_size < dp_size or args.global_batch_size % dp_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         f'should be divisible by the world_size{world_size}.')

    if (args.global_batch_size / dp_size) % args.mirco_batch_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         f'should be divisible by the world_size{world_size}*'
                         f'`mirco_batch_size`({args.mirco_batch_size})')

    if args.sp_size > 1 and args.mirco_batch_size > 1:
        raise NotImplementedError('Not support mirco_batch_size>1 when sp_size')


def set_logger_envs(args):
    rank = get_world_mesh().get_rank()
    world_size = get_world_mesh().size()

    mkdir_or_exist(args.work_dir)

    log_file = os.path.join(args.work_dir, f'rank{rank}.log')

    logger.remove()
    # Change the log format printed in the terminal
    lvl = 'INFO'
    logger.add(sys.stderr, level=lvl, format=log_format(rank))
    # Change the format saved in the log file
    logger.add(log_file, format=log_format(rank), backtrace=True, catch=True)

    logger.info(args)
    if rank == 0:
        env = collect_env()
        import transformers

        import xtuner
        env['Transformers'] = transformers.__version__
        env['XTuner'] = f'{xtuner.__version__}+{get_git_hash(digits=6)}'
        runtime_env = OrderedDict()
        runtime_env.update(env)
        runtime_env['Seed'] = args.seed
        runtime_env['World Size'] = world_size

        branch, commit_id, remote_url = get_repo_git_info(os.path.dirname(os.path.abspath(__file__)))
        if branch is not None:
            runtime_env['xtuner_branch'] = branch
            runtime_env['xtuner_commit_id'] = commit_id
            runtime_env['xtuner_remote_url'] = remote_url

        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60

        logger.info('\n' + dash_line + '\nRuntime environment:' +
                    runtime_env_info + '\n' + dash_line + '\n')

        shutil.copy(__file__, args.work_dir)


def resume(args, fsdp_model, dpo_model, optimizer, warmup_scheduler, cosine_scheduler, start_step, total_steps,
           inf_nan_skip_batches):
    logger.info(f'[Resume] Resume from {args.resume_from}')
    _options = StateDictOptions(
        cpu_offload=True, ignore_frozen_params=True)
    (shard_model_state_dict,
     shard_optimizer_state_dict) = get_state_dict(
        fsdp_model, optimizer, options=_options)
    meta_stateful = MetaStateful(step=start_step, total_steps=total_steps, inf_nan_skip_batches=inf_nan_skip_batches)
    state_dict = {
        'model': shard_model_state_dict,
        'optimizer': shard_optimizer_state_dict,
        'meta_stateful': meta_stateful,
        'warmup_scheduler': warmup_scheduler,
        'cosine_scheduler': cosine_scheduler
    }
    # inplace state_dict
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=args.resume_from,
    )

    _options = StateDictOptions(
        cpu_offload=True, strict=False)
    set_state_dict(
        fsdp_model,
        optimizer,
        model_state_dict=state_dict["model"],
        optim_state_dict=state_dict["optimizer"],
        options=_options
    )

    start_step = meta_stateful['step']
    inf_nan_skip_batches = meta_stateful['inf_nan_skip_batches']
    logger.info(f'[Resume] start_step to {start_step}')

    if os.path.exists(os.path.join(args.resume_from, 'bco_pair_running.json')):
        if hasattr(dpo_model, 'running'):
            dpo_model.running.load_from_json(os.path.join(args.resume_from, 'bco_pair_running.json'))
        logger.info(f'[Resume] dpo_model bco_pair_running')
    dist.barrier()
    return start_step, inf_nan_skip_batches


def save_ckpt(args, step, total_steps, inf_nan_skip_batches, fsdp_model, rank0_model, warmup_scheduler,
              cosine_scheduler, optimizer,
              max_keep_ckpts, save_hf_ckpt_names, save_pt_ckpt_names, tokenizer, processor, dpo_model, future,
              save_pt=True,
              save_hf=True):
    digits = len(str(abs(total_steps)))
    work_dir = args.work_dir

    ckpt_id = f'{(step + 1):0{digits}}-of-{total_steps:0{digits}}'
    ckpt_dir = os.path.join(work_dir, f'ckpt-{ckpt_id}')
    hf_dir = os.path.join(work_dir, f'hf-{ckpt_id}')

    rank = dist.get_rank()
    if save_hf:
        with profile_time_and_memory('[HF Checkpoint]'):
            from torch.distributed._tensor import DTensor

            if rank == 0:
                llm_state_dict = {}

            with torch.no_grad():
                for name, param in fsdp_model.state_dict().items():
                    if isinstance(param, DTensor):
                        full_param = param.full_tensor().cpu()
                    else:
                        full_param = param.cpu()

                    if rank == 0:
                        llm_state_dict[name] = full_param

            if rank == 0:
                rank0_model.load_state_dict(llm_state_dict)
                rank0_model.save_pretrained(hf_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(hf_dir)
                if processor is not None:
                    processor.save_pretrained(hf_dir)

                try:
                    shutil.copy(os.path.join(args.model, 'modeling_internlm2.py'), hf_dir)
                    shutil.copy(os.path.join(args.model, 'tokenization_internlm2.py'), hf_dir)
                    shutil.copy(os.path.join(args.model, 'tokenization_internlm2_fast.py'), hf_dir)
                    shutil.copy(os.path.join(args.model, 'configuration_internlm2.py'), hf_dir)
                except Exception as e:
                    pass

                save_hf_ckpt_names.append(hf_dir)
                if len(save_hf_ckpt_names) > max_keep_ckpts:
                    remove_hf_ckpt_name = save_hf_ckpt_names.pop(0)
                    os.system(f'rm -rf {remove_hf_ckpt_name}')
        dist.barrier()

    if save_pt:
        if future is not None:
            wait([future])

        if args.checkpoint_drop_optimizer:
            logger.warning('[Checkpoint] The saved checkpoint cannot be '
                           'resumed. If you want to save a resumable '
                           'checkpoint, please remove '
                           '`--checkpoint-drop-optimizer` '
                           'from the command.')
        else:
            xtuner_load_timeout = timedelta(minutes=60)
            group_gloo = dist.new_group(backend='gloo', timeout=xtuner_load_timeout)
            with profile_time_and_memory('[PT Checkpoint]'):
                if dist.get_rank() == 0:
                    mkdir_or_exist(ckpt_dir)
                dist.barrier()

                if hasattr(dpo_model, 'running'):
                    dpo_model.running.save_to_json(os.path.join(ckpt_dir, 'bco_pair_running.json'))
                    dist.barrier()

                # FSDP cannot be saved via torch.save
                # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
                _options = StateDictOptions(
                    cpu_offload=True, ignore_frozen_params=True)
                (shard_model_state_dict,
                 shard_optimizer_state_dict) = get_state_dict(
                    fsdp_model, optimizer, options=_options)
                meta_stateful = MetaStateful(step=step + 1, total_steps=total_steps,
                                             inf_nan_skip_batches=inf_nan_skip_batches)
                state_dict = {
                    'model': shard_model_state_dict,
                    'optimizer': shard_optimizer_state_dict,
                    'meta_stateful': meta_stateful,
                    'warmup_scheduler': warmup_scheduler.state_dict(),
                    'cosine_scheduler': cosine_scheduler.state_dict()
                }
                future = dcp.async_save(state_dict, checkpoint_id=ckpt_dir, process_group=group_gloo)

                def send_to_oss_and_remove(future):
                    # send to oss and remove local file
                    # TODO: send to oss

                    if dist.get_rank() == 0:
                        save_pt_ckpt_names.append(ckpt_dir)
                        if len(save_pt_ckpt_names) > max_keep_ckpts:
                            remove_pt_ckpt_name = save_pt_ckpt_names.pop(0)
                            os.system(f'rm -rf {remove_pt_ckpt_name}')
                    # print('============send_to_oss_and_remove callback==================')

                future.add_done_callback(send_to_oss_and_remove)
    return future


class DPOTokenizeFunction:

    def __init__(self, tokenizer, max_length=32768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 暂时只支持 qwen2.5 和 internlm2.5
        if 'Qwen2Tokenizer' in tokenizer.__class__.__name__:
            self.tokenizer_name = 'qwen2.5'
            # You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
            self.chat_template = dict(system='<|im_start|>system\n{system}<|im_end|>\n',
                                      user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
                                      assistant='{assistant}<|im_end|>\n')
        else:
            self.tokenizer_name = 'internlm2.5'
            self.chat_template = dict(
                system='<|im_start|>system\n{system}<|im_end|>\n',
                user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
                assistant='{assistant}<|im_end|>')

        logger.info(f'[DPOTokenizeFunction] tokenizer_name: {self.tokenizer_name}')

    def process_conversations(self, conversations):
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]

        assert len(conversations) % 2 == 0, f'Invalid conversation length: {len(conversations)}'

        input_ = ''
        out_conversation = []
        for msg in conversations:
            if msg['from'] == 'human':
                input_ += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input_,
                    'output': msg['value'].strip()
                })
                input_ = ''
            else:
                raise NotImplementedError(f'Unsupported message type: {msg}')

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input_ = single_turn_conversation.get('input', '')
            if input_ is None:
                input_ = ''
            input_ = self.chat_template['user'].format(user=input_)

            if i == 0:
                # qwen2 加固定的 system
                if self.tokenizer_name == 'qwen2.5':
                    input_ = self.chat_template['system'].\
                                 format(system='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.') \
                                 + input_
                input_encode = self.tokenizer.encode(input_, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(input_, add_special_tokens=False)

            input_ids += input_encode
            labels += [-100] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            output_encode = self.chat_template['assistant'].format(assistant=output_text)
            output_encode = self.tokenizer.encode(output_encode, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            logger.info(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}')
        return input_ids, labels

    def __call__(self, data_item):

        chosen_ids, chosen_labels = self.process_conversations(data_item['chosen_conversations'])
        rejected_ids, rejected_labels = self.process_conversations(data_item['rejected_conversations'])

        ret = dict(
            chosen_ids=chosen_ids,
            rejected_ids=rejected_ids,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            num_tokens=[len(chosen_ids), len(rejected_ids)],
        )
        return ret


class DPOWrapper:
    def __init__(self, model, model_ref, sp_mesh, args):
        self.sp_mesh = sp_mesh
        self.sp_group = sp_mesh.get_group()
        self.model = model
        self.model_ref = model_ref

        loss_types = args.loss_type.split(',')  # sigmoid,bco_pair
        loss_types = [loss_type.strip() for loss_type in loss_types]
        loss_weights = args.loss_weight.split(',')  # 1.0,1.0
        loss_weights = [float(w) for w in loss_weights]

        if len(loss_weights) >= 1:
            assert len(loss_types) == len(loss_weights), 'loss_types and loss_weights should have the same length, ' \
                                                         f'loss_types={loss_types}, loss_weights={loss_weights}'
        else:
            loss_weights = [1.0] * len(loss_types)

        # ipo loss 暂时不支持组合，只能单独用
        if 'ipo' in loss_types:
            assert len(
                loss_types) == 1, 'The IPO loss does not currently support combinations and can only be used individually.'

        self.loss_types = loss_types
        self.loss_weights = loss_weights
        logger.info(f'[DPOWrapper] loss_types: {loss_types}, loss_weights: {loss_weights}')

        if "bco_pair" in self.loss_types:
            self.running = RunningMoments()

        self.use_sft_loss = args.rpo_alpha > 0
        self.rpo_alpha = args.rpo_alpha
        self.beta = args.beta
        self.label_smoothing = args.label_smoothing
        self._count = 0

        self.use_liger = args.use_liger
        if self.use_liger:
            logger.info('[DPOWrapper] ========= Use Liger ==========')
            self.liger_dpo_loss = LigerFusedLinearPackDPOLoss(
                beta=self.beta,
                rpo_alpha=self.rpo_alpha,
                compiled=True,  # 可以选择自己关闭
                loss_types=self.loss_types,
                loss_weights=self.loss_weights,
                use_ref_model=True
            )

    def _gather_masked_logits(self, logits, labels, mask):
        logits = torch.gather(
            logits.log_softmax(-1), dim=2,
            index=labels.unsqueeze(2)).squeeze(2)
        return logits * mask

    def get_pack_logps(
            self,
            all_logits_list,  # seqlen,vocab_size
            all_ref_logits_list,  # seqlen,vocab_size
            loss_mask_list,  # seqlen
    ):
        def compute_logps(_logps, _mask):
            _logps = _logps.sum(-1)
            if self.loss_types[0] == 'ipo':
                _logps /= _mask.sum(-1)
            return _logps

        (policy_chosen_logps, policy_rejected_logps, reference_chosen_logps,
         reference_rejected_logps) = [], [], [], []
        for i in range(len(all_logits_list) // 2):
            chosen = all_logits_list[2 * i]
            rejected = all_logits_list[2 * i + 1]
            chosen_ref = all_ref_logits_list[2 * i]
            rejected_ref = all_ref_logits_list[2 * i + 1]
            chosen_mask = loss_mask_list[2 * i]
            rejected_mask = loss_mask_list[2 * i + 1]

            policy_chosen_logps.append(compute_logps(chosen, chosen_mask))
            policy_rejected_logps.append(
                compute_logps(rejected, rejected_mask))
            reference_chosen_logps.append(
                compute_logps(chosen_ref, chosen_mask))
            reference_rejected_logps.append(
                compute_logps(rejected_ref, rejected_mask))

        return (torch.stack(policy_chosen_logps),
                torch.stack(policy_rejected_logps),
                torch.stack(reference_chosen_logps),
                torch.stack(reference_rejected_logps))

    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios
        if self.loss_type == 'sigmoid':
            loss = (-F.logsigmoid(self.beta * logits) *
                    (1 - self.label_smoothing) -
                    F.logsigmoid(-self.beta * logits) * self.label_smoothing)
        elif self.loss_type == 'robust':
            loss = (-F.logsigmoid(self.beta * logits) *
                    (1 - self.label_smoothing) +
                    F.logsigmoid(-self.beta * logits) *
                    self.label_smoothing) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == 'hinge':
            loss = torch.relu(1 - self.beta * logits)
        elif self.loss_type == 'ipo':
            # eqn (17) of the paper where beta is the regularization
            # parameter for the IPO loss, denoted by tau in the paper.  # noqa
            loss = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean
            loss = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == 'kto_pair':
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps -
                         reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps -
                           reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = \
                policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected)
            # is estimated using the rejected (chosen) half.  # noqa
            loss = torch.cat(
                (
                    1 - F.sigmoid(self.beta *
                                  (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta *
                                  (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.loss_type == 'sppo_hard':
            # In the paper (https://arxiv.org/pdf/2405.00675),
            # SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation
            # is conducted outside of the trainer class.
            # The version described here is the hard probability version,
            # where P in Equation (4.7) of Algorithm 1 is set to 1 for
            # the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            loss = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == 'nca_pair':
            chosen_rewards = (policy_chosen_logps -
                              reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps -
                                reference_rejected_logps) * self.beta
            loss = (-F.logsigmoid(chosen_rewards) -
                    0.5 * F.logsigmoid(-chosen_rewards) -
                    0.5 * F.logsigmoid(-rejected_rewards))
        else:
            raise ValueError(
                f'Unknown loss type: {self.loss_type}. Should be one of '
                "['sigmoid', 'hinge', 'ipo','bco_pair', 'kto_pair', "
                "'sppo_hard', 'nca_pair', 'robust']")

        # for logging
        chosen_rewards = self.beta * (
                policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (
                policy_rejected_logps - reference_rejected_logps)
        return loss, chosen_rewards, rejected_rewards

    def __call__(self, data, chosen_grad_nums=None):

        def cross_entropy_loss(logits, labels):
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels).sum()

            if torch.isnan(loss) and (labels != -100).sum() == 0:
                # When all labels are -100, the CE loss will return NaN and requires special handling.
                loss = logits.sum() * 0
                logger.info('All labels are -100, set loss to 0.')
            return loss

        ctx = MessageHub.get_instance('packed_sequence')
        sp_size = dist.get_world_size(self.sp_group)
        num_tokens = ctx.get_info('num_tokens')

        # split before
        labels = data.pop('labels')

        if self.use_liger:
            all_hidden_states = self.model(**data, use_cache=False).logits
            with torch.no_grad():
                all_ref_hidden_states = self.model_ref(**data, use_cache=False).logits

            loss, chosen_rewards, rejected_rewards, reward_margin, reward_acc, policy_nll_loss, bco_pair_rewards_batch = self.liger_dpo_loss(
                self.model.lm_head.weight,
                all_hidden_states,
                labels,
                self.model.lm_head.bias,
                all_ref_hidden_states,
                self.model_ref.lm_head.weight,
                self.model_ref.lm_head.bias,
                num_tokens,
                chosen_grad_nums,
                torch.tensor(self.running.mean) if "bco_pair" in self.loss_types else None
            )
            loss_dict = {
                'dpo_losses': loss - policy_nll_loss,
                'chosen_rewards': chosen_rewards,
                'rejected_rewards': rejected_rewards,
                'reward_margin': reward_margin,
                'reward_acc': reward_acc,
                'is_liger': True
            }
            if self.use_sft_loss:
                loss_dict['policy_nll_loss'] = policy_nll_loss

            if "bco_pair" in self.loss_types:
                self.running.update(bco_pair_rewards_batch)

            return loss_dict


        if sp_size > 1 and self.use_sft_loss:
            chosen_masks = torch.ones(num_tokens.sum(), dtype=torch.bool, device=labels.device)
            start_index = 0
            for i, num_token in enumerate(num_tokens):
                if i % 2 == 1:
                    chosen_masks[start_index:start_index + num_token] = False
                start_index += num_token.item()

            chosen_masks_pre_rank = split_for_sequence_parallel(chosen_masks, dim=0, sp_mesh=self.sp_mesh)

        if sp_size > 1:
            input_ids_pre_rank = split_for_sequence_parallel(data['input_ids'], dim=1, sp_mesh=self.sp_mesh)
            labels_pre_rank = split_for_sequence_parallel(labels, dim=1, sp_mesh=self.sp_mesh)
            data['input_ids'] = input_ids_pre_rank

        all_logits = self.model(**data, use_cache=False).logits
        with torch.no_grad():
            all_ref_logits = self.model_ref(**data, use_cache=False).logits

        # unpack the logits and labels
        if sp_size > 1:
            global_labels = labels.clone()
            global_labels[global_labels == -100] = 0
            global_loss_mask = global_labels != 0
            _labels = labels_pre_rank.clone()
            _labels[_labels == -100] = 0
            loss_mask = _labels != 0
        else:
            _labels = labels.clone()
            _labels[_labels == -100] = 0
            loss_mask = _labels != 0
            global_loss_mask = loss_mask

        policy_logps = self._gather_masked_logits(all_logits, _labels,
                                                  loss_mask)
        ref_logps = self._gather_masked_logits(all_ref_logits, _labels,
                                               loss_mask)

        if sp_size > 1:
            policy_logps = all_gather(policy_logps, self.sp_group)
            policy_logps = torch.cat(policy_logps, dim=-1)
            ref_logps = all_gather(ref_logps, self.sp_group)
            ref_logps = torch.cat(ref_logps, dim=-1)

        policy_logps_list = unpack_sequence(policy_logps, num_tokens)
        all_ref_logits_list = unpack_sequence(ref_logps, num_tokens)
        loss_mask_list = unpack_sequence(global_loss_mask, num_tokens)

        if self.use_sft_loss:
            labels_list = unpack_sequence(labels, num_tokens)
            if sp_size > 1:
                # To avoid all gather `all_logits` objects, some tricks are needed.
                chosen_logits_pre_rank = all_logits[0][chosen_masks_pre_rank]
                chosen_labels_pre_rank = labels_pre_rank[0][chosen_masks_pre_rank]
                policy_nll_loss = cross_entropy_loss(chosen_logits_pre_rank, chosen_labels_pre_rank)
                policy_nll_loss = all_reduce(policy_nll_loss, group=self.sp_group)
            else:
                # add sft loss
                all_logits_list = unpack_sequence(all_logits, num_tokens)
                shift_logits = torch.cat(all_logits_list[::2], dim=1)
                shift_labels = torch.cat(labels_list[::2], dim=1)
                policy_nll_loss = cross_entropy_loss(shift_logits, shift_labels)

        (policy_chosen_logps, policy_rejected_logps,
         reference_chosen_logps,
         reference_rejected_logps) = self.get_pack_logps(
            policy_logps_list, all_ref_logits_list, loss_mask_list)

        dpo_losses, chosen_rewards, rejected_rewards = 0, 0, 0
        current_dpo_losses = {}
        for curr_type, curr_weight in zip(self.loss_types, self.loss_weights):
            self.loss_type = curr_type
            curr_losses, curr_chosen_rewards, curr_rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            dpo_losses = dpo_losses + curr_losses * curr_weight
            chosen_rewards = chosen_rewards + curr_chosen_rewards * curr_weight
            rejected_rewards = rejected_rewards + curr_rejected_rewards * curr_weight

            if len(self.loss_types) > 1:
                current_dpo_losses[curr_type] = round(curr_losses.mean().item() * curr_weight, 2)

        if self._count % 50 == 0 and len(current_dpo_losses) > 1:
            rank = dist.get_rank()
            logger.info(f'[DPO Loss][{rank}] {current_dpo_losses}')
        self._count += 1

        reward_acc = (chosen_rewards > rejected_rewards).float()

        loss_dict = {
            'dpo_losses': dpo_losses,
            'chosen_rewards': chosen_rewards,
            'rejected_rewards': rejected_rewards,
            'reward_acc': reward_acc,
            'reward_margin': (chosen_rewards - rejected_rewards),
        }

        if self.use_sft_loss:
            policy_nll_loss = policy_nll_loss * self.rpo_alpha
            loss_dict['policy_nll_loss'] = policy_nll_loss
        return loss_dict


def packing_collate(features, pack_batch=True):
    _features = []
    # pack level
    for ins in features:
        if isinstance(ins, list):
            _features.extend(ins)
        else:
            _features.append(ins)
    features = _features

    chosen_grad_nums = torch.tensor(0)
    # dpo level
    double_features = []
    for feat in features:
        double_features.append({
            'input_ids': feat['chosen_ids'],
            'labels': feat['chosen_labels'],
            'num_tokens': [feat['num_tokens'][0]],
        })
        double_features.append({
            'input_ids': feat['rejected_ids'],
            'labels': feat['rejected_labels'],
            'num_tokens': [feat['num_tokens'][1]],
        })

        chosen_grad_nums += (torch.tensor(feat['chosen_labels'])[:-1] >= 0).sum()

    input_ids = []
    labels = []
    num_tokens = []

    for data in double_features:
        input_ids.append(torch.LongTensor(data['input_ids']))
        labels.append(torch.LongTensor(data['labels']))
        num_tokens.extend(data['num_tokens'])

    num_tokens = torch.IntTensor(num_tokens)

    if pack_batch:
        # packing
        input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        labels = torch.cat(labels, dim=0).unsqueeze(0)
    else:
        raise NotImplementedError

    data_dict = {
        'input_ids': input_ids,
        'labels': labels,
        'num_tokens': num_tokens,
        'chosen_grad_nums': chosen_grad_nums,
    }

    return data_dict


class DPOJsonlDataset(JsonlDataset):
    def _tokenize_by_offset(self, offset):
        with open(self.path, 'r') as f:
            f.seek(offset)
            data = json.loads(f.readline())
        num_tokens = self.tokenize_fn(data)['num_tokens']
        return {'num_tokens': sum(num_tokens)}


def build_model(args, dtype=torch.float32, device='cpu'):
    with torch.device(device):
        with LoadWoInit():
            llm = AutoModelForCausalLM.from_pretrained(
                args.model,
                attn_implementation='flash_attention_2',
                trust_remote_code=True,
                torch_dtype=dtype)

        llm.to(dtype)

    for module in llm.modules():
        for p_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                setattr(module, p_name, param_fp32)
    return llm


def vlm_train(args):
    assert args.tp_size == 1
    setup_parallel(tp_size=args.tp_size, sp_size=args.sp_size)
    set_random_seed(args.seed)

    dp_mesh = get_dp_mesh()
    tp_mesh = get_tp_mesh()
    sp_mesh = get_sp_mesh()
    fsdp_mesh = get_fsdp_mesh()  # dp_size * sp_size
    world_mesh = get_world_mesh()  # dp_size * sp_size * tp_size

    dp_size = dp_mesh.size()
    sp_size = sp_mesh.size()
    tp_size = tp_mesh.size()

    rank = world_mesh.get_rank()

    set_logger_envs(args)
    check_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    try:
        pad_token_id = tokenizer.pad_token_id
    except Exception as e:
        logger.warning('Tokenizer does not have pad_token_id attribute. Use 0 instead.')
        pad_token_id = 0

    with profile_time_and_memory('[Dataset & Dataloader]'):
        tokenize_fn = DPOTokenizeFunction(tokenizer, max_length=args.max_length)
        datasets = []
        for dset_info in args.datasets:
            _path, _ratio, _sys_type, _sys_prompt = parse_dataset_info(dset_info)
            _dataset = DPOJsonlDataset(_path, _ratio, tokenize_fn, cache_dir=args.dset_cache_dir,
                                       max_length=args.max_length)
            logger.info(f'[Orig Dataset] {os.path.basename(_path)}:{len(_dataset)} samples.')
            datasets.append(_dataset)

        assert is_flash_attn_2_available()
        pack_dataset = RichSoftPackDataset(datasets,
                                           target=args.pack_max_length,
                                           blend=args.concat_before_pack,
                                           pack_len_type=args.pack_len_type,
                                           pack_extra_buffer_size=args.pack_extra_buffer_size)

        if rank == 0:
            ori_samples = sum([len(dset) for dset in datasets])
            packed_samples = len(pack_dataset)
            logger.info(f'[Dataset] (Original) {ori_samples} samples.')
            logger.info(f'[Dataset] (Packed) {packed_samples} samples.')

        if args.group_by_length:
            sampler = LengthGroupedSampler(pack_dataset, dp_mesh,
                                           args.global_batch_size)
        else:
            sampler = ParallelSampler(
                pack_dataset, dp_mesh, args.global_batch_size, shuffle=True)

        train_dataloader = DataLoader(
            pack_dataset,
            batch_size=args.mirco_batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
            collate_fn=packing_collate,
            persistent_workers=args.num_workers > 0)

        if rank == 0:
            logger.info(f'[Dataloader] {len(train_dataloader)} batches.')
            _first_batch = pack_dataset[0]
            _first_batch = packing_collate(_first_batch)
            _decoded = tokenizer.batch_decode(_first_batch['input_ids'])
            logger.debug(f'[Dataloader] Training Batch:\n{_first_batch}')
            logger.debug(f'[Dataloader] Training Batch(Decoded):\n{_decoded}')

    args.dtype = 'bf16'
    dtype = torch.bfloat16
    with profile_time_and_memory('[Model]'):
        fsdp_model = build_model(args, dtype=dtype, device='meta')
        dispatch_hf_code(fsdp_model)

        if dist.get_rank() == 0:
            logger.info(fsdp_model)

        timeout = timedelta(
            minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=45)))
        group = dist.new_group(backend='gloo', timeout=timeout)
        if rank == 0:
            logger.info(f'=====[Build CPU Model]=======')
            rank0_model = build_model(args, dtype=dtype, device='cpu')
        else:
            rank0_model = None
        dist.monitored_barrier(group=group, timeout=timeout)

        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
        megatron_parallelize(fsdp_model,
                             rank0_model,
                             fsdp_mesh,
                             tp_mesh=tp_mesh,
                             mp_policy=mp_policy,
                             reshard_after_forward=True if args.reshard_after_forward else False)
        fsdp_model.train()
        if dist.get_rank() == 0:
            logger.info(fsdp_model)
    with profile_time_and_memory('[Ref Model]'):
        fsdp_ref_model = build_model(args, dtype=dtype, device='meta')
        dispatch_hf_code(fsdp_ref_model)

        fsdp_ref_model.eval()
        fsdp_ref_model.requires_grad_(False)

        megatron_parallelize(fsdp_ref_model,
                             rank0_model,
                             fsdp_mesh,
                             tp_mesh=tp_mesh,
                             mp_policy=mp_policy,
                             reshard_after_forward=True if args.reshard_after_forward else False)

    requried_grad_params = [
        param for param in fsdp_model.parameters() if param.requires_grad
    ]
    requried_grad_name = [name for name, param in fsdp_model.named_parameters() if param.requires_grad]
    if rank == 0:
        logger.info(f'[Optimizer] {requried_grad_name}')

    optimizer = AdamW(
        requried_grad_params, lr=args.lr, weight_decay=args.wd, fused=False)

    max_memory = get_torch_device_module().max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024 ** 3):.1f}GB')

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    # `iter` means once forward+backward
    # `step` means once optimizer step
    # `per_step_iters` means gradient accumulative counts
    per_step_iters = global_batch_size // mirco_batch_size // dp_size
    per_epoch_iters = len(train_dataloader)
    per_epoch_steps = math.ceil(per_epoch_iters / per_step_iters)
    logger.info(f'[Optimizer] Global batch size: {global_batch_size}, Gradient accumulative counts: {per_step_iters}')
    total_epochs = args.epochs
    total_steps = per_epoch_steps * total_epochs

    if args.checkpoint_interval == -1:
        checkpoint_interval = total_steps
    elif args.checkpoint_interval < 1:
        checkpoint_interval = int(total_steps * args.checkpoint_interval)
    else:
        checkpoint_interval = int(args.checkpoint_interval)

    if args.hf_interval == -1:
        hf_interval = total_steps
    elif args.hf_interval < 1:
        hf_interval = int(total_steps * args.hf_interval)
    else:
        hf_interval = int(args.hf_interval)

    warmup_steps = int(args.warmup_ratio * total_steps)

    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr_min)

    dpo_model = DPOWrapper(fsdp_model, fsdp_ref_model, sp_mesh, args=args)

    start_step = 0
    inf_nan_skip_batches = 0
    if args.resume:
        start_step, inf_nan_skip_batches = resume(args, fsdp_model, dpo_model, optimizer, warmup_scheduler,
                                                  cosine_scheduler, start_step, total_steps,
                                                  inf_nan_skip_batches)

    start_train_t = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()

    save_hf_ckpt_names = []
    save_pt_ckpt_names = []
    ckpt_dirs = [os.path.join(args.work_dir, d) for d in os.listdir(args.work_dir) if
                 os.path.isdir(os.path.join(args.work_dir, d)) and d.startswith('ckpt-')]
    if len(ckpt_dirs) > 0:
        ckpt_dirs.sort()
        save_pt_ckpt_names = ckpt_dirs

    hf_dirs = [os.path.join(args.work_dir, d) for d in os.listdir(args.work_dir) if
               os.path.isdir(os.path.join(args.work_dir, d)) and d.startswith('hf-')]
    if len(hf_dirs) > 0:
        hf_dirs.sort()
        save_hf_ckpt_names = hf_dirs

    max_keep_ckpts = args.max_keep_ckpts
    if max_keep_ckpts <= 0:
        max_keep_ckpts = 100000000

    if rank == 0:
        if args.sp_size > 1:
            logger.info(
                f'======= Using SP mode. sp_size: {args.sp_size}======')

        logger.info('[Train] Begin Train Loop. The current GPU memory is '
                    f'{(max_memory / 1024 ** 3):.1f}GB')
        logger.info('The FSDP adopts a lazy design, so the first iteration will be slow.')

    processor = None
    future = None

    if args.tensorboard:
        tbwriter = SummaryWriterWrapper(log_dir=args.work_dir + f'/rank_{dist.get_rank()}',
                                        only_rank0=not args.debug)
    else:
        tbwriter = None

    torch.cuda.empty_cache()

    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024 ** 3):.1f}GB')

    total_consumed_tokens = 0
    time_used_by_val_and_save_ckpt = 0

    for step in range(start_step, total_steps):
        torch.cuda.reset_peak_memory_stats()

        if is_interval(step + 1, total_steps, args.gc_interval):
            # torch.cuda.empty_cache()
            gc.collect()

        epoch = step // per_epoch_steps
        epoch_inner_step = step % per_epoch_steps
        if epoch_inner_step == 0 or step == start_step:
            # For the first step of each epoch, the data order needs to be
            # readjusted.
            # Or after resuming, for the first step, the dataloader needs to
            # be adjusted to the position before resume.
            # train_dataloader.sampler.set_epoch(epoch, inner_step)
            train_dataloader.sampler.set_epoch(epoch, epoch_inner_step * per_step_iters)
            data_iterator = iter(train_dataloader)

        if step <= warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_lr()[0]

        step_loss = 0
        step_start_t = time.time()
        step_consumed_tokens = 0
        step_consumed_img_tokens = 0

        _data_start_t = time.time()
        chosen_grad_nums = 0
        step_data_list = [next(data_iterator) for _ in range(per_step_iters)]
        if args.rpo_alpha > 0:
            for _iter in range(per_step_iters):
                _chosen_grad_nums = step_data_list[_iter].pop('chosen_grad_nums')
                chosen_grad_nums += _chosen_grad_nums.sum()
            chosen_grad_nums = chosen_grad_nums.to(DEVICE)
            dist.all_reduce(chosen_grad_nums)
            chosen_grad_nums = chosen_grad_nums / sp_size / tp_size
        else:
            for _iter in range(per_step_iters):
                step_data_list[_iter].pop('chosen_grad_nums')

        step_data_time = time.time() - _data_start_t

        reward_acc = 0
        reward_margin = 0
        chosen_rewards = 0
        rejected_rewards = 0
        nll_loss = 0

        for i in range(per_step_iters):
            data = step_data_list[i]
            data = move_data_to_device(data)

            num_tokens = data.pop('num_tokens')

            data['input_ids'] = data['input_ids'][:, :-1]
            data['labels'] = data['labels'][:, 1:]

            if num_tokens[-1] == 1:
                num_tokens = num_tokens[:-1]
            else:
                num_tokens[-1] = num_tokens[-1] - 1

            if sp_size > 1:
                input_ids = pad_for_sequence_parallel(data['input_ids'], pad_token_id, sp_mesh, dim=1)
                _num_pad = input_ids.numel() - num_tokens.sum()
                if _num_pad > 0:
                    _num_pad = num_tokens.new_tensor([_num_pad])
                    num_tokens = torch.cat([num_tokens, _num_pad], dim=-1)
                data['input_ids'] = input_ids
                data['labels'] = pad_for_sequence_parallel(data['labels'], -100, sp_mesh, dim=1)

            packed_ctx = packed_sequence(num_tokens, enable=True, sp_mesh=sp_mesh)

            with packed_ctx:
                outputs = dpo_model(data)

                # Gradient Accumulation Loss Correction
                bs = outputs['dpo_losses'].size(0)
                dpo_losses = outputs['dpo_losses'].sum() / (per_step_iters * bs)
                avg_iter_loss = dpo_losses
                if args.rpo_alpha > 0:
                    policy_nll_loss = outputs['policy_nll_loss'] / chosen_grad_nums * dp_size
                    avg_iter_loss += policy_nll_loss

                # if args.gradient_sync_after_accumulate and per_step_iters > 1:
                #     is_accumulating = i < per_step_iters - 1
                #     fsdp_model.set_is_last_backward(not is_accumulating)
                #     fsdp_model.set_requires_gradient_sync(not is_accumulating)

                avg_iter_loss.backward()

                if args.rpo_alpha > 0:
                    nll_loss += policy_nll_loss.item()

                reward_acc += outputs['reward_acc'].sum().item() / (per_step_iters * bs)
                reward_margin += outputs['reward_margin'].sum().item() / (per_step_iters * bs)
                chosen_rewards += outputs['chosen_rewards'].sum().item() / (per_step_iters * bs)
                rejected_rewards += outputs['rejected_rewards'].sum().item() / (per_step_iters * bs)

            step_loss += avg_iter_loss.item()
            if sp_size > 1:
                if _num_pad > 0:
                    step_consumed_tokens += num_tokens[:-1].sum() / sp_size / tp_size
                else:
                    step_consumed_tokens += num_tokens.sum() / sp_size / tp_size
            else:
                step_consumed_tokens += num_tokens.sum() / sp_size / tp_size

        grad_norm = clip_grad_norm_(requried_grad_params, fsdp_mesh, args.max_grad_norm)
        if grad_norm.isnan() or grad_norm.isinf():
            inf_nan_skip_batches += 1
            logger.info(f"The grad norm is NaN={grad_norm.isnan()} or Inf={grad_norm.isinf()}, skip this batch.")
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

        step_text_tokens = step_consumed_tokens - step_consumed_img_tokens
        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        tgs = int(step_consumed_tokens / step_time)
        max_memory = torch.cuda.max_memory_allocated()

        total_consumed_tokens += step_consumed_tokens
        end2end_tgs = int(total_consumed_tokens / (time.time() - start_train_t - time_used_by_val_and_save_ckpt))

        if tbwriter is not None:
            tensorboard_start_time = time.time()
            tbwriter.add_optimize_info(grad_norm.detach().clone(), inf_nan_skip_batches, cur_lr, step + 1)
            tbwriter.add_speed_info(tgs, end2end_tgs, step + 1)
            tbwriter.add_scalar('loss/total_loss', step_loss, step + 1)
            if args.rpo_alpha > 0:
                tbwriter.add_scalar('loss/nll_loss', nll_loss, step + 1)
            tbwriter.add_scalar('reward/reward_acc', reward_acc, step + 1)
            tbwriter.add_scalar('reward/reward_margin', reward_margin, step + 1)
            tbwriter.add_scalar('reward/chosen_rewards', chosen_rewards, step + 1)
            tbwriter.add_scalar('reward/rejected_rewards', rejected_rewards, step + 1)
            tensorboard_time = time.time() - tensorboard_start_time
        else:
            tensorboard_time = -1

        if is_interval(step, total_steps, args.log_interval):
            if args.rpo_alpha > 0:
                logger.info(
                    f'[Train] (Epoch {epoch}) Step {step + 1}/{total_steps}  '  # noqa: E501
                    f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                    f'grad_norm: {grad_norm:.2f}  '
                    f'reward_acc: {reward_acc:.3f}  reward_margin: {reward_margin:.3f}  '
                    f'chosen_rewards: {chosen_rewards:.3f}  rejected_rewards: {rejected_rewards:.3f}  '
                    f'nll_loss: {nll_loss:.3f} '
                    f'max_memory: {(max_memory / 1024 ** 3):.1f}GB  '
                    f'text_tokens: {step_text_tokens}  '
                    f'tgs: {tgs} e2e_tgs: {end2end_tgs} data_time: {step_data_time:.2f}s  '
                    f'time: {step_time:.2f}s  '
                    f'tb_time: {tensorboard_time:.2f}s  '
                    f'eta: {eta}')
            else:
                logger.info(
                    f'[Train] (Epoch {epoch}) Step {step + 1}/{total_steps}  '  # noqa: E501
                    f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                    f'grad_norm: {grad_norm:.2f}  '
                    f'reward_acc: {reward_acc:.3f}  reward_margin: {reward_margin:.3f}  '
                    f'chosen_rewards: {chosen_rewards:.3f}  rejected_rewards: {rejected_rewards:.3f}  '
                    f'max_memory: {(max_memory / 1024 ** 3):.1f}GB  '
                    f'text_tokens: {step_text_tokens}  '
                    f'tgs: {tgs} e2e_tgs: {end2end_tgs} data_time: {step_data_time:.2f}s  '
                    f'time: {step_time:.2f}s  '
                    f'tb_time: {tensorboard_time:.2f}s  '
                    f'eta: {eta}')

        time_before_save = time.time()
        if is_interval(step, total_steps, hf_interval):
            future = save_ckpt(args, step, total_steps, inf_nan_skip_batches, fsdp_model, rank0_model, warmup_scheduler,
                               cosine_scheduler,
                               optimizer, max_keep_ckpts, save_hf_ckpt_names, save_pt_ckpt_names, tokenizer, processor,
                               dpo_model,
                               future, save_pt=False)

        if is_interval(step, total_steps, checkpoint_interval):
            future = save_ckpt(args, step, total_steps, inf_nan_skip_batches, fsdp_model, rank0_model, warmup_scheduler,
                               cosine_scheduler,
                               optimizer, max_keep_ckpts, save_hf_ckpt_names, save_pt_ckpt_names, tokenizer, processor,
                               dpo_model,
                               future, save_hf=False)
        time_used_by_val_and_save_ckpt += time.time() - time_before_save

    if tbwriter is not None:
        tbwriter.close()

    if future is not None:
        wait([future])

    train_cost_time = time.time() - start_train_t
    m, s = divmod(train_cost_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    logger.info("[Train] Cost: %d day, %d:%d:%d" % (d, h, m, s))
    # ------------------------    Training  End  ---------------------------- #
    dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    vlm_train(args)
