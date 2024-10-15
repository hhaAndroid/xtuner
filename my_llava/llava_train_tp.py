# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import os
import shutil
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate.utils import set_module_tensor_to_device
from datasets import Dataset
from mmengine import load, mkdir_or_exist
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
from peft import LoraConfig, get_peft_model
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    apply_activation_checkpointing
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_state_dict, set_state_dict)
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import _or_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          CLIPVisionModel, LlavaConfig, CLIPImageProcessor,
                          LlavaForConditionalGeneration, LlavaProcessor)
from transformers.utils.import_utils import (is_flash_attn_2_available,
                                             is_torch_sdpa_available)
from torch.distributed.nn.functional import all_gather
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               RowwiseParallel,
                                               parallelize_module)
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import loss_parallel

from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.accelerate import (LORA_TARGET_MAP, LoadWoInit,
                                     dispatch_modules, packed_sequence)
from xtuner._lite.accelerate.fsdp import (RECOMPUTE_MODULES,
                                          all_required_grad_wrap_policy,
                                          checkpoint_check_fn, dp_lazy_init,
                                          dp_tp_lazy_init,
                                          layer_auto_wrap_policy)
from xtuner._lite.chat import CHAT_TEMPLATE_MAP
from xtuner._lite.datasets import (LlavaCollator, LlavaRawDataset,
                                   LlavaTokenizeFunction, SoftPackerForLlava)
from xtuner._lite.datasets.load import (LOAD_FN_MAP, load_datasets,
                                        load_from_cache)
from xtuner._lite.modelings import register_remote_code
from xtuner._lite.parallel import (LengthGroupedSampler, ParallelSampler,
                                   get_dp_mesh, get_tp_mesh, setup_parallel,
                                   get_tp_world_size, get_tp_group)
from llava_model import LlavaForConditionalGeneration
from torch.distributed.checkpoint.stateful import Stateful
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX

logger = get_logger()


def log_format(rank, debug=False):
    formatter = f'[XTuner][RANK {rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


def new_conv2d_forward(self, input):
    tp_world_size = get_tp_world_size()
    tp_group = get_tp_group()
    tp_weight = self.weight.chunk(tp_world_size, dim=0)[dist.get_rank(tp_group)]
    output = F.conv2d(input, tp_weight, None, self.stride,
                      self.padding, self.dilation, self.groups)
    if tp_world_size > 1:
        output = all_gather(output, tp_group)
        output = torch.cat(output, dim=1)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Model Related Settings')

    # pretrain
    model_args.add_argument('--llm', help='repo id or local path of the model')
    model_args.add_argument(
        '--vit',
        default='openai/clip-vit-large-patch14-336',
        help='repo id or local path of the model')
    # sft
    model_args.add_argument(
        '--llava', help='repo id or local path of the model')

    model_args.add_argument(
        '-t',
        '--tokenizer',
        help=('repo id or local path of the tokenizer. '
              'Defaults to the same as `model`'))
    model_args.add_argument(
        '--chat-template',
        choices=CHAT_TEMPLATE_MAP.keys(),
        help=('repo id or local path of the tokenizer. '
              'Defaults to the same as `model`'))
    model_args.add_argument(
        '--tp-size', type=int, default=1, help='Tensor Parallel Size')
    model_args.add_argument(
        '--tp-vit',
        action='store_true',
        help="to apply tensor parallel to vit's parameters")
    model_args.add_argument(
        '--loss-parallel',
        action='store_true',
        help="to apply loss parallel")
    model_args.add_argument(
        '--freeze-llm',
        action='store_true',
        help="Not updating LLM's parameters")
    model_args.add_argument(
        '--freeze-vit',
        action='store_true',
        help="Not updating vit's parameters")
    model_args.add_argument(
        '--llm-use-lora',
        action='store_true',
        help='Apply the adapter to LLM.')
    model_args.add_argument(
        '--llm-lora-targets',
        default=None,
        nargs='*',
        help='The names of the modules to apply the adapter to. ')
    model_args.add_argument(
        '--llm-lora-r',
        default=64,
        type=int,
        help="Not updating vit's parameters")
    model_args.add_argument(
        '--llm-lora-alpha',
        default=16,
        type=int,
        help='The alpha parameter for Lora scaling.')
    model_args.add_argument(
        '--llm-lora-dropout',
        default=0.1,
        type=float,
        help='The dropout probability for Lora layers.')
    model_args.add_argument(
        '--llm-lora-bias',
        default='none',
        help='The dropout probability for Lora layers.')
    model_args.add_argument(
        '--vit-use-lora',
        action='store_true',
        help='Apply the adapter to Vit.')
    model_args.add_argument(
        '--vit-lora-targets',
        default=None,
        type=str,
        help='The names of the modules to apply the adapter to. ')
    model_args.add_argument(
        '--vit-lora-r',
        default=64,
        type=int,
        help="Not updating vit's parameters")
    model_args.add_argument(
        '--vit-lora-alpha',
        default=16,
        type=int,
        help='The alpha parameter for vit Lora scaling.')
    model_args.add_argument(
        '--vit-lora-dropout',
        default=0.1,
        type=float,
        help='The dropout probability for vit Lora layers.')
    model_args.add_argument(
        '--vit-lora-bias',
        default='none',
        help='The dropout probability for Lora layers.')
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
    model_args.add_argument(
        '--shard-strategy',
        default='full',
        choices=['full', 'hybrid', 'no', 'zero2'],
        help=('The sharding strategy to be used for distributed training.'))
    data_args = parser.add_argument_group('data', 'Dataset Related Settings')
    data_args.add_argument(
        '--datasets',
        help=('repo id or local path or dir of the datasets. For repo ids, '
              'the `dset-sources` needs to be appropriately set to '
              '`modelscope` or `huggingface`. For local dir, all json and '
              'jsonl files will be loaded by default. The type of loaded '
              'files can be controlled by setting `dset-file-type`'))
    data_args.add_argument(
        '--dset-file-types',
        nargs='*',
        default=LOAD_FN_MAP.keys(),
        choices=LOAD_FN_MAP.keys(),
        help='the file type that needs to be loaded')
    data_args.add_argument(
        '--dset-cache-dir',
        help=('the cache dir of the loaded datasets. When the `datasets` is '
              'set, the loaded datasets will be cached to this dir. If the '
              '`datasets` are not set, the cached dataset in this dir will be '
              'loaded.'))
    data_args.add_argument(
        '--dset-from-cache',
        action='store_true',
        help=('Load data directly from `dset-cache-dir`. This can save time '
              'on online tokenization, but if the tokenizer changed, '
              'recaching is needed.'))
    data_args.add_argument(
        '--dset-pack-level',
        choices=['soft'],
        help=('the level of data packing. When `hard`, multiple data will be '
              'packed to `max_length`, potentially causing some data to be '
              'truncated, and the length of the packed data will always '
              'be `max_length`; When `soft`, it will pack multiple  data '
              'into nearly `max_length` without truncating the data.'))
    data_args.add_argument('--group-by-length', action='store_true')
    data_args.add_argument('--group-by-modality-length', action='store_true')
    data_args.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help=('the maximum length of each piece of data, any excess will be '
              'truncated.'))
    data_args.add_argument(
        '--pack-max-length',
        type=int,
        default=2048,
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
        '--num-proc',
        type=int,
        default=8,
        help='how many subprocesses to use for data mapping.')

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

    parser.add_argument('-c', '--config', default=None)
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='the dir to save logs and checkpoints')
    parser.add_argument(
        '--checkpoint-interval',
        default=0.25,
        type=float,
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


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


def map_meta_modules(model, meta_model):
    modules = {name: mod for name, mod in model.named_modules()}
    meta_module_map = {
        mod: modules[name]
        for name, mod in meta_model.named_modules()
    }
    return meta_module_map


def build_llava_model(args, config, world_size, dtype=torch.float32,
                      tokenizer=None, device='cpu', resize_emb=False,
                      is_pretrain=False):
    with torch.device(device):
        _cfg = copy.deepcopy(config)

        if is_pretrain:
            llava = LlavaForConditionalGeneration(_cfg)
            if device != 'meta':
                del llava.language_model
                del llava.vision_tower
                with LoadWoInit():
                    llm = AutoModelForCausalLM.from_pretrained(
                        args.llm, config=_cfg.text_config,
                        attn_implementation=_cfg._attn_implementation)
                    vit = CLIPVisionModel.from_pretrained(
                        args.vit, config=_cfg.vision_config)
                llava.language_model = llm
                llava.vision_tower = vit
        else:
            with LoadWoInit():
                llava = LlavaForConditionalGeneration.from_pretrained(
                    args.llava, config=_cfg)

        llava.to(dtype)

        if resize_emb:
            ori_emb_shape = llava.get_input_embeddings().weight.shape
            llava.resize_token_embeddings(len(tokenizer))
            new_emb_shape = llava.get_input_embeddings().weight.shape
            logger.info('Pad the parameters of `embbedings` and `output` from '
                        f'shape {ori_emb_shape} to shape {new_emb_shape}')

        if args.freeze_llm or args.llm_use_lora:
            llava.language_model.eval()
            llava.language_model.requires_grad_(False)
            llava.language_model.to(dtype)

        if args.freeze_vit or args.vit_use_lora:
            llava.vision_tower.eval()
            llava.vision_tower.requires_grad_(False)
            llava.vision_tower.to(dtype)

        if args.llm_use_lora:
            llm = llava.language_model
            if args.llm_lora_targets is None:
                llm_cls = llm.__class__.__name__
                args.llm_lora_targets = LORA_TARGET_MAP[llm_cls]
            llm_lora_cfg = LoraConfig(
                target_modules=args.llm_lora_targets,
                r=args.llm_lora_r,
                lora_alpha=args.llm_lora_alpha,
                lora_dropout=args.llm_lora_dropout,
                bias=args.llm_lora_bias,
                task_type='CAUSAL_LM')
            lora_llm = get_peft_model(llm, llm_lora_cfg)
            llava.language_model = lora_llm

        if args.vit_use_lora:
            vit = llava.vision_tower
            if args.vit_lora_targets is None:
                vit_cls = vit.__class__.__name__
                args.vit_lora_targets = LORA_TARGET_MAP[vit_cls]
            vit_lora_cfg = LoraConfig(
                target_modules=args.vit_lora_targets,
                r=args.vit_lora_r,
                lora_alpha=args.vit_lora_alpha,
                lora_dropout=args.vit_lora_dropout,
                bias=args.vit_lora_bias,
            )
            llava.vision_tower = get_peft_model(vit, vit_lora_cfg)

    return llava


class MetaStateful(Stateful):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def state_dict(self):
        return self.kwargs

    def load_state_dict(self, state_dict) -> None:
        self.kwargs = state_dict

    def __getitem__(self, key):
        return self.kwargs[key]


# @logger.catch
def llava_sft(args):
    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    if args.llm_use_lora:
        args.freeze_llm = True

    if args.vit_use_lora:
        args.freeze_vit = True

    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(args.seed)

    world_size = int(os.environ['WORLD_SIZE'])
    tp_size = args.tp_size
    dp_size = world_size // tp_size

    assert args.pack_max_length % args.max_length == 0

    if args.llm is not None:
        is_pretrain = True
        pad_image_to_square = False
        assert args.vit, 'Please specify the `vit` model.'
        logger.info(f'============ Pretraining mode ============')
    else:
        is_pretrain = False
        pad_image_to_square = True
        assert args.llava, 'Please specify the `llava` model.'
        logger.info(f'============ SFT mode ============')

    # 如果 args.resume 和 args.resume_from 同时都设置了，则以 args.resume_from 为准
    if args.resume_from and args.resume is False:
        args.resume = True
    if args.resume is True and args.resume_from is None:
        # find last checkpoint
        save_file = os.path.join(args.work_dir, 'last_checkpoint')
        if os.path.exists(save_file):
            with open(save_file) as f:
                args.resume_from = f.read().strip()
        else:
            logger.warning('Did not find last_checkpoint to be resumed. training from scratch.')
            args.resume = False
    if args.resume:
        assert not args.checkpoint_drop_optimizer, '`resume` and `checkpoint_drop_optimizer` cannot be set at the same time.'

    if args.global_batch_size < dp_size or args.global_batch_size % dp_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         f'should be divisible by the world_size{world_size}.')

    if (args.global_batch_size / dp_size) % args.mirco_batch_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         f'should be divisible by the world_size{world_size}*'
                         f'`mirco_batch_size`({args.mirco_batch_size})')

    # During data packing, it is essential to tokenize the data in
    # advance, cache the tokenized data, so that it can be quickly
    # loaded for the second training without the need to re-tokenize.
    if args.dset_cache_dir and os.path.isdir(args.dset_cache_dir):
        if len(os.listdir(args.dset_cache_dir)):
            logger.warning(f'`{args.dset_cache_dir}` is not an empty '
                           'folder, which may lead to inaccurate '
                           'cache results.')

    if args.group_by_length and args.group_by_modality_length:
        print('if you set both `group_by_length` and `group_by_modality_length`,'
              ' the `group_by_modality_length` will be used.')

    setup_parallel(tp_size=tp_size)
    dp_mesh = get_dp_mesh()
    tp_mesh = get_tp_mesh()

    rank = dist.get_rank()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    objects = [timestamp]
    dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    args.work_dir = os.path.join(args.work_dir, timestamp)
    mkdir_or_exist(args.work_dir)

    log_file = os.path.join(args.work_dir, f'rank{rank}.log')

    # Change the log format printed in the terminal
    lvl = 'DEBUG' if args.debug else 'INFO'
    logger.add(sys.stderr, level=lvl, format=log_format(rank, args.debug))
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
        runtime_env['Distributed launcher'] = dist_launcher

        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        logger.info('\n' + dash_line + '\nRuntime environment:' +
                    runtime_env_info + '\n' + dash_line + '\n')

        shutil.copy(__file__, args.work_dir)
    # -------------------    Environment  End  ------------------------------ #
    chat_template = CHAT_TEMPLATE_MAP[args.chat_template]

    if args.tokenizer is None:
        if is_pretrain:
            args.tokenizer = args.llm
        else:
            args.tokenizer = args.llava
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        padding_side='right')

    # If you directly use the pre-trained tokenizer, you may encounter
    # a pickle error of InternLM2TokenizerFast, but the reason is currently unclear.
    # Therefore, we need to append the tokenizer again.
    img_token = '<image>'
    tokenizer.add_tokens([img_token], special_tokens=True)
    img_token_id = tokenizer.convert_tokens_to_ids([img_token])[0]
    logger.info(f'[Tokenizer] Added a new token `{img_token}`, '
                f'token id is {img_token_id}, the new vocab size is '
                f'{len(tokenizer)}')

    register_remote_code()
    if is_pretrain:
        _text_config = AutoConfig.from_pretrained(args.llm)
        _vision_config = AutoConfig.from_pretrained(args.vit).vision_config
        llava_config = LlavaConfig(
            _vision_config, _text_config,
            image_token_index=img_token_id,
            pad_token_id=DEFAULT_PAD_TOKEN_INDEX)
        _img_processor = CLIPImageProcessor.from_pretrained(args.vit)
        processor = LlavaProcessor(_img_processor, tokenizer)
    else:
        llava_config = AutoConfig.from_pretrained(args.llava)
        if hasattr(llava_config.text_config, 'auto_map'):
            delattr(llava_config.text_config, 'auto_map')
        processor = AutoProcessor.from_pretrained(
            args.llava, trust_remote_code=True)

    img_processor = processor.image_processor
    _crop_size = img_processor.crop_size
    patch_size = llava_config.vision_config.patch_size
    img_size = (_crop_size['height'], _crop_size['width'])
    per_img_tokens = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    need_resize_emb = False
    _llm_vocab_size = llava_config.text_config.vocab_size
    if _llm_vocab_size < len(tokenizer):
        need_resize_emb = True
    ###########################################################################
    #                     2. Dataset & Dataloader                             #
    ###########################################################################
    start_load_data_t = time.time()
    if args.dset_from_cache:
        _datasets = load_from_cache(args.dset_cache_dir)
        dist.barrier()
    else:
        dset_infos = load(args.datasets)

        sample_ratios = []
        annotations = []
        init_fns = []
        tokenize_fns = []
        for _, info in dset_infos.items():
            if 'format' in info:
                dset_format = info['format']
            else:
                dset_format = 'llava'

            if 'image_dir' in info:
                image_dir = info['image_dir']
            else:
                image_dir = None

            # If your data format is not in `SUPPORT_DATA_FORMATS`, you should
            # redefine a `tokenize_fn`, defining how to convert a piece of raw
            # data into tokenized data.
            # The tokenized data must include `input_ids`, `labels``,
            # and `num_tokens`.
            tokenize_fn = LlavaTokenizeFunction(tokenizer, chat_template,
                                                per_img_tokens, image_dir,
                                                dset_format, max_length=args.max_length)

            if args.dset_pack_level:
                init_fn = Dataset.from_list
            else:
                init_fn = partial(
                    LlavaRawDataset,
                    image_processor=img_processor,
                    pad_image_to_square=pad_image_to_square,
                    tokenize_fn=tokenize_fn)
                # Online tokenization is used when not using a pack dataset,
                # saving startup time.
                tokenize_fn = None

            init_fns.append(init_fn)
            tokenize_fns.append(tokenize_fn)
            sample_ratios.append(info['sample_ratio'])
            annotations.append(info['annotations'])

        _datasets = load_datasets(
            paths=annotations,
            sources='local',
            cache_dir=args.dset_cache_dir if args.dset_pack_level else None,
            file_types=args.dset_file_types,
            sample_ratios=sample_ratios,
            num_proc=args.num_proc,
            map_fns=tokenize_fns,
            init_fns=init_fns)

    if args.dset_pack_level and rank == 0:
        # Only the tokenized datasets can count the number of tokens
        num_tokens = sum(sum(dset['num_tokens']) for dset in _datasets)
        logger.debug(f'[Dataset] {num_tokens} tokens.')

    num_datasets = len(_datasets)
    datasets = []
    if args.dset_pack_level and args.dset_pack_level == 'soft':
        pack_infos = SoftPackerForLlava.get_pack_infos(_datasets,
                                                       args.pack_max_length)
        for i in range(num_datasets):
            _infos = pack_infos[i]
            _dset = _datasets[i]
            _packed_dset = SoftPackerForLlava(_dset, img_processor,
                                              args.pack_max_length, _infos,
                                              pad_image_to_square=pad_image_to_square)
            datasets.append(_packed_dset)
    else:
        for i, dset in enumerate(_datasets):
            datasets.append(dset)

    train_dataset = ConcatDataset(datasets)

    if args.dset_pack_level and rank == 0:
        ori_samples = sum([len(dset) for dset in _datasets])
        packed_samples = len(train_dataset)
        logger.info(f'[Dataset] (Original) {ori_samples} samples.')
        logger.info(f'[Dataset] (Packed) {packed_samples} samples.')

    pack_batch = is_flash_attn_2_available()
    collator = LlavaCollator(pack_batch=pack_batch)

    if args.group_by_length:
        if args.dset_pack_level:
            length_property = 'max_length_per_pack'
        else:
            length_property = 'length'
        sampler = LengthGroupedSampler(train_dataset, dp_mesh,
                                       args.global_batch_size,
                                       seed=args.seed,
                                       length_property=length_property)
    elif args.group_by_modality_length:
        # 当开启 soft packing 时，暂时不支持模态区分
        if args.dset_pack_level:
            raise NotImplementedError
        else:
            sampler = LengthGroupedSampler(train_dataset, dp_mesh,
                                           args.global_batch_size,
                                           seed=args.seed,
                                           length_property='modality_length')
    else:
        sampler = ParallelSampler(
            train_dataset, dp_mesh, args.global_batch_size, seed=args.seed, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0)

    if rank == 0:
        logger.info(f'[Dataloader] {len(train_dataloader)} batches.')
        _first_batch = [train_dataset[i] for i in range(args.mirco_batch_size)]
        _first_batch = collator(_first_batch)
        _decoded = tokenizer.batch_decode(_first_batch['input_ids'])
        logger.debug(f'[Dataloader] Training Batch:\n{_first_batch}')
        logger.debug(f'[Dataloader] Training Batch(Decoded):\n{_decoded}')
    dist.barrier()

    load_data_cost_time = time.time() - start_load_data_t
    logger.info(f'[Dataset & Dataloader] Cost {load_data_cost_time:.2f}s')
    # -------------------    Dataset & Dataloader  End  --------------------- #

    ###########################################################################
    #                          3. FSDP                                        #
    ###########################################################################

    start_model_t = time.time()

    if args.dtype == 'auto':
        args.dtype = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'

    if args.dtype == 'fp16':
        dtype = torch.float16
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype)
        scaler = ShardedGradScaler()
    elif args.dtype == 'bf16':
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype)
            scaler = None
        else:
            raise RuntimeError('The device does not support `bf16`, '
                               'please set `dtype` to `fp16`.')
    else:
        raise RuntimeError('`dtype` only supports `fp16`，`bf16`, or `auto`, '
                           f'but found {args.dtype}.')

    use_lora = args.llm_use_lora or args.vit_use_lora
    if not use_lora:
        autocast = nullcontext()
        scaler = None

    if is_flash_attn_2_available():
        if llava_config.text_config.model_type == 'internlm2':
            llava_config.text_config.attn_implementation = 'flash_attention_2'  # for InternLM
        else:
            llava_config.text_config._attn_implementation = 'flash_attention_2'
        llava_config._attn_implementation = 'flash_attention_2'
    elif is_torch_sdpa_available():
        llava_config.text_config.attn_implementation = 'sdpa'
    llava_config.text_config.use_cache = False

    # Ensure all numerical values in the optimizer are fp32.
    # FSDP will use low precision during forward.
    meta_llava = build_llava_model(args, llava_config, world_size,
                                   dtype,
                                   tokenizer=tokenizer,
                                   device='meta',
                                   resize_emb=need_resize_emb,
                                   is_pretrain=is_pretrain)

    # ！！！ 非常重要，确保 FSDP 优化器状态和 backward 时的梯度(反向传播时候梯度是 bf16 的，
    # 但是由于有一次同步导致会全部转换为 fp32的，方便后面优化器使用) 都是 fp32，梯度累加时候也是 fp32 的
    # 在开启 meta_llava.no_sync() 时，由于开启混合精度，backward 后的梯度并不是 fp32 的，而是 bf16 的
    # 所以可能不推荐在梯度累加时候开启 no_sync 上下文，虽然训练会变快，但是可能会导致梯度累加的精度问题
    # 如果一定要开启且运行不报错，假设梯度累加是 4，那么前 3 次使用 no_sync 上下文，最后一次不使用，才能保证程序不报错
    for module in meta_llava.modules():
        for p_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                setattr(module, p_name, param_fp32)

    if pack_batch or args.dset_pack_level:
        dispatch_modules(meta_llava)

    if tp_size > 1:
        # 专门处理 vit 中的 patch_embedding
        meta_llava.vision_tower.vision_model.embeddings.patch_embedding.forward = \
            new_conv2d_forward.__get__(meta_llava.vision_tower.vision_model.embeddings.patch_embedding)
        logger.info(f'[Model] Patch embedding is replaced by `new_conv2d_forward`.')

    # Only load parameters on rank 0 to avoid each rank repeatedly loading the
    # same model into the CPU, wasting memory
    timeout = timedelta(
        minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=45)))
    group = dist.new_group(backend='gloo', timeout=timeout)

    if rank == 0:
        logger.info(f'=====[Build Model]=======')
        llava = build_llava_model(args, llava_config, world_size, dtype,
                                  tokenizer=tokenizer,
                                  device='cpu',
                                  resize_emb=need_resize_emb,
                                  is_pretrain=is_pretrain)
        for module in llava.modules():
            for p_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                    setattr(module, p_name, param_fp32)
        # 必须位于 tp 包装前,否则保存会有问题
        rank0_meta_llava = copy.deepcopy(meta_llava)
        meta_llava_map = map_meta_modules(llava, meta_llava)
    else:
        meta_llava_map = None

    dist.monitored_barrier(group=group, timeout=timeout)

    # tp model
    if args.tp_size > 1:
        if args.tp_vit:
            # vit
            logger.info(f'[Model] Apply tensor parallel to ViT.')
            layer_tp_plan = {
                'self_attn.k_proj': ColwiseParallel(),
                'self_attn.q_proj': ColwiseParallel(),
                'self_attn.v_proj': ColwiseParallel(),
                'self_attn.out_proj': RowwiseParallel(),
                'mlp.fc1': ColwiseParallel(),
                'mlp.fc2': RowwiseParallel(),
            }

            for layer in meta_llava.vision_tower.vision_model.encoder.layers:
                attention = layer.self_attn
                attention.num_heads = attention.num_heads // tp_mesh.size()
                parallelize_module(
                    module=layer,
                    device_mesh=tp_mesh,
                    parallelize_plan=layer_tp_plan,
                )

        # llm
        logger.info(f'[Model] Apply tensor parallel to LLM.')
        layer_tp_plan = {
            'attention.wqkv': ColwiseParallel(),
            'attention.wo': RowwiseParallel(),
            'feed_forward.w1': ColwiseParallel(),
            'feed_forward.w2': RowwiseParallel(),
            'feed_forward.w3': ColwiseParallel(),
        }

        for layer in meta_llava.language_model.model.layers:
            attention = layer.attention
            attention.num_heads = attention.num_heads // tp_mesh.size()
            attention.hidden_size = attention.hidden_size // tp_mesh.size()
            parallelize_module(
                module=layer,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan,
            )

        if args.loss_parallel:
            logger.info(f'[Model] Apply loss parallel to LLM.')
            meta_llava = parallelize_module(
                module=meta_llava,
                device_mesh=tp_mesh,
                parallelize_plan={
                    'language_model.model.tok_embeddings':
                        RowwiseParallel(input_layouts=Replicate(), ),
                    'language_model.output': ColwiseParallel(use_local_output=False),
                })
        else:
            meta_llava = parallelize_module(
                module=meta_llava,
                device_mesh=tp_mesh,
                parallelize_plan={
                    'language_model.model.tok_embeddings':
                        RowwiseParallel(input_layouts=Replicate(), ),
                    'language_model.output': ColwiseParallel(output_layouts=Replicate(), ),
                })

        # tp 会重置所有包装的模块参数为 requires_grad=true，需要再次设置，但是也没有用，FSDP 包装后依然是 requires_grad=true
        # 应该是 FSDP1 对于这种情况没有支持
        if args.freeze_vit:
            meta_llava.vision_tower.eval()
            meta_llava.vision_tower.requires_grad_(False)
        if args.freeze_llm:
            meta_llava.language_model.eval()
            meta_llava.language_model.requires_grad_(False)

    # requried_grad_params1 = [
    #     (name, param) for name, param in meta_llava.named_parameters() if param.requires_grad
    # ]
    # if rank == 0:
    #     print('xxxx', requried_grad_params1)

    if args.tp_size > 1:
        param_init_fn = partial(
            dp_tp_lazy_init,
            module_map=meta_llava_map,
            dp_mesh=dp_mesh,
            tp_mesh=tp_mesh)
    else:
        param_init_fn = partial(
            dp_lazy_init, module_map=meta_llava_map, dp_mesh=dp_mesh)

    policies = [layer_auto_wrap_policy]
    if args.llm_use_lora or args.vit_use_lora:
        policies.append(all_required_grad_wrap_policy)

    if args.shard_strategy == 'full':
        fsdp_device_mesh = dp_mesh
        strategy = ShardingStrategy.FULL_SHARD
    elif args.shard_strategy == 'no':
        fsdp_device_mesh = dp_mesh
        strategy = ShardingStrategy.NO_SHARD
    elif args.shard_strategy == 'zero2':
        fsdp_device_mesh = dp_mesh
        strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args.shard_strategy == 'hybrid':
        fsdp_device_mesh = init_device_mesh('cuda', (dp_size // 8, 8))
        strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise ValueError

    torch.cuda.reset_peak_memory_stats()
    shard_llava = FSDP(
        meta_llava,
        device_mesh=fsdp_device_mesh,
        sharding_strategy=strategy,
        auto_wrap_policy=partial(_or_policy, policies=policies),
        mixed_precision=MixedPrecision(
            param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        param_init_fn=param_init_fn,
        sync_module_states=True,
    )
    # TODO HHA: 不管咋弄，被 TP 包装后的 DTensor 依然是 requires_grad=True, 这是不对的
    # 暂时只能强制再设置一遍
    # 如果不设置会导致 freeze 的模块依然是 requires_grad=True，这不符合预期，而且程序会报错，因为有 fp32 和 bf16 参数混合一起
    if args.freeze_vit:
        shard_llava.vision_tower.eval()
        shard_llava.vision_tower.requires_grad_(False)
    if args.freeze_llm:
        shard_llava.language_model.eval()
        shard_llava.language_model.requires_grad_(False)

    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Model] The peak GPU memory when building the FSDP model is '
                f'{max_memory / 1024 ** 3:.1f}GB.')

    if args.selective_recompute:
        check_fn = partial(
            checkpoint_check_fn,
            target=RECOMPUTE_MODULES,
            selective=args.selective_recompute)
        apply_activation_checkpointing(shard_llava, check_fn=check_fn)

    fsdp_cost_time = time.time() - start_model_t
    logger.info(f'[Model] Cost {fsdp_cost_time:.2f}s')
    # --------------------------    FSDP  End  ------------------------------ #

    ###########################################################################
    #                      4. Optimizer & Scheduler                           #
    ###########################################################################
    requried_grad_params = [
        param for param in shard_llava.parameters() if param.requires_grad
    ]
    # requried_grad_params1 = [
    #     (name, param) for name, param in shard_llava.named_parameters() if param.requires_grad
    # ]
    # if rank == 0:
    #     print(requried_grad_params1, requried_grad_params1[0][1].dtype)
    optimizer = AdamW(
        requried_grad_params, lr=args.lr, weight_decay=args.wd, fused=True)

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    # `iter` means once forward+backward
    # `step` means once optimizer step
    # `per_step_iters` means gradient accumulative counts
    per_step_iters = global_batch_size // mirco_batch_size // dp_size
    logger.info(f'[Optimizer] Global batch size: {global_batch_size}, Gradient accumulative counts: {per_step_iters}')
    per_epoch_iters = len(train_dataloader)
    per_epoch_steps = math.ceil(per_epoch_iters / per_step_iters)

    total_epochs = args.epochs
    total_steps = per_epoch_steps * total_epochs

    if args.checkpoint_interval == -1:
        checkpoint_interval = total_steps
    elif args.checkpoint_interval < 1:
        checkpoint_interval = int(total_steps * args.checkpoint_interval)
    else:
        checkpoint_interval = int(args.checkpoint_interval)

    warmup_steps = int(args.warmup_ratio * total_steps)

    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr_min)

    start_step = 0

    # ----------------    Optimizer & Scheduler End   ----------------------- #
    if args.resume:
        logger.info(f'[Resume] Resume from {args.resume_from}')
        _options = StateDictOptions(
            cpu_offload=True, ignore_frozen_params=True)
        (shard_model_state_dict,
         shard_optimizer_state_dict) = get_state_dict(
            shard_llava, optimizer, options=_options)
        meta_stateful = MetaStateful(step=start_step, total_steps=total_steps)
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
            shard_llava,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
            options=_options
        )

        start_step = meta_stateful['step']
        logger.info(f'[Resume] start_step to {start_step}')

    ###########################################################################
    #                          5. Training                                    #
    ###########################################################################

    start_train_t = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024 ** 3):.1f}GB')
    save_hf_ckpt_names = []
    save_pt_ckpt_names = []
    max_keep_ckpts = args.max_keep_ckpts
    if max_keep_ckpts <= 0:
        # 全部都保存
        max_keep_ckpts = 100000000

    for step in range(start_step, total_steps):

        epoch = step // per_epoch_steps
        epoch_inner_step = step % per_epoch_steps
        if epoch_inner_step == 0 or step == start_step:
            # For the first step of each epoch, the data order needs to be
            # readjusted.
            # Or after resuming, for the first step, the dataloader needs to
            # be adjusted to the position before resume.
            # train_dataloader.sampler.set_epoch(epoch, inner_step)
            train_dataloader.sampler.set_epoch(epoch, epoch_inner_step)
            data_iterator = iter(train_dataloader)

        if step <= warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_lr()[0]

        torch.cuda.reset_peak_memory_stats()

        step_loss = 0
        step_data_time = 0
        step_start_t = time.time()
        step_consumed_tokens = 0
        step_consumed_img_tokens = 0
        for _ in range(per_step_iters):

            _data_start_t = time.time()
            data = next(data_iterator)
            step_data_time += time.time() - _data_start_t

            input_ids = data['input_ids'].cuda(non_blocking=True)
            pixel_values = data['pixel_values']
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.cuda(non_blocking=True)
            labels = data['labels'].cuda(non_blocking=True)
            attention_mask = data['attention_mask'].cuda(non_blocking=True)
            num_tokens = data['num_tokens'].cuda(non_blocking=True)
            num_img_tokens = data['num_img_tokens'].cuda(non_blocking=True)

            packed_ctx = packed_sequence(num_tokens, enable=pack_batch)

            with packed_ctx:
                if args.loss_parallel:
                    with autocast if use_lora else nullcontext():
                        outputs = shard_llava(
                            input_ids=input_ids,
                            labels=None,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask)

                        shift_attention_mask = attention_mask[..., 1:]
                        shift_logits = outputs.logits[..., :-1, :][
                            shift_attention_mask.to(outputs.logits.device) != 0].contiguous()
                        shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()

                        with loss_parallel():
                            loss_fct = nn.CrossEntropyLoss()
                            loss = loss_fct(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1).to(shift_logits.device)
                            )
                            avg_iter_loss = loss / per_step_iters

                            if scaler and use_lora:
                                scaler.scale(avg_iter_loss).backward()
                            else:
                                avg_iter_loss.backward()
                else:
                    with autocast if use_lora else nullcontext():
                        outputs = shard_llava(
                            input_ids=input_ids,
                            labels=labels,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask)
                        avg_iter_loss = outputs.loss / per_step_iters

                    if scaler and use_lora:
                        scaler.scale(avg_iter_loss).backward()
                    else:
                        avg_iter_loss.backward()

            step_loss += avg_iter_loss.item()
            step_consumed_tokens += num_tokens.sum() / tp_size
            step_consumed_img_tokens += num_img_tokens.sum() / tp_size

        grad_norm = shard_llava.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        step_text_tokens = step_consumed_tokens - step_consumed_img_tokens
        step_img_tokens = step_consumed_img_tokens
        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        tgs = int(step_consumed_tokens / step_time)
        max_memory = torch.cuda.max_memory_allocated()
        if is_interval(step, total_steps, args.log_interval):
            logger.info(
                f'[Train] (Epoch {epoch}) Step {step + 1}/{total_steps}  '  # noqa: E501
                f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                f'grad_norm: {grad_norm:.2f}  '
                f'max_memory: {(max_memory / 1024 ** 3):.1f}GB  '
                f'text_tokens: {step_text_tokens}  '
                f'image_tokens: {step_img_tokens}  '
                f'tgs: {tgs}  data_time: {step_data_time:.2f}s  '
                f'time: {step_time:.2f}s  '
                f'eta: {eta}')

        if is_interval(step, total_steps, checkpoint_interval):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            max_memory = torch.cuda.max_memory_allocated()
            logger.info('[Checkpoint] Before saving checkpoint, the peak GPU '
                        f'memory is {max_memory / 1024 ** 3:.1f}GB.')

            digits = len(str(abs(total_steps)))
            work_dir = args.work_dir

            ckpt_id = f'{(step + 1):0{digits}}-of-{total_steps:0{digits}}'
            ckpt_dir = os.path.join(work_dir, f'ckpt-{ckpt_id}')
            hf_dir = os.path.join(work_dir, f'hf-{ckpt_id}')
            _options = StateDictOptions(cpu_offload=True, full_state_dict=True)

            full_model_state_dict = get_model_state_dict(
                shard_llava, options=_options)
            if rank == 0:
                saved_llava = copy.deepcopy(rank0_meta_llava)
                saved_llava.to(dtype)
                for name, param in full_model_state_dict.items():
                    set_module_tensor_to_device(saved_llava, name, 'cpu',
                                                param)

                if args.llm_use_lora:
                    merged_llm = saved_llava.language_model.merge_and_unload()
                    saved_llava.language_model = merged_llm

                if args.vit_use_lora:
                    merged_vit = saved_llava.vision_tower.merge_and_unload()
                    saved_llava.vision_tower = merged_vit

                saved_llava.save_pretrained(hf_dir)
                tokenizer.save_pretrained(hf_dir)
                processor.save_pretrained(hf_dir)
                del saved_llava

            dist.barrier()
            del full_model_state_dict

            if rank == 0:
                save_hf_ckpt_names.append(hf_dir)
                if len(save_hf_ckpt_names) > max_keep_ckpts:
                    # 移除最先加入的
                    remove_hf_ckpt_name = save_hf_ckpt_names.pop(0)
                    os.system(f'rm -rf {remove_hf_ckpt_name}')

            if args.checkpoint_drop_optimizer:
                logger.warning('[Checkpoint] The saved checkpoint cannot be '
                               'resumed. If you want to save a resumable '
                               'checkpoint, please remove '
                               '`--checkpoint-drop-optimizer` '
                               'from the command.')
            else:
                # FSDP cannot be saved via torch.save
                # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
                _options = StateDictOptions(
                    cpu_offload=True, ignore_frozen_params=True)
                (shard_model_state_dict,
                 shard_optimizer_state_dict) = get_state_dict(
                    shard_llava, optimizer, options=_options)
                meta_stateful = MetaStateful(step=step + 1, total_steps=total_steps)
                # warmup_scheduler/cosine_scheduler 并不是 Stateful 对象，但是是 work 的，
                # 怀疑是这个对象有啥特殊，存储后就变成了 Stateful 对象
                # 常数对象没有这个功能，需要自己封装
                state_dict = {
                    'model': shard_model_state_dict,
                    'optimizer': shard_optimizer_state_dict,
                    'meta_stateful': meta_stateful,
                    'warmup_scheduler': warmup_scheduler.state_dict(),
                    'cosine_scheduler': cosine_scheduler.state_dict()
                }
                dcp.save(state_dict, checkpoint_id=ckpt_dir)

                if rank == 0:
                    # 只能存储到 work_dir/../last_checkpoint 这，否则不好找
                    save_file = os.path.join(args.work_dir, '../', 'last_checkpoint')
                    with open(save_file, 'w') as f:
                        f.write(ckpt_dir)  # type: ignore

                    save_pt_ckpt_names.append(ckpt_dir)
                    if len(save_pt_ckpt_names) > max_keep_ckpts:
                        # 移除最先加入的
                        remove_pt_ckpt_name = save_pt_ckpt_names.pop(0)
                        os.system(f'rm -rf {remove_pt_ckpt_name}')

                max_memory = torch.cuda.max_memory_allocated()
                logger.info(
                    '[Checkpoint] During saving checkpoint, the peak GPU '
                    f'memory is {max_memory / 1024 ** 3:.1f}GB.')

    train_cost_time = time.time() - start_train_t
    m, s = divmod(train_cost_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    logger.info("[Train] Cost: %d day, %d:%d:%d" % (d, h, m, s))
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':
    args = parse_args()
    llava_sft(args)
