# Copyright (c) OpenMMLab. All rights reserved.
import warnings

# display once
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*")

import argparse
import math
import os
import time
import copy
from datetime import timedelta
from functools import partial
import random
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from PIL import Image
import json
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from xtuner._lite.parallel.new_setup import setup_parallel, \
    get_fsdp_mesh, get_dp_mesh, get_tp_mesh, get_world_mesh, get_sp_mesh, \
    profile_time_and_memory, get_torch_device_module
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.accelerate import (LoadWoInit,
                                     dispatch_modules, packed_sequence)
from xtuner._lite.accelerate.fsdp import clip_grad_norm_
from torch.distributed._composable import checkpoint
from torch.distributed._composable.fsdp import fully_shard
from xtuner._lite.modelings import register_remote_code
from llava_model import LlavaForConditionalGeneration
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          CLIPVisionModel, LlavaConfig, CLIPImageProcessor,
                          LlavaForConditionalGeneration, LlavaProcessor)

from xtuner._lite.datasets.dataset_fn import check_args, \
    set_logger_envs, build_train_dataloader, build_dataset, BaseOrigDataset, \
    _apply_exif_orientation, expand2square, _prepare_input, is_interval
from xtuner._lite.modelings.model_fn import map_meta_modules, lazy_init_megatron, save_ckpt, resume

logger = get_logger()


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
    parser.add_argument(
        '--liger', action='store_true', help='use liger kernel')
    model_args.add_argument(
        '--freeze-llm',
        action='store_true',
        help="Not updating LLM's parameters")
    model_args.add_argument(
        '--freeze-vit',
        action='store_true',
        help="Not updating vit's parameters")
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
    model_args.add_argument('--sp-size', type=int, default=1, help='')
    model_args.add_argument(
        '--tp-size',
        default=1,
        type=int,
        help="tp size")
    model_args.add_argument(
        '--ring-size',
        default=1,
        type=int,
        help='The ring size. if it is 1, it is the same as sp ulysses')
    data_args = parser.add_argument_group('data', 'Dataset Related Settings')
    data_args.add_argument(
        '--datasets',
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
    data_args.add_argument('--dset-pack', action='store_true')
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
    args = parser.parse_args()
    return args


class LazyLLaVADataset(BaseOrigDataset):
    def __init__(self, data_name, data, tokenizer, image_processor,
                 pad_image_to_square=False,
                 patch_size=14, max_length=2048,
                 group_by_length=False,
                 pack_data=False, pack_data_cache_dir=None):
        self.img_processor = image_processor
        self.pad_image_to_square = pad_image_to_square
        self.chat_template = dict(system='<|im_start|>system\n{system}<|im_end|>\n',
                                  user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
                                  assistant='{assistant}<|im_end|>')

        _crop_size = image_processor.crop_size
        img_size = (_crop_size['height'], _crop_size['width'])
        self.per_img_tokens = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        super().__init__(data_name, data, self.chat_template,
                         tokenizer=tokenizer,
                         max_length=max_length,
                         group_by_length=group_by_length,
                         pack_data=pack_data,
                         pack_data_cache_dir=pack_data_cache_dir)

    def calc_group_len(self):
        group_length = []
        conv2length_text = {}
        print('Calculating the length of text data...')
        for data_item in self.raw_data:
            if self._is_jsonl:
                data_item = json.loads(data_item)
            conversations = '\n'.join(
                [temp['value'] for temp in data_item['conversations']])
            str_length = len(conversations)
            if str_length not in conv2length_text:
                token_length = self.tokenizer(
                    conversations,
                    return_tensors='pt',
                    padding=False,
                    truncation=False,
                ).input_ids.size(1)
                conv2length_text[str_length] = token_length
            else:
                token_length = conv2length_text[str_length]
            if 'image' in data_item and data_item['image'] is not None:
                token_length += self.per_img_tokens
            else:
                token_length = -token_length
            group_length.append(token_length)
        print('Finished calculating the length of text data...')
        return group_length

    def _pre_tokenize_fn_for_pack(self, data_item):
        if self._is_jsonl:
            data_item = json.loads(data_item)
        if 'image' in data_item and data_item['image'] is not None:
            if type(data_item['image']) == list and len(data_item['image']) > 1:
                raise NotImplementedError
            else:
                num_tokens = self.multi_modal_get_item(data_item, pack_data=True)
        elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
            raise NotImplementedError
        else:
            num_tokens = self.pure_text_get_item(data_item, pack_data=True)
        return {'num_tokens': num_tokens}

    def multi_modal_get_item(self, data_item, pack_data=False):
        if pack_data:
            ret = self.process_text(data_item['conversations'], media_type='image')
            return len(ret['input_ids']) + self.per_img_tokens - 1

        image_path = data_item['image']
        if isinstance(image_path, list):
            image_path = image_path[0]

        image_path = os.path.join(self.root, image_path)
        image = Image.open(image_path).convert('RGB')
        image = _apply_exif_orientation(image)
        if self.pad_image_to_square:
            image = expand2square(
                image,
                tuple(
                    int(x * 255) for x in self.img_processor.image_mean))

        outputs = self.img_processor(image, return_tensors='pt')
        pixel_values = outputs['pixel_values']
        ret = self.process_text(data_item['conversations'], media_type='image')

        data = {
            'input_ids': ret['input_ids'],
            'labels': ret['labels'],
            'pixel_values': pixel_values,
            'num_tokens': [len(ret['input_ids'])],
            'num_img_tokens': [self.per_img_tokens],
        }

        return data

    def pure_text_get_item(self, data_item, pack_data=False):
        ret = self.process_text(data_item['conversations'], media_type='text')
        if pack_data:
            return len(ret['input_ids'])

        data = {
            'input_ids': ret['input_ids'],
            'labels': ret['labels'],
            'num_tokens': [len(ret['input_ids'])],
            'num_img_tokens': [0],
        }
        return data

    def _process_media_format_first_round(self, input_, media_type, image_grids):
        return input_

    def __getitem__(self, i):
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = self.raw_data[i]
                if self._is_jsonl:
                    data_item = json.loads(data_item)
                if 'image' in data_item and data_item['image'] is not None:
                    if type(data_item['image']) == list and len(data_item['image']) > 1:
                        raise NotImplementedError
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    raise NotImplementedError
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                logger.info(f'Exception: {e} of {self.data_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def packing_collate(features, pack_batch=True, pad_id=0):
    _features = []
    for ins in features:
        if isinstance(ins, list):
            _features.extend(ins)
        else:
            _features.append(ins)
    features = _features

    input_ids = []
    labels = []
    pixel_values = []
    num_tokens = []
    num_img_tokens = []

    for data in features:
        input_ids.append(torch.LongTensor(data['input_ids']))
        labels.append(torch.LongTensor(data['labels']))
        num_tokens.extend(data['num_tokens'])
        num_img_tokens.extend(data['num_img_tokens'])
        if 'pixel_values' in data:
            pixel_values.append(data['pixel_values'])

    attention_mask = [ids.ne(pad_id) for ids in input_ids]
    num_tokens = torch.IntTensor(num_tokens)
    num_img_tokens = torch.IntTensor(num_img_tokens)

    if len(features) > 1 and pack_batch:
        # packing
        input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        labels = torch.cat(labels, dim=0).unsqueeze(0)
        attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)
        if len(pixel_values) > 0:
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None
    elif len(features) > 1 and not pack_batch:
        raise NotImplementedError
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        if len(pixel_values) > 0:
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None

    data_dict = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask.bool(),
        'pixel_values': pixel_values,
        'num_tokens': num_tokens,
        'num_img_tokens': num_img_tokens,
    }

    return data_dict


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
                        args.llm, config=_cfg.text_config)
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

        if args.freeze_llm:
            llava.language_model.requires_grad_(False)
            llava.language_model.eval()

        if args.freeze_vit:
            llava.vision_tower.requires_grad_(False)
            llava.vision_tower.eval()

    for module in llava.modules():
        for p_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                setattr(module, p_name, param_fp32)
    return llava


def build_fsdp_model(rank0_model, meta_model, dp_mesh, tp_mesh, dtype, args):
    if dp_mesh.get_rank() == 0:
        rank0_map = map_meta_modules(rank0_model, meta_model)
    else:
        rank0_map = None

    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
    )

    if args.shard_strategy == 'full':
        reshard_after_forward = True
    else:
        reshard_after_forward = False

    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

    # visual
    meta_model.vision_tower.apply(param_init_fn)
    fully_shard(
        meta_model.vision_tower,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    for i, block in enumerate(meta_model.vision_tower.blocks):
        checkpoint(block)

    # llm
    num_layers = len(meta_model.language_model.model.layers)
    num_recompute_layers = int(num_layers * 1.0)
    for i, block in enumerate(meta_model.language_model.model.layers):
        block.apply(param_init_fn)

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        if i < num_recompute_layers:
            # 有 bug，无法处理当模块输入是 **kwargs 的情况，暂时只能 dispatch 这个模块的 forward
            checkpoint(block)

    meta_model.language_model.model.tok_embeddings.apply(param_init_fn)
    meta_model.language_model.model.norm.apply(param_init_fn)
    meta_model.language_model.output.apply(param_init_fn)

    model = fully_shard(
        meta_model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)  # False is zero2, True is zero3

    return model


def llava_train(args):
    if args.liger:
        from xtuner._lite.modelings import apply_liger_kernel_to_llava_clip_internlm2
        try:
            from liger_kernel.transformers.geglu import LigerGEGLUMLP
        except ImportError:
            raise ImportError('Please install liger_kernel to use liger.')
        apply_liger_kernel_to_llava_clip_internlm2()

    setup_parallel(tp_size=args.tp_size, sp_size=args.sp_size)
    set_random_seed(args.seed)

    dp_mesh = get_dp_mesh()
    tp_mesh = get_tp_mesh()
    sp_mesh = get_sp_mesh()
    fsdp_mesh = get_fsdp_mesh()  # dp_size * sp_size
    world_mesh = get_world_mesh()  # dp_size * sp_size * tp_size

    dp_size = dp_mesh.size()
    tp_size = tp_mesh.size()
    sp_size = sp_mesh.size()
    world_size = world_mesh.size()

    rank = world_mesh.get_rank()

    check_args(args)
    set_logger_envs(args)

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
            pad_token_id=0)
        _img_processor = CLIPImageProcessor.from_pretrained(args.vit)
        processor = LlavaProcessor(_img_processor, tokenizer)
    else:
        llava_config = AutoConfig.from_pretrained(args.llava)
        if hasattr(llava_config.text_config, 'auto_map'):
            delattr(llava_config.text_config, 'auto_map')
        processor = AutoProcessor.from_pretrained(
            args.llava, trust_remote_code=True)

    with profile_time_and_memory('[Dataset & Dataloader]'):
        ds_collections = json.loads(open(args.datasets).read())
        _datasets = []
        for name, _data in ds_collections.items():
            _dataset = LazyLLaVADataset(name, _data, tokenizer, processor.image_processor,
                                        patch_size=llava_config.vision_config.patch_size,
                                        pad_image_to_square=pad_image_to_square,
                                        max_length=args.max_length,
                                        group_by_length=args.group_by_length,
                                        pack_data=args.dset_pack_level,
                                        pack_data_cache_dir=args.dset_cache_dir)
            if dist.get_rank() == 0:
                logger.info(f'[Dataset] (Original) {name}: {len(_dataset)} samples.')
            _datasets.append(_dataset)
        train_dataset = build_dataset(args, _datasets)
        logger.warning(f'{dist.get_rank()} ===== End of all dataset =====')
        train_dataloader = build_train_dataloader(args, train_dataset, packing_collate)

    args.dtype = 'bf16'
    dtype = torch.bfloat16
    with profile_time_and_memory('[Model]'):
        need_resize_emb = False
        _llm_vocab_size = llava_config.text_config.vocab_size
        if _llm_vocab_size < len(tokenizer):
            need_resize_emb = True

        llava_config.text_config.attn_implementation = 'flash_attention_2'
        llava_config.text_config.use_cache = False
        meta_model = build_llava_model(args, llava_config, world_size, dtype=dtype,
                                       tokenizer=tokenizer, device='meta',
                                       resize_emb=need_resize_emb, is_pretrain=is_pretrain)
        dispatch_modules(meta_model)
        if dist.get_rank() == 0:
            logger.info(meta_model)

        timeout = timedelta(
            minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=45)))
        group = dist.new_group(backend='gloo', timeout=timeout)
        if rank == 0:
            # 用于初始化 meta_model 权重和后续保存权重
            logger.info(f'=====[Build CPU Model]=======')
            rank0_model = build_llava_model(args, llava_config, world_size, dtype=dtype,
                                            tokenizer=tokenizer, device='cpu',
                                            resize_emb=need_resize_emb, is_pretrain=is_pretrain)
        else:
            rank0_model = None
        dist.monitored_barrier(group=group, timeout=timeout)

        fsdp_model = build_fsdp_model(rank0_model, meta_model, fsdp_mesh, world_mesh, dtype, args)
        fsdp_model.train()

    requried_grad_params = [
        param for param in fsdp_model.parameters() if param.requires_grad
    ]
    optimizer = AdamW(
        requried_grad_params, lr=args.lr, weight_decay=args.wd, fused=True)

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

    warmup_steps = int(args.warmup_ratio * total_steps)

    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr_min)

    start_step = 0

    if args.resume:
        start_step = resume(args, fsdp_model, optimizer, warmup_scheduler, cosine_scheduler, start_step, total_steps)

    start_train_t = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()

    save_hf_ckpt_names = []
    save_pt_ckpt_names = []
    max_keep_ckpts = args.max_keep_ckpts
    if max_keep_ckpts <= 0:
        # 全部都保存
        max_keep_ckpts = 100000000

    if rank == 0:
        logger.info('[Train] Begin Train Loop. The current GPU memory is '
                    f'{(max_memory / 1024 ** 3):.1f}GB')
        logger.info('The FSDP adopts a lazy design, so the first iteration will be slow.')
        if args.liger:
            logger.info('====== use liger kernel =====')

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

            data = _prepare_input(data)
            num_tokens = data.pop('num_tokens')
            num_img_tokens = data.pop('num_img_tokens')

            packed_ctx = packed_sequence(num_tokens, enable=True)

            with packed_ctx:
                outputs = fsdp_model(**data, use_cache=False)
                avg_iter_loss = outputs.loss / per_step_iters
                avg_iter_loss.backward()

            step_loss += avg_iter_loss.item()
            step_consumed_tokens += num_tokens.sum()
            step_consumed_img_tokens += num_img_tokens.sum()

        grad_norm = clip_grad_norm_(requried_grad_params, fsdp_mesh, args.max_grad_norm)
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
            save_ckpt(args, step, total_steps, fsdp_model, rank0_model, warmup_scheduler, cosine_scheduler,
                      optimizer, max_keep_ckpts, save_hf_ckpt_names, save_pt_ckpt_names)

    train_cost_time = time.time() - start_train_t
    m, s = divmod(train_cost_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    logger.info("[Train] Cost: %d day, %d:%d:%d" % (d, h, m, s))
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':
    args = parse_args()
    llava_train(args)
