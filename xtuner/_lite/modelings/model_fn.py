from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor, distribute_tensor
import torch
import torch.distributed as dist
from ..parallel.new_setup import get_torch_device_module
import os
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_state_dict, set_state_dict)
import copy
from accelerate.utils import set_module_tensor_to_device
import torch.distributed.checkpoint as dcp
from .. import get_logger

logger = get_logger()


def map_meta_modules(model, meta_model):
    modules = {name: mod for name, mod in model.named_modules()}
    meta_module_map = {
        mod: modules[name]
        for name, mod in meta_model.named_modules()
    }
    return meta_module_map


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


@torch.no_grad
def lazy_init_megatron(module, rank0_map, dp_mesh, tp_mesh=None, pp_mesh=None):
    device = get_torch_device_module().current_device()

    if dp_mesh.get_rank() == 0:
        rank0_module = rank0_map[module]
        rank0_params = {
            name: param
            for name, param in rank0_module.named_parameters(recurse=False)
        }
        rank0_buffers = {
            name: buffer
            for name, buffer in rank0_module.named_buffers(recurse=False)
        }
    else:
        rank0_params = None
        rank0_buffers = None

    param_shapes = {
        name: param.full_tensor().shape
        if isinstance(param, DTensor) else param.shape
        for name, param in module.named_parameters(recurse=False)
    }

    module.to_empty(device=get_torch_device_module().current_device(), recurse=False)

    for name, param in module.named_parameters(recurse=False):
        dtype = param.dtype
        if dp_mesh.get_rank() == 0:
            rank0_param = rank0_params[name].to(device).to(dtype)
        else:
            full_shape = param_shapes[name]
            rank0_param = torch.zeros(full_shape, dtype=dtype, device=device)

        dist.broadcast(rank0_param, src=0)

        if isinstance(param, DTensor):
            mesh = param.device_mesh
            assert mesh == tp_mesh
            placements = param.placements
            rank0_param = distribute_tensor(rank0_param, mesh, placements)

        param.data.copy_(rank0_param)
        dist.barrier()

    # TP does not shard buffers
    for name, buffer in module.named_buffers(recurse=False):
        if dp_mesh.get_rank() == 0:
            rank0_buffer = rank0_buffers[name].to(device)
        else:
            rank0_buffer = torch.empty_like(buffer).to(device)

        dist.broadcast(rank0_buffer, src=0)
        buffer.data.copy_(rank0_buffer)


def resume(args, fsdp_model, optimizer, warmup_scheduler, cosine_scheduler, start_step, total_steps):
    logger.info(f'[Resume] Resume from {args.resume_from}')
    _options = StateDictOptions(
        cpu_offload=True, ignore_frozen_params=True)
    (shard_model_state_dict,
     shard_optimizer_state_dict) = get_state_dict(
        fsdp_model, optimizer, options=_options)
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
        fsdp_model,
        optimizer,
        model_state_dict=state_dict["model"],
        optim_state_dict=state_dict["optimizer"],
        options=_options
    )

    start_step = meta_stateful['step']
    logger.info(f'[Resume] start_step to {start_step}')
    return start_step

def save_ckpt(args, step, total_steps, fsdp_model, rank0_model, warmup_scheduler, cosine_scheduler, optimizer,
              max_keep_ckpts, save_hf_ckpt_names, save_pt_ckpt_names, tokenizer, processor):
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

    full_model_state_dict = get_model_state_dict(fsdp_model, options=_options)
    if dist.get_rank() == 0:
        saved_model = copy.deepcopy(rank0_model)
        saved_model.to(torch.bfloat16)
        for name, param in full_model_state_dict.items():
            set_module_tensor_to_device(saved_model, name, 'cpu',
                                        param)

        saved_model.save_pretrained(hf_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(hf_dir)
        if processor is not None:
            processor.save_pretrained(hf_dir)
        del saved_model

    dist.barrier()
    del full_model_state_dict

    if dist.get_rank() == 0:
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
            fsdp_model, optimizer, options=_options)
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

        if dist.get_rank() == 0:
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
