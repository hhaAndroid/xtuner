from functools import partial
from packaging import version

import torch
from torch import nn
from torch.distributed._tensor import Replicate, distribute_tensor
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               PrepareModuleOutput,
                                               RowwiseParallel,
                                               parallelize_module)

from xtuner._lite import get_logger
from ..fsdp.lazy import lazy_init_megatron
from .utils import map_rank0_modules

import copy
from torch.distributed.pipelining.schedules import (
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining import PipelineStage
from typing import Any, Dict, Optional, Tuple, List

logger = get_logger()


def _tp_internlm2(model, tp_mesh):

    layer_tp_plan = {
        # by default ColwiseParallel input layouts is replicated
        # and RowwiseParallel output layouts is replicated
        'attention.wqkv':
        ColwiseParallel(),
        'attention.wo':
        RowwiseParallel(),
        'attention_norm':
        PrepareModuleInput(
            input_layouts=(Replicate(), ),
            desired_input_layouts=(Replicate(), ),
            use_local_output=True
        ),
        'feed_forward.w1':
        ColwiseParallel(),
        'feed_forward.w2':
        RowwiseParallel(),
        'feed_forward.w3':
        ColwiseParallel(),
        'ffn_norm':
        PrepareModuleInput(
            input_layouts=(Replicate(), ),
            desired_input_layouts=(Replicate(), ),
            use_local_output=True
        )
    }

    tp_size = tp_mesh.size()
    for layer in model.layers:
        attention = layer.attention
        num_key_value_heads = attention.num_key_value_heads
        num_heads = attention.num_heads
        hidden_size = attention.hidden_size

        attention.num_heads = num_heads // tp_size
        attention.num_key_value_heads = num_key_value_heads // tp_size
        attention.hidden_size = hidden_size // tp_size

        attn_norm = layer.attention_norm
        attn_norm.register_parameter(
            'weight',
            nn.Parameter(
                distribute_tensor(attn_norm.weight, tp_mesh, [Replicate()])))

        ffn_norm = layer.ffn_norm
        ffn_norm.register_parameter(
            'weight',
            nn.Parameter(
                distribute_tensor(ffn_norm.weight, tp_mesh, [Replicate()])))

        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    norm = model.norm
    dist_norm_w = nn.Parameter(
        distribute_tensor(norm.weight, tp_mesh, [Replicate()]))
    norm.register_parameter('weight', dist_norm_w)

    # emb = model.tok_embeddings
    # dist_emb_w = nn.Parameter(
    #     distribute_tensor(emb.weight, tp_mesh, [Replicate()]))
    # emb.register_parameter('weight', dist_emb_w)

    model = parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan={
            'model.tok_embeddings':
            RowwiseParallel(input_layouts=Replicate(), ),
            'model.norm':PrepareModuleInput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Replicate(),),
                use_local_output=True
            ),
        })


def megatron_internlm2(model,
                       rank0_model,
                       dp_mesh,
                       tp_mesh=None,
                       pp_mesh=None,
                       mp_policy=None,
                       recompute_ratio=1.0,
                       reshard_after_forward=True):

    if dp_mesh.get_rank() == 0:
        rank0_map = map_rank0_modules(model, rank0_model)
    else:
        rank0_map = None

    if tp_mesh and tp_mesh.size() > 1:
        _tp_internlm2(model, tp_mesh)

    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
    )

    from torch.distributed._composable import checkpoint
    from torch.distributed._composable.fsdp import fully_shard
    num_layers = len(model.layers)
    num_recompute_layers = int(num_layers * recompute_ratio)

    for i, block in enumerate(model.layers):

        block.apply(param_init_fn)

        # # As an optimization, do not reshard after forward for the last
        # # transformer block since FSDP would prefetch it immediately
        # if i < num_layers - 1:
        #     _reshard = reshard_after_forward
        # else:
        #     _reshard = False

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        if i < num_recompute_layers:
            checkpoint(block)

    if version.parse(torch.__version__) >= version.parse("2.5.0"):
        for layer_cur, layer_next in zip(model.layers[:-1], model.layers[1:]):
            layer_cur.set_modules_to_forward_prefetch([layer_next])

    model.tok_embeddings.apply(param_init_fn)
    model.norm.apply(param_init_fn)

    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)

def generate_split_points(pp_schedule, pp_dim, num_layers):
    schedule_class = get_schedule_class(pp_schedule)
    if issubclass(schedule_class, PipelineScheduleSingle):
        num_stages_per_rank = 1
    elif issubclass(schedule_class, PipelineScheduleMulti):
        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified
        num_stages_per_rank = 2
    else:
        raise ValueError(
            f"Unsupported pipeline schedule: {pp_schedule}"
        )
    total_stages = pp_dim * num_stages_per_rank

    if total_stages > num_layers:
        raise ValueError("Total stages cannot be greater than the number of layers")

    base_interval = num_layers // total_stages
    extra_layers = num_layers % total_stages

    splits = []
    current_layer = 0
    for i in range(total_stages - 1):
        if i == 0:
            current_layer += base_interval
        else:
            # Middle stages get an extra layer if there are any remaining
            if extra_layers > 0:
                current_layer += base_interval + 1
                extra_layers -= 1
            else:
                current_layer += base_interval
        splits.append("layers." + str(current_layer))
    return splits


def stage_ids_this_rank(
        pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
):
    """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
    assert (
            num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert (
                stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
        )
        return stage_v_pairs[pp_rank]


def pipeline_manual_split(
        whole_model: nn.Module,
        pp_mesh,
        device,
        pp_schedule,
):
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    splits = generate_split_points(pp_schedule, pp_size, len(whole_model.model.layers))  # [layers.12]

    def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
        model = copy.deepcopy(whole_model)
        if not is_first:
            model.model.tok_embeddings = None

        drop_layers = start_layer is not None

        need_del_models = []
        for name in list(range(0, len(model.model.layers))):
            # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
            if f"layers.{name}" == start_layer:
                drop_layers = False
            if f"layers.{name}" == stop_layer:
                drop_layers = True
            if drop_layers:
                need_del_models.append(int(name))

        for i in range(len(need_del_models) - 1, -1, -1):
            del model.model.layers[need_del_models[i]]
        del need_del_models

        if not is_last:
            model.model.norm = None
            model.output = None

        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group(),
        )
        return stage, model

    num_stages = len(splits) + 1

    stages = []
    models = []
    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style="loop"):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}: model chunk \n{model_chunk}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models


def split_list(lst, m):
    n = len(lst)
    length = n // m
    return [lst[i * length:(i + 1) * length] for i in range(m)]


def split_args_kwargs_into_chunks(
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]],
        chunks: int,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
) -> Tuple[List[Tuple], List[Dict]]:
    # TODO: 直接 hard code 写死，后续可以考虑更通用
    assert len(args) == 0
    args_split = [() for _ in range(chunks)]
    kwargs_split = [{} for _ in range(chunks)]
    copy_keys = ['use_cache', 'return_dict']
    for key in copy_keys:
        if key in kwargs:
            for i in range(chunks):
                kwargs_split[i][key] = kwargs[key]

    spilt_keys = ['input_ids', 'position_ids', 'target', 'losses']
    for key in spilt_keys:
        if key in kwargs:
            data = kwargs[key]
            chunk_tensors = torch.tensor_split(data, chunks)
            assert len(chunk_tensors) == chunks
            for i, chunk_tensor in enumerate(chunk_tensors):
                kwargs_split[i][key] = chunk_tensor

    split_seq = ['max_seqlen', 'cumulative_lengths']
    for key in split_seq:
        if key in kwargs:
            data = kwargs[key]
            assert len(data) % chunks == 0
            result = split_list(data, chunks)
            for i, chunk_tensor in enumerate(result):
                kwargs_split[i][key] = chunk_tensor

    return args_split, kwargs_split


def _new_split_inputs(self,
                      args: Tuple[Any, ...],
                      kwargs: Optional[Dict[str, Any]] = None):
    if args or kwargs:
        args_split, kwargs_split = split_args_kwargs_into_chunks(
            args,
            kwargs,
            self._n_microbatches,
            self._args_chunk_spec,
            self._kwargs_chunk_spec,
        )
        return args_split, kwargs_split
    else:
        # Empty inputs (e.g. when called on middle stages)
        # Return a list of empty tuples/dicts with matching length as chunks
        return [()] * self._n_microbatches, [{}] * self._n_microbatches


def build_pipeline_schedule(pp_mb, pp_schedule, pp_size, stages, loss_fn):
    schedule_class = get_schedule_class(pp_schedule)
    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    logger.info(
        f"Using pipeline schedule {pp_schedule}"
    )
    n_microbatches = pp_mb
    if n_microbatches == -1:
        n_microbatches = pp_size

    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )
    schedule._split_inputs = _new_split_inputs.__get__(schedule)
    return schedule


def megatron_internlm2_casual(model,
                              rank0_model,
                              dp_mesh,
                              tp_mesh=None,
                              pp_mesh=None,
                              mp_policy=None,
                              recompute_ratio=1.0,
                              reshard_after_forward=True,
                              pp_mb=-1,
                              pp_schedule=None,
                              loss_fn=None):
    # 先利用 pp 切分模型，然后在对每个 pp 内部的模型进行 fsdp 包装
    pp_size = pp_mesh.size()
    if pp_size > 1:
        # TODO： 为了代码简单暂时先对 meta model 进行初始化，这样会找出显存浪费。
        # 由于只有 rank0 model 不太好处理, 合理逻辑应该是 rank_model 进行 pp 切分，然后将非 pp rank0 的 model 广播到其他 pp rank
        if dp_mesh.get_rank() == 0:
            rank0_map = map_rank0_modules(model, rank0_model)
        else:
            rank0_map = None
        param_init_fn = partial(
            lazy_init_megatron,
            rank0_map=rank0_map,
            dp_mesh=dp_mesh,
            tp_mesh=tp_mesh,
        )
        model.apply(param_init_fn)

        stages, model_parts = pipeline_manual_split(
            model, pp_mesh, 'cuda', pp_schedule,
        )
        pp_schedule = build_pipeline_schedule(pp_mb, pp_schedule, pp_size, stages, loss_fn)

        # 进行 fsdp 切分
        from torch.distributed._composable import checkpoint
        from torch.distributed._composable.fsdp import fully_shard

        for model_part in model_parts:
            num_layers = len(model_part.model.layers)
            num_recompute_layers = int(num_layers * recompute_ratio)

            for i, block in enumerate(model_part.model.layers):
                fully_shard(
                    block,
                    mesh=dp_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                )
                if i < num_recompute_layers:
                    checkpoint(block)

            if version.parse(torch.__version__) >= version.parse("2.5.0"):
                for layer_cur, layer_next in zip(model_part.model.layers[:-1], model_part.model.layers[1:]):
                    layer_cur.set_modules_to_forward_prefetch([layer_next])

            fully_shard(
                model_part,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward)

        return pp_schedule, model_parts

    else:
        megatron_internlm2(
            model.model,
            rank0_model.model if dp_mesh.get_rank() == 0 else None,
            dp_mesh,
            tp_mesh=tp_mesh,
            pp_mesh=pp_mesh,
            mp_policy=mp_policy,
            recompute_ratio=recompute_ratio,
            reshard_after_forward=reshard_after_forward)

        if tp_mesh and tp_mesh.size() > 1:
            model = parallelize_module(
                module=model,
                device_mesh=tp_mesh,
                parallelize_plan={
                    'output': ColwiseParallel(output_layouts=Replicate(), ),
                })

        if dp_mesh.get_rank() == 0:
            rank0_map = map_rank0_modules(model, rank0_model)
        else:
            rank0_map = None

        param_init_fn = partial(
            lazy_init_megatron,
            rank0_map=rank0_map,
            dp_mesh=dp_mesh,
            tp_mesh=tp_mesh,
        )
        model.output.apply(param_init_fn)

        from torch.distributed._composable.fsdp import fully_shard
        fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward)

        return model


def megatron_internlm2_reward(model,
                              rank0_model,
                              dp_mesh,
                              tp_mesh=None,
                              pp_mesh=None,
                              mp_policy=None,
                              recompute_ratio=1.0,
                              reshard_after_forward=True):
    megatron_internlm2(
        model.model,
        rank0_model.model if dp_mesh.get_rank() == 0 else None,
        dp_mesh,
        tp_mesh=tp_mesh,
        pp_mesh=pp_mesh,
        mp_policy=mp_policy,
        recompute_ratio=recompute_ratio,
        reshard_after_forward=reshard_after_forward)

    if tp_mesh and tp_mesh.size() > 1:

        head_0 = model.v_head[0]
        dist_head_0 = nn.Parameter(
            distribute_tensor(head_0.weight, tp_mesh, [Replicate()]))
        head_0.register_parameter('weight', dist_head_0)

        head_norm = model.v_head[1]
        dist_head_norm = nn.Parameter(
            distribute_tensor(head_norm.weight, tp_mesh, [Replicate()]))
        dist_head_bias = nn.Parameter(
            distribute_tensor(head_norm.bias, tp_mesh, [Replicate()]))
        head_norm.register_parameter('weight', dist_head_norm)
        head_norm.register_parameter('bias', dist_head_bias)

        head_1 = model.v_head[3]
        dist_head_1 = nn.Parameter(
            distribute_tensor(head_1.weight, tp_mesh, [Replicate()]))
        head_1.register_parameter('weight', dist_head_1)

        
        parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan={
                'v_head.0': PrepareModuleInput(
                    input_layouts=(Replicate(),),
                    desired_input_layouts=(Replicate(),),
                ),
                'v_head.0': PrepareModuleOutput(
                    output_layouts=(Replicate(),),
                    desired_output_layouts=(Replicate(),),
                    use_local_output=True
                ),
                'v_head.1': PrepareModuleInput(
                    input_layouts=(Replicate(),),
                    desired_input_layouts=(Replicate(),),
                ),
                'v_head.1': PrepareModuleOutput(
                    output_layouts=(Replicate(),),
                    desired_output_layouts=(Replicate(),),
                    use_local_output=True
                ),
                'v_head.3': PrepareModuleInput(
                    input_layouts=(Replicate(),),
                    desired_input_layouts=(Replicate(),),
                ),
                'v_head.3': PrepareModuleOutput(
                    output_layouts=(Replicate(),),
                    desired_output_layouts=(Replicate(),),
                    use_local_output=True
                ),
                
            })

    if dp_mesh.get_rank() == 0:
        rank0_map = map_rank0_modules(model, rank0_model)
    else:
        rank0_map = None

    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
    )
    model.v_head.apply(param_init_fn)

    from torch.distributed._composable.fsdp import fully_shard
    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward)
