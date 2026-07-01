import contextlib
import gc
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Sequence,
    TypeAlias,
    TypedDict,
    cast,
)


# export ONLY_CALC_MISMATCH_RATIO=1
# export XTUNER_DEBUG_FSDP_DEFERRED=1
# export XTUNER_DEBUG_OFFLOAD_MEMORY=1
# export XTUNER_DEBUG_OFFLOAD_MEMORY_SNAPSHOT=1


if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

import numpy as np
import ray
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorClass, ActorProxy
from torch.distributed.device_mesh import init_device_mesh
from typing_extensions import NotRequired

from transformers import AutoTokenizer
from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.engine.train_engine import TrainEngine, TrainStepInfo
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.loss import BaseLossContext, CELossConfig, LogProbConfig
from xtuner.v1.loss.ce_loss import CELossContext, LMHeadLossContext
from xtuner.v1.loss.mtp_loss import MTPLossConfig, MTPLossContext
from xtuner.v1.model.base import BaseModel as XtunerBaseModel
from xtuner.v1.model.base import ModelItem, TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig, BaseComposeModel
from xtuner.v1.model.utils.misc import ModelForwardExtraLogInfo
from xtuner.v1.profiler import profiling_memory, profiling_time
from xtuner.v1.rl.loss import BaseRLLossConfig, BaseRLLossContext, finalize_train_policy_metrics, kl_penalty
from xtuner.v1.rl.utils import SingleAcceleratorWorker
from xtuner.v1.rl.weight_update import UpdateWeighter
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import (
    XTUNER_DETERMINISTIC,
    ParallelConfigException,
    get_device,
    get_logger,
    get_torch_device_module,
    ray_method,
    set_deterministic,
)

from ..rollout_is import merge_rollout_is_metrics


DeviceMeshRaw: TypeAlias = List[List[int]]  # A list of lists representing device mesh indices
ServiceUrlMap: TypeAlias = Dict[int, str]  # A dictionary mapping service names to their URLs
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def calculate_entropy(
    shifted_labels_list: Sequence[torch.Tensor],
    old_logprobs_list: Sequence[torch.Tensor | None],
    global_grad_tokens: torch.Tensor,
) -> torch.Tensor | None:
    if len(old_logprobs_list) == 0 or old_logprobs_list[0] is None:
        return None
    sum_entropy: torch.Tensor | None = None
    for i, shifted_labels in enumerate(shifted_labels_list):
        mask = shifted_labels != -100
        assert old_logprobs_list[i] is not None
        entropy = -(cast(torch.Tensor, old_logprobs_list[i]) * mask).sum()
        sum_entropy = entropy if sum_entropy is None else sum_entropy + entropy
    sum_entropy = cast(torch.Tensor, sum_entropy)
    dist.all_reduce(sum_entropy, op=dist.ReduceOp.SUM)
    avg_sum_entropy = sum_entropy / global_grad_tokens if global_grad_tokens > 0 else torch.tensor(0.0)
    return avg_sum_entropy


class WorkerConfig(BaseModel):
    """Training worker configuration for XTuner RL.

    Configuration for RL training workers managing model training, optimization,
    and distributed computing in reinforcement learning workflows.

    Args:
        model_cfg (TransformerConfig): Model architecture configuration.
        optim_cfg (OptimConfig): Optimizer configuration for training.
        loss_cfg (BaseRLLossConfig): Loss function configuration for RL training.
        lr_cfg (LRConfig): Learning rate scheduler configuration.
        fsdp_cfg (FSDPConfig): Fully Sharded Data Parallel configuration.
        load_from (str | Path): Path to load the main model from.
        optimizer_steps (int): Number of optimizer steps per training iteration. Defaults to 1.
        sp_size (int): Sequence parallel size for distributed training. Defaults to 1.
        pack_max_length (int): Maximum sequence length for data packing.
        ref_load_from (str | Path | None): Path to load reference model from.
            If None, uses same as load_from. Defaults to None.
        ref_model_fsdp_cfg (FSDPConfig | None): FSDP configuration for reference model.
            Defaults to None.
        log_dir (str | Path | None): Directory for training logs. Defaults to None.
        update_weight_bucket_size_in_gb (float): Bucket size used when syncing
            updated weights to rollout workers. Defaults to 0.5.
        seed (int | None): Training worker random seed. When None, the RL
            trainer seed is used. Defaults to None.

    **Examples:**

    Example configuration for Basic worker::

        config = WorkerConfig(
            model_cfg=TransformerConfig(model_name="llama2-7b"),
            optim_cfg=OptimConfig(optimizer="adamw"),
            loss_cfg=GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"}),
            lr_cfg=LRConfig(lr=1e-5),
            fsdp_cfg=FSDPConfig(),
            load_from="meta-llama/Llama-2-7b-hf",
            pack_max_length=2048,
        )

    .. note::
       When ``use_kl_loss=True`` in loss_cfg, a reference model will be loaded
       for KL divergence computation during training.
    """

    model_config = ConfigDict(title="Worker config", extra="forbid", arbitrary_types_allowed=True)
    model_cfg: TransformerConfig | BaseComposeConfig
    optim_cfg: OptimConfig
    loss_cfg: BaseRLLossConfig
    lr_cfg: LRConfig
    fsdp_cfg: FSDPConfig
    load_from: str | Path  # TODO: 把 actor 和 ref 配置分离
    optimizer_steps: int = 1
    sp_size: int = 1
    pack_max_length: int
    ref_load_from: str | Path | None = None
    ref_model_fsdp_cfg: FSDPConfig | None = None
    log_dir: str | Path | None = None
    update_weight_bucket_size_in_gb: float = 0.5  # 512MB
    seed: None | int = None  # if None, use RLTrainer seed
    profile_step: list[int] | int | None = None  # 1-based global RL train_step ids to profile.
    profile_time: bool = True
    profile_memory: bool = False
    free_rollout_routed_experts_in_worker: bool = True  # 默认不需要用户配置

    # sft config
    sft_dataloader_cfg: DataloaderConfig | None = None
    sft_global_batch_size: int = -1
    rollout_steps_per_sft: int = 1
    sft_loss_cfg: CELossConfig = CELossConfig()

    def build(self, placement_group: "PlacementGroup"):
        """Build training workers and controller from this config and placement
        group."""
        # import here to avoid circular import
        from xtuner.v1.rl.trainer.controller import TrainingController
        from xtuner.v1.rl.utils import AutoAcceleratorWorkers

        TrainingWorkerCls = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                }
            }
        )(TrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(TrainingWorkerCls, self, placement_group)
        ray.wait([w.ready.remote() for w in train_workers])
        return TrainingController(workers=train_workers)


class WorkerInputItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.LongTensor
    advantages: torch.Tensor
    rollout_logprobs: torch.Tensor | None


class WorkerTrainLogItem(TypedDict, total=False):
    step_consumed_tokens: int
    efficient_attn_ratio: float
    grad_norm: float


class WorkerLogItem(TypedDict):
    train_entropy: float
    rollout_entropy: NotRequired[float]
    mismatch_metrics: NotRequired[dict[str, float]]
    rollout_is_metrics: NotRequired[dict[str, float]]
    train_metrics: List[WorkerTrainLogItem]
    sft_train_metrics: NotRequired[dict[str, float]]


class TrainingWorker(SingleAcceleratorWorker):
    _SAVE_WEIGHTS_DIR = "weights"
    _SAVE_SFT_DATALOADER_DIR = "sft_dataloader"
    _SAVE_SFT_TRAIN_STATE_PATH = "sft_train_state.json"

    def __init__(
        self,
        worker_cfg: WorkerConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        super().__init__(worker_cfg, rank, master_addr, master_port, world_size, accelerator)
        self.config = cast(WorkerConfig, self.config)
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
        self.rank = rank

        # TODO: add lr scheduler
        log_dir = worker_cfg.log_dir
        self.log_dir = None
        if log_dir is not None:
            self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
            self.logger = get_logger(log_dir=self.log_dir, tag="TrainingWorker")
        else:
            self.logger = get_logger()

        if os.environ.get("XTUNER_DEBUG_OFFLOAD_MEMORY", "0") == "1":
            self._enable_cuda_memory_history()

        self._set_deterministic()
        self._set_random_seed(worker_cfg.seed)

        self.data_mesh = self._init_data_mesh(sp_size=worker_cfg.sp_size)
        self.sp_mesh = self.data_mesh["sp"]

        self._init_sft(worker_cfg)

        if not worker_cfg.fsdp_cfg.torch_compile:
            worker_cfg.model_cfg.compile_cfg = False
        self._engine = self._build_engine(worker_cfg)

        self._has_ref = False
        if worker_cfg.loss_cfg.use_kl_loss:
            self._has_ref = True
            if worker_cfg.ref_load_from is None:
                worker_cfg.ref_load_from = worker_cfg.load_from
            self._ref_model = self._build_ref_model(
                worker_cfg.model_cfg, worker_cfg.ref_load_from, worker_cfg.ref_model_fsdp_cfg
            )

        self._optimizer_steps = worker_cfg.optimizer_steps
        profile_step = worker_cfg.profile_step
        if isinstance(profile_step, int):
            profile_step = [profile_step]
        self._profile_step = set(profile_step or [])
        self._profile_time = worker_cfg.profile_time
        self._profile_memory = worker_cfg.profile_memory
        self._global_train_step = 0

        if worker_cfg.loss_cfg.chunk_size is not None:
            mode = "chunk"
        else:
            mode = "eager"
        self.logprob_cfg = LogProbConfig(chunk_size=worker_cfg.loss_cfg.chunk_size, mode=mode)
        self.mtp_config = None
        if isinstance(worker_cfg.model_cfg, BaseComposeConfig):
            if hasattr(worker_cfg.model_cfg.text_config, "mtp_config"):
                self.mtp_config = worker_cfg.model_cfg.text_config.mtp_config

        self.update_weighter = UpdateWeighter(
            rank=self.rank,
            logger=self.logger,
            config=self.config,
            engine=self._engine,
        )

    @ray_method
    def update_rollout_info(self, *args, **kwargs):
        return self.update_weighter.update_rollout_info(*args, **kwargs)

    @ray_method
    def update_weights(self):
        return self.update_weighter.update_weights()

    def _init_sft(self, worker_cfg: WorkerConfig):
        self._sft_dataloader_config = worker_cfg.sft_dataloader_cfg
        self._sft_dataloader: Dataloader | None = None
        self._sft_dataloader_iter: Iterable | None = None
        self._sft_loss_cfg: CELossConfig | None = None
        self._rollout_steps_per_sft = worker_cfg.rollout_steps_per_sft

        self._rollout_step = 0
        self._sft_cur_epoch = 0
        self._sft_total_consumed_tokens = 0

        if self._sft_dataloader_config is not None:
            assert worker_cfg.sft_global_batch_size > 0, "sft_global_batch_size must be greater than 0"
            assert worker_cfg.seed is not None, "seed must be set when sft_dataloader_config is not None"
            tokenizer = AutoTokenizer.from_pretrained(worker_cfg.load_from, trust_remote_code=True)
            self._sft_dataloader = self._sft_dataloader_config.build(
                tokenizer=tokenizer,
                dp_mesh=self.data_mesh["dp"],
                global_batch_size=worker_cfg.sft_global_batch_size,
                micro_batch_size=1,
                seed=worker_cfg.seed,
            )
            self.logger.info(f"Sft Dataloader len: {len(self._sft_dataloader)}")

            sft_loss_cfg = worker_cfg.sft_loss_cfg
            if worker_cfg.sft_loss_cfg is None:
                sft_loss_cfg = CELossConfig()
            self._sft_loss_cfg = sft_loss_cfg

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
            self.logger.info("Setting deterministic algorithms of TrainingWorker.")
            set_deterministic()

    def _set_random_seed(self, seed: None | int):
        set_random_seed(seed)

    def _build_engine(self, worker_cfg: WorkerConfig) -> TrainEngine:
        engine = TrainEngine(  # type: ignore
            optim_cfg=worker_cfg.optim_cfg,
            fsdp_cfg=worker_cfg.fsdp_cfg,
            model_cfg=worker_cfg.model_cfg,
        )
        if worker_cfg.load_from is not None:
            engine.from_hf(worker_cfg.load_from)

        if engine.model.compile_cfg is not None and self.rank == 0:
            self.logger.info(f"The `compile_cfg` of model is {json.dumps(engine.model.compile_cfg, indent=4)}")
        return engine

    def _build_ref_model(
        self,
        ref_model_cfg: TransformerConfig | BaseComposeConfig,
        load_from: str | Path,
        ref_model_fsdp_cfg: FSDPConfig | None = None,
    ):
        # TODO: 需要重构，使得能更优雅的兼容 mllm
        model: BaseComposeModel | XtunerBaseModel
        with torch.device("meta"):
            model = ref_model_cfg.build()

        if isinstance(ref_model_cfg, BaseComposeConfig):
            assert ref_model_cfg.text_config.float8_cfg is None, "BaseComposeConfig does not support float8"
            if ref_model_fsdp_cfg is None:
                ref_model_fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
            model = model.fully_shard(ref_model_fsdp_cfg)
            model.from_hf(hf_path=load_from)
            model.eval()  # type: ignore
        else:
            ref_model_cfg = cast(TransformerConfig, ref_model_cfg)
            if ref_model_cfg.float8_cfg is not None and ref_model_cfg.float8_cfg.enable_float8:
                float8_handler = Float8Handler(
                    scaling_granularity_gemm=ref_model_cfg.float8_cfg.scaling_granularity_gemm,
                    scaling_granularity_grouped_gemm=ref_model_cfg.float8_cfg.scaling_granularity_grouped_gemm,
                )
            else:
                float8_handler = None
            if ref_model_fsdp_cfg is None:
                ref_model_fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
            model = model.fully_shard(ref_model_fsdp_cfg)  # type: ignore

            model.from_hf(hf_path=load_from)
            model.eval()  # type: ignore
            if float8_handler is not None:
                # As the ref model is not updated, we only compute params' scales once
                float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)  # type: ignore
        model.to_device("cpu")  # type: ignore
        DEVICE_MODULE.empty_cache()
        return model

    def _init_data_mesh(
        self,
        sp_size: int,
    ):
        world_size = dist.get_world_size()
        if world_size % sp_size != 0:
            raise ParallelConfigException(
                f"Found sp_size {sp_size}, world_size {world_size}."
                "sequence parallel size must be a divisor of world size."
            )
        dp_size = world_size // sp_size

        # TODO: fsdp_config could be None
        device = str(DEVICE) if not self.config.fsdp_cfg.cpu_offload else "cpu"

        data_mesh = init_device_mesh(
            device,
            (dp_size, sp_size),
            mesh_dim_names=("dp", "sp"),
        )
        return data_mesh

    def compute_actor_logprobs(
        self,
        seq_ctx_list: list[SequenceContext],
        shifted_labels_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # precompute float8 dynamic scale only once
        self._engine._maybe_precompute_float8_dynamic_scale_for_fsdp()
        old_logprobs_list: list[torch.Tensor] = []
        for seq_ctx, shifted_labels in zip(seq_ctx_list, shifted_labels_list):
            loss_ctx = self.logprob_cfg.build(data={"shifted_labels": shifted_labels})
            assert loss_ctx is not None
            output = self._engine.forward_only(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
            old_logprobs_list.append(output["loss"])
        return old_logprobs_list

    def compute_ref_logprobs(
        self, seq_ctx_list: list[SequenceContext], shifted_labels_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        assert self._has_ref
        self._ref_model.to_device(DEVICE)
        ref_logprobs_list: list[torch.Tensor] = []
        for seq_ctx, shifted_labels in zip(seq_ctx_list, shifted_labels_list):
            with torch.no_grad():
                loss_ctx = self.logprob_cfg.build(data={"shifted_labels": shifted_labels})
                assert loss_ctx is not None
                ref_output = self._ref_model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
                ref_logprobs_list.append(ref_output["loss"])
        self._ref_model.to_device("cpu")
        return ref_logprobs_list

    def _add_rollout_routed_experts(
        self, seq_ctx: SequenceContext, rollout_routed_experts: torch.Tensor | list[torch.Tensor | ray.ObjectRef]
    ):
        language_cfg = (
            self.config.model_cfg.text_config
            if isinstance(self.config.model_cfg, BaseComposeConfig)
            else self.config.model_cfg
        )

        to_free_routed_expert_refs: list[ray.ObjectRef] = []
        if isinstance(rollout_routed_experts, list):
            # list[n,l,e]
            out_rollout_routed_expert = []
            for rollout_routed_expert in rollout_routed_experts:
                if isinstance(rollout_routed_expert, torch.Tensor):
                    rollout_routed_experts_tensor = torch.randint(
                        low=0,
                        high=language_cfg.n_routed_experts,
                        size=(
                            rollout_routed_expert.size(0),
                            language_cfg.num_hidden_layers,
                            language_cfg.num_experts_per_tok,
                        ),
                    )
                    out_rollout_routed_expert.append(rollout_routed_experts_tensor)
                else:
                    rollout_routed_expert_refs = rollout_routed_expert
                    if isinstance(rollout_routed_expert_refs, ray.ObjectRef):
                        rollout_routed_expert = ray.get(rollout_routed_expert_refs)
                    elif isinstance(rollout_routed_expert_refs, list):
                        _rollout_routed_expert = []
                        _rollout_routed_expert_refs = rollout_routed_expert_refs
                        for rollout_routed_expert_ref in rollout_routed_expert_refs:
                            rollout_routed_expert = ray.get(rollout_routed_expert_ref)  # np
                            _rollout_routed_expert.append(rollout_routed_expert)
                        rollout_routed_expert = np.concatenate(_rollout_routed_expert, axis=0)[1:, ...]
                        rollout_routed_expert_refs = _rollout_routed_expert_refs
                    else:
                        raise ValueError(
                            f"Invalid rollout_routed_expert_refs type: {type(rollout_routed_expert_refs)}"
                        )
                    # Some agent loops export routed-expert refs from the rollout trace store.
                    # Those refs may be shared by multiple trainable segments and replicated
                    # train workers, so they must be released by the trainer after all workers
                    # finish consuming the batch.
                    if self.config.free_rollout_routed_experts_in_worker:
                        if self.sp_mesh is None or self.sp_mesh.size() == 1:
                            ray.internal.free(rollout_routed_expert_refs, local_only=False)
                        else:
                            if self.sp_mesh.get_local_rank() == 0:
                                # only free once of sp mesh
                                to_free_routed_expert_refs.append(rollout_routed_expert_refs)
                    rollout_routed_expert = torch.as_tensor(rollout_routed_expert, dtype=torch.long)
                    rollout_routed_expert = rollout_routed_expert.reshape(
                        -1, language_cfg.num_hidden_layers, language_cfg.num_experts_per_tok
                    )
                    out_rollout_routed_expert.append(rollout_routed_expert)

            seq_ctx.rollout_routed_experts = torch.cat(out_rollout_routed_expert, dim=0)  # max_len,l,e
        else:
            assert isinstance(rollout_routed_experts, torch.Tensor), (
                f"padding experts should be a dummy tensor, bug got {type(rollout_routed_experts)}"
            )
            rollout_routed_experts_tensor = torch.randint(
                low=0,
                high=language_cfg.n_routed_experts,
                size=(
                    self.config.pack_max_length,
                    language_cfg.num_hidden_layers,
                    language_cfg.num_experts_per_tok,
                ),
            )
            seq_ctx.rollout_routed_experts = rollout_routed_experts_tensor

        assert seq_ctx.input_ids is not None, "input_ids is None"
        assert seq_ctx.rollout_routed_experts.size(0) == seq_ctx.input_ids.size(1), (
            f"rollout_routed_experts.size(0) {seq_ctx.rollout_routed_experts.size(0)} != input_ids.size(1) {seq_ctx.input_ids.size(1)}"
        )

        if self.config.free_rollout_routed_experts_in_worker and self.sp_mesh is not None and self.sp_mesh.size() > 1:
            dist.barrier()
            for free_routed_expert_refs in to_free_routed_expert_refs:
                ray.internal.free(free_routed_expert_refs, local_only=False)
            del to_free_routed_expert_refs

    @contextmanager
    def _maybe_profiling(self, global_train_step: int, phase: str):
        if global_train_step not in self._profile_step:
            yield
            return

        if self.log_dir is not None:
            profile_home = self.log_dir.parent
        else:
            profile_home = Path(os.environ.get("WORK_DIR", "."))

        with contextlib.ExitStack() as stack:
            if self._profile_time:
                time_dir = profile_home / "profiling_time" / phase / f"global-step-{global_train_step}"
                stack.enter_context(profiling_time(time_dir))
            if self._profile_memory:
                memory_dir = profile_home / "profiling_memory" / phase / f"global-step-{global_train_step}"
                stack.enter_context(profiling_memory(memory_dir))
            yield

    @ray_method
    def fit(self, data_batches: list[WorkerInputItem], rollout_idx: int) -> WorkerLogItem:
        # NOTE: sglang会清除logger handle, 重新创建
        self.logger = get_logger(log_dir=self.log_dir, tag="TrainingWorker")
        loss_cfg: BaseRLLossConfig = self.config.loss_cfg
        num_batches = len(data_batches)
        iters_per_step = math.ceil(num_batches / self._optimizer_steps)
        if num_batches < self._optimizer_steps:
            self.logger.info(
                f"Optimizer only step once because num_batches {num_batches} < optimizer_steps {self._optimizer_steps}."
            )

        # Update seq_ctx: pixel_values, rollout_routed_experts
        # Init loss_ctx: shifted_labels, advantages, rollout_logprobs
        seq_ctx_list: list[SequenceContext] = []
        loss_ctx_list: list[BaseRLLossContext] = []
        mtp_loss_ctx_list: list[list[MTPLossContext]] = []
        prepare_inputs_begin = time.perf_counter()
        for data in data_batches:
            # update seq_ctx
            seq_ctx = data["seq_ctx"]
            pixel_values = seq_ctx.pixel_values
            if pixel_values is not None:
                if not isinstance(pixel_values, np.ndarray):
                    assert isinstance(pixel_values, list), (
                        f"pixel_values should be list of tensor, got {type(pixel_values)}"
                    )
                    pixel_values = ray.get(list(pixel_values))
                    pixel_values = [torch.as_tensor(pixel_value) for pixel_value in pixel_values]
                    pixel_values = torch.cat(pixel_values, dim=0)
                    seq_ctx.pixel_values = pixel_values
                else:
                    raise NotImplementedError("The case where pixel_values is a numpy array is not implemented yet.")

            rollout_routed_experts = seq_ctx.rollout_routed_experts
            if rollout_routed_experts is not None:
                self._add_rollout_routed_experts(seq_ctx, rollout_routed_experts)

            seq_ctx = data["seq_ctx"].to(DEVICE)
            if self.sp_mesh.size() > 1:
                seq_ctx = seq_ctx.split(self.sp_mesh)

            # init loss_ctx
            shifted_labels = data["shifted_labels"].to(DEVICE)
            advantages = data["advantages"].to(DEVICE)
            rollout_logprobs = data.get("rollout_logprobs", None)
            rollout_logprobs = rollout_logprobs.to(DEVICE) if rollout_logprobs is not None else None
            loss_ctx = loss_cfg.build(
                data={
                    "shifted_labels": shifted_labels,
                    "advantages": advantages,
                    "rollout_logprobs": rollout_logprobs,
                },
                sp_mesh=self.sp_mesh,
            )

            seq_ctx_list.append(seq_ctx)
            assert loss_ctx is not None
            loss_ctx_list.append(loss_ctx)
            if self.mtp_config is not None:
                mtp_loss_ctxs_per_batch: list[MTPLossContext] = []
                for mtp_idx in range(self.mtp_config.num_layers):
                    mtp_loss_cfg = MTPLossConfig(
                        **loss_cfg.model_dump(include={"mode", "chunk_size"}),
                        mtp_depth=mtp_idx + 1,
                        detach_mtp_lm_head_weight=self.mtp_config.detach_mtp_lm_head_weight,
                    )
                    mtp_ctx = mtp_loss_cfg.build(
                        data={
                            "shifted_labels": shifted_labels,
                            "seq_ctx": seq_ctx,
                            "logprobs": rollout_logprobs,
                        },
                        sp_mesh=self.sp_mesh,
                    )
                    if mtp_ctx is not None:
                        mtp_loss_ctxs_per_batch.append(mtp_ctx)
                mtp_loss_ctx_list.append(mtp_loss_ctxs_per_batch)
        self.logger.debug(
            f"Rank{self.rank} Rollout {rollout_idx} prepare_inputs elapsed="
            f"{time.perf_counter() - prepare_inputs_begin:.4f}s"
        )
        self._maybe_log_memory_stage(f"rollout_{rollout_idx}/after_prepare_inputs")

        del data_batches
        self._maybe_log_memory_stage(f"rollout_{rollout_idx}/after_del_data_batches")

        # When sp_mesh.size() > 1, get the sp_split shifted_labels and rollout_logprobs
        shifted_labels_list = [loss_ctx.loss_kwargs.shifted_labels for loss_ctx in loss_ctx_list]
        rollout_logprobs_list = [loss_ctx.loss_kwargs.rollout_logprobs for loss_ctx in loss_ctx_list]

        # compute old logprobs
        self._maybe_log_memory_stage(f"rollout_{rollout_idx}/before_compute_actor_logprobs")
        self._maybe_log_deferred_fsdp_all_gathers(f"rollout_{rollout_idx}/before_compute_actor_logprobs")
        old_logprobs_list = self.compute_actor_logprobs(seq_ctx_list, shifted_labels_list)
        self._maybe_log_memory_stage(f"rollout_{rollout_idx}/after_compute_actor_logprobs")
        self._maybe_log_deferred_fsdp_all_gathers(f"rollout_{rollout_idx}/after_compute_actor_logprobs")
        for old_logprobs, loss_ctx in zip(old_logprobs_list, loss_ctx_list):
            loss_ctx.loss_kwargs.old_logprobs = old_logprobs
        self._maybe_log_memory_stage(f"rollout_{rollout_idx}/after_attach_old_logprobs")
        self._maybe_log_deferred_fsdp_all_gathers(f"rollout_{rollout_idx}/after_attach_old_logprobs")

        worker_log_item: WorkerLogItem = {"train_entropy": 0.0, "train_metrics": [], "sft_train_metrics": {}}
        logger_msg = f"Rollout {rollout_idx}: "

        # compute entropy
        rank_grad_tokens: torch.Tensor | None = None
        for shifted_labels in shifted_labels_list:
            mask = shifted_labels != -100
            grad_tokens = mask.sum()
            rank_grad_tokens = grad_tokens if rank_grad_tokens is None else rank_grad_tokens + grad_tokens
        rank_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
        global_grad_tokens = rank_grad_tokens
        dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)

        avg_sum_entropy = calculate_entropy(shifted_labels_list, old_logprobs_list, global_grad_tokens)
        avg_rollout_entropy = calculate_entropy(shifted_labels_list, rollout_logprobs_list, global_grad_tokens)
        self._maybe_log_memory_stage(f"rollout_{rollout_idx}/after_entropy")

        assert avg_sum_entropy is not None
        worker_log_item["train_entropy"] = avg_sum_entropy.item()
        logger_msg += f"avg entropy: {avg_sum_entropy:.4f}"
        if avg_rollout_entropy is not None:
            worker_log_item["rollout_entropy"] = avg_rollout_entropy.item()
            logger_msg += f", avg rollout entropy: {avg_rollout_entropy:.4f}"

        # compute rollout importance sampling metrics
        all_rollout_is_metrics = []
        all_mismatch_metrics = []
        for i, loss_ctx in enumerate(loss_ctx_list):
            if loss_ctx.loss_kwargs.rollout_logprobs is not None:
                # calculate importance sampling weights
                num_tokens = seq_ctx_list[i].seq_lens_q
                mismatch_metrics, rollout_is_metrics = loss_ctx.compute_rollout_is(self.sp_mesh, num_tokens)
                all_rollout_is_metrics.append(rollout_is_metrics)
                all_mismatch_metrics.append(mismatch_metrics)

        if len(all_mismatch_metrics) > 0:
            mismatch_metrics = merge_rollout_is_metrics(all_mismatch_metrics, DEVICE)
            if len(mismatch_metrics) > 0:
                worker_log_item["mismatch_metrics"] = mismatch_metrics
                logger_msg += f"\n rollout mismatch metrics:\n{json.dumps(mismatch_metrics, indent=4)}"

        if len(all_rollout_is_metrics) > 0:
            rollout_is_metrics = merge_rollout_is_metrics(all_rollout_is_metrics, DEVICE)
            if len(rollout_is_metrics) > 0:
                worker_log_item["rollout_is_metrics"] = rollout_is_metrics
                logger_msg += f"\n rollout importance sampling metrics:\n{json.dumps(rollout_is_metrics, indent=4)}"
        self._maybe_log_memory_stage(f"rollout_{rollout_idx}/after_rollout_is_metrics")

        if self.rank == 0:
            self.logger.info(logger_msg)

        only_calc_mismatch_ratio = os.environ.get("ONLY_CALC_MISMATCH_RATIO", "0") == "1"
        if only_calc_mismatch_ratio:
            self._maybe_log_memory_stage(f"rollout_{rollout_idx}/before_only_calc_mismatch_return")
            self._maybe_log_deferred_fsdp_all_gathers(f"rollout_{rollout_idx}/before_only_calc_mismatch_return")
            return worker_log_item

        # compute reference logprobs
        ref_logprobs_list: list[torch.Tensor] | None = None
        if self._has_ref:
            self._maybe_log_memory_stage(f"rollout_{rollout_idx}/before_compute_ref_logprobs")
            ref_logprobs_list = self.compute_ref_logprobs(seq_ctx_list, shifted_labels_list)
            self._maybe_log_memory_stage(f"rollout_{rollout_idx}/after_compute_ref_logprobs")

            for i, loss_ctx in enumerate(loss_ctx_list):
                loss_ctx.loss_kwargs.ref_logprobs = ref_logprobs_list[i]

            kl_div_sum: torch.Tensor | None = None
            for i, shifted_labels in enumerate(shifted_labels_list):
                mask = shifted_labels != -100
                kl_div = kl_penalty(
                    cast(torch.Tensor, old_logprobs_list[i]),
                    cast(torch.Tensor, ref_logprobs_list[i]),
                    loss_weights=mask,
                    kl_penalty="low_var_kl",
                )
                kl_div_sum = kl_div if kl_div_sum is None else kl_div_sum + kl_div

            kl_div_sum = cast(torch.Tensor, kl_div_sum)
            dist.all_reduce(kl_div_sum, op=dist.ReduceOp.SUM)
            avg_kl_div = kl_div_sum / global_grad_tokens if global_grad_tokens > 0 else 0
            self.logger.info(f"Rollout {rollout_idx}: avg KL divergence: {avg_kl_div:.4f}")
            self._maybe_log_memory_stage(f"rollout_{rollout_idx}/after_ref_kl")

        # compute batched loss context
        batched_loss_ctx_list: list[BaseRLLossContext] = []
        batched_mtp_loss_ctx_list: list[list[MTPLossContext]] = []
        LossContext = loss_cfg.loss_ctx_cls
        for i in range(0, len(loss_ctx_list), iters_per_step):
            batches_loss_ctx = loss_ctx_list[i : i + iters_per_step]
            batched_loss_ctx_list.extend(
                LossContext.build_batches(batches_loss_ctx)  # type: ignore[arg-type]
            )

            if self.mtp_config is not None:
                batches_seq_ctx = seq_ctx_list[i : i + iters_per_step]
                cu_seq_lens_list = [seq_ctx.cu_seq_lens_q for seq_ctx in batches_seq_ctx]
                # mtp_loss_ctx_list: list[list[MTPLossContext]], outer=batch, inner=mtp_depth
                num_mtp_depths = len(mtp_loss_ctx_list[0]) if mtp_loss_ctx_list else 0
                for mtp_idx in range(num_mtp_depths):
                    depth_mtp_loss_ctxs: list[LMHeadLossContext] = [
                        mtp_loss_ctx_list[j][mtp_idx]
                        for j in range(i, min(i + iters_per_step, len(mtp_loss_ctx_list)))
                    ]
                    batched_mtp_depth_ctxs = cast(
                        list[MTPLossContext],
                        MTPLossContext.build_batches(
                            depth_mtp_loss_ctxs,
                            cu_seq_lens_list=cu_seq_lens_list,
                            sp_mesh=self.sp_mesh,
                        ),
                    )
                    # Append each depth's batched ctx to the corresponding batch index
                    for batch_offset, mtp_ctx in enumerate(batched_mtp_depth_ctxs):
                        global_batch_idx = i + batch_offset
                        if global_batch_idx >= len(batched_mtp_loss_ctx_list):
                            batched_mtp_loss_ctx_list.append([mtp_ctx])
                        else:
                            batched_mtp_loss_ctx_list[global_batch_idx].append(mtp_ctx)
        self._maybe_log_memory_stage(f"rollout_{rollout_idx}/after_build_batched_loss_ctx")

        # train optimizer steps
        for i in range(0, len(seq_ctx_list), iters_per_step):
            global_train_step = self._global_train_step + 1
            batches_seq_ctx = seq_ctx_list[i : i + iters_per_step]
            batches_loss_ctx = batched_loss_ctx_list[i : i + iters_per_step]

            engine_input = [
                ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
                for seq_ctx, loss_ctx in zip(batches_seq_ctx, batches_loss_ctx)
            ]

            if self.mtp_config is not None:
                batches_mtp_loss_ctxs = batched_mtp_loss_ctx_list[i : i + iters_per_step]
                engine_input = [
                    ModelItem(
                        seq_ctx=seq_ctx,
                        loss_ctx=cast(
                            dict[str, BaseLossContext],
                            {"mtp": mtp_loss_ctx_depths, "lm": loss_ctx},
                        ),
                    )
                    for seq_ctx, loss_ctx, mtp_loss_ctx_depths in zip(
                        batches_seq_ctx, batches_loss_ctx, batches_mtp_loss_ctxs
                    )
                ]

            train_step_begin = time.perf_counter()
            self._maybe_log_memory_stage(f"rollout_{rollout_idx}/global_step_{global_train_step}/before_engine_train_step")
            self._maybe_log_deferred_fsdp_all_gathers(
                f"rollout_{rollout_idx}/global_step_{global_train_step}/before_engine_train_step"
            )
            with self._maybe_profiling(global_train_step, "train_step"):
                train_step_info = self._engine.train_step(
                    data_batches=engine_input,
                )
            self._maybe_log_memory_stage(f"rollout_{rollout_idx}/global_step_{global_train_step}/after_engine_train_step")
            self._maybe_log_deferred_fsdp_all_gathers(
                f"rollout_{rollout_idx}/global_step_{global_train_step}/after_engine_train_step"
            )
            self.logger.debug(
                f"Rank{self.rank} Rollout {rollout_idx} GlobalStep {global_train_step} "
                f"train_step[{i}].engine_train_step elapsed={time.perf_counter() - train_step_begin:.4f}s"
            )
            grad_norm = self._engine.clip_grad_norm()
            self._maybe_log_memory_stage(f"rollout_{rollout_idx}/global_step_{global_train_step}/after_clip_grad_norm")
            self._engine.step_optimizer(grad_norm)
            self._maybe_log_memory_stage(f"rollout_{rollout_idx}/global_step_{global_train_step}/after_step_optimizer")

            engine_logs_info = cast(dict[str, float], train_step_info.pop("logs_info"))  # type: ignore[misc]
            engine_extra_info = train_step_info.pop("extra_info")  # type: ignore[misc]

            if isinstance(engine_extra_info, ModelForwardExtraLogInfo):
                extra_info_dict = engine_extra_info.get()
            else:
                extra_info_dict = cast(dict, engine_extra_info)

            extra_info_dict = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in extra_info_dict.items()
                if isinstance(v, (torch.Tensor, int, float))
            }
            extra_info_dict = finalize_train_policy_metrics(extra_info_dict, DEVICE)
            train_step_info.pop("total_loss")  # type: ignore[misc]

            train_log_item = WorkerTrainLogItem(
                **engine_logs_info,  # type: ignore[typeddict-item]
                **train_step_info,
                **extra_info_dict,
                grad_norm=grad_norm.item(),
            )
            worker_log_item["train_metrics"].append(train_log_item)

            # Extract logs_info for logging
            log_str = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in train_log_item.items()
                if not key.startswith("reduced_train_policy_") and key != "max_ratio"
            )
            log_str = f"Rank{self.rank} Rollout {rollout_idx} Step {i}: " + log_str
            self.logger.info(log_str)
            self._global_train_step = global_train_step
            self._maybe_log_memory_stage(f"rollout_{rollout_idx}/global_step_{global_train_step}/after_train_log")

        self._rollout_step += 1
        if self._sft_dataloader is not None and self._rollout_step % self._rollout_steps_per_sft == 0:
            train_step_info = self._fit_sft()
            engine_logs_info = train_step_info["logs_info"]
            worker_log_item["sft_train_metrics"] = {
                **engine_logs_info,
                **train_step_info["extra_info"].get(),
                "efficient_attn_ratio": train_step_info["efficient_attn_ratio"],
            }

        self._maybe_log_memory_stage(f"rollout_{rollout_idx}/fit_end")
        return worker_log_item

    def _fit_sft(self):
        self.logger.info(f"Train SFT after {self._rollout_step} RL steps")
        if self._sft_dataloader_iter is None:
            self._sft_dataloader_iter = iter(self._sft_dataloader)

        time_before_get_data = time.time()
        data_batch = self._next_sft_data_batch()
        time_before_train_step = time.time()
        data_time = time_before_train_step - time_before_get_data
        DEVICE_MODULE.reset_peak_memory_stats()

        train_step_info, grad_norm = self._train_one_step_sft(data_batch)

        time_after_train_step = time.time()
        step_time = time_after_train_step - time_before_train_step
        step_consumed_tokens = train_step_info["step_consumed_tokens"]

        reduced_step_consumed_tokens = self._reduce_number_across_rank(step_consumed_tokens)
        self._sft_total_consumed_tokens += reduced_step_consumed_tokens

        self._sft_log_step(
            train_step_info=train_step_info,
            local_step_consumed_tokens=step_consumed_tokens,
            step_consumed_tokens=reduced_step_consumed_tokens,
            total_consumed_tokens=self._sft_total_consumed_tokens,
            data_time=data_time,
            step_time=step_time,
            grad_norm=grad_norm,
        )

        return train_step_info

    def _next_sft_data_batch(self):
        try:
            data = next(self._sft_dataloader_iter)  # type: ignore[assignment]
        except StopIteration:
            self._sft_cur_epoch += 1
            self._sft_dataloader.set_epoch(self._sft_cur_epoch)
            self._sft_dataloader_iter = iter(self._sft_dataloader)
            data = next(self._sft_dataloader_iter)
        return data

    def _train_one_step_sft(self, data_batch):
        seq_ctx_list: list[SequenceContext] = []
        loss_cfg: CELossConfig = self._sft_loss_cfg
        loss_ctx_list: list[CELossContext] = []
        for data in data_batch:
            seq_ctx = data["seq_ctx"].to(DEVICE)
            if self.sp_mesh.size() > 1:
                seq_ctx = seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)
            seq_ctx_list.append(seq_ctx)
            loss_ctx = loss_cfg.build(data={"shifted_labels": data["shifted_labels"]}, sp_mesh=self.sp_mesh)
            loss_ctx_list.append(loss_ctx)

        del data_batch

        cu_seq_lens_list = [seq_ctx.cu_seq_lens_q for seq_ctx in seq_ctx_list]
        loss_ctx_list = CELossContext.build_batches(
            loss_ctx_list, cu_seq_lens_list=cu_seq_lens_list, sp_mesh=self.sp_mesh
        )

        engine_input = [
            ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
            for seq_ctx, loss_ctx in zip(seq_ctx_list, loss_ctx_list)
        ]

        train_step_info = self._engine.train_step(engine_input)
        grad_norm = self._engine.clip_grad_norm()
        self._engine.step_optimizer(grad_norm)
        return train_step_info, grad_norm

    def _sft_log_step(
        self,
        train_step_info: TrainStepInfo,
        local_step_consumed_tokens: int,
        step_consumed_tokens: int,
        total_consumed_tokens: int,
        data_time: float,
        step_time: float,
        grad_norm: torch.Tensor,
    ):
        tgs = local_step_consumed_tokens / step_time
        logs_info = train_step_info.get("logs_info", {})
        log_items = [f"{k}: {v:.8f}" for k, v in logs_info.items() if "loss" in k]
        log_items.append(f"total_loss: {train_step_info['total_loss']:.8f}")
        loss_log_str = ", ".join(log_items)

        max_memory = DEVICE_MODULE.max_memory_allocated()  # type: ignore[attr-defined]
        reserved_memory = DEVICE_MODULE.max_memory_reserved()  # type: ignore[attr-defined]

        self.logger.info(
            f"Rank{self.rank} Step {self._rollout_step}: data_time: {data_time:.4f} time: {step_time:.4f} "
            f"text_tokens: {local_step_consumed_tokens} "
            f"step_consumed_tokens: {step_consumed_tokens} "
            f"total_consumed_tokens: {total_consumed_tokens} "
            f"efficient_attn_ratio: {train_step_info['efficient_attn_ratio']:.4f} "
            f"{loss_log_str} "
            f"grad_norm: {grad_norm:.8f} "
            f"max_memory: {max_memory / (1024**3):.2f} GB "
            f"reserved_memory: {reserved_memory / (1024**3):.2f} GB "
            f"tgs: {tgs:.4f}"
        )

    def _reduce_number_across_rank(self, rank_number: int) -> int:
        _gathered_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(_gathered_list, rank_number)
        reduced_number = sum(_gathered_list)  # type: ignore[arg-type]
        return reduced_number

    @ray_method
    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        self._engine.save_hf(hf_dir, save_dtype)

    @ray_method
    def get_data_replicate_size(self) -> int:
        """Get the data replicate size for the training worker."""
        # tp and pp will affect the data replicate size in engine
        # sp will affect the data replicate size in worker
        return self._engine.data_replicate_size * self.sp_mesh.size()

    @ray_method
    def get_model_cfg(self):
        model_cfg = self._engine.model_cfg
        return model_cfg

    @ray_method
    def offload_model(self):
        self._maybe_log_memory_stage("offload_model/before_model_to_cpu")
        self._maybe_log_deferred_fsdp_all_gathers("offload_model/before_model_to_cpu")
        self._engine.put_model_to_device("cpu")
        if os.environ.get("XTUNER_DEBUG_OFFLOAD_MEMORY", "0") == "1":
            self._log_offload_memory_debug("after_model_to_cpu_before_deferred_fsdp_release")
        self._maybe_log_deferred_fsdp_all_gathers("offload_model/after_model_to_cpu_before_deferred_release")
        self._release_deferred_fsdp_all_gathers("offload_model")
        DEVICE_MODULE.empty_cache()
        self.logger.info(
            f"Offloaded model to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )
        if os.environ.get("XTUNER_DEBUG_OFFLOAD_MEMORY", "0") == "1":
            self._log_offload_memory_debug("after_deferred_fsdp_release")
        self._maybe_log_deferred_fsdp_all_gathers("offload_model/after_deferred_release")

    def _release_deferred_fsdp_all_gathers(self, log_tag: str) -> None:
        """Free FSDP2 deferred all-gather buffers before long offload gaps.

        Two FSDP2 overlap states can keep CUDA flat buffers alive outside model
        parameters:
        - ``comm_ctx.all_gather_state`` for a result already consumed by a
          module's ``wait_for_unshard()`` and deferred for overlap.
        - ``FSDPParamGroup._all_gather_result`` for an explicit prefetch whose
          target module did not subsequently run forward and consume it.

        The latter is the important colocated RL case when old-logprob forward
        only runs the LM loss but the last decoder layer prefetched the MTP
        layer. Before switching back to rollout, release both forms explicitly.
        """

        try:
            from torch.distributed._composable_state import _get_module_state
        except Exception:
            return

        released_states = 0
        released_param_groups = 0
        debug = os.environ.get("XTUNER_DEBUG_OFFLOAD_MEMORY", "0") == "1"
        debug_items: list[str] = []

        def get_all_gather_result(holder):
            result = getattr(holder, "_all_gather_result", None)
            if result is not None:
                return result
            result = getattr(holder, "all_gather_result", None)
            if result is not None:
                return result
            result = getattr(holder, "result", None)
            if result is not None:
                return result
            if isinstance(holder, tuple) and len(holder) > 0:
                return holder[0]
            return None

        def record_debug_item(source: str, holder) -> None:
            if not debug or holder is None:
                return
            all_gather_result = get_all_gather_result(holder)
            all_gather_output = getattr(all_gather_result, "all_gather_output", None)
            if not torch.is_tensor(all_gather_output):
                return
            debug_items.append(
                f"{source}: {all_gather_output.numel() * all_gather_output.element_size() / (1024**2):.2f} MB "
                f"shape={tuple(all_gather_output.shape)} dtype={all_gather_output.dtype} "
                f"device={all_gather_output.device}"
            )

        def wait_all_gather_result(all_gather_result) -> None:
            event = getattr(all_gather_result, "all_gather_event", None)
            if event is not None:
                try:
                    torch.accelerator.current_stream().wait_event(event)
                except Exception:
                    DEVICE_MODULE.synchronize()
            work = getattr(all_gather_result, "all_gather_work", None)
            if isinstance(work, dist.distributed_c10d.Work):
                work.wait()

        for module in self._engine.model.modules():
            try:
                state = _get_module_state(module)
            except Exception:
                continue
            if state is None:
                continue

            comm_ctx = getattr(state, "_comm_ctx", None)
            all_gather_state = getattr(comm_ctx, "all_gather_state", None)
            if all_gather_state is not None:
                record_debug_item(f"{module.__class__.__name__}.comm_ctx.all_gather_state", all_gather_state)
                event = getattr(all_gather_state, "event", None)
                if event is not None:
                    try:
                        event.synchronize()
                    except Exception:
                        DEVICE_MODULE.synchronize()
                comm_ctx.all_gather_state = None
                released_states += 1

            param_group = getattr(state, "_fsdp_param_group", None)
            if (all_gather_result := getattr(param_group, "_all_gather_result", None)) is not None:
                record_debug_item(f"{module.__class__.__name__}._fsdp_param_group._all_gather_result", param_group)
                wait_all_gather_result(all_gather_result)
                param_group._all_gather_result = None
                released_param_groups += 1

        if debug and (released_states or released_param_groups):
            detail = "\n".join(debug_items) if debug_items else "<no tensor details>"
            self.logger.info(
                f"[{log_tag}] released deferred FSDP all-gathers: "
                f"comm_states={released_states}, param_groups={released_param_groups}\n{detail}"
            )

    def _fsdp_deferred_debug_enabled(self) -> bool:
        return os.environ.get("XTUNER_DEBUG_FSDP_DEFERRED", "0") == "1"

    def _maybe_log_deferred_fsdp_all_gathers(self, tag: str) -> None:
        if not self._fsdp_deferred_debug_enabled():
            return
        try:
            from torch.distributed._composable_state import _get_module_state
        except Exception as e:
            self.logger.warning(f"[{tag}] failed to import FSDP composable state helper: {e}")
            return

        module_names = {id(module): name or "<root>" for name, module in self._engine.model.named_modules()}
        fsdp_groups = []
        comm_ctx_states = []
        seen_param_groups: set[int] = set()
        seen_comm_ctx: set[int] = set()
        deferred_rows: list[str] = []

        for module in self._engine.model.modules():
            try:
                state = _get_module_state(module)
            except Exception:
                continue
            if state is None:
                continue

            module_name = module_names.get(id(module), module.__class__.__name__)
            param_group = getattr(state, "_fsdp_param_group", None)
            if param_group is not None and id(param_group) not in seen_param_groups:
                seen_param_groups.add(id(param_group))
                fsdp_groups.append((module_name, param_group))

            comm_ctx = getattr(state, "_comm_ctx", None)
            if comm_ctx is None or id(comm_ctx) in seen_comm_ctx:
                continue
            seen_comm_ctx.add(id(comm_ctx))
            comm_ctx_states.append((module_name, comm_ctx))

        for module_name, comm_ctx in comm_ctx_states:
            all_gather_state = getattr(comm_ctx, "all_gather_state", None)
            if all_gather_state is None:
                continue
            deferred_rows.append(
                self._format_deferred_fsdp_holder(
                    source=f"comm_ctx@0x{id(comm_ctx):x} first_seen_module={module_name}",
                    holder=all_gather_state,
                    fsdp_groups=fsdp_groups,
                )
            )

        param_group_rows = []
        for module_name, param_group in fsdp_groups:
            holder = getattr(param_group, "_all_gather_result", None)
            if holder is None:
                continue
            param_group_rows.append(
                self._format_deferred_fsdp_holder(
                    source=f"param_group@0x{id(param_group):x} module={module_name}",
                    holder=param_group,
                    fsdp_groups=fsdp_groups,
                )
            )

        rows = deferred_rows + param_group_rows
        if not rows:
            self.logger.info(
                f"[{tag}] deferred FSDP all-gather states: none "
                f"(fsdp_groups={len(fsdp_groups)}, comm_ctx={len(seen_comm_ctx)})"
            )
            return
        self.logger.info(
            f"[{tag}] deferred FSDP all-gather states: count={len(rows)} "
            f"(fsdp_groups={len(fsdp_groups)}, comm_ctx={len(seen_comm_ctx)})\n" + "\n".join(rows)
        )

    def _format_deferred_fsdp_holder(self, source: str, holder, fsdp_groups: list[tuple[str, object]]) -> str:
        all_gather_result = self._extract_all_gather_result(holder)
        all_gather_output = getattr(all_gather_result, "all_gather_output", None)
        if not torch.is_tensor(all_gather_output):
            return f"{source}: holder={type(holder).__name__}, all_gather_output=<none>"

        output_numel = int(all_gather_output.numel())
        output_dtype = all_gather_output.dtype
        output_mb = output_numel * all_gather_output.element_size() / (1024**2)
        split_sizes = list(getattr(all_gather_result, "all_gather_input_split_sizes", []) or [])
        input_numels_raw = list(getattr(all_gather_result, "param_all_gather_input_numels", []) or [])
        input_numels = self._flatten_int_items(input_numels_raw)
        input_dtypes = list(getattr(all_gather_result, "param_all_gather_input_dtypes", []) or [])
        candidate_lines = self._match_fsdp_param_groups_for_all_gather(
            output_numel=output_numel,
            output_dtype=output_dtype,
            input_numels=input_numels,
            fsdp_groups=fsdp_groups,
        )
        return (
            f"{source}: holder={type(holder).__name__}, result={type(all_gather_result).__name__}, "
            f"output={output_mb:.2f} MB shape={tuple(all_gather_output.shape)} dtype={output_dtype} "
            f"device={all_gather_output.device} ptr=0x{all_gather_output.data_ptr():x}, "
            f"split_sizes={self._summarize_int_list(self._flatten_int_items(split_sizes))}, "
            f"param_input_numels={self._summarize_int_list(input_numels)}, "
            f"param_input_dtypes={self._summarize_obj_list(input_dtypes)}\n"
            f"    candidate_param_groups:\n{candidate_lines}"
        )

    def _extract_all_gather_result(self, holder):
        result = getattr(holder, "_all_gather_result", None)
        if result is not None:
            return result
        result = getattr(holder, "all_gather_result", None)
        if result is not None:
            return result
        result = getattr(holder, "result", None)
        if result is not None:
            return result
        if isinstance(holder, tuple) and len(holder) > 0:
            return holder[0]
        return None

    def _match_fsdp_param_groups_for_all_gather(
        self,
        output_numel: int,
        output_dtype: torch.dtype,
        input_numels: list[int],
        fsdp_groups: list[tuple[str, object]],
        limit: int = 5,
    ) -> str:
        candidates: list[tuple[int, str]] = []
        expected_input_numel = sum(input_numels) if input_numels else None
        for module_name, param_group in fsdp_groups:
            try:
                group_world_size = int(param_group._all_gather_process_group.size())
            except Exception:
                group_world_size = 0
            fsdp_params = list(getattr(param_group, "fsdp_params", []) or [])
            group_input_numels = []
            param_descriptions = []
            dtype_matches = 0
            for fsdp_param in fsdp_params:
                sharded_data = getattr(fsdp_param, "_sharded_param_data", None)
                sharded_numel = int(sharded_data.numel()) if torch.is_tensor(sharded_data) else 0
                group_input_numels.append(sharded_numel)
                param_dtype = getattr(fsdp_param, "param_dtype", None) or (
                    sharded_data.dtype if torch.is_tensor(sharded_data) else None
                )
                if param_dtype == output_dtype:
                    dtype_matches += 1
                param_descriptions.append(
                    f"{getattr(fsdp_param, '_param_fqn', None)}"
                    f" orig={tuple(getattr(fsdp_param, '_orig_size', ())) }"
                    f" shard={tuple(getattr(fsdp_param, 'sharded_size', ())) }"
                    f" gather_numel={sharded_numel}"
                    f" orig_dtype={getattr(fsdp_param, 'orig_dtype', None)}"
                    f" param_dtype={getattr(fsdp_param, 'param_dtype', None)}"
                    f" state={getattr(fsdp_param, 'sharded_state', None)}"
                )

            group_input_total = sum(group_input_numels)
            group_output_numel = group_input_total * group_world_size if group_world_size else -1
            score = 0
            if group_output_numel == output_numel:
                score += 100
            if expected_input_numel is not None and group_input_total == expected_input_numel:
                score += 50
            if len(group_input_numels) == len(input_numels):
                score += 10
            if dtype_matches == len(fsdp_params) and fsdp_params:
                score += 5
            if score == 0:
                continue
            candidates.append(
                (
                    score,
                    f"      score={score} module={module_name} module_fqn={getattr(param_group, '_module_fqn', None)} "
                    f"world_size={group_world_size} group_input={group_input_total} "
                    f"group_output={group_output_numel} training_state={getattr(param_group, '_training_state', None)} "
                    f"reshard_after_forward={getattr(param_group, '_reshard_after_forward', None)} "
                    f"is_unsharded={getattr(param_group, 'is_unsharded', None)} "
                    f"params={param_descriptions[:8]}"
                ),
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            return "      <no matching param group by output/input numel>"
        return "\n".join(line for _, line in candidates[:limit])

    def _summarize_int_list(self, values: list[int], limit: int = 8) -> str:
        if not values:
            return "[]"
        suffix = "" if len(values) <= limit else f", ... total={len(values)}"
        return f"[{', '.join(str(v) for v in values[:limit])}{suffix}] sum={sum(values)}"

    def _flatten_int_items(self, values) -> list[int]:
        flattened: list[int] = []

        def visit(value) -> None:
            if isinstance(value, int):
                flattened.append(value)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
                return
            try:
                flattened.append(int(value))
            except Exception:
                return

        visit(values)
        return flattened

    def _summarize_obj_list(self, values: list[object], limit: int = 8) -> str:
        if not values:
            return "[]"
        suffix = "" if len(values) <= limit else f", ... total={len(values)}"
        return f"[{', '.join(str(v) for v in values[:limit])}{suffix}]"

    def _log_offload_memory_debug(self, tag: str) -> None:
        self._log_cuda_allocator_debug(tag)
        self._log_live_cuda_tensors(tag)

    def _memory_stage_debug_enabled(self) -> bool:
        return os.environ.get("XTUNER_DEBUG_MEMORY_STAGES", "0") == "1"

    def _memory_stage_live_tensor_debug_enabled(self) -> bool:
        return os.environ.get("XTUNER_DEBUG_MEMORY_STAGE_LIVE_TENSORS", "0") == "1"

    def _maybe_log_memory_stage(self, tag: str) -> None:
        if not self._memory_stage_debug_enabled():
            return
        self._log_cuda_allocator_debug(tag)
        if self._memory_stage_live_tensor_debug_enabled():
            self._log_live_cuda_tensors(tag, limit=8)

    def _enable_cuda_memory_history(self) -> None:
        if os.environ.get("XTUNER_DEBUG_OFFLOAD_MEMORY_SNAPSHOT", "0") != "1":
            return
        try:
            torch.cuda.memory._record_memory_history(enabled="all", stacks="python")
        except Exception as e:
            self.logger.warning(f"Failed to enable CUDA memory history for offload debug: {e}")

    def _log_cuda_allocator_debug(self, tag: str) -> None:
        try:
            stats = torch.cuda.memory_stats()
        except Exception as e:
            self.logger.warning(f"[{tag}] failed to get CUDA memory stats: {e}")
            return

        def mb(key: str) -> float:
            return float(stats.get(key, 0)) / (1024**2)

        free_mb = total_mb = 0.0
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mb = free_bytes / (1024**2)
            total_mb = total_bytes / (1024**2)
        except Exception:
            pass

        self.logger.info(
            f"[{tag}] CUDA allocator stats: "
            f"allocated={DEVICE_MODULE.memory_allocated() / (1024**2):.2f} MB, "
            f"reserved={DEVICE_MODULE.memory_reserved() / (1024**2):.2f} MB, "
            f"max_allocated={DEVICE_MODULE.max_memory_allocated() / (1024**2):.2f} MB, "
            f"max_reserved={DEVICE_MODULE.max_memory_reserved() / (1024**2):.2f} MB, "
            f"active={mb('active_bytes.all.current'):.2f} MB, "
            f"active_large={mb('active_bytes.large_pool.current'):.2f} MB, "
            f"active_small={mb('active_bytes.small_pool.current'):.2f} MB, "
            f"inactive_split={mb('inactive_split_bytes.all.current'):.2f} MB, "
            f"segments={int(stats.get('segment.all.current', 0))}, "
            f"active_allocs={int(stats.get('allocation.all.current', 0))}, "
            f"device_free={free_mb:.2f} MB, device_total={total_mb:.2f} MB"
        )
        if os.environ.get("XTUNER_DEBUG_OFFLOAD_MEMORY_SNAPSHOT", "0") == "1":
            self._log_cuda_memory_snapshot(tag)

    def _log_cuda_memory_snapshot(self, tag: str, limit: int = 12) -> None:
        try:
            snapshot = torch.cuda.memory_snapshot()
        except Exception as e:
            self.logger.warning(f"[{tag}] failed to get CUDA memory snapshot: {e}")
            return

        active_blocks: list[tuple[int, int, int, str, str]] = []
        for segment in snapshot:
            segment_type = str(segment.get("segment_type", "<unknown>"))
            stream = str(segment.get("stream", "<unknown>"))
            segment_address = int(segment.get("address", 0) or 0)
            block_offset = 0
            for block in segment.get("blocks", []):
                state = str(block.get("state", ""))
                size = int(block.get("size", 0))
                if not state.startswith("active"):
                    block_offset += size
                    continue
                requested_size = int(block.get("requested_size", size))
                address = int(block.get("address", 0) or 0)
                if address == 0 and segment_address:
                    address = segment_address + block_offset
                frames = block.get("frames", [])
                frame_text = "<no python frames>"
                if frames:
                    frame_lines = []
                    for frame in frames[:6]:
                        filename = frame.get("filename", "<unknown>")
                        line = frame.get("line", "?")
                        name = frame.get("name", "<unknown>")
                        frame_lines.append(f"{filename}:{line}:{name}")
                    frame_text = " <- ".join(frame_lines)
                active_blocks.append((size, requested_size, address, segment_type, f"stream={stream} {frame_text}"))
                block_offset += size

        active_blocks.sort(key=lambda item: item[0], reverse=True)
        total_size = sum(item[0] for item in active_blocks)
        lines = [
            (
                f"{idx:02d}: size={size / (1024**2):.2f} MB "
                f"requested={requested_size / (1024**2):.2f} MB address=0x{address:x} "
                f"segment={segment_type} {frame_text}"
            )
            for idx, (size, requested_size, address, segment_type, frame_text) in enumerate(
                active_blocks[:limit], start=1
            )
        ]
        body = "\n".join(lines) if lines else "<none>"
        self.logger.info(
            f"[{tag}] CUDA memory snapshot active blocks: total={total_size / (1024**2):.2f} MB "
            f"count={len(active_blocks)}\n{body}"
        )
        if os.environ.get("XTUNER_DEBUG_ACTIVE_BLOCK_OWNERS", "0") == "1":
            self._log_active_block_python_owners(tag, active_blocks[:limit])

    def _log_active_block_python_owners(self, tag: str, active_blocks: list[tuple[int, int, int, str, str]]) -> None:
        gc.collect()
        tensor_rows: list[tuple[int, int, str, str, str, str]] = []
        seen_storages: set[tuple[int, int, str]] = set()
        for obj in gc.get_objects():
            try:
                if not torch.is_tensor(obj) or not obj.is_cuda:
                    continue
                storage = obj.untyped_storage()
                ptr = int(storage.data_ptr())
                nbytes = int(storage.nbytes())
                device = str(obj.device)
                key = (ptr, nbytes, device)
                if key in seen_storages:
                    continue
                seen_storages.add(key)
                tensor_rows.append((ptr, nbytes, device, str(tuple(obj.shape)), str(obj.dtype), type(obj).__name__))
            except Exception:
                continue

        lines: list[str] = []
        for idx, (block_size, requested_size, block_address, segment_type, frame_text) in enumerate(
            active_blocks, start=1
        ):
            if block_address == 0:
                lines.append(
                    f"{idx:02d}: block={block_size / (1024**2):.2f} MB address=<unknown> "
                    f"requested={requested_size / (1024**2):.2f} MB owners=<cannot_match_without_address>"
                )
                continue
            block_end = block_address + block_size
            matches = []
            for ptr, nbytes, device, shape, dtype, obj_type in tensor_rows:
                tensor_end = ptr + nbytes
                if ptr < block_end and tensor_end > block_address:
                    matches.append(
                        f"{nbytes / (1024**2):.2f} MB ptr=0x{ptr:x} {device} "
                        f"shape={shape} dtype={dtype} type={obj_type}"
                    )
            owner_text = "; ".join(matches[:6]) if matches else "<no Python tensor/storage owner found>"
            lines.append(
                f"{idx:02d}: block={block_size / (1024**2):.2f} MB "
                f"requested={requested_size / (1024**2):.2f} MB address=0x{block_address:x} "
                f"segment={segment_type} owners={owner_text} frame={frame_text}"
            )

        body = "\n".join(lines) if lines else "<none>"
        self.logger.info(f"[{tag}] active CUDA block Python owner probe:\n{body}")

    def _log_live_cuda_tensors(self, tag: str, limit: int = 15) -> None:
        gc.collect()
        rows: list[tuple[int, str, str, str, str, bool, torch.Tensor]] = []
        seen_storages: set[tuple[int, int, str]] = set()
        for obj in gc.get_objects():
            try:
                if not torch.is_tensor(obj) or not obj.is_cuda:
                    continue
                storage = obj.untyped_storage()
                storage_bytes = storage.nbytes()
                key = (storage.data_ptr(), storage_bytes, str(obj.device))
                if key in seen_storages:
                    continue
                seen_storages.add(key)
                rows.append(
                    (
                        storage_bytes,
                        str(tuple(obj.shape)),
                        str(obj.dtype),
                        str(obj.device),
                        type(obj).__name__,
                        bool(obj.requires_grad),
                        obj,
                    )
                )
            except Exception:
                continue

        rows.sort(key=lambda row: row[0], reverse=True)
        total_bytes = sum(row[0] for row in rows)
        top_lines = [
            (
                f"{idx:02d}: {storage_bytes / (1024**2):.2f} MB "
                f"shape={shape} dtype={dtype} device={device} type={obj_type} requires_grad={requires_grad}"
            )
            for idx, (storage_bytes, shape, dtype, device, obj_type, requires_grad, _) in enumerate(
                rows[:limit], start=1
            )
        ]
        top_text = "\n".join(top_lines) if top_lines else "<none>"
        self.logger.info(
            f"[{tag}] live CUDA tensors visible to Python: total={total_bytes / (1024**2):.2f} MB "
            f"unique_storages={len(rows)}\n{top_text}"
        )
        if rows:
            self.logger.info(f"[{tag}] largest CUDA tensor referrers:\n{self._format_tensor_referrers(rows[0][-1])}")

    def _format_tensor_referrers(self, tensor: torch.Tensor, limit: int = 20) -> str:
        lines: list[str] = []
        for ref in gc.get_referrers(tensor):
            if ref is lines:
                continue
            try:
                ref_type = type(ref).__name__
                detail = ""
                if isinstance(ref, dict):
                    keys = []
                    for key, value in list(ref.items())[:200]:
                        if value is tensor:
                            keys.append(repr(key))
                    detail = f" keys=[{', '.join(keys[:8])}]" if keys else f" len={len(ref)}"
                elif isinstance(ref, (list, tuple)):
                    indices = [str(idx) for idx, value in enumerate(ref[:200]) if value is tensor]
                    detail = f" indices=[{', '.join(indices[:8])}]" if indices else f" len={len(ref)}"
                else:
                    attrs = []
                    for name, value in vars(ref).items() if hasattr(ref, "__dict__") else []:
                        if value is tensor:
                            attrs.append(name)
                    detail = f" attrs=[{', '.join(attrs[:8])}]" if attrs else ""

                lines.append(f"{len(lines) + 1:02d}: type={ref_type}{detail}")
                if len(lines) >= limit:
                    break
            except Exception:
                continue
        return "\n".join(lines) if lines else "<none>"

    @ray_method
    def offload_optimizer(self):
        """Offload the optimizer of the training worker."""
        self._maybe_log_memory_stage("offload_optimizer/before_optimizer_to_cpu")
        self._engine.put_optimizer_to_device("cpu")
        self._maybe_log_memory_stage("offload_optimizer/after_optimizer_to_cpu_before_empty_cache")
        DEVICE_MODULE.empty_cache()
        self.logger.info(
            f"Offloaded optimizer to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, "
            f"reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )
        self._maybe_log_memory_stage("offload_optimizer/after_empty_cache")

    @ray_method
    def onload_model(self):
        self._maybe_log_memory_stage("onload_model/before_model_to_device")
        self._engine.put_model_to_device(DEVICE)
        self._maybe_log_memory_stage("onload_model/after_model_to_device")

    @ray_method
    def onload_optimizer(self):
        self._maybe_log_memory_stage("onload_optimizer/before_optimizer_to_device")
        self._engine.put_optimizer_to_device(DEVICE)
        self._maybe_log_memory_stage("onload_optimizer/after_optimizer_to_device")

    @ray_method
    def save(self, checkpoint_path: Path | str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training worker."""
        if not isinstance(checkpoint_path, Path):
            checkpoint_path = Path(checkpoint_path)
        weights_path = checkpoint_path / self._SAVE_WEIGHTS_DIR

        # Save model and optimizer
        self._engine.save_dcp(
            weights_dir=weights_path,
            save_optimizer=not no_save_optimizer,
        )

        # Save sft dataloader
        if self._sft_dataloader is not None:
            sft_dataloader_path = checkpoint_path / self._SAVE_SFT_DATALOADER_DIR
            dataloader_state = self._sft_dataloader.get_state_dict()
            total_consumed_samples = dataloader_state["total_consumed_samples"]
            if self.rank != 0:
                return

            torch.save(dataloader_state, sft_dataloader_path)

            train_state_path = checkpoint_path / self._SAVE_SFT_TRAIN_STATE_PATH
            with train_state_path.open("w") as f:
                f.write(
                    json.dumps(
                        {
                            "cur_step": self._rollout_step,
                            "cur_epoch": self._sft_cur_epoch,
                            "total_consumed_samples": total_consumed_samples,
                            "total_consumed_tokens": self._sft_total_consumed_tokens,
                        }
                    )
                )

    @ray_method
    def resume(self, load_checkpoint_cfg: LoadCheckpointConfig):
        """Resume the training worker from the checkpoint."""
        resume_from = load_checkpoint_cfg.checkpoint_path
        if resume_from is None:
            return
        if isinstance(resume_from, str):
            resume_from = Path(resume_from)
        self.logger.info(f"Resume from checkpoint: {resume_from}")

        if not resume_from.exists():
            raise FileNotFoundError(f"Checkpoint path {resume_from} does not exist.")

        weights_path = resume_from / self._SAVE_WEIGHTS_DIR
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint at {resume_from} has no '{self._SAVE_WEIGHTS_DIR}/' directory.")

        self._engine.load_dcp(
            weights_dir=weights_path,
            load_states=load_checkpoint_cfg.load_optimizer_states,
            load_args=load_checkpoint_cfg.load_optimizer_args,
        )

        # Resume sft dataloader
        if self._sft_dataloader is not None:
            train_state_path = resume_from / self._SAVE_SFT_TRAIN_STATE_PATH
            if not train_state_path.exists():
                raise FileNotFoundError(f"Train state path {train_state_path} does not exist.")
            with train_state_path.open("r") as f:
                train_state = json.loads(f.read())
            self._rollout_step = train_state["cur_step"]
            self._sft_cur_epoch = train_state["cur_epoch"]
            self._sft_total_consumed_tokens = train_state["total_consumed_tokens"]
            self.logger.info(f"Resume sft train state from {train_state_path}")

            sft_dataloader_path = resume_from / self._SAVE_SFT_DATALOADER_DIR
            if not sft_dataloader_path.exists():
                raise FileNotFoundError(f"Dataloader path {sft_dataloader_path} does not exist.")
            dataloader_state = torch.load(sft_dataloader_path, map_location=DEVICE)
            self._sft_dataloader.load_state_dict(dataloader_state)
            self.logger.info(f"Resume sft dataloader from {sft_dataloader_path}")

    @ray_method
    def ready(self) -> bool:
        return True


TrainingWorkerClass = ActorClass[TrainingWorker]
TrainingWorkerProxy = ActorProxy[TrainingWorker]
