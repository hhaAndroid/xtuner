import json
import os
from pathlib import Path

from recipe.frontierscience_bench.xtuner_dataset import RLSearchTokenizeFnConfig as FrontierScienceTokenizeFnConfig
from recipe.math_ci_eval.xtuner_dataset import RLMathTokenizeFnConfig as MathEvalTokenizeFnConfig
from recipe.math_code_interpreter.xtuner_dataset import RLMathTokenizeFnConfig as MathTrainTokenizeFnConfig
from recipe.paper_review.xtuner_dataset import RLReviewTokenizeFnConfig
from recipe.search_eval.xtuner_dataset import RLSearchTokenizeFnConfig as SearchEvalTokenizeFnConfig
from recipe.science_search_train.xtuner_dataset import RLSearchTokenizeFnConfig as ScienceSearchTokenizeFnConfig
from recipe.search_train.xtuner_dataset import RLSearchTokenizeFnConfig
from recipe.sgi_bench.xtuner_dataset import RLSearchTokenizeFnConfig as SGITokenizeFnConfig
from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.float8 import Float8Config, ScalingGranularity
from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5_VLMoE397BA17SplitConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import AgentInLocalhostLoopConfig
from xtuner.v1.rl.agent_loop_manager import AgentLoopManagerConfig, AsyncProduceStrategyConfig, SamplerConfig, TaskSpecConfig
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig
from intern_s1_delivery.modules.kpop_importance_sampling import KpopRolloutImportanceSampling
from intern_s1_delivery.advantage.rloo_entropy_badword import (
    OverlongRLOOGroupEntropyBadwordAdvantageConfig,
)
from intern_s1_delivery.dataset.filter_func import group_sample_filter_func

TOKENIZE_FN_CONFIGS = {
    "search_train": RLSearchTokenizeFnConfig,
    "science_search_train": ScienceSearchTokenizeFnConfig,
    "search_eval": SearchEvalTokenizeFnConfig,
    "math_code_interpreter": MathTrainTokenizeFnConfig,
    "math_ci_eval": MathEvalTokenizeFnConfig,
    "paper_review": RLReviewTokenizeFnConfig,
    "sgi_bench": SGITokenizeFnConfig,
    "frontierscience_bench": FrontierScienceTokenizeFnConfig,
}


OLD_LLMIT1_PREFIX = "/mnt/shared-storage-user/llmit1"
NEW_LLMIT1_PREFIX = "/mnt/shared-storage-gpfs2/llmit1"

DATASET_CONFIG_PATH = Path(os.environ["DATA_PATH"])
with DATASET_CONFIG_PATH.open("r", encoding="utf-8") as f:
    DATASET_CONFIG = json.load(f)
TRAIN_DATASETS = DATASET_CONFIG["train"]
EVAL_DATASETS = DATASET_CONFIG.get("eval", [])


def _iter_dataset_entries(specs, default_recipe):
    entries = ((spec["name"], spec) for spec in specs)
    for name, spec in entries:
        if not spec.get("enabled", True):
            continue
        annotation = spec.get("annotation")
        if not annotation:
            continue
        annotations = annotation if isinstance(annotation, list) else [annotation]
        annotations = [item.replace(OLD_LLMIT1_PREFIX, NEW_LLMIT1_PREFIX) for item in annotations]

        if spec.get("expand_json_meta", False):
            meta_path = Path(annotations[0])
            if not meta_path.exists():
                continue
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            for child_name, child in meta.items():
                if not child.get("enabled", True):
                    continue
                child_annotation = child["annotation"]
                child_annotations = child_annotation if isinstance(child_annotation, list) else [child_annotation]
                yield child_name, {
                    **child,
                    "annotation": [item.replace(OLD_LLMIT1_PREFIX, NEW_LLMIT1_PREFIX) for item in child_annotations],
                    "recipe": spec.get("recipe", default_recipe),
                    "data_source": child.get("data_source") or spec.get("data_source") or child_name,
                    "sample_ratio": child.get("sample_ratio", spec.get("sample_ratio", 1.0)),
                    "media_root": child.get("media_root", spec.get("media_root", "")),
                    "class_name": child.get("class_name", spec.get("class_name", "JsonlDataset")),
                }
            continue

        yield name, {
            **spec,
            "annotation": annotations,
            "recipe": spec.get("recipe", default_recipe),
            "data_source": spec.get("data_source") or name,
            "sample_ratio": spec.get("sample_ratio", 1.0),
        }


def _build_dataset_cfg(specs, default_recipe):
    dataset_cfg = []
    for name, data in _iter_dataset_entries(specs, default_recipe):
        recipe = data.get("recipe") or data.get("tokenize_fn") or default_recipe
        if recipe not in TOKENIZE_FN_CONFIGS:
            raise KeyError(f"Unknown localhost recipe/tokenize_fn {recipe!r}. Choices: {sorted(TOKENIZE_FN_CONFIGS)}")
        annotations = data["annotation"] if isinstance(data["annotation"], list) else [data["annotation"]]
        for annotation in annotations:
            if not Path(annotation).exists():
                continue
            dataset_cfg.append(
                {
                    "dataset": DatasetConfig(
                        name=name,
                        anno_path=annotation,
                        media_root=data.get("media_root", ""),
                        sample_ratio=data.get("sample_ratio", 1.0),
                        class_name=data.get("class_name", "JsonlDataset"),
                    ),
                    "tokenize_fn": TOKENIZE_FN_CONFIGS[recipe](default_data_source=data.get("data_source") or name),
                }
            )
    if not dataset_cfg:
        raise ValueError("No readable localhost dataset annotations were configured.")
    return dataset_cfg


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
model_name = os.environ["MODEL_NAME"]

debug_rollout_dir = os.environ.get("DEBUG_ROLLOUT_DIR", "")
debug_train = os.environ.get("DEBUG_TRAIN", "False") == "True"
debug_rollout = os.environ.get("DEBUG_ROLLOUT", "False") == "True"

experimental_name = os.environ.get("EXPERIMENT_NAME", "localhost_agent_rl")
total_epochs = 10
total_train_steps = int(os.environ["TOTAL_TRAIN_STEPS"]) if os.environ.get("TOTAL_TRAIN_STEPS") else None
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", 128))
prompt_repeat_k = int(os.environ.get("PROMPT_REPEAT_K", 16))
max_concurrent_samples = int(os.environ.get("MAX_CONCURRENT_SAMPLES", min(global_batch_size, 64)))
eval_max_concurrent_samples = int(os.environ.get("EVAL_MAX_CONCURRENT_SAMPLES", min(global_batch_size, 64)))
localhost_agent_sample_timeout_s = float(os.environ["LOCALHOST_AGENT_SAMPLE_TIMEOUT_S"]) if os.environ.get("LOCALHOST_AGENT_SAMPLE_TIMEOUT_S") else None
max_prompt_length = 4 * 1024
max_response_length = 64 * 1024
pack_max_length = 68 * 1024
train_optimizer_steps = 8
hf_interval = 10
train_sp_size = 1
train_ep_size = 8
rollout_tp_size = 1
rollout_ep_size = 8
enable_float8_rollout = True
enable_return_routed_experts = True
fp32_lm_head = True
skip_load_weights = True
enable_initial_evaluate = os.environ.get("ENABLE_INITIAL_EVALUATE", "False") == "True"
evaluate_step = 5
rollout_max_batch_size_per_instance = 32 * rollout_ep_size

rollout_extra_config = dict(
    lmdeploy_log_level="INFO",
    lmdeploy_uvicorn_log_level="INFO",
    lmdeploy_tool_call_parser="qwen3coder",
    lmdeploy_reasoning_parser="default",
)

resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=int(os.environ.get("NUM_WORKERS", 8)),
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,
)

rollout_config = RolloutConfig(
    model_name=model_name,
    fp32_lm_head=fp32_lm_head,
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    gpus_per_node=8,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    dist_port_base=25000,
    gpu_memory_utilization=0.62,
    enable_float8=enable_float8_rollout,
    skip_load_weights=skip_load_weights,
    context_length=max_response_length,
    enable_return_routed_experts=enable_return_routed_experts,
    rollout_max_batch_size_per_instance=rollout_max_batch_size_per_instance,
    allow_over_concurrency_ratio=1.0,
    rollout_timeout=36000,
    session_server_timeout=36000,
    health_check_interval_seconds=300.0,
    health_check_failure_threshold=3,
    extra_rollout_config=rollout_extra_config,
)

training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=0.999,
    temperature=1.0,
    min_tokens=0,
)
evaluation_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=0.999,
    temperature=0.8,
    min_tokens=0,
)

train_dataset_cfg = _build_dataset_cfg(
    TRAIN_DATASETS,
    "search_train",
)
dataloader_cfg = DataloaderConfig(
    dataset_config_list=train_dataset_cfg,
    num_workers=8,
    collator="fake_collator",
    pack_level="none",
    pack_max_length=pack_max_length,
)

enable_evaluate = os.environ.get("ENABLE_EVALUATE", "True") == "True"
if enable_evaluate:
    eval_dataset_cfg = _build_dataset_cfg(
        EVAL_DATASETS,
        "search_eval",
    )
    eval_dataloader_cfg = DataloaderConfig(
        dataset_config_list=eval_dataset_cfg,
        num_workers=8,
        collator="fake_collator",
        pack_level="none",
        pack_max_length=pack_max_length,
    )
else:
    eval_agent_loop_manager_cfg = None

float8_cfg = Float8Config(
    scaling_granularity_gemm=None,
    scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
)
model_cfg = Qwen3_5_VLMoE397BA17SplitConfig(freeze_vision=True, freeze_projector=True)
model_cfg.float8_cfg = float8_cfg
model_cfg.text_config.ep_size = train_ep_size
model_cfg.text_config.z_loss_cfg = None
model_cfg.text_config.balancing_loss_cfg = None
model_cfg.text_config.freeze_routers = True
model_cfg.compile_cfg = None
model_cfg.text_config.mtp_config = MTPConfig(
    num_layers=4,
    loss_scaling_factor=1.0,
    detach_mtp_lm_head_weight=True,
    detach_mtp_inputs=True,
    share_weights=True,
)
model_cfg.text_config.vocab_size = 251392
optim_cfg = AdamWConfig(
    lr=1e-6,
    betas=(0.9, 0.95),
    max_grad_norm=1.0,
    weight_decay=0.1,
    foreach=False,
    skip_grad_norm_threshold=5,
    eps=1e-15,
)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.28,
        cliprange_low=0.2,
        loss_type="intern_s1_delivery.modules.pg_loss.pg_loss_fn",
        clip_ratio_c=3.0,
        log_prob_diff_min=-20.0,
        log_prob_diff_max=20.0,
    ),
    ignore_idx=-100,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    mode="chunk",
    chunk_size=512,
    rollout_is=KpopRolloutImportanceSampling(
        rollout_is_level="token",
        rollout_is_mode="mask",
        rollout_is_threshold=(5.0, 0.5),
        rollout_is_mask_type="binary_kpop",
        rollout_is_kpop_delta=0.9,
        rollout_is_kpop_use_ratio_weight=True,
    ),
)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=train_ep_size, fp32_lm_head=fp32_lm_head)
train_worker_cfg = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=train_sp_size,
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

agent_loop_config = AgentInLocalhostLoopConfig(
    hf_checkpoint=model_path,
    sample_params=training_sample_params,
    max_concurrent_samples=max_concurrent_samples,
    sample_timeout_s=localhost_agent_sample_timeout_s,
)
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=agent_loop_config,
        produce_strategy_config=AsyncProduceStrategyConfig(
            over_sample_threshold=0.0,
            is_valid_sample_fn=group_sample_filter_func,
        ),
        sampler_config=SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=prompt_repeat_k),
    ),
)

if enable_evaluate:
    eval_agent_loop_config = AgentInLocalhostLoopConfig(
        hf_checkpoint=model_path,
        sample_params=evaluation_sample_params,
        max_concurrent_samples=eval_max_concurrent_samples,
        sample_timeout_s=localhost_agent_sample_timeout_s,
        mode="eval",
    )
    eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
        tasks=TaskSpecConfig(
            task_name="eval_task",
            agent_loop_config=eval_agent_loop_config,
            produce_strategy_config=AsyncProduceStrategyConfig(),
            sampler_config=SamplerConfig(dataloader_cfg=eval_dataloader_cfg, prompt_repeat_k=1),
        ),
    )

trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=SyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=EvaluatorConfig(compute_metric_func=None),
    load_from=model_path,
    total_train_steps=total_train_steps,
    total_epochs=total_epochs if total_train_steps is None else None,
    train_batch_size=global_batch_size,
    advantage_estimator_config=OverlongRLOOGroupEntropyBadwordAdvantageConfig(
        entropy_upper_bound=0.65,
        entropy_lower_bound=0.25,
        tau_upper=0.0,
        tau_lower=0.0,
        coeff_min_upper=0.2,
        coeff_min_lower=0.5,
        overlong_filer=True,
        badword_ratio_cost_factor=1.0,
        tokenizer_path=model_path,
        enable_length_reweight=True,
    ),
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=enable_initial_evaluate,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    auto_resume=True,
    load_checkpoint_cfg=LoadCheckpointConfig(load_optimizer_states=False, load_optimizer_args=False),
    checkpoint_interval=2,
    checkpoint_maxkeep=1,
    checkpoint_no_save_optimizer=True,
    checkpoint_no_save_replay_buffer=True,
    skip_checkpoint_validation=True,
    hf_interval=hf_interval,
    debug_rollout_dir=debug_rollout_dir,
    debug_train=debug_train,
    debug_rollout=debug_rollout,
)
