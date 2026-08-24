import os
from copy import deepcopy

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
import json
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    TaskSpecConfig,
    SamplerConfig,
    AsyncProduceStrategyConfig,
)
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.rl.judger import ComposedJudgerConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from intern_s1_delivery.judger_utils import weighted_merge_fn
from intern_s1_delivery.dataset.xpuyu_dataset_vl import parse_xpuyu_json_cfg
from intern_s1_delivery.judgers import (
    CompassVerifierV2Config,
    NemoIFJudgerConfig,
    JudgeDataJudgerConfig,
    ChemistryJudgerConfig,
    MaterialJudgerConfig,
    BiologyJudgerConfig,
    NemoCodeGenJudgerConfig,
    NemoSingleStepToolUseJudgerConfig,
    NemoStructureJudgerConfig,
    NemoGenRMCompareJudgerConfig,
)
from intern_s1_delivery.dataset.filter_func import group_sample_filter_func
from intern_s1_delivery.evaluate.compute_metric import compute_metric
from intern_s1_delivery.advantage.rloo_entropy import (
    OverlongRLOOGroupEntropyAdvantageConfig,
)
from intern_s1_delivery.advantage.rloo_entropy_badword import (
    OverlongRLOOGroupEntropyBadwordAdvantageConfig,
)


# async config
max_concurrent = int(os.environ.get("MAX_CONCURRENT", "256"))
work_dir = os.environ["WORK_DIR"]
enable_partial_rollout = True
train_optimizer_steps = 8

# model_path = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns2_preview_sft/sft_interns2_pre_base03_20260417b_lr2e5_128gpu/20260417150331/hf-4975"
# model_path = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns2_preview_sft/sft_interns2_pre_base03_20260418a_lr2e5_192gpu/20260419041127/hf-4636"
# model_path = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns2_preview_sft/sft_interns2_pre_base04_20260420a_lr2e5_128gpu/20260420153326/hf-7721"
model_path = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns2_preview_sft/sft_interns2_pre_base05_20260424a_lr2e5_256gpu/20260424143245/hf-4073"

stop_word = "<|im_end|>"

# basic settings
experimental_name = "dapo_math"

global_batch_size = os.environ.get("GLOBAL_BATCH_SIZE", "1024")
prompt_repeat_k = os.environ.get("PROMPT_REPEAT_K", "8")
rollout_max_batch_size_per_instance = 128  # 128

max_prompt_length = 8192
pack_max_length = 64 * 1024
max_response_length = 56 * 1024

# global_batch_size = 32
# prompt_repeat_k = 8
# rollout_max_batch_size_per_instance = 128
# group_sample_filter_func = failed_sample_filter_func
# max_prompt_length = 8192
# pack_max_length = 16 * 1024
# max_response_length = 8 * 1024

train_ep_size = 1
train_sp_size = 1
rollout_tp_size = 1
rollout_ep_size = 1
enable_float8_rollout = False
enable_return_routed_experts = True
fp32_lm_head = True
# enable_partial_rollout = True

lr = 1e-6
# train_optimizer_steps = 8  # mini batch steps
hf_interval = 100
total_epochs = 100

# evaluation settings
enable_evaluate = True
enable_initial_evaluate = False
evaluate_step = 10

# dataset settings
train_datasets = "/mnt/shared-storage-user/llmit/user/lvchengqi/data/intern-s2-xtuner-rl-format/train_interns2_260426rc0.json" # train_interns2_260426rc0
eval_datasets = "/mnt/shared-storage-user/llmit/user/lvchengqi/data/intern-s2-xtuner-rl-format-valset/val_v3.json"
# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=os.environ.get("NUM_WORKERS", "8"),
    num_cpus_per_worker=4, # 12
    cpu_memory_per_worker=16 * 1024**3,  # 16 GB
)

# 2. rollout
rollout_config = RolloutConfig(
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    # LMDeploy PyTorch: this field is passed as PytorchEngineConfig.cache_max_entry_count
    # (fraction of *free* VRAM after weights for KV blocks). Lower if rollout OOMs.
    gpu_memory_utilization=0.52,  # 0.8
    enable_float8=enable_float8_rollout,
    skip_load_weights=True,
    context_length=65536*2,
    # chunked_prefill_size=4096,
    # Slightly lower peak concurrent /generate pressure vs GPU KV budget.
    allow_over_concurrency_ratio=1.0,  # 1.2
    rollout_timeout=36000,
    rollout_max_batch_size_per_instance=rollout_max_batch_size_per_instance,
    enable_return_routed_experts=enable_return_routed_experts,
    extra_rollout_config=dict(
        lmdeploy_log_level="INFO", 
        lmdeploy_uvicorn_log_level="INFO",
        lmdeploy_speculative_algorithm='qwen3_5_mtp',
        # MTP draft tokens trade throughput for extra activation memory; try 3 if still tight.
        lmdeploy_speculative_num_draft_tokens=4,
    ),
    fp32_lm_head=fp32_lm_head,
    health_check_interval_seconds=100,
    health_check_failure_threshold=10,
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
    return_routed_experts=True,
)
evaluation_sample_params = deepcopy(training_sample_params)
evaluation_sample_params.temperature = 0.8
evaluation_sample_params.max_tokens = 64 * 1024
evaluation_sample_params.return_routed_experts = False

# dataset
data_judger_mapping = dict(
    math={"compass_verifier_v2": 1.0},
    math_vl={"compass_verifier_v2": 1.0},
    mmpr={"compass_verifier_v2": 1.0},
    # intern_bootcamp={"intern_bootcamp": 1.0},
    cif_v3={"cif": 1.0},
    nano_v3_sft_profiled_instruction_following={"nemo_if": 1.0},
    nano_v3_sft_profiled_stem_mcqa={"compass_verifier_v2": 1.0},
    physics={"hipho_judger": 1.0},
    cmphysics={"cmphysics_judger": 1.0},
    judge={"judge_data_judger": 1.0},
    rapidapi={"rapidapi": 1.0},
    rl_rmp_prompt={"sglang_reward_service_rmp_cot": 1.0},
    MathVerse_MINIVOnly={"compass_verifier_v2": 1.0},
    MathVista_MINI={"compass_verifier_v2": 1.0},
    MMMU_DEV_VAL={"compass_verifier_v2": 1.0},
    MathVision={"compass_verifier_v2": 1.0},
    GPQA_diamond={"compass_verifier_v2": 1.0},
    aime2026={"compass_verifier_v2": 1.0},
    hmmt26={"compass_verifier_v2": 1.0},
    UGD_hard={"compass_verifier_v2": 1.0},
    DynaMath={"compass_verifier_v2": 1.0},
    MMMU_Pro={"compass_verifier_v2": 1.0},
    IMO_Answer_Bench={"compass_verifier_v2": 1.0},
    chemistry_mol={"chemistry_judger": 1.0},
    material_science={"material_judger": 1.0},
    biology_science={"biology_judger": 1.0},
    nemotron_super_comp_coding={"nemo_code_gen": 1.0},
    nemotron_super_tau_pivot={"nemo_single_step_tool_use": 1.0},
    knowledge_easy_verify={"compass_verifier_v2": 1.0},
    nemotron_super_structured_outputs={"nemo_structure": 1.0},
    helpsteer_3={"nemo_genrm_compare": 1.0},
)
tokenize_fn_cfg = RLQwen3VLTokenizeFnConfig(
    processor_path=model_path,
    max_length=max_prompt_length,
    chat_template="qwen3.5-vl",
    add_generation_prompt=True,
    enable_thinking=True,
    data_judger_mapping=data_judger_mapping,
)
eval_tokenize_fn_cfg = RLQwen3VLTokenizeFnConfig(
    processor_path=model_path,
    max_length=max_prompt_length,
    chat_template="qwen3.5-vl",
    add_generation_prompt=True,
    enable_thinking=True,
    data_judger_mapping=data_judger_mapping,
    ignore_multimodal_info=True
)
train_dataset_cfg = []
eval_dataset_cfg = []

def _as_list(value):
    return value if isinstance(value, list) else [value]

with open(train_datasets, "r", encoding="utf-8") as f:
    train_ds_collections = json.load(f)

for name, data in train_ds_collections.items():
    annotations = _as_list(data["annotation"])
    for annotation in annotations:
        train_dataset_cfg.append(
            {
                "dataset": DatasetConfig(
                    name=name,
                    anno_path=annotation,
                    media_root=data.get("media_root", ""),
                    sample_ratio=data.get("sample_ratio", 1.0),
                    class_name="VLMJsonlDataset",
                ),
                "tokenize_fn": tokenize_fn_cfg,
            }
        )

with open(eval_datasets, "r", encoding="utf-8") as f:
    eval_ds_collections = json.load(f)

for name, data in eval_ds_collections.items():
    annotations = _as_list(data["annotation"])
    for annotation in annotations:
        eval_dataset_cfg.append(
            {
                "dataset": DatasetConfig(
                    name=name,
                    anno_path=annotation,
                    media_root=data.get("media_root", ""),
                    sample_ratio=data.get("sample_ratio", 1.0),
                    class_name="VLMJsonlDataset",
                ),
                "tokenize_fn": eval_tokenize_fn_cfg,
            }
        )


dataloader_config = DataloaderConfig(
    pack_max_length=pack_max_length,
    dataset_config_list=train_dataset_cfg,
    num_workers=4,
    collator="fake_collator",
    pack_level="none",
)

# 3. judger
judger_cfg = ComposedJudgerConfig(
    merge_fn=weighted_merge_fn,
    branches={
        "compass_verifier_v2": CompassVerifierV2Config(
            hosts=[
                "10.102.139.22:12345",
                "10.102.139.22:12346",
                "10.102.139.22:12347",
                "10.102.139.22:12348",
                "10.102.139.22:12349",
                "10.102.139.22:12350",
                "10.102.139.22:12351",
                "10.102.139.22:12352",
            ],
            cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
        ),
        "nemo_if": NemoIFJudgerConfig(
            cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
        ),
        "judge_data_judger": JudgeDataJudgerConfig(
            cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
        ),
        "chemistry_judger": ChemistryJudgerConfig(
            hosts=[
                "http://10.102.138.54:30030/v1",
                "http://10.102.138.52:30030/v1",
                "http://10.102.138.54:30031/v1",
                "http://10.102.138.52:30031/v1",
                "http://10.102.138.54:30032/v1",
                "http://10.102.138.52:30032/v1",
                "http://10.102.138.54:30033/v1",
                "http://10.102.138.52:30033/v1",
            ],
            model_name="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e",
            cpu_resources=CPUResourcesConfig(num_workers=4, num_cpus_per_worker=1),
        ),
        "material_judger": MaterialJudgerConfig(
            hosts=[
                "http://10.102.138.54:30030/v1",
                "http://10.102.138.52:30030/v1",
                "http://10.102.138.54:30031/v1",
                "http://10.102.138.52:30031/v1",
                "http://10.102.138.54:30032/v1",
                "http://10.102.138.52:30032/v1",
                "http://10.102.138.54:30033/v1",
                "http://10.102.138.52:30033/v1",
            ],
            model_name="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e",
            cpu_resources=CPUResourcesConfig(num_workers=4, num_cpus_per_worker=1),
        ),
        "biology_judger": BiologyJudgerConfig(
            cpu_resources=CPUResourcesConfig(num_workers=4, num_cpus_per_worker=1),
        ),
        "nemo_code_gen": NemoCodeGenJudgerConfig(
            cpu_resources=CPUResourcesConfig(
                num_workers=8,
                num_cpus_per_worker=4,
                cpu_memory_per_worker=16 * 1024**3,
            ),
        ),
        "nemo_single_step_tool_use": NemoSingleStepToolUseJudgerConfig(
            cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
        ),
        "nemo_structure": NemoStructureJudgerConfig(
            cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
        ),
        "nemo_genrm_compare": NemoGenRMCompareJudgerConfig(
            cpu_resources=CPUResourcesConfig(num_workers=4, num_cpus_per_worker=1),
            genrm_hosts=[
                "10.103.26.53:5000",
                "10.103.11.48:5000",
                "10.103.28.36:5000"
            ],
            genrm_model_name="nvidia/Qwen3-Nemotron-235B-A22B-GenRM-2603",
        ),
    },
)

# judger_cfg = ComposedJudgerConfig(
#     select_fn=weighted_select_fn,
#     merge_fn=weighted_merge_fn,
#     branches={
#         "compass_verifier_v2": CompassVerifierV2Config(
#             hosts=[
#                 "10.102.139.22:12345",
#                 "10.102.139.22:12346",
#                 "10.102.139.22:12347",
#                 "10.102.139.22:12348",
#                 "10.102.139.22:12349",
#                 "10.102.139.22:12350",
#                 "10.102.139.22:12351",
#                 "10.102.139.22:12352",
#             ],
#             cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
#         ),
#         "nemo_if": NemoIFJudgerConfig(
#             cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
#         ),
#         "judge_data_judger": JudgeDataJudgerConfig(
#             cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
#         ),
#         "chemistry_judger": ChemistryJudgerConfig(
#             hosts=[
#                 "http://10.102.138.54:30030/v1",
#                 "http://10.102.138.52:30030/v1",
#                 "http://10.102.138.54:30031/v1",
#                 "http://10.102.138.52:30031/v1",
#                 "http://10.102.138.54:30032/v1",
#                 "http://10.102.138.52:30032/v1",
#                 "http://10.102.138.54:30033/v1",
#                 "http://10.102.138.52:30033/v1",
#             ],
#             model_name="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e",
#             cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
#         ),
#         "material_judger": MaterialJudgerConfig(
#             hosts=[
#                 "http://10.102.138.54:30030/v1",
#                 "http://10.102.138.52:30030/v1",
#                 "http://10.102.138.54:30031/v1",
#                 "http://10.102.138.52:30031/v1",
#                 "http://10.102.138.54:30032/v1",
#                 "http://10.102.138.52:30032/v1",
#                 "http://10.102.138.54:30033/v1",
#                 "http://10.102.138.52:30033/v1",
#             ],
#             model_name="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e",
#             cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
#         ),
#         "biology_judger": BiologyJudgerConfig(
#             cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
#         ),
#         "nemo_code_gen": NemoCodeGenJudgerConfig(
#             cpu_resources=CPUResourcesConfig(
#                 num_workers=1,
#                 num_cpus_per_worker=1,
#                 cpu_memory_per_worker=16 * 1024**3,
#             ),
#         ),
#         "nemo_single_step_tool_use": NemoSingleStepToolUseJudgerConfig(
#             cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
#         ),
#         "nemo_structure": NemoStructureJudgerConfig(
#             cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
#         ),
#         "nemo_genrm_compare": NemoGenRMCompareJudgerConfig(
#             cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
#             genrm_hosts=[
#                 "10.103.26.53:5000",
#                 "10.103.11.48:5000",
#                 "10.103.28.36:5000"
#             ],
#             genrm_model_name="nvidia/Qwen3-Nemotron-235B-A22B-GenRM-2603",
#         ),
#     },
# )

# 4. sampler, produce strategy, and evaluator
train_sampler_config = SamplerConfig(
    dataloader_cfg=dataloader_config,
    prompt_repeat_k=prompt_repeat_k,
)

produce_strategy_config = AsyncProduceStrategyConfig(
    over_sample_threshold=1,
    enable_partial_rollout=1,
    is_valid_sample_fn=group_sample_filter_func,
    max_staleness=3,
    # max_retry_times=3,
)

evaluator_cfg = EvaluatorConfig(
    eval_sample_ratio=1,
    compute_metric_func=compute_metric,
)
eval_sampler_config = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=eval_dataset_cfg if enable_evaluate else train_dataset_cfg,
        collator="fake_collator",
        pack_level="none",
        num_workers=0,
    ),
    prompt_repeat_k=1,
)

# # 5. Train worker
model_cfg = Qwen3_5_VLMoE35BA3Config(
    freeze_vision=True,
    freeze_projector=True,
)
model_cfg.float8_cfg = None
model_cfg.text_config.ep_size = 1
model_cfg.text_config.z_loss_cfg = None
model_cfg.text_config.balancing_loss_cfg = None
model_cfg.text_config.freeze_routers = True
model_cfg.text_config.mtp_config = MTPConfig(
    num_layers=4, 
    loss_scaling_factor=1.0,
    detach_mtp_lm_head_weight=True,
    detach_mtp_inputs=True,
    share_weights=True,
)
model_cfg.text_config.vocab_size = 251392
# model_cfg.text_config.embed_grad_max_token_id = 251173

optim_cfg = AdamWConfig(
    lr=lr,
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
        clip_ratio_c=2.0,
        log_prob_diff_min=-20.0,
        log_prob_diff_max=20.0,
    ),
    ignore_idx=-100,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    mode="chunk",
    chunk_size=512,
    rollout_is=RolloutImportanceSampling(
        rollout_is_level="token",
        rollout_is_mode="both",
        rollout_is_threshold=(5, 0),
        rollout_is_mask_threshold=(5, 0.5),
        rollout_is_veto_threshold=(5, 0),
    ),
)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=lr)
fsdp_cfg = FSDPConfig(
    torch_compile=False,
    cpu_offload=False,
    ep_size=train_ep_size,
    fp32_lm_head=fp32_lm_head,
)
train_worker_cfg: WorkerConfig = WorkerConfig(
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

# 6. RL Trainer
train_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=str(model_path),
    sample_params=training_sample_params,
    enable_batch_judge=True,
)
eval_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=str(model_path),
    sample_params=evaluation_sample_params,
    enable_batch_judge=True,
)

train_agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name=experimental_name,
        agent_loop_config=train_agent_loop_config,
        judger_config=judger_cfg,
        produce_strategy_config=produce_strategy_config,
        sampler_config=train_sampler_config,
    )
)

eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name=f"{experimental_name}_eval",
        agent_loop_config=eval_agent_loop_config,
        judger_config=judger_cfg,
        sampler_config=eval_sampler_config,
    )
)

trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=AsyncReplayBufferConfig(),
    agent_loop_manager_cfg=train_agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_cfg,
    load_from=model_path,
    total_epochs=total_epochs,
    train_batch_size=global_batch_size,
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=enable_initial_evaluate,
    evaluate_step=evaluate_step,
    hf_interval=hf_interval,
    hf_max_keep=1,
    work_dir=work_dir,
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
    ),
)
