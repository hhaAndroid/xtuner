import os
from copy import deepcopy

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.module.rope.rope import RopeScalingConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.model import Qwen3VLMoE30BA3Config
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.ray.base import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.evaluator import EvaluatorConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.base.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.grpo import GRPOLossConfig
from xtuner.v1.train.rl_trainer import RLTrainerConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig
from xtuner.v1.float8 import Float8Config, ScalingGranularity

from intern_s1_delivery.dataset.xpuyu_dataset_vl import parse_xpuyu_json_cfg
from intern_s1_delivery.judgers import (
    CompassVerifierV2Config,
    BootcampJudgerConfig,
    CIFJudgerConfig,
    SGLangRewardServiceRMPCotConfig,
)
from intern_s1_delivery.dataset.filter_func import group_sample_filter_func, failed_sample_filter_func
from intern_s1_delivery.evaluate.compute_metric import compute_metric


model_path = (
    "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns1_1/interns1_1_g8_30b_a3b_cpt_tiny_bs512_epoch1_lr1e-5_max2k_retry1/20251217070648/hf-740/"
)
stop_word = "<|im_end|>"

# basic settings
experimental_name = "dapo_math"

global_batch_size = 256
prompt_repeat_k = 16
max_concurrent_groups = 512
# global_batch_size = 64
# prompt_repeat_k = 8
# max_concurrent_groups = 128

max_prompt_length = 4096
pack_max_length = 36 * 1024
max_response_length = 32 * 1024
# pack_max_length = 8 * 1024
# max_response_length = 4 * 1024

train_ep_size=1
rollout_tp_size = 4
rollout_ep_size = 1
enable_float8_rollout = False
rollout_max_batch_size = 256 * rollout_ep_size
max_prefill_token_num = 1024
enable_return_routed_experts = True
# enable_return_routed_experts = False
enable_partial_rollout = True

lr = 1e-6
train_optimizer_steps = 8  # mini batch steps
hf_interval = 20
total_epochs = 100

# evaluation settings
enable_evaluate = True
enable_initial_evaluate = False
evaluate_step = 5

# dataset settings
train_datasets = "/mnt/shared-storage-user/llmit/user/lvchengqi/projects/moe_rl/xtuner_v1_projects/src/intern_s1_delivery/configs/data_configs/math_text_train_06-1_xtuner_format.json"
# train_datasets = "/mnt/shared-storage-user/llmit/user/lvchengqi/projects/moe_rl/xtuner_v1_projects/src/intern_s1_delivery/configs/data_configs/mmpr250701_xtuner_format.json"
eval_datasets = "/mnt/shared-storage-user/llmit/user/lvchengqi/projects/moe_rl/xtuner_v1_projects/src/intern_s1_delivery/configs/data_configs/math_text_val_xtuner_format.json"


# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,  # 16 GB
)
judger_cpu_resources = CPUResourcesConfig.from_total(
    total_cpus=16,
    num_workers=16,
    total_memory=64 * 1024**3
)

# 2. rollout
rollout_config = RolloutConfig(
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=0.7,
    enable_float8=enable_float8_rollout,
    skip_load_weights=True,
    context_length=65536,
    rollout_max_batch_size_per_instance=rollout_max_batch_size,
    allow_over_concurrency_ratio=2,
    rollout_timeout=36000,
    enable_return_routed_experts=enable_return_routed_experts,
    router_n_groups=4,
    max_prefill_token_num=max_prefill_token_num,
    extra_rollout_config=dict(lmdeploy_log_level="ERROR", lmdeploy_uvicorn_log_level="ERROR"),
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=0.999,
    temperature=1.0,
    min_tokens=0,
)
evaluation_sample_params = deepcopy(training_sample_params)
evaluation_sample_params.temperature = 0.8
evaluation_sample_params.max_tokens = 62 * 1024

# dataset
data_judger_mapping = dict(
    math={"compass_verifier_v2": 1.0},
    # intern_bootcamp={"intern_bootcamp": 1.0},
    # cif_v3={"cif": 1.0},
    # rl_rmp_prompt={"sglang_reward_service_rmp_cot": 1.0},
    MathVerse_MINIVOnly={"compass_verifier_v2": 1.0},
    MathVista_MINI={"compass_verifier_v2": 1.0},
    MMMU_DEV_VAL={"compass_verifier_v2": 1.0},
    MathVision={"compass_verifier_v2": 1.0},
    GPQA_diamond={"compass_verifier_v2": 1.0},
    aime2024={"compass_verifier_v2": 1.0},
    aime2025={"compass_verifier_v2": 1.0},
    AIME2024={"compass_verifier_v2": 1.0},
    AIME2025={"compass_verifier_v2": 1.0},
)
tokenize_fn_cfg = Qwen3VLTokenizeFnConfig(
    max_length=pack_max_length,
    processor_path=model_path,
    min_pixels=None,
    # max_pixels=None,
    max_pixels=2097152, # ----------------------------------------------------------------
    video_min_total_pixels=None,
    video_max_total_pixels=None,
    video_min_frames=None,
    video_max_frames=None,
    fps=None,
    rand_video_max_frames=24,
    add_vision_id=True,
    system_message=None,
    hash=None,
    enable_3d_rope=False,
    oss_loader_cfg=None,
    debug=True,
    oss_time_log_thr=10
)
train_dataset_cfg = parse_xpuyu_json_cfg(train_datasets, tokenize_fn_cfg, max_prompt_length, data_judger_mapping)
eval_dataset_cfg = (
    parse_xpuyu_json_cfg(eval_datasets, tokenize_fn_cfg, max_prompt_length, data_judger_mapping, ignore_multimodal_info=True) if enable_evaluate else []
)

dataloader_config = DataloaderConfig(pack_max_length=pack_max_length, collator="fake_collator", pack_level="none")

# 3. judger
judger_cfg = JudgerConfig(
    enable_weighted_judgers=True,
    reward_judger_configs=[
        CompassVerifierV2Config(
            hosts=[
                "10.103.12.31:12345",
                "10.103.12.31:12346",
                "10.103.12.31:12347",
                "10.103.12.31:12348",
                "10.103.12.31:12349",
                "10.103.12.31:12350",
                "10.103.12.31:12351",
                "10.103.12.31:12352",
            ]
        ),
        # BootcampJudgerConfig(),
        # CIFJudgerConfig(
        #     hosts=["0.0.0.0:8080"],
        #     stop_word=stop_word,
        #     thinking_finish_words=["<conclude>", "**Final Answer**", "</think>"],
        # ),
        # SGLangRewardServiceRMPCotConfig(
        #     hosts=["100.97.184.163:30030"],
        #     tokenizer_path="/mnt/shared-storage-user/songdemin/user/lishuaibin/mv2yidian/RM_SFT_reward_pt_7b_final_DATA_HH_puyu_mixed_Node_2_LR_2e-5_STEP_905_hf",
        # ),
    ],
)

# 4. dataflow and evaluator
dataflow_config = DataFlowConfig(
    env=experimental_name,
    max_concurrent=max_concurrent_groups,
    enable_partial_rollout=enable_partial_rollout,
    max_retry_times=3,
    prompt_repeat_k=prompt_repeat_k,
    global_batch_size=global_batch_size,
    sample_params=training_sample_params,
)

evaluator_cfg = (
    EvaluatorConfig(
        enable_evaluate=enable_evaluate,
        enable_initial_evaluate=enable_initial_evaluate,
        dataset_cfg=eval_dataset_cfg,
        tokenizer=model_path,
        evaluate_step=evaluate_step,
        compute_metric_func=compute_metric,
        sample_params=evaluation_sample_params,
        max_concurrent=8192,
    )
    if enable_evaluate
    else None
)

replay_buffer_cfg = ReplayBufferConfig(
    dataset_cfg=train_dataset_cfg,
    dataloader_cfg=dataloader_config,
    tokenizer=model_path,
    postprocessor_func=group_sample_filter_func,
)

# 5. Train worker
float8_cfg = Float8Config(
    scaling_granularity_gemm=None,
    scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
)

model_cfg = Qwen3VLMoE30BA3Config(
    freeze_vision=True,
    freeze_projector=True,
    text_config=Qwen3MoE30BA3Config(
        freeze_routers=True, 
        balancing_loss_cfg=None, 
        max_position_embeddings=32768, 
        rope_theta=1000000,
        n_routed_experts=512))
model_cfg.vision_config.depth = 24
model_cfg.vision_config.hidden_size = 1024
model_cfg.vision_config.intermediate_size = 4096
model_cfg.vision_config.deepstack_visual_indexes = []

model_cfg.projector_config.vision_hidden_size = 1024
model_cfg.projector_config.deepstack_visual_indexes = []

model_cfg.text_config.rope_scaling_cfg = RopeScalingConfig(
            fope_init_factor=0.1,
            fope_sep_head=True,
            num_inv_freq=None,
            )
model_cfg.text_config.vocab_size = 155008
# model_cfg.text_config.float8_cfg = float8_cfg
model_cfg.text_config.router.use_grouped_router = True
model_cfg.text_config.router.router_n_groups = 4

optim_cfg = AdamWConfig(lr=lr, betas=(0.9, 0.95), max_grad_norm=1.0, weight_decay=0.1, foreach=False, skip_grad_norm_threshold=0.9, eps=1e-15)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.2,
        cliprange_low=0.2,
        loss_type="intern_s1_delivery.modules.pg_loss.pg_loss_fn",
        clip_ratio_c=10.0,
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
        rollout_is_veto_threshold=(20, 0),
    ),
)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=lr)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=train_ep_size)
train_worker_cfg: WorkerConfig = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=1,
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

# 6. RL Trainer
trainer = RLTrainerConfig(
    load_from=model_path,
    resources=resources,
    cpu_resources=judger_cpu_resources,
    rollout_config=rollout_config,
    dataflow_config=dataflow_config,
    judger_config=judger_cfg,
    replay_buffer_config=replay_buffer_cfg,
    evaluator_config=evaluator_cfg,
    train_worker_config=train_worker_cfg,
    tokenizer_path=model_path,
    total_epochs=total_epochs,
    hf_interval=hf_interval,
)
