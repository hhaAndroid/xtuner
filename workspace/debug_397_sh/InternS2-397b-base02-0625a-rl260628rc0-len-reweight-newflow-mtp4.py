import os
from copy import deepcopy

from xtuner.v1.config import (
    AdamWConfig,
    # MuonConfig,
    FSDPConfig,
    LRConfig,
)
import json
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
# from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5_VLMoE397BA17SplitConfig
from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5_VLMoE397BA17SplitConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    TaskSpecConfig,
    SamplerConfig,
    ProgressiveProduceStrategyConfig,
    GroupPolicyConfig,
    TrajectorySchedulerConfig,
)
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig
from xtuner.v1.rl.trainer import WorkerConfig
# from xtuner.v1.rl.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
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
    SafeWorkSafetyJudgerConfig,
    SafeWorkValueJudgerConfig,
    SafeWorkDoudiJudgerConfig,
    GPTOssRubricJudgerConfig,
    ScidocLayoutJudgerConfig,
    MP20JudgerConfig,
    NemoSwePivotJudgerConfig,
)
from intern_s1_delivery.dataset.filter_func import group_sample_filter_func, failed_sample_filter_func
from intern_s1_delivery.evaluate.compute_metric import compute_metric
from intern_s1_delivery.advantage.rloo_entropy import (
    OverlongRLOOGroupEntropyAdvantageConfig,
)
from intern_s1_delivery.advantage.rloo_entropy_badword import (
    OverlongRLOOGroupEntropyBadwordAdvantageConfig,
)
from xtuner.v1.float8 import Float8Config, ScalingGranularity
from intern_s1_delivery.modules.kpop_importance_sampling import KpopRolloutImportanceSampling


# async config
world_size = int(os.environ["WORLD_SIZE"])
max_concurrent = 8192
work_dir = os.environ["WORK_DIR"]
enable_partial_rollout = True
tail_batch_candidate_steps = 2
tail_batch_trigger_size = int(os.environ.get("TAIL_BATCH_TRIGGER_SIZE", "256"))
train_optimizer_steps = 8

# model_path = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns2_preview_sft/sft_interns2_pre_base03_20260417b_lr2e5_128gpu/20260417150331/hf-4975"
# model_path = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns2_preview_sft/sft_interns2_pre_base03_20260418a_lr2e5_192gpu/20260419041127/hf-4636"
# model_path = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns2_preview_sft/sft_interns2_pre_base04_20260420a_lr2e5_128gpu/20260420153326/hf-7721"
# model_path = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/interns2/verify/sft/verify_interns2_397b_sft_base05_w_mtp_ep1_sp4_fix_rope_fix_fp8/20260521093800/hf-681"
model_path = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/interns2/sft_official/sft_interns2_base02_20260625a_lr2e5_512gpu/20260625041103/hf-5338"

stop_word = "<|im_end|>"

# basic settings
experimental_name = "dapo_math"

# NOTE: with the progressive producer, global_batch_size is counted in
# *trajectories* (not groups). Under the legacy fixed-k config this value
# meant 128 groups × prompt_repeat_k trajectories; users preserving the old
# trajectory throughput should scale it accordingly (e.g. 128 * 8 = 1024).
global_batch_size = 8192
# Progressive sampling range. The aggregator collects at least min_repeat
# trajectories per prompt, emits READY when reward variance appears, and
# continues up to max_repeat before falling back to STOPPED.
min_repeat = 8
max_repeat = 16

max_prompt_length = 56 * 1024
pack_max_length = 68 * 1024
max_response_length = 62 * 1024

# global_batch_size = 32
# prompt_repeat_k = 8
# rollout_max_batch_size_per_instance = 128
# group_sample_filter_func = failed_sample_filter_func  # for debug
# max_prompt_length = 8192
# pack_max_length = 16 * 1024
# max_response_length = 8 * 1024

train_ep_size = 4
train_sp_size = 1
rollout_tp_size = 1
rollout_ep_size = 8
enable_float8_rollout = True
enable_return_routed_experts = True
fp32_lm_head = True
rollout_max_batch_size_per_instance = 32 * rollout_ep_size
# rollout_max_batch_size_per_instance = 8 * rollout_ep_size # for debug
# max_concurrent = 256 # for debug


lr = 1e-6
# train_optimizer_steps = 8  # mini batch steps
hf_interval = 10
total_epochs = 100

# evaluation settings
enable_evaluate = True
enable_initial_evaluate = False
evaluate_step = 10

# dataset settings
train_datasets = "/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_xtuner_rl_design/crg_rl_projects/src/intern_s1_delivery/configs/interns2_397b/data_config/train_interns2_260628rc0.json"
eval_datasets = "/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_xtuner_rl_design/crg_rl_projects/src/intern_s1_delivery/configs/interns2_397b/data_config/val_v4.json"

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8,
    num_cpus_per_worker=12,
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
    gpu_memory_utilization=0.5,  # 0.8
    enable_float8=enable_float8_rollout,
    skip_load_weights=True,
    context_length=1024*65,
    # chunked_prefill_size=4096,
    # Slightly lower peak concurrent /generate pressure vs GPU KV budget.
    allow_over_concurrency_ratio=1.0,  # 1.2
    rollout_timeout=36000,
    rollout_max_batch_size_per_instance=rollout_max_batch_size_per_instance,
    enable_return_routed_experts=enable_return_routed_experts,
    extra_rollout_config=dict(
        lmdeploy_log_level="ERROR", 
        lmdeploy_uvicorn_log_level="ERROR",
        lmdeploy_speculative_algorithm='qwen3_5_mtp',
        # MTP draft tokens trade throughput for extra activation memory; try 3 if still tight.
        lmdeploy_speculative_num_draft_tokens=3,
    ),
    fp32_lm_head=fp32_lm_head,
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
)
evaluation_sample_params = deepcopy(training_sample_params)
evaluation_sample_params.temperature = 0.8
evaluation_sample_params.max_tokens = 64 * 1024

# dataset
data_judger_mapping = {
    "math": "compass_verifier_v2",
    "math_vl": "compass_verifier_v2",
    "mmpr": "compass_verifier_v2",
    # "intern_bootcamp": "intern_bootcamp",
    "cif_v3": "cif",
    "nano_v3_sft_profiled_instruction_following": "nemo_if",
    "nano_v3_sft_profiled_stem_mcqa": "compass_verifier_v2",
    "physics": "hipho_judger",
    "cmphysics": "cmphysics_judger",
    "judge": "judge_data_judger",
    "rapidapi": "rapidapi",
    "rl_rmp_prompt": "sglang_reward_service_rmp_cot",
    "MathVerse_MINIVOnly": "compass_verifier_v2",
    "MathVista_MINI": "compass_verifier_v2",
    "MMMU_DEV_VAL": "compass_verifier_v2",
    "MathVision": "compass_verifier_v2",
    "GPQA_diamond": "compass_verifier_v2",
    "aime2026": "compass_verifier_v2",
    "hmmt26": "compass_verifier_v2",
    "UGD_hard": "compass_verifier_v2",
    "DynaMath": "compass_verifier_v2",
    "MMMU_Pro": "compass_verifier_v2",
    "IMO_Answer_Bench": "compass_verifier_v2",
    "chemistry_mol": "chemistry_judger",
    "material_science": "material_judger",
    "biology_science": "biology_judger",
    "nemotron_super_comp_coding": "nemo_code_gen",
    "nemotron_super_tau_pivot": "nemo_single_step_tool_use",
    "knowledge_easy_verify": "compass_verifier_v2",
    "nemotron_super_structured_outputs": "nemo_structure",
    "helpsteer_3": "nemo_genrm_compare",
    "interns2_identity": "nemo_genrm_compare",
    "safework_safety_en": "safework_safety",
    "safework_safety_cn": "safework_safety",
    "safework_value_en": "safework_value",
    "safework_value_cn": "safework_value",
    "WXB": "safework_doudi",
    "rl_rubric_prompt": "gpt_oss_rubric",
    "rl_rubricbench_prompt": "gpt_oss_rubric",
    "rl_openrubric_v2_prompt": "gpt_oss_rubric",
    "rl_openrubric_science_prompt": "gpt_oss_rubric",
    "scidoc_layout_verifier_v1": "scidoc_layout_verifier_v1",
    "mp20": "mp20_judger",
    "nemotron_swe_pivot": "nemo_swe_pivot",
    "hle_enhance": "compass_verifier_v2",
}
tokenize_fn_cfg = RLQwen3VLTokenizeFnConfig(
    processor_path=model_path,
    max_length=max_prompt_length,
    chat_template="qwen3.5-vl",
    add_generation_prompt=True,
    enable_thinking=True,
    data_judger_mapping=data_judger_mapping,
    random_system_prompt="你是上海人工智能实验室开发的对话式AI模型\"Intern-S2\"。\n- 对于日常问候、闲聊、事实性问题或可以直接回答的问题，直接给出简洁答案。\n- 仅当任务本身客观需要多步逻辑推导（如复杂数学、证明、严谨推演）时，才进行详细推理。\n- 不要假设用户有未说明的格式要求或隐藏指令。\n- 默认以自然、直接的方式回应用户输入。",
    random_system_prompt_prob=0.25,
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
    collator="fake_collator",
    pack_level="none",
)

# 3. judger
judger_cfg = {
    "compass_verifier_v2": CompassVerifierV2Config(
        judger_name="compass_verifier_v2",
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
        judger_name="nemo_if",
        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
    ),
    "judge_data_judger": JudgeDataJudgerConfig(
        judger_name="judge_data_judger",
        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
    ),
    "chemistry_judger": ChemistryJudgerConfig(
        judger_name="chemistry_judger",
        hosts=[
            "http://10.102.138.54:30030/v1",
            "http://10.102.138.52:30030/v1",
            "http://10.102.138.54:30031/v1",
            "http://10.102.138.52:30031/v1",
            "http://10.102.138.54:30032/v1",
            "http://10.102.138.52:30032/v1",
            "http://10.102.138.54:30033/v1",
            "http://10.102.138.52:30033/v1",
            "http://10.103.4.52:30030/v1",
            "http://10.103.4.52:30031/v1",
            "http://10.103.4.52:30032/v1", 
            "http://10.103.4.52:30033/v1",
        ],
        model_name="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e",
        cpu_resources=CPUResourcesConfig(num_workers=4, num_cpus_per_worker=1),
    ),
    "material_judger": MaterialJudgerConfig(
        judger_name="material_judger",
        hosts=[
            "http://10.102.138.54:30030/v1",
            "http://10.102.138.52:30030/v1",
            "http://10.102.138.54:30031/v1",
            "http://10.102.138.52:30031/v1",
            "http://10.102.138.54:30032/v1",
            "http://10.102.138.52:30032/v1",
            "http://10.102.138.54:30033/v1",
            "http://10.102.138.52:30033/v1",
            "http://10.103.4.52:30030/v1",
            "http://10.103.4.52:30031/v1",
            "http://10.103.4.52:30032/v1", 
            "http://10.103.4.52:30033/v1",
        ],
        model_name="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e",
        cpu_resources=CPUResourcesConfig(num_workers=4, num_cpus_per_worker=1),
    ),
    "biology_judger": BiologyJudgerConfig(
        judger_name="biology_judger",
        cpu_resources=CPUResourcesConfig(num_workers=4, num_cpus_per_worker=1),
    ),
    "nemo_code_gen": NemoCodeGenJudgerConfig(
        judger_name="nemo_code_gen",
        cpu_resources=CPUResourcesConfig(num_workers=8, num_cpus_per_worker=4, cpu_memory_per_worker=16 * 1024 ** 3),
    ),
    "nemo_single_step_tool_use": NemoSingleStepToolUseJudgerConfig(
        judger_name="nemo_single_step_tool_use",
        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
    ),
    "nemo_structure": NemoStructureJudgerConfig(
        judger_name="nemo_structure",
        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
    ),
    "nemo_genrm_compare": NemoGenRMCompareJudgerConfig(
        judger_name="nemo_genrm_compare",
        cpu_resources=CPUResourcesConfig(num_workers=4, num_cpus_per_worker=1),
        genrm_hosts=[
            "10.103.26.53:5000",
            "10.103.11.48:5000",
            "10.103.28.36:5000"
        ],
        genrm_model_name="nvidia/Qwen3-Nemotron-235B-A22B-GenRM-2603",
    ),
    "safework_safety": SafeWorkSafetyJudgerConfig(            
        hosts=[                
            "10.102.147.3:23330",
            "10.102.147.3:23331",
        ],
        key_hosts=[                
            "10.102.147.3:23340",
            "10.102.147.3:23341",      
        ],
        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
    ),        
    "safework_value": SafeWorkValueJudgerConfig(            
        hosts=[                
            "10.103.22.24:23332",                
            "10.103.22.24:23333",                
            "10.103.22.24:23334",                
            "10.103.22.24:23335",                
            "10.103.22.24:23336",                
            "10.103.22.24:23337",                
            "10.103.22.24:23338",                
            "10.103.22.24:23339",            
        ],            
        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),         
    ),
    "safework_doudi": SafeWorkDoudiJudgerConfig(            
        hosts=[                
            "10.103.4.52:30030",
            "10.103.4.52:30031",
            "10.103.4.52:30032", 
            "10.103.4.52:30033",
            "10.102.138.54:30030",
            "10.102.138.52:30030",
            "10.102.138.54:30031",
            "10.102.138.52:30031",
            "10.102.138.54:30032",
            "10.102.138.52:30032",
            "10.102.138.54:30033",
            "10.102.138.52:30033",
        ],            
        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),      
    ),
    "gpt_oss_rubric": GPTOssRubricJudgerConfig(
        hosts=[
            "10.103.4.52:30030",
            "10.103.4.52:30031",
            "10.103.4.52:30032", 
            "10.103.4.52:30033",
            "10.102.138.54:30030",
            "10.102.138.52:30030",
            "10.102.138.54:30031",
            "10.102.138.52:30031",
            "10.102.138.54:30032",
            "10.102.138.52:30032",
            "10.102.138.54:30033",
            "10.102.138.52:30033",
        ],
        model_name="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e",
        request_timeout=180.0,
        max_tokens=4096,
        cpu_resources=CPUResourcesConfig(num_workers=2, num_cpus_per_worker=1),   
    ),
    "scidoc_layout_verifier_v1": ScidocLayoutJudgerConfig(
        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),  
    ),
    "mp20_judger": MP20JudgerConfig(
        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),  
    ),
    "nemo_swe_pivot": NemoSwePivotJudgerConfig(
        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),  
    ),
}

# 4. sampler, produce strategy, and evaluator
train_sampler_config = SamplerConfig(
    dataloader_cfg=dataloader_config,
)

produce_strategy_config = ProgressiveProduceStrategyConfig(
    group_policy=GroupPolicyConfig(
        min_repeat=min_repeat,
        max_repeat=max_repeat,
        stop_when_all_equal=True,
    ),
    scheduler=TrajectorySchedulerConfig(
        max_on_fly=max_concurrent,
        enable_partial_rollout=bool(enable_partial_rollout),
        max_staleness=3,
        tail_batch_trigger_size=tail_batch_trigger_size,
    ),
    # over_sample_threshold=1.0,
    is_valid_sample_fn=group_sample_filter_func,
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
    # single_epoch=True 让 eval sampler 在 val 数据集耗尽时抛 SamplerExhausted，
    # producer 据此停止追加 prompt。否则 producer 会按 max_on_fly 满载预取，
    # sampler 自动 wrap 到下一个 epoch，导致一次 evaluation 发出去的 trajectory
    # 数远大于 val 数据集大小（max_on_fly / dataset_size 倍）。
    single_epoch=True,
)

# Eval 走单条采样 (k=1), 但仍需要打满 max_on_fly 才能追上 rollout 引擎的并发预算。
# SyncProduceStrategyConfig 的 max_on_fly 是写死的 prompt_repeat_k * 64 = 64，所以这里
# 改用 ProgressiveProduceStrategyConfig + min=max=1 暴露 max_on_fly。
eval_produce_strategy_config = ProgressiveProduceStrategyConfig(
    group_policy=GroupPolicyConfig(
        min_repeat=1,
        max_repeat=1,
        stop_when_all_equal=False,
    ),
    scheduler=TrajectorySchedulerConfig(
        max_on_fly=max_concurrent,
        enable_partial_rollout=False,
        max_staleness=3,
    ),
)

# # 5. Train worker
float8_cfg = Float8Config(
    scaling_granularity_gemm=None,
    scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
)
model_cfg = Qwen3_5_VLMoE397BA17SplitConfig(
    freeze_vision=True,
    freeze_projector=True,
)
model_cfg.float8_cfg = float8_cfg
# model_cfg.float8_cfg = None
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
# model_cfg.text_config.dispatcher = "deepep"
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
# optim_cfg = MuonConfig(
#     lr=lr,
#     betas=(0.9, 0.95),
#     max_grad_norm=1.0,
#     weight_decay=0.1,
#     # foreach=False,
#     skip_grad_norm_threshold=5,
#     eps=1e-15,
# )
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
        # rollout_is_veto_threshold=(100, 0),
        rollout_is_mask_type="binary_kpop",
        rollout_is_kpop_delta=0.9,
        rollout_is_kpop_use_ratio_weight=True,
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
)
eval_agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=str(model_path),
    sample_params=evaluation_sample_params,
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
        produce_strategy_config=eval_produce_strategy_config,
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
    work_dir=work_dir,
    advantage_estimator_config=OverlongRLOOGroupEntropyBadwordAdvantageConfig(
        entropy_upper_bound=0.75,
        entropy_lower_bound=0.25,
        tau_upper=0.0,
        tau_lower=0.0,
        coeff_min_upper=0.2,
        coeff_min_lower=0.5,
        overlong_filer=True,
        badword_ratio_cost_factor=0.7,
        tokenizer_path=model_path,
        enable_length_reweight=True,
        positive_ratio_threshold=0.5,
        length_reweight_alpha=0.7,
        length_reweight_gamma=1.0,
    ),
)
