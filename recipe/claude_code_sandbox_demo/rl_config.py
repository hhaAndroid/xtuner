"""RL Colocate Trainer config for Claude Code black-box RL on SkillsBench.

Usage (run from repo root):

    WORK_DIR=./work_dirs/claudecode_rl \\
    MODEL_PATH=/path/to/Qwen3.5-35B-A3B \\
    DATA_PATH=/path/to/skillsbench_train.jsonl \\
    EVAL_DATA_PATH=/path/to/skillsbench_eval.jsonl \\
    TASKS_DIR=/path/to/skillsbench/tasks \\
    python -m xtuner.v1.train.cli train recipe/claude_code_sandbox_demo/rl_config.py

Required environment variables
--------------------------------
WORK_DIR        Output directory for checkpoints, logs and trajectories.
MODEL_PATH      HuggingFace model path (e.g. Qwen/Qwen3.5-35B-A3B).
DATA_PATH       Training JSONL file in SkillsBench format (see rl_tokenize_fn.py).
EVAL_DATA_PATH  Eval JSONL file.
TASKS_DIR       SkillsBench tasks root directory.  Used only by ``prepare_dataset.py``
                to generate the training JSONL; ``task_dir`` per sample is stored in
                the JSONL and forwarded to the agent loop via ``extra_fields``.

Optional environment variables
--------------------------------
WORLD_SIZE                  Number of nodes (default: 1).
ENV_GATEWAY_URL             env-gateway service URL
                            (default: http://env-gateway.ailab.ailab.ai).
INSTALL_CLAUDE_CODE         Set to "0" when the sandbox image already has
                            claude-code installed (default: "1").
ENABLE_RETURN_ROUTED_EXPERTS  Set to "1" to enable router-replay / return
                            routed experts from the rollout controller (default: "0").
LOSS_TYPE                   GRPO policy loss type (default: vanilla).
LOSS_MODE                   Loss computation mode (default: chunk).
SP_SIZE                     Sequence parallelism size (default: 1).

Dataset JSONL format
---------------------
Each line must be a JSON object:

    {
        "data_source": "skillsbench",
        "prompt": [{"role": "user", "content": "<task instruction>"}],
        "reward_model": {},
        "extra_info": {
            "task_name": "offer-letter-generator",
            "image_tag": "hb_offer-letter-generator",  // optional
            "agent_timeout": 900,                       // optional (seconds)
            "verifier_timeout": 600                     // optional (seconds)
        }
    }
"""

import os
import sys
from pathlib import Path

# Make the recipe directory importable so rl_tokenize_fn / sandbox_agent_loop can be found.
sys.path.insert(0, str(Path(__file__).parent))

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.gateway.config import GatewayConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig

from rl_tokenize_fn import SkillsBenchTokenizeFnConfig
from sandbox_agent_loop import SandboxClaudeCodeAgentLoopConfig

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]

env_gateway_url = os.environ.get("ENV_GATEWAY_URL", "http://env-gateway.ailab.ailab.ai")
install_claude_code = os.environ.get("INSTALL_CLAUDE_CODE", "1") != "0"
enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "0") == "1"
NNODE = int(os.environ.get("WORLD_SIZE", "1"))

# ---------------------------------------------------------------------------
# Basic settings
# ---------------------------------------------------------------------------
experimental_name = "claudecode_skillsbench"
total_train_steps = 100
evaluate_step = 100          # run full eval once at the end
train_optimizer_steps = 1
train_batch_size = 16 * train_optimizer_steps
prompt_repeat_k = 4          # GRPO: 4 independent sandbox executions per prompt
rollout_tp_size = 1
rollout_ep_size = 1
max_prompt_length = 2048     # approximate upper bound for filtering; real lengths
                             # come from gateway trace records at training time
max_response_length = 16 * 1024
pack_max_length = 64 * 1024

# ---------------------------------------------------------------------------
# 1. Resources
# ---------------------------------------------------------------------------
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8 * NNODE,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,  # 16 GB
)

# ---------------------------------------------------------------------------
# 2. Rollout (inference engine + XTuner gateway)
# ---------------------------------------------------------------------------
rollout_config = RolloutConfig(
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=0.8,
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=enable_return_routed_experts,
    tool_call_parser="qwen3p5",
    reasoning_parser="qwen3",
)

# The gateway must be started so Claude Code (running inside sandboxes) can reach
# the XTuner inference engine over the network.
gateway_config = GatewayConfig(auto_start=True)

# ---------------------------------------------------------------------------
# 3. Train worker
# ---------------------------------------------------------------------------
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)

model_cfg = get_model_config_from_hf(Path(model_path))
if hasattr(model_cfg, "balancing_loss_cfg"):
    model_cfg.balancing_loss_cfg = None
if hasattr(model_cfg, "z_loss_cfg"):
    model_cfg.z_loss_cfg = None

optim_cfg = AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg={
        "cliprange_high": 0.28,
        "cliprange_low": 0.2,
        "loss_type": os.environ.get("LOSS_TYPE", "vanilla"),
        "clip_ratio_c": 10.0,
        "log_prob_diff_min": -20.0,
        "log_prob_diff_max": 20.0,
    },
    ignore_idx=-100,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    mode=os.environ.get("LOSS_MODE", "chunk"),
    chunk_size=512,
)
train_worker_cfg = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=int(os.environ.get("SP_SIZE", "1")),
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

# ---------------------------------------------------------------------------
# 4. Tokenize function  (raw JSONL records → RolloutState)
# ---------------------------------------------------------------------------
# Note: max_length here filters prompts that are already too long even before
# the multi-turn context builds up.  The real per-turn prompt_ids used during
# training are taken from the gateway trace records.
tokenizer_cfg = SkillsBenchTokenizeFnConfig(
    max_length=max_prompt_length,
    default_agent_timeout=900,
    default_verifier_timeout=600,
    default_sandbox_ttl=3600,
)

# ---------------------------------------------------------------------------
# 5. Train agent loop manager
# ---------------------------------------------------------------------------
train_dataset = DatasetConfig(name=experimental_name, anno_path=data_path)
train_dataloader_cfg = DataloaderConfig(
    dataset_config_list=[{"dataset": train_dataset, "tokenize_fn": tokenizer_cfg}],
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
train_sampler_cfg = SamplerConfig(
    dataloader_cfg=train_dataloader_cfg,
    prompt_repeat_k=prompt_repeat_k,
)
# SampleParams are required by AgentLoopConfig but not used for LLM sampling inside
# SandboxClaudeCodeAgentLoop (the inference engine is driven by Claude Code via the
# gateway; temperature / top_p are controlled at the rollout-controller level).
train_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
)
train_agent_loop_cfg = SandboxClaudeCodeAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=train_sample_params,
    env_gateway_url=env_gateway_url,
    max_turns=50,
    agent_timeout=900,
    verifier_timeout=600,
    sandbox_ttl=3600,
    install_claude_code=install_claude_code,
    permission_mode="bypassPermissions",
    max_concurrent_sandboxes=16,
)
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=train_agent_loop_cfg,
        # judger_config=None: reward is computed inside SandboxClaudeCodeAgentLoop
        # from the sandbox verifier (test.sh).  Set a judger here to override.
        judger_config=None,
        produce_strategy_config=SyncProduceStrategyConfig(),
        sampler_config=train_sampler_cfg,
    ),
)

# ---------------------------------------------------------------------------
# 6. Eval agent loop manager
# ---------------------------------------------------------------------------
eval_dataset = DatasetConfig(name=experimental_name, anno_path=eval_data_path, sample_ratio=1.0)
eval_dataloader_cfg = DataloaderConfig(
    dataset_config_list=[{"dataset": eval_dataset, "tokenize_fn": tokenizer_cfg}],
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
eval_sampler_cfg = SamplerConfig(
    dataloader_cfg=eval_dataloader_cfg,
    prompt_repeat_k=1,
)
eval_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=1,
    top_p=1.0,
    temperature=0.0,
    min_tokens=0,
)
eval_agent_loop_cfg = SandboxClaudeCodeAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=eval_sample_params,
    env_gateway_url=env_gateway_url,
    max_turns=50,
    agent_timeout=900,
    verifier_timeout=600,
    sandbox_ttl=3600,
    install_claude_code=install_claude_code,
    permission_mode="bypassPermissions",
    max_concurrent_sandboxes=8,
)
eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="eval_task",
        agent_loop_config=eval_agent_loop_cfg,
        judger_config=None,
        sampler_config=eval_sampler_cfg,
    ),
)

# ---------------------------------------------------------------------------
# 7. Evaluator
# ---------------------------------------------------------------------------
evaluator_config = EvaluatorConfig(compute_metric_func=None)

# ---------------------------------------------------------------------------
# 8. RL Colocate Trainer  (CLI entry point: config["trainer"].build().fit())
# ---------------------------------------------------------------------------
trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    gateway_config=gateway_config,
    tokenizer_path=model_path,
    replay_buffer_config=SyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    total_train_steps=total_train_steps,
    train_batch_size=train_batch_size,
    advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
    enable_evaluate=True,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=42,
    debug_rollout=False,
)
