"""Minimal example: use HarborAgentLoop in XTuner RL.

Run idea:
    XTUNER_USE_VLLM=1 python examples/harbor_agentloop_config_example.py

This file focuses on the AgentLoop part; plug into your existing RLColocateTrainer
config file and resource settings.
"""

from xtuner.v1.data_proto import SampleParams
from xtuner.v1.rl.agent_loop import AgentLoopManagerConfig, HarborAgentLoopConfig, SamplerConfig
from xtuner.v1.rl.agent_loop.producer import SyncProduceStrategyConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn.text import RLTextTokenizeFnConfig


model_path = "/path/to/your/model-or-tokenizer"

sampler_cfg = SamplerConfig(
    dataloader_cfg=DataloaderConfig(
        dataset_cfg=[
            {
                "dataset": DatasetConfig(name="gsm8k", anno_path="/path/to/train.jsonl"),
                "tokenize_fn": RLTextTokenizeFnConfig(max_length=1024),
            }
        ],
        collator="fake_collator",
        pack_level="none",
        pack_max_length=4096,
    ),
    prompt_repeat_k=4,
)

agent_loop_cfg = HarborAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=SampleParams(max_tokens=1024, temperature=1.0, top_p=1.0),
    bridge_import_path="xtuner.v1.rl.agent_loop.harbor_bridge:generate_with_harbor",
    # Prefer rollout controller metadata (gateway base_url/api_key) when available.
    prefer_rollout_gateway=True,
    rollout_metadata_ttl_sec=10,
    bridge_kwargs={
        "harbor_repo": "/home/huanghaian/.openclaw/workspace/harbor",
        "harbor_bin": "harbor",
        "bridge_workspace": "/tmp/xtuner_harbor_bridge",
        "task_template_path": "/home/huanghaian/.openclaw/workspace/harbor/examples/tasks/hello-world",
        "job_name_prefix": "xtuner-harbor",
        "env": "docker",
        "agent": "terminus-2",
        # Fallback if rollout metadata is unavailable:
        "llm_backend": "litellm",
        "api_base": "http://127.0.0.1:8000/v1",
        "model_name": "openai/your-model",
        # Optional extra kwargs forwarded as `--ak key=value`
        "extra_agent_kwargs": {
            "collect_rollout_details": True,
            "max_turns": 20,
        },
        "timeout_sec": 1800,
        "keep_job_dir": False,
    },
)

agent_loop_manager_cfg = AgentLoopManagerConfig(
    task_name="gsm8k",
    agent_loop_config=agent_loop_cfg,
    sampler_config=sampler_cfg,
    produce_strategy_config=SyncProduceStrategyConfig(),
)

print(agent_loop_manager_cfg)
