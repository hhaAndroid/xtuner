ray stop --force

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PYTHONPATH="$(pwd)"

export MEDIA_ROOT='/mnt/shared-storage-user/llmrazor-share/data/geometry3k/'
export QWEN25_MODEL_PATH=/mnt/shared-storage-user/llmrazor-share/model/Qwen2.5-Math-7B/
# export MODEL_PATH='/mnt/shared-storage-user/llmit/user/lvchengqi/ckpt/xpuyu/qwen3-30ba3b_cold-start/20250924081143/hf-170'
# export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-30B-A3B'
export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'
# export QWEN3_MODEL_PATH='/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/89e5e822ba31507f5f79dc3422c7c5345c422737/'
# export QWEN3_MODEL_PATH="/mnt/shared-storage-user/llmit/user/lvchengqi/ckpt/release/interns1-mini-language-model"
# export QWEN3VL_MODEL_PATH='/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/cadac78306af287f801b75a5565ede58f323f472'
export QWEN3VL_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-VL-30B-A3B-Instruct_MOE'

export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
# export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/geometry3k/train.jsonl'
# export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/dapo_math/dapo-math-17k.jsonl'
# export ROLLOUT_DEBUG_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train-mini.jsonl'
# export DAPO_EVAL_DATA_PATH=/mnt/shared-storage-user/llmrazor-share/data/dapo_math/aime-2024.jsonl


export ROLLOUT_TEST_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'
# export ROLLOUT_TEST_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/geometry3k/test.jsonl'

# export ENABLE_RETURN_ROUTED_EXPERTS=1
# export XTUNER_DETERMINISTIC=1
# export XTUNER_ENABLE_LOGPROB_ZERO_DIFF=1
# export XTUNER_RL_MEM_DIR='work_dirs11/moe_8b_mem_replay_del'

# export RAY_PROFILING=1
# export RAY_task_events_report_interval_ms=1

export WORK_DIR='work_dirs11/qwen3vl_8b_grpo_gsm8k'

export GSM8K_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
export GSM8K_EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'
export DAPO_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/dapo_math/dapo-math-17k.jsonl'
export DAPO_EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/dapo_math/aime-2024.jsonl'
export GSM8K_TASK_WEIGHT=3.0
export DAPO_TASK_WEIGHT=1.0

bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_multi_task_gsm8k_dapo_math.py "lmdeploy" $QWEN3_MODEL_PATH $ROLLOUT_DATA_PATH $ROLLOUT_TEST_DATA_PATH
