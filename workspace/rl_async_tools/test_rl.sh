ray stop --force

# source /mnt/shared-storage-user/huanghaian/.bashrc 
# source /mnt/shared-storage-user/huanghaian/proxy_off
# conda activate verl
cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export PYTHONPATH="$(pwd)"

export QWEN25_MODEL_PATH=/mnt/shared-storage-user/llmrazor-share/model/Qwen2.5-Math-7B/
# export MODEL_PATH='/mnt/shared-storage-user/llmit/user/lvchengqi/ckpt/xpuyu/qwen3-30ba3b_cold-start/20250924081143/hf-170'
export DAPO_DATA_PATH=/mnt/shared-storage-user/llmrazor-share/data/dapo_math/dapo-math-17k.jsonl
export EVAL_DATA_PATH=/mnt/shared-storage-user/llmrazor-share/data/dapo_math/aime-2024.jsonl

export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-30B-A3B'
# export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'
# export QWEN3_MODEL_PATH='/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/89e5e822ba31507f5f79dc3422c7c5345c422737/'
# export QWEN3_MODEL_PATH="/mnt/shared-storage-user/llmit/user/lvchengqi/ckpt/release/interns1-mini-language-model"

export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
# export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/dapo_math/dapo-math-17k.jsonl'
# export ROLLOUT_DEBUG_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train-mini.jsonl'
# export DAPO_EVAL_DATA_PATH=/mnt/shared-storage-user/llmrazor-share/data/dapo_math/aime-2024.jsonl
export ROLLOUT_DEBUG_DATA_PATH=''
export DAPO_EVAL_DATA_PATH=""
# export ROLLOUT_TEST_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'

# export ENABLE_RETURN_ROUTED_EXPERTS=1
# export XTUNER_DETERMINISTIC=1
# export XTUNER_ENABLE_LOGPROB_ZERO_DIFF=1
# export XTUNER_RL_MEM_DIR='work_dirs11/moe_8b_mem_replay_del'

# export RAY_PROFILING=1
# export RAY_task_events_report_interval_ms=1

bash workspace/rl_async_tools/run_rl.sh workspace/rl_async_tools/rl_qwen25_7B_dapo.py "sglang" $QWEN25_MODEL_PATH $DAPO_DATA_PATH $ROLLOUT_DEBUG_DATA_PATH
# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3_30B_grpo_dapo.py "lmdeploy" $QWEN3_MODEL_PATH $DAPO_DATA_PATH $DAPO_EVAL_DATA_PATH
# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3_30B_dapo.py "lmdeploy" $QWEN3_MODEL_PATH $ROLLOUT_DATA_PATH $DAPO_EVAL_DATA_PATH
# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3_30B_grpo.py "lmdeploy" $QWEN3_MODEL_PATH $ROLLOUT_DATA_PATH $DAPO_EVAL_DATA_PATH
