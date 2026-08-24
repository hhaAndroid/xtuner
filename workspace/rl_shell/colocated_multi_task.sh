export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
export ROLLOUT_TEST_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'
export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'

export GSM8K_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
export GSM8K_EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'
export DAPO_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/dapo_math/dapo-math-17k.jsonl'
export DAPO_EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/dapo_math/aime-2024.jsonl'
export GSM8K_TASK_WEIGHT=3.0
export DAPO_TASK_WEIGHT=1.0

export WORK_DIR='work_dirs12/qwen3_8b_grpo_multi_task'
bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_multi_task_gsm8k_dapo_math.py "lmdeploy" $QWEN3_MODEL_PATH $ROLLOUT_DATA_PATH $ROLLOUT_TEST_DATA_PATH
