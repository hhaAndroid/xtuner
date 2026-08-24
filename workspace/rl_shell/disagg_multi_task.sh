export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'
export GSM8K_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
export GSM8K_EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'
export DAPO_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/dapo_math/dapo-math-17k.jsonl'
export DAPO_EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/dapo_math/aime-2024.jsonl'

export MODEL_PATH="$QWEN3_MODEL_PATH"
export DATA_PATH="$GSM8K_DATA_PATH"
export EVAL_DATA_PATH="$GSM8K_EVAL_DATA_PATH"

export TRAIN_NUM_WORKERS=4
export ROLLOUT_NUM_WORKERS=4
export TRAIN_BATCH_SIZE=64
export TOTAL_TRAIN_STEPS=4
export TRIGGER_PARAMETER_SYNC_STEP=1
export OVER_SAMPLE_THRESHOLD=0.0
export PARTIAL_ROLLOUT=0
export GSM8K_TASK_WEIGHT=3.0
export DAPO_TASK_WEIGHT=1.0
export ENABLE_EVALUATE=0

export WORK_DIR='work_dirs11/qwen3_8b_grpo_multi_task_disagg'
# run_rl.sh 仍然需要一组通用 positional 参数来设置 MODEL_PATH / DATA_PATH / EVAL_DATA_PATH。
# 这个 multi-task config 实际读取的是上面的 GSM8K_* / DAPO_* 环境变量，而不是 DATA_PATH 本身。
bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_disaggregated_multi_task_gsm8k_dapo_math.py "lmdeploy" $QWEN3_MODEL_PATH $GSM8K_DATA_PATH $GSM8K_EVAL_DATA_PATH
