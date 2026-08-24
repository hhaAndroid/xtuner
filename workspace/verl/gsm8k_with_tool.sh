export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'
export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k_with_tool/train.jsonl'
export ROLLOUT_TEST_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k_with_tool/test.jsonl'

export LMDEPLOY_PATH=/mnt/shared-storage-user/duanyanhui/workspace/code/lmdeploy-main/lmdeploy/
export PYTHONPATH="$LMDEPLOY_PATH:$(pwd)"

export WORK_DIR='./work_dir_VERL/gsm8k_tool_example'

bash examples/v1/scripts/run_rl.sh recipe/verl_agent/gsm8k_tool_example/gsm8k_tool_grpo_config.py "lmdeploy" $QWEN3_MODEL_PATH $ROLLOUT_DATA_PATH $ROLLOUT_TEST_DATA_PATH