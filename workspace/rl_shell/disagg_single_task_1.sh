export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'
export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
export ROLLOUT_TEST_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'

export WORK_DIR='work_dirs12/qwen3_8b_grpo_gsm8k_disagg_1'

export TRIGGER_PARAMETER_SYNC_STEP=1
export OVER_SAMPLE_THRESHOLD=0.0
export PARTIAL_ROLLOUT=0

bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_disaggregated_grpo_gsm8k.py "lmdeploy" $QWEN3_MODEL_PATH $ROLLOUT_DATA_PATH $ROLLOUT_TEST_DATA_PATH
