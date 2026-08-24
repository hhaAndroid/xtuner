export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'
export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
export ROLLOUT_TEST_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'

# source /mnt/shared-storage-user/huanghaian/miniconda3/bin/activate pt28_all_env
# cd /mnt/shared-storage-user/huanghaian/code/xtuner/

export WORK_DIR='work_dirs121/qwen3_8b_grpo_gsm8k'

export XTUNER_CENTRAL_MEM_TRACE=1 
export XTUNER_CENTRAL_MEM_TRACE_MAX_NODES=1200000

bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_grpo_gsm8k_judge.py "lmdeploy" $QWEN3_MODEL_PATH $ROLLOUT_DATA_PATH $ROLLOUT_TEST_DATA_PATH
