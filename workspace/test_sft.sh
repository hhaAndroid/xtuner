export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export QWEN3_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'
export QWEN3_4B_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-4B-Instruct-2507'
export INTERN_VL_1B_PATH="/mnt/shared-storage-user/llmrazor-share/model/InternVL3_5-1B-HF"
export QWEN3_VL_PATH="/mnt/shared-storage-user/llmrazor-share/model/Qwen2.5-VL-3B-Instruct"
export GPT_OSS_MINI_PATH='/mnt/shared-storage-user/llmrazor-share/model/gpt-oss-20b-bf16'
export QWEN3_MOE_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-30B-A3B'
export INTERNS1_DENSE_PATH='/mnt/shared-storage-user/llmrazor-share/model/intern-s1-mini/'
export ROLLOUT_MODEL_PATH=$QWEN3_PATH
export ALPACA_PATH='/mnt/shared-storage-user/llmrazor-share/data/alpaca'

export INTERNS1_DATA_META='/mnt/shared-storage-user/llmrazor-share/data/vlm_ci_data.json'

export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
export ROLLOUT_TEST_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'

export DEEPSEEK_V3_PATH='/mnt/shared-storage-user/llmrazor-share/model/DeepSeek-V3.1'

export PYTHONPATH="$(pwd)"
export PYTHONPATH=/mnt/shared-storage-user/huanghaian/code/lmdeploy/:$PYTHONPATH
export XTUNER_USE_LMDEPLOY=1

export VERL_ROLLOUT_DATA_PATH=/mnt/shared-storage-user/llmrazor-share/data/verl-rollout-step0.jsonl
export VIDEO_ROOT='/mnt/shared-storage-user/llmrazor-share/data/images/'


export WORK_DIR="work_dir/qwen_moe"

current_time=$(date "+%m%d%H")
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi

torchrun --nproc-per-node=8 \
    ci/scripts/test_sft_trainer.py \
    ${WORK_DIR} \
    2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"
