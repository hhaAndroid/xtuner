set -x

cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

SAVE_DIR='work_dirs'
if [ ! -d "$SAVE_DIR" ]; then
  mkdir -p "$SAVE_DIR"
fi
SCRIPT_NAME=$(basename "$0")
cp "$0" "${SAVE_DIR}/${SCRIPT_NAME}"

export PYTHONPATH="$(pwd)"
export PYTHONPATH=$DION_PATH:$XTUNER_PATH:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export XTUNER_USE_CUTLASS_GROUP_GEMM=1

DATA_PATH=/mnt/shared-storage-user/llmit/user/guyuzhe/projects/imo/work_dirs/data_process/2025-09-09/oss_sft_data/AM_Thinking_GPT_OSS_high_16k_unfinished_queries_64k_rollout_sft_data_oss_format.jsonl

echo "Save dir: ${SAVE_DIR}"
cp $0 $SAVE_DIR/

current_time=$(date "+%m%d%H%M")

torchrun --nproc-per-node=8 \
    --master_addr=${MASTER_ADDR} \
    --nnodes=${NODE_COUNT} \
    --node_rank=${NODE_RANK} \
    xtuner/v1/train/cli/sft.py \
    --config examples/oss_sft_config.py \
    2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"
