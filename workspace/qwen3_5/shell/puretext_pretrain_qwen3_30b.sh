
set -ex

cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/

# export TORCH_LOGS=recompiles

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export XTUNER_DECORD_VIDEO_THREADS=16

export PYTHONPATH="$(pwd)"
export XTUNER_GC_ENABLE=1
export GLOBAL_BATCH_SIZE=64

export WORK_DIR="work_dirs/qwen30b/pure_pretrain"
export META_DATA_PATH="/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/meta_data/text_pretrain.json"
export TOKENIZER_CACHE_DIR='workspace/qwen30ba3/puretext_pretrain_tokenizer_cache'
CONFIG_PATH="/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/configs/puretext_pretrain_qwen3vl_30b_config.py"

current_time=$(date "+%m%d%H%M")
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi

SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"

torchrun --nproc-per-node=8 \
    --master_addr=${MASTER_ADDR} \
    --nnodes=${NODE_COUNT} \
    --node_rank=${NODE_RANK} \
    xtuner/v1/train/cli/sft.py \
    --config $CONFIG_PATH \
    2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"
