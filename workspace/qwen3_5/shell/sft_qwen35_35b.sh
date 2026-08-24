
set -ex

cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/
bash /mnt/shared-storage-user/huanghaian/env.sh

export TORCH_LOGS=recompiles

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export XTUNER_DECORD_VIDEO_THREADS=16
export CEPH_CONFIG_PATH="/mnt/shared-storage-user/huanghaian/petreloss.conf"

export PYTHONPATH="$(pwd)"
export XTUNER_GC_ENABLE=1
export GLOBAL_BATCH_SIZE=8

# sft
export WORK_DIR="work_dirs_local/qwen35b/sft_all_leak"
export META_DATA_PATH="/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/meta_data/interns1_1_base02_20260120b_tiny_rollout_v2.json"
export TOKENIZER_CACHE_DIR='workspace/qwen35ba3_v2/sft_tokenizer_cache'
# CONFIG_PATH="/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/configs/sft_qwen35vl_35b_config.py"
CONFIG_PATH="/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/configs/sft_qwen35vl_35b_config_leak.py"

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
