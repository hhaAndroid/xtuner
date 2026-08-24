
set -ex

cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/
# bash /mnt/shared-storage-user/huanghaian/env.sh

# export TORCH_LOGS=recompiles

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export XTUNER_DECORD_VIDEO_THREADS=16

export PYTHONPATH="$(pwd)"
export XTUNER_GC_ENABLE=1
export GLOBAL_BATCH_SIZE=1
export XTUNER_SKIP_EMPTY_THINK=1

export WORK_DIR="work_dirs_pretrain_debug/qwen35b/sft"

CONFIG_PATH='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/configs/interns2_preview_verify_decay_20260304d_32k_debug.py'

current_time=$(date "+%m%d%H%M")
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi

SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"

torchrun --nproc-per-node=1 \
    xtuner/v1/train/cli/sft.py \
    --config $CONFIG_PATH \
    2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"
