
set -ex

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export XTUNER_DECORD_VIDEO_THREADS=16

export CEPH_CONFIG_PATH="/mnt/shared-storage-user/huanghaian/petreloss.conf"
export META_DATA_PATH="/mnt/shared-storage-user/huanghaian/code/xtuner/vlm_pretrain_ci_data.json"
export TOKENIZER_CACHE_DIR="./hf_tokenizer_cache_xtuner"
export GLOBAL_BATCH_SIZE=8
export WORK_DIR="work_dir/internvl3.5-8B-pretrain-xtuner"
export PYTHONPATH="$(pwd)"

CONFIG_PATH="examples/v1/pretrain_internvl_3p5_8b_config.py"

current_time=$(date "+%m%d%H%M")
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi

SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"

# -m debugpy --connect 5680 
torchrun --nproc-per-node=8 \
    xtuner/v1/train/cli/sft.py \
    --config $CONFIG_PATH \
    2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"
