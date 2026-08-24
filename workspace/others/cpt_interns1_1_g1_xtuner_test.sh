
set -ex

export TORCH_LOGS=recompiles

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export XTUNER_DECORD_VIDEO_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export GLOBAL_BATCH_SIZE=8
export WORK_DIR="work_dirs_qwen3_test/interns1_1_g1_cpt"
export PYTHONPATH="$(pwd)"
export XTUNER_GC_ENABLE=1

CONFIG_PATH="examples/v1/cpt_interns1_1_g1_config_test.py"

current_time=$(date "+%m%d%H%M")
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi


SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"


torchrun --nproc-per-node=8 \
    xtuner/v1/train/cli/sft.py \
    --config $CONFIG_PATH \
    2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"
