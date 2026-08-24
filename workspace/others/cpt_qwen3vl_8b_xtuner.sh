
set -ex

# bash /mnt/shared-storage-user/huanghaian/env.sh
# pip install rdkit==2025.3.6 transformers==4.57.0 -i http://mirrors.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.h.pjlab.org.cn  --break-system-packages

cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/

export TORCH_LOGS=recompiles

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export XTUNER_DECORD_VIDEO_THREADS=16

export GLOBAL_BATCH_SIZE=8
export WORK_DIR="work_dir_test22222/qwen3vl_cpt_8b-nocompile-novit"
export PYTHONPATH="$(pwd)"
export XTUNER_GC_ENABLE=1
export XTUNER_SKIP_EMPTY_THINK=1

CONFIG_PATH="examples/v1/cpt_qwen3vl_8b_config.py"

current_time=$(date "+%m%d%H%M")
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi


SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"


# export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc-per-node=8 \
    xtuner/v1/train/cli/sft.py \
    --config $CONFIG_PATH \
    2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"