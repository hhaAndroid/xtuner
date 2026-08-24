
set -ex

cd /mnt/shared-storage-user/huanghaian/code/xtuner/
# bash /mnt/shared-storage-user/huanghaian/env.sh

# export TORCH_LOGS=recompiles
# conda activate /mnt/shared-storage-user/yehaochen/miniconda3/envs/py312-pt29/

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export XTUNER_DECORD_VIDEO_THREADS=16

export PYTHONPATH="$(pwd)"
export DEEPSEEK_V4_PATH=/mnt/shared-storage-user/llmrazor-share/yehaochen/model/DeepSeek-V4-Flash
export ALPACA_PATH='/mnt/shared-storage-user/llmrazor-share/data/alpaca'

export WORK_DIR="work_dirs/dsv4_l4/sft_config"

current_time=$(date "+%m%d%H%M")
if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi

SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"

XTUNER_USE_MHC_KERNELS=1 XTUNER_ACTIVATION_OFFLOAD=1 XTUNER_USE_NATIVE_RMSNORM=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --master-port 12345 --nproc-per-node 8 -m xtuner.v1.train.cli.sft --config workspace/dsv4/deepseek_v4_flash.py 2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"


