set -x

source /mnt/shared-storage-user/llmit/user/guyuzhe/projects/imo/env/xtuner/bin/activate

git config --global --add safe.directory /mnt/shared-storage-user/llmit/user/guyuzhe/projects/imo
git config --global --add safe.directory /mnt/shared-storage-user/llmit/user/guyuzhe/projects/imo/xtuner

XTUNER_PATH="/mnt/shared-storage-user/llmit/user/guyuzhe/projects/imo/xtuner"
SAVE_DIR=/mnt/shared-storage-user/llmit/user/guyuzhe/projects/imo/work_dirs/data_process/2025-10-29/oss_sft/gpt_oss_120b_test
if [ ! -d "$SAVE_DIR" ]; then
  mkdir -p "$SAVE_DIR"
fi
SCRIPT_NAME=$(basename "$0")
cp "$0" "${SAVE_DIR}/${SCRIPT_NAME}"

export PYTHONPATH=$XTUNER_PATH:"/mnt/shared-storage-user/llmit/user/guyuzhe/projects/imo/env/xtuner/lib/python3.12/site-packages":$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export HF_HUB_CACHE=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub
export HUGGINGFACE_HUB_CACHE=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1

export XTUNER_USE_CUTLASS_GROUP_GEMM=1
export XTUNER_DISABLE_GIT_INFO=1
export XTUNER_ACTIVATION_OFFLOAD=1

echo "Save dir: ${SAVE_DIR}"
cp $0 $SAVE_DIR/

torchrun --nproc-per-node=8 ci/scripts/test_sft_trainer.py --config /mnt/shared-storage-user/llmit/user/guyuzhe/projects/imo/work_dirs/data_process/2025-10-29/oss_sft/config.py

