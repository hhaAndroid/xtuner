
set -ex

cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/

# export TORCH_LOGS=recompiles
export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export XTUNER_TOKENIZE_WORKERS=16
export XTUNER_USE_FA3=1
export PYTHONPATH="$(pwd)"
export XTUNER_GC_ENABLE=1

export WORK_DIR="work_dir/internvl3.5-8B-sft-xtuner-1"
export META_DATA_PATH="/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/interns1_1_base02_20260120b_tiny.json"
export CEPH_CONFIG_PATH="/mnt/shared-storage-user/huanghaian/petreloss.conf"
export TOKENIZER_CACHE_DIR="./workspace/qwen3_interns1_1_base02_20260120b_tiny/qwen3_interns1_1_base02_20260120b_tiny"

CONFIG_PATH="examples/v1/sft_qwen3vl_8b_config.py"
# CONFIG_PATH='examples/v1/sft_gptoss_20b_config.py'

torchrun --nproc-per-node=8 \
    xtuner/v1/train/cli/sft.py \
    --config $CONFIG_PATH
