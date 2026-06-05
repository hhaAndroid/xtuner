set -ex
export XTUNER_PACK_WORKERS=8
export XTUNER_TOKENIZE_WORKERS=16
export NCCL_TIMEOUT=10800
export TORCH_DISTRIBUTED_TIMEOUT=10800
export XTUNER_USE_FA3=1
export PYTHONPATH="$(pwd)"
export HF_HOME="$(pwd)/"
export TORCHDYNAMO_VERBOSE=1

MASTER_PORT=20500
config_file="examples/v1/config/mpo_qwen3_vl_8B.py"
# NODE_COUNT=1
# NODE_RANK=0
# MASTER_ADDR=127.0.0.1
PROC_PER_NODE=8

#  --nnodes=$NODE_COUNT \
# --node_rank=$NODE_RANK \
# --master_addr=$MASTER_ADDR \
# --master_port=$MASTER_PORT \
export META_DATA_PATH="/mnt/llm-razor/data/dpo_meta_data.json"
export MODEL_PATH="/mnt/llm-razor/model/Qwen3-VL-4B-Instruct"
export WORK_DIR="/mnt/yehaochen/huanghaian/xtuner/work_dirs/dpo"

torchrun \
  --nproc_per_node=$PROC_PER_NODE \
  xtuner/v1/train/cli/dpo.py --config ${config_file}
