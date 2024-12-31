set -x

export HOME='/cpfs01/shared/llm_razor/huanghaian/'
export ENV_PATH="/cpfs01/shared/llm_razor/huanghaian/miniconda3/envs/torchtitan"
export TRITON_CACHE_DIR="/tmp/triton"

OUTPUT_DIR='work_dirs/xpuyu_sft_pp'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
SCRIPT_NAME=$(basename "$0")
cp "$0" "${OUTPUT_DIR}/${SCRIPT_NAME}"

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export PYTHONPATH="$(pwd):$(pwd)/../"

ACCUMULATIVE_COUNTS=${ACCUMULATIVE_COUNTS:-1}
# -m debugpy --connect 5680
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $ENV_PATH/bin/torchrun \
  --nproc-per-node=$GPUS_PER_NODE \
  fsdp_sft.py \
  --llm /cpfs01/shared/llm_razor/huanghaian/model/internlm2_5-1_8b-chat \
  --chat-template internlm2 \
  --datasets /cpfs01/shared/llm_razor/huanghaian/data/llm_sft_data/xpuyu_sft \
  --dset-cache-dir /cpfs01/shared/llm_razor/huanghaian/data/llm_sft_data/xpuyu_sft_internlm2_1_8b_cache \
  --debug \
  --pp-size 2 \
  --pp-mb 1 \
  --mirco-batch-size 2 \
  --num-workers 4 \
  --dset-pack-level soft \
  --global-pack \
  --max-length 8192 \
  --group-by-length \
  --global-batch-size 4 \
  --lr 2e-6 \
  --wd 0.0 \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 1 \
  --seed 42 \
  --checkpoint-interval 20000 \
  --resume \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
