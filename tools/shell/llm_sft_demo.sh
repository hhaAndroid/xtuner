set -x

export HOME='/cpfs01/shared/llm_razor/huanghaian/'
export ENV_PATH="/cpfs01/shared/llm_razor/huanghaian/miniconda3/envs/xtuner_25"
export TRITON_CACHE_DIR="/tmp/triton"

OUTPUT_DIR='work_dirs/xpuyu_sft'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
SCRIPT_NAME=$(basename "$0")
cp "$0" "${OUTPUT_DIR}/${SCRIPT_NAME}"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export PYTHONPATH="$(pwd):$(pwd)/../"

ACCUMULATIVE_COUNTS=${ACCUMULATIVE_COUNTS:-2}

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $ENV_PATH/bin/torchrun \
  --nproc-per-node=$GPUS_PER_NODE  \
  fsdp_sft.py \
  --llm /cpfs01/shared/llm_razor/huanghaian/model/internlm2_5-1_8b-chat \
  --chat-template internlm2 \
  --datasets /cpfs01/shared/llm_razor/huanghaian/data/llm_sft_data/xpuyu_sft \
  --dset-formats processed \
  --dset-cache-dir /cpfs01/shared/llm_razor/huanghaian/data/llm_sft_data/xpuyu_sft_internlm2_1_8b_cache \
  --num-workers 4 \
  --dset-pack-level soft \
  --global-pack \
  --max-length 16384 \
  --group-by-length \
  --global-batch-size $((GPUS_PER_NODE*ACCUMULATIVE_COUNTS)) \
  --lr 2e-6 \
  --wd 0.0 \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 1 \
  --seed 42 \
  --checkpoint-interval 20000 \
  --resume \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
