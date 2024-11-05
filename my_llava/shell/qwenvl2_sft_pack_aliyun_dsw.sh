set -x

export HOME='/cpfs01/shared/llm_razor/huanghaian/'
export ENV_PATH="/cpfs01/shared/llm_razor/huanghaian/miniconda3/envs/xtuner_24"

OUTPUT_DIR='work_dirs/qwenvl2-2b-sft-test'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
SCRIPT_NAME=$(basename "$0")
cp "$0" "${OUTPUT_DIR}/${SCRIPT_NAME}"

# -m debugpy --connect 5688

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export PYTHONPATH="$(pwd):$(pwd)/../"

MIRCO_BATCH_SIZE=${MIRCO_BATCH_SIZE:-2}
ACCUMULATIVE_COUNTS=${ACCUMULATIVE_COUNTS:-2}

# --group-by-modality-length \
# --group-by-length \
# --liger \
# -m debugpy --connect 5680
MAX_LENGHT = 32768
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $ENV_PATH/bin/torchrun \
  --nproc-per-node=$GPUS_PER_NODE  \
  unify_qwenvl2_train.py \
  --model /cpfs01/shared/llm_razor/huanghaian/new_model/Qwen2-VL-2B-Instruct \
  --datasets data/qwenvl2_sft.json \
  --liger \
  --num-workers 4 \
  --global-batch-size $((GPUS_PER_NODE*ACCUMULATIVE_COUNTS)) \
  --lr 2e-5 \
  --wd 0.0 \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 2 \
  --seed 42 \
  --checkpoint-interval 2000 \
  --checkpoint-drop-optimizer \
  --shard-strategy 'zero2' \
  --dset-pack \
  --dset-cache-dir /cpfs01/shared/llm_razor/huanghaian/code/xtuner/my_llava/qwenvl2_2b_sft_cache \
  --mirco-batch-size 1 \
  --max-length $MAX_LENGHT \
  --pack-max-length $((MIRCO_BATCH_SIZE * MAX_LENGHT)) \
  --concat-before-pack \
  --group-by-length \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

