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

MIRCO_BATCH_SIZE=${MIRCO_BATCH_SIZE:-4}
ACCUMULATIVE_COUNTS=${ACCUMULATIVE_COUNTS:-2}

# --group-by-modality-length \
# -m debugpy --connect 5680
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $ENV_PATH/bin/torchrun \
  --nproc-per-node=$GPUS_PER_NODE  \
  unify_internvl2_train.py \
  --model /cpfs01/shared/llm_razor/huanghaian/new_model/InternVL2-2B \
  --datasets data/internvl2_sft.json \
  --freeze-vit \
  --num-workers 4 \
  --mirco-batch-size $MIRCO_BATCH_SIZE \
  --global-batch-size $((MIRCO_BATCH_SIZE*GPUS_PER_NODE*ACCUMULATIVE_COUNTS)) \
  --lr 2e-5 \
  --wd 0.0 \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 2 \
  --seed 42 \
  --checkpoint-interval 2000 \
  --checkpoint-drop-optimizer \
  --shard-strategy 'zero2' \
  --group-by-length \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

