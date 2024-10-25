set -x

PARTITION=${PARTITION:-"llm_razor"}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
MIRCO_BATCH_SIZE=${MIRCO_BATCH_SIZE:-2}
ACCUMULATIVE_COUNTS=${ACCUMULATIVE_COUNTS:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

export PYTHONPATH="$(pwd):$(pwd)/../"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/qwenvl2_sft_2b_pack'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
SCRIPT_NAME=$(basename "$0")
cp "$0" "${OUTPUT_DIR}/${SCRIPT_NAME}"

# --group-by-modality-length \
# --group-by-length \
# --liger \

MAX_LENGHT=32768
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 srun -p ${PARTITION} --time 1-00:00:00 \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python -u qwenvl2_sft.py \
  --model /mnt/hwfile/xtuner/huanghaian/model/Qwen2-VL-2B-Instruct \
  --datasets data/qwenvl2_sft_test.json \
  --max-length $MAX_LENGHT \
  --pack-max-length $((MIRCO_BATCH_SIZE * MAX_LENGHT)) \
  --concat-before-pack \
  --num-workers 4 \
  --mirco-batch-size 1 \
  --group-by-length \
  --global-batch-size $((GPUS*ACCUMULATIVE_COUNTS)) \
  --lr 3e-5 \
  --wd 0.0 \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 10 \
  --seed 42 \
  --checkpoint-interval 2000 \
  --checkpoint-drop-optimizer \
  --shard-strategy 'zero2' \
  --dset-pack-level 'soft' \
  --dset-cache-dir /mnt/petrelfs/huanghaian/code/mm/xtuner/my_llava/qwenvl2_2b_sft_cache \
  --liger \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
