set -x

PARTITION=${PARTITION:-"llm_razor"}
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SRUN_ARGS=${SRUN_ARGS:-""}

export PYTHONPATH="$(pwd):$(pwd)/../"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/qwen2_pretrain'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 32
# batch size per gpu: 8
# gradient accumulation steps: 2
# total token per batch: 32gx8bsx2accx2048len = 1m
# epoch: 1
#
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python -u fsdp_pretrain.py \
  --llm 'qwen2' \
  --mirco-batch-size 8 \
  --global-batch-size 512 \
  --lr 1e-4 \
  --wd 0.01 \
  --warmup-ratio 2000 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 10 \
  --num-workers 4 \
  --seed 42 \
  --max-length 2048 \
  --dset-pack-level 'hard' \
  --dset-from-cache \
  --checkpoint-interval 5000 \
  --shard-strategy 'full' \
  --dset-cache-dir '/mnt/hwfile/xtuner/huanghaian/data/llm/wanjuan_1/dataset_cache/' '/mnt/hwfile/xtuner/huanghaian/data/llm/SkyPile-150B/dataset_cache/' \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
