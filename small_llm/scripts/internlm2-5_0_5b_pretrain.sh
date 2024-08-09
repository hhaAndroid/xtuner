set -x

PARTITION=${PARTITION:-"llm_razor"}
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SRUN_ARGS=${SRUN_ARGS:-""}

export PYTHONPATH="$(pwd):$(pwd)/../"
export MASTER_PORT=34230
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/internlm2_0_5b_pretrain'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 32
# batch size per gpu: 16
# gradient accumulation steps: 4
# total token per batch: 32gx16bsx4accx2048len = 4m
# step: 25000, 25000x4m = 100b token
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
  python -u fsdp_train.py \
  --llm '../internlm2_5-05b' \
  --dset-length 25000 \
  --mirco-batch-size 16 \
  --global-batch-size 2048 \
  --lr 3e-4 \
  --lr-min 3e-5 \
  --wd 0.01 \
  --warmup-ratio 2000 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 10 \
  --seed 42 \
  --max-length 2048 \
  --checkpoint-interval 5000 \
  --shard-strategy 'zero2' \
  --datasets '/mnt/hwfile/xtuner/huanghaian/data/llm/wanjuan_1/orig_jsonl/' '/mnt/hwfile/xtuner/huanghaian/data/llm/SkyPile-150B/data/' \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
