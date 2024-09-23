set -x

PARTITION=${PARTITION:-"llm_razor"}
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
MIRCO_BATCH_SIZE=${MIRCO_BATCH_SIZE:-4}
ACCUMULATIVE_COUNTS=${ACCUMULATIVE_COUNTS:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

export PYTHONPATH="$(pwd):$(pwd)/../"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/internvl1_5_phi3_sft'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 64
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 512
# epoch: 1
MAX_LENGHT=8192
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 srun -p ${PARTITION} --time 4-00:00:00 \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python -u internvl2_train.py \
  --vit '/mnt/hwfile/xtuner/huanghaian/model/InternViT-300M-448px' \
  --projector '/mnt/hwfile/xtuner/huanghaian/model/InternViT-300M-448px/mlp_projector/phi_3_mini_128k_instruct.pth' \
  --llm '/mnt/hwfile/xtuner/huanghaian/model/Phi-3-mini-128k-instruct' \
  --internvl '/mnt/hwfile/xtuner/huanghaian/model/Mini-InternVL-Chat-4B-V1-5' \
  --meta-path 'aa' \
  --chat-template 'phi3-chat' \
  --drop-path-rate 0.1 \
  --group-by-length \
  --max-length $MAX_LENGHT \
  --pack-max-length $((MIRCO_BATCH_SIZE * MAX_LENGHT)) \
  --num-workers 4 \
  --mirco-batch-size 1 \
  --global-batch-size $((GPUS*ACCUMULATIVE_COUNTS)) \
  --lr 3.5e-5 \
  --lr-min 3.5e-6 \
  --wd 0.05 \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 10 \
  --seed 42 \
  --checkpoint-interval 2000 \
  --checkpoint-drop-optimizer \
  --shard-strategy 'zero2' \
  --use-fast-tokenizer \
  --dset-pack-level 'soft' \
  --dset-cache-dir /mnt/hwfile/xtuner/huanghaian/internvl_meta13_cache \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
