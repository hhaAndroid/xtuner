set -x

export HOME='/cpfs01/shared/llm_razor/huanghaian/'
export ENV_PATH="/cpfs01/shared/llm_razor/huanghaian/miniconda3/envs/xtuner_25"
export TRITON_CACHE_DIR="/tmp/triton"

OUTPUT_DIR='work_dirs/xpuyu_dpo'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
SCRIPT_NAME=$(basename "$0")
cp "$0" "${OUTPUT_DIR}/${SCRIPT_NAME}"

export PYTHONPATH="$(pwd):$(pwd)/../"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MIRCO_BATCH_SIZE=${MIRCO_BATCH_SIZE:-1}
ACCUMULATIVE_COUNTS=${ACCUMULATIVE_COUNTS:-1}

#   --tensorboard \
# qwen2.5
# --model /cpfs01/shared/llm_razor/huanghaian/new_model/Qwen2.5-1.5B-Instruct \
# --dset-cache-dir ./ultrafeedback_xpuyu_cache \

# internlm2.5
#  --model /cpfs01/shared/llm_razor/huanghaian/model/internlm2_5-1_8b-chat \
#  --dset-cache-dir ./ultrafeedback_xpuyu_cache_interlm2 \

# --use-liger
# export PYTHONPATH="$(pwd):$(pwd)/../:/cpfs01/shared/llm_razor/huanghaian/code/Liger-Kernel/src/"

MAX_LENGHT=16384
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $ENV_PATH/bin/torchrun \
  --nproc-per-node=$GPUS_PER_NODE\
  fsdp2_dpo_train.py \
  --model /cpfs01/shared/llm_razor/huanghaian/new_model/Qwen2.5-1.5B-Instruct \
  --datasets /cpfs01/shared/llm_razor/huanghaian/data/ultrafeedback-binarized-preferences-cleaned/data/ultrafeedback_xpuyu.jsonl::1.0 \
  --dset-cache-dir ./ultrafeedback_xpuyu_cache \
  --loss-type sigmoid,bco_pair \
  --loss-weight '0.8,0.2' \
  --rpo-alpha 1.0 \
  --beta 0.1 \
  --sp-size 1 \
  --max-length $MAX_LENGHT \
  --pack-max-length $((MIRCO_BATCH_SIZE * MAX_LENGHT)) \
  --mirco-batch-size 1 \
  --global-batch-size $((GPUS_PER_NODE*ACCUMULATIVE_COUNTS)) \
  --lr 2e-6 \
  --wd 0.01 \
  --warmup-ratio 0.03 \
  --work-dir ${OUTPUT_DIR} \
  --log-interval 1 \
  --seed 42 \
  --checkpoint-interval 2000 \
  --hf-interval 2000 \
  --num-workers 4 \
  --concat-before-pack \
  --group-by-length \
  --reshard-after-forward \
  --tensorboard \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
