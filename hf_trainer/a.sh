set -x

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34227
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/hf_llava_4x2'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 srun -p llm_razor \
  --job-name=hf_phi3_4x2 --cpus-per-task=16  --nodes=2 --gres=gpu:8 --ntasks-per-node=8 --kill-on-bad-exit=1 \
  python trainer.py config.py \
  --work-dir ${OUTPUT_DIR} \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
