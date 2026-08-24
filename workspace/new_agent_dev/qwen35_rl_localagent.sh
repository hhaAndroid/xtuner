ray stop --force

source /mnt/shared-storage-user/huanghaian/miniconda3/bin/activate pt29_all_env
cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/

# TOTAL_TRAIN_STEPS=1 \
# GLOBAL_BATCH_SIZE=4 \
# PROMPT_REPEAT_K=1 \
# MAX_CONCURRENT_SAMPLES=4 \

export GLOBAL_BATCH_SIZE=256 # 256
export PROMPT_REPEAT_K=16 # 16
export MAX_CONCURRENT_SAMPLES=2048

RAY_HEAD_PORT=6391 RAY_DASHBOARD_PORT=8268 \
ENABLE_EVALUATE=False \
bash /mnt/shared-storage-user/huanghaian/code/gitlab/crg_rl_projects/scripts/run_agentic_rl.sh \
  /mnt/shared-storage-user/huanghaian/code/gitlab/crg_rl_projects/scripts/configs/agent_localhost_rl_qwen3p5_rl.py \
  /mnt/shared-storage-user/huanghaian/code/gitlab/crg_rl_projects/scripts/configs/dataset_metas/localhost_datasets.json \
  localhost_smoke
