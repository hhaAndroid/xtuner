export PYTHONPATH="$(pwd):/mnt/shared-storage-user/huanghaian/code/temp/xtuner/recipe"
export RL_LLM_MODEL='xtuner_qwen3p5_vl_35b'
export LAGENT_SRC_DIR='/mnt/shared-storage-user/huanghaian/code/gateway/lagent'

export QWEN3P5_VL_MODEL_PATH='/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'
export WORK_DIR='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/work_dirs_rl/output'

# python -m tb2_eval.local_run --limit 1 --output /mnt/shared-storage-user/huanghaian/code/temp/xtuner/work_dirs_rl/output.json --mode agentloop --hf-checkpoint $QWEN3P5_VL_MODEL_PATH
python -m tb2_rl.local_run --limit 1 --output /mnt/shared-storage-user/huanghaian/code/temp/xtuner/work_dirs_rl/output.json --mode agentloop --hf-checkpoint $QWEN3P5_VL_MODEL_PATH