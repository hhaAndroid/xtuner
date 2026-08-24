export PYTHONPATH="$(pwd):/mnt/shared-storage-user/huanghaian/code/temp/xtuner/recipe"
export RL_LLM_MODEL='xtuner_qwen3p5_vl_35b'
export DEFAULT_LAGENT_SRC='/mnt/shared-storage-user/huanghaian/code/gateway/lagent'
# python xtuner/v1/ray/environment/rl_task/runner.py --config projects/claw_bench/configs/calendar.py --limit 1
python -m tb2_eval.local_run --limit 1