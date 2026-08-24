ray stop --force

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PYTHONPATH="$(pwd)"
export WORK_DIR='work_dirs_rl/xiaomi_rl_mix_data_ci'

export MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/MiMo-7B-RL'
export META_DATA_PATH='/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/meta_data/rl_mix_data.json'
 
export XTUNER_USE_FA3=0
# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3p5_vl_35B_grpo_mixdata.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $META_DATA_PATH
bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_mimo_7B_grpo_mixdata.py "sglang" $MODEL_PATH $META_DATA_PATH
