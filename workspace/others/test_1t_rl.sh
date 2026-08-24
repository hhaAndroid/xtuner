ray stop --force

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

source /mnt/shared-storage-user/llmit/user/lvchengqi/uv_venvs/xtuner_sglang/bin/activate

cd /mnt/shared-storage-user/huanghaian/code/temp/xtuner/

lmdeploy_dir=/mnt/shared-storage-user/llmit/user/lvchengqi/projects/agent_rl/lmdeploy
xtuner_dir=/mnt/shared-storage-user/huanghaian/code/temp/xtuner/

export PYTHONPATH=$lmdeploy_dir:$xtuner_dir:$PYTHONPATH 
export XTUNER_LOG_LEVEL="INFO"
export UVICORN_LOG_LEVEL="CRITICAl"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export QWEN25_MODEL_PATH=/mnt/shared-storage-user/llmrazor-share/model/Qwen2.5-Math-7B/
# export MODEL_PATH='/mnt/shared-storage-user/llmit/user/lvchengqi/ckpt/xpuyu/qwen3-30ba3b_cold-start/20250924081143/hf-170'
export DAPO_DATA_PATH=/mnt/shared-storage-user/llmrazor-share/data/dapo_math/dapo-math-17k.jsonl
export EVAL_DATA_PATH=/mnt/shared-storage-user/llmrazor-share/data/dapo_math/aime-2024.jsonl

export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-30B-A3B'
# export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'
# export QWEN3_MODEL_PATH='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/work_dirs/Qwen3-8B_gsm8k_sglang/20251118065035/hf-1'
# export QWEN3_MODEL_PATH="/mnt/shared-storage-user/llmit/user/lvchengqi/ckpt/release/interns1-mini-language-model"
export QWEN3_1T_MODEL='/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/Turner_1T/sft/official_Turner_1T_stable_20251118a_4k_0_0_2400_SFT_d027025/20251119161741/hf-2479'

export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
# export ROLLOUT_DEBUG_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train-mini.jsonl'
# export DAPO_EVAL_DATA_PATH=/mnt/shared-storage-user/llmrazor-share/data/dapo_math/aime-2024.jsonl
export ROLLOUT_DEBUG_DATA_PATH=''
export DAPO_EVAL_DATA_PATH=""
# export ROLLOUT_TEST_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'

export ENABLE_RETURN_ROUTED_EXPERTS=1
# export XTUNER_DETERMINISTIC=1
# export XTUNER_ENABLE_LOGPROB_ZERO_DIFF=1

# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen25_7B_dapo.py "lmdeploy" $QWEN25_MODEL_PATH $DAPO_DATA_PATH $ROLLOUT_DEBUG_DATA_PATH
# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3_30B_grpo_dapo.py "lmdeploy" $QWEN3_MODEL_PATH $DAPO_DATA_PATH $DAPO_EVAL_DATA_PATH
bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3_1t_grpo.py "lmdeploy" $QWEN3_1T_MODEL $ROLLOUT_DATA_PATH $DAPO_EVAL_DATA_PATH
# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3_8B_grpo.py "sglang" $QWEN3_MODEL_PATH $ROLLOUT_DATA_PATH $DAPO_EVAL_DATA_PATH