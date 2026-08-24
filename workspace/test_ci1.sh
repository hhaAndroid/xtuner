ray stop --force

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

export QWEN3_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'
# export QWEN3_VL_DENSE_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-4B-Instruct-2507'
export INTERN_VL_1B_PATH="/mnt/shared-storage-user/llmrazor-share/model/InternVL3_5-1B-HF"
export QWEN3_VL_DENSE_PATH="/mnt/shared-storage-user/llmrazor-share/model/Qwen3-VL-4B-Instruct"
export QWEN3_4B_PATH="/mnt/shared-storage-user/llmrazor-share/model/Qwen3-4B-Instruct-2507"
# export QWEN3_VL_DENSE_PATH="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/f2981ed65cf8ac8b860135c9115dff9dd7c7c80c"
export QWEN3_VL_MOE_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-VL-30B-A3B-Instruct_MOE'
export GPT_OSS_MINI_PATH='/mnt/shared-storage-user/llmrazor-share/model/gpt-oss-20b-bf16'
export QWEN3_MOE_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-30B-A3B'
export QWEN3_MOE_FOPE_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3_30B_fope_g0.1_sephead'
export INTERNS1_DENSE_PATH='/mnt/shared-storage-user/llmrazor-share/model/intern-s1-mini/'
export QWEN3_5_MOE_PATH='/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'
export ROLLOUT_MODEL_PATH=$QWEN3_PATH
export ALPACA_PATH='/mnt/shared-storage-user/llmrazor-share/data/alpaca'
export VIDEO_ROOT='/mnt/shared-storage-user/llmrazor-share/data/images'
export GEO3K_MEDIA_ROOT='/mnt/shared-storage-user/llmrazor-share/data/geometry3k'

export INTERNS1_DATA_META='/mnt/shared-storage-user/llmrazor-share/data/vlm_ci_data.json'

export ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
export ROLLOUT_TEST_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/test.jsonl'

export DEEPSEEK_V3_PATH='/mnt/shared-storage-user/llmrazor-share/model/DeepSeek-V3.1'

export PYTHONPATH="$(pwd)"
# export PYTHONPATH=/mnt/shared-storage-user/caoweihan/duanyanhui/lmdeploy/:$PYTHONPATH
export XTUNER_USE_LMDEPLOY=1
# export XTUNER_USE_SGLANG=1

export VERL_ROLLOUT_DATA_PATH=/mnt/shared-storage-user/llmrazor-share/data/verl-rollout-step0.jsonl
export GEO_ROLLOUT_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/rl_test_judge_geo_data.jsonl'
export ROLLOUT_DAPO_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/rl_test_judger_dapo_math_data.jsonl'

export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export XTUNER_DETERMINISTIC=1

# --ignore tests/module/dispatcher/test_deepep.py 
pytest /mnt/shared-storage-user/huanghaian/code/xtuner/tests/