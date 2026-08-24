ray stop --force

export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# /mnt/shared-storage-user/duanyanhui/workspace/code/lmdeploy-main/lmdeploy/:
export PYTHONPATH="$(pwd):/mnt/shared-storage-user/huanghaian/code/temp/xtuner/mathruler":$PYTHONPATH 

export MODEL_PATH='/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/cadac78306af287f801b75a5565ede58f323f472'
# export MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/intern-s1-mini-hha-fix_tokenizer'
# export MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B'

# export DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/gsm8k/train.jsonl'
export DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/geometry3k/train.jsonl'
# export DATA_PATH='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/xtuner/test_rl.jsonl'
# export EVAL_DATA_PATH='/mnt/shared-storage-user/llmrazor-share/data/geometry3k/test.jsonl'
export EVAL_DATA_PATH=''
export MEDIA_ROOT='/mnt/shared-storage-user/llmrazor-share/data/geometry3k/'

export MMPR_DATA_PATH='/mnt/shared-storage-user/llmit/user/guyuzhe/20250708/wangweiyun/share_data/MMPR-250701/annotations_xtuner_format_multi_image.jsonl'
# export MMPR_DATA_PATH='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/xtuner/test_rl.jsonl'
# export MEDIA_ROOT='/mnt/shared-storage-user/llmit/user/guyuzhe/20250708/wangweiyun/share_data/MMPR-250701/'
# export MEDIA_ROOT=''

export WORK_DIR='work_dirs11/qwen3_8b_grpo_gsm8k'

bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen3_vl_8B_grpo.py "lmdeploy" $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH
# bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_interns1_mini_grpo.py "lmdeploy" $MODEL_PATH $DATA_PATH $EVAL_DATA_PATH
