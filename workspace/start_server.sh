export LMDEPLOY_PATH=/mnt/shared-storage-user/duanyanhui/workspace/code/lmdeploy
export PYTHONPATH=$LMDEPLOY_PATH:$PYTHONPATH
model=/mnt/shared-storage-user/llmit1/user/lvchengqi/ckpt/xtuner_v1/qwen3_5/cpt_qwen3_5_moe_30b_a3b_adamw_full_math_sft_mtp_g64_mtpfactor1/20260317125314/hf-5030-fuck
# export QWEN3_MODEL_PATH='/mnt/shared-storage-user/llmrazor-share/model/Qwen3-VL-30B-A3B-Instruct_MOE'
# model=$QWEN3_MODEL_PATH
port=24546
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
# model=internlm/Intern-S1

# NCCL_CUMEM_ENABLE=0 \
# CUDA_LAUNCH_BLOCKING=1 \

# UVICORN_LOG_LEVEL=error \
# RAY_DEBUG=1 \

# model=/nvme3/InternS1-235b-rc23-fp8-remote

#  CUDA_LAUNCH_BLOCKING=1 \

# UVICORN_LOG_LEVEL=error \
# RAY_ADDRESS='10.130.8.158:8345' \
cd /mnt/shared-storage-user/duanyanhui/workspace/code/lmdeploy
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lmdeploy serve api_server  \
$model \
--backend pytorch \
--server-port $port \
--log-level INFO \
--tp 2 \
--logprobs-mode "raw_logprobs" \
--enable-abort-handling \
--allow-terminate-by-client \
--enable-return-routed-experts
# --logprobs-mode raw_logprobs \
# --enable-prefix-caching \
# --distributed-executor-backend ray \



# LMDEPLOY_DP_MASTER_ADDR=10.130.8.158 \
# LMDEPLOY_DP_MASTER_PORT=29666 \
# RAY_DEBUG=1 \
# RAY_ADDRESS='10.130.8.158:8345' \
# HF_HOME=/nvme4/huggingface_hub \
# TRANSFORMERS_OFFLINE=1 \
# CUDA_VISIBLE_DEVICES=5,6 \
# lmdeploy serve api_server \
# Qwen/Qwen3-30B-A3B-FP8 \
# --backend pytorch \
# --server-port 24545 \
# --log-level INFO \
# --dp 4  \
# --ep 4 \
# --proxy-url http://0.0.0.0:24444 