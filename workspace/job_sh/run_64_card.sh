set -ex
# ray stop --force
# cd /mnt/shared-storage-user/duanyanhui/workspace/dev/xtuner-interns2-refactor
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
cd /mnt/shared-storage-user/duanyanhui/workspace/dev/xtuner-dev-trace-store/xtuner
source /mnt/shared-storage-user/llmit/user/lvchengqi/uv_venvs/interns2_rl_397b/bin/activate

export PYTHONUNBUFFERED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export HF_HUB_CACHE=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub
export HUGGINGFACE_HUB_CACHE=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1

bootcamp_dir=/mnt/shared-storage-user/llmit/user/guyuzhe/20250708/lvchengqi/projects/moe_rl/InternBootcamp
# lmdeploy_dir=/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_rl/mtp_rl_dev/lmdeploy
lmdeploy_dir=/mnt/shared-storage-user/llmit/user/lvchengqi/projects/interns2_xtuner_rl_design/lmdeploy
#xtuner_dir=/mnt/shared-storage-user/duanyanhui/workspace/dev/xtuner-refactor-v2
xtuner_dir=/mnt/shared-storage-user/duanyanhui/workspace/dev/xtuner-dev-trace-store/xtuner
intern_s2_delivery_dir=/mnt/shared-storage-user/duanyanhui/workspace/dev/xtuner-interns2-refactor/refactor/crg_rl_projects/src

export PYTHONPATH=$intern_s2_delivery_dir:$lmdeploy_dir:$xtuner_dir:$bootcamp_dir:$PYTHONPATH 
export NLTK_DATA=/mnt/shared-storage-user/llmit/user/lishuaibin/mv2yidian/nltk_data

export XTUNER_USE_FA3=1
export XTUNER_MAX_CONCURRENCY=8192
export RAY_MAX_CONCURRENCY=8192
export XTUNER_LOG_LEVEL="INFO"
export UVICORN_LOG_LEVEL="CRITICAl"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export XTUNER_USE_SGLANG=0
export XTUNER_USE_LMDEPLOY=1
export XTUNER_USE_VLLM=0

export MASTER_PORT=6000
export WORLD_SIZE=$NODE_COUNT
export RANK=$NODE_RANK
# export WORLD_SIZE=1
# export RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PERMUTE_COMPUTE_DTYPE=fp32

if [ "$XTUNER_USE_SGLANG" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
fi

export MAX_CONCURRENT=128
export ENABLE_PARTIAL_ROLLOUT=1
export TAIL_BATCH_CANDIDATE_STEPS=2
export TAIL_BATCH_TRIGGER_SIZE=10000000
export STALENESS_THRESHOLD=1
export TRAIN_OPTIMIZER_STEPS=8

current_time=$(date "+%m%d%H")

# export CONFIG_PATH='/mnt/shared-storage-user/duanyanhui/workspace/dev/xtuner-interns2-refactor/refactor/crg_rl_projects/src/intern_s1_delivery/configs/interns2_preview_rldesign/interns2-35ba3-base05-20260424a-rl-data260426rc1-56k-badword-mtp4-raw-data.py'
export CONFIG_PATH='/mnt/shared-storage-user/duanyanhui/workspace/dev/xtuner-interns2-refactor/refactor/crg_rl_projects/src/intern_s1_delivery/configs/interns2_preview_rldesign/interns2-35ba3-base05-20260424a-rl-data260426rc1-56k-badword-mtp4-0528.py'
# export WORK_DIR='/mnt/shared-storage-user/duanyanhui/workspace/dev/xtuner-interns2-refactor/log-128card-0503'
export WORK_DIR="/mnt/shared-storage-user/llmrazor-share/duanyanhui/workspace/dev/xtuner-interns2-refactor/64card-delivery-exp-0529-${current_time}"
export LMDEPLOY_LOG_FILE="/mnt/shared-storage-user/llmrazor-share/duanyanhui/workspace/dev/xtuner-interns2-refactor/64card-delivery-exp-0529-${current_time}/lmdeploy_log_${current_time}.txt"

if [ ! -d "$WORK_DIR" ]; then
  mkdir -p "$WORK_DIR"
fi

# ray 环境变量
export RAY_MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_RANK=${RANK:-0} # 0 代表主节点, >0 代表工作节点
export RAY_HEAD_PORT=${RAY_HEAD_PORT:-"6379"}
export RAY_CLIENT_PORT=${RAY_CLIENT_PORT:-"10001"}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-"8265"}
# export RAY_TEMP_DIR=${RAY_TEMP_DIR:-"/tmp/xtuner_ray_${ray_temp_owner}_${RAY_HEAD_PORT}_$$"}
# mkdir -p "$RAY_TEMP_DIR"

# TODO: support NPU RL Memory Monitor
export XTUNER_RL_MEM_DIR="${WORK_DIR}/mem_${current_time}"
export XTUNER_RL_MEM_INTERVAL="${XTUNER_RL_MEM_INTERVAL:-60}"
export XTUNER_RL_OBJECT_LIMIT="${XTUNER_RL_OBJECT_LIMIT:-5000}"
export XTUNER_RL_OBJECT_TOP_K="${XTUNER_RL_OBJECT_TOP_K:-10}"

node_count=${NODE_COUNT:-1}
total_cpus=$((node_count * 128))

if [ "$RAY_RANK" -eq 0 ]; then
  rm -rf /tmp/ray_log
  export RAY_LOG_DIR="${WORK_DIR}/ray_${current_time}/"
  mkdir -p ${RAY_LOG_DIR}
  ln -sfn "${RAY_LOG_DIR}" /tmp/ray_log
  ray start --head \
    --node-ip-address="$RAY_MASTER_ADDR" \
    --port="$RAY_HEAD_PORT" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --include-dashboard=true \
    --disable-usage-stats \
    --temp-dir="/tmp/ray_log/"
else
  sleep 60
  ray start --address="$RAY_MASTER_ADDR:$RAY_HEAD_PORT" --block --disable-usage-stats
fi

sleep 60

SCRIPT_NAME=$(basename "$0")
cp "$0" "${WORK_DIR}/${SCRIPT_NAME}"
cp "$CONFIG_PATH" "${WORK_DIR}/config.py"

export NCCL_IB_DISABLE=0 \
NCCL_SOCKET_IFNAME=bond0 \
NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
NCCL_IB_GID_INDEX=3 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_IB_GID_INDEX=3 \
NCCL_DEBUG=INFO

# 4. Submit training job on Head node
if [ "$RAY_RANK" -eq 0 ]; then
  RUNTIME_ENV_JSON="{
      \"env_vars\": {
        \"XTUNER_USE_FA3\": \"${XTUNER_USE_FA3}\",
        \"XTUNER_MAX_CONCURRENCY\": \"${XTUNER_MAX_CONCURRENCY}\",
        \"RAY_MAX_CONCURRENCY\": \"${RAY_MAX_CONCURRENCY}\",
        \"XTUNER_LOG_LEVEL\": \"${XTUNER_LOG_LEVEL}\",
        \"PYTHONPATH\": \"${PYTHONPATH}\",
        \"MASTER_ADDR\": \"${RAY_MASTER_ADDR}\",
        \"XTUNER_USE_SGLANG\": \"${XTUNER_USE_SGLANG:-}\",
        \"XTUNER_USE_LMDEPLOY\": \"${XTUNER_USE_LMDEPLOY:-}\",
        \"XTUNER_USE_VLLM\": \"${XTUNER_USE_VLLM:-}\",
        \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE:-}\",
        \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-}\",
        \"NCCL_IB_HCA\": \"${NCCL_IB_HCA:-}\",
        \"NCCL_IB_GID_INDEX\": \"${NCCL_IB_GID_INDEX:-}\",
        \"NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME\": \"${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-}\",
        \"PERMUTE_COMPUTE_DTYPE\": \"${PERMUTE_COMPUTE_DTYPE:-}\",
        \"NVSHMEM_IB_GID_INDEX\": \"${NVSHMEM_IB_GID_INDEX:-}\",
        \"NCCL_DEBUG\": \"${NCCL_DEBUG:-}\",
        \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF:-}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE\": \"1\",
        \"CUBLAS_WORKSPACE_CONFIG\": \":16:8\"
      }
    }"

  ray job submit --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" \
       --runtime-env-json="$RUNTIME_ENV_JSON" \
       -- /mnt/shared-storage-user/llmit/user/lvchengqi/uv_venvs/interns2_rl_397b/bin/python $intern_s2_delivery_dir/intern_s1_delivery/main.py \
       $CONFIG_PATH \
       --num-workers $((node_count * 8)) \
       --work-dir $WORK_DIR \
       2>&1 | tee -a "${WORK_DIR}/training_log.txt"
  # echo $WORK_DIR
  echo "训练任务提交完成。日志文件: ${WORK_DIR}/training_log.txt"
fi
