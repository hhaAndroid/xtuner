#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy all_proxy no_proxy ftp_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY FTP_PROXY

CONFIG_PATH="/mnt/shared-storage-user/huanghaian/code/xtuner/workspace/job_sh/agent_localhost_rl_qwen3p5_397b_rl.py"
DATA_PATH="/mnt/shared-storage-user/huanghaian/code/agent_dev/crg_rl_projects/scripts/configs/dataset_metas/math_coder.json"
EXPERIMENT_NAME="agent_397b_rl_code"

export GLOBAL_BATCH_SIZE=8
export PROMPT_REPEAT_K=4
BASE_WORK_DIR="/mnt/shared-storage-user/huanghaian/code/xtuner/work_dirs_agent/exp/insterns2"
WORK_DIR="$(realpath -m "${BASE_WORK_DIR}/${EXPERIMENT_NAME}")"
export DEBUG_ROLLOUT_DIR="${WORK_DIR}/debug_rollout"
export DEBUG_TRAIN=False
export DEBUG_ROLLOUT=False
export ONLY_CALC_MISMATCH_RATIO=1


export DATA_PATH
RUN_SCRIPT_PATH="$(realpath -m "${BASH_SOURCE[0]}")"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"

# User-local paths and service addresses. Edit these lines when moving repos,
# changing Python envs, or switching model/service endpoints.
PYTHON_ENV="/mnt/shared-storage-user/llmit/user/lvchengqi/uv_venvs/interns2_rl_397b"
CRG_PATH="/mnt/shared-storage-user/huanghaian/code/agent_dev/crg_rl_projects"
CRG_PATH_SRC="/mnt/shared-storage-user/huanghaian/code/agent_dev/crg_rl_projects/src"

XTUNER_PATH="/mnt/shared-storage-user/huanghaian/code/xtuner"
LAGENT_PATH="/mnt/shared-storage-user/huanghaian/code/agent_dev/lagent"
LMDEPLOY_PATH="/mnt/shared-storage-user/huanghaian/code/lmdeploy"
MODEL_PATH="/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/interns2/sft_official/sft_interns2_base02_20260614a_lr2e5_512gpu/20260615054647/hf-2312/"
RL_LLM_BASE_URL="http://s-20260605163846-jznwc-decode.ailab-evalservice.svc:8000/v1"
COMPASS_VERIFIER_V2_HOSTS="10.102.237.13:23333,10.102.237.13:23334,10.102.237.13:23335,10.102.237.13:23336,10.102.237.13:23337,10.102.237.13:23338,10.102.237.13:23339,10.102.237.13:23340"
SERPER_MCP_URLS="http://10.102.103.157:8091/mcp,http://10.102.103.155:8096/mcp,http://10.102.103.155:8092/mcp,http://10.102.103.155:8095/mcp,http://10.102.103.155:8097/mcp,http://10.102.103.155:8098/mcp,http://10.102.103.155:8094/mcp,http://10.102.103.155:8093/mcp"
JINA_MCP_URLS="http://10.102.103.155:8104/mcp,http://10.102.103.155:8100/mcp,http://10.102.103.148:8101/mcp,http://10.102.103.157:8105/mcp,http://10.102.103.155:8099/mcp,http://10.102.103.155:8102/mcp,http://10.102.103.155:8103/mcp,http://10.102.103.155:8106/mcp"
SANDBOX_PROVIDER_KEY="lkk-as8dHd2Q"

source "${PYTHON_ENV}/bin/activate"


export PATH="/usr/local/nvidia/bin/:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${CRG_PATH}:${XTUNER_PATH}:${LAGENT_PATH}:${LMDEPLOY_PATH}:${CRG_PATH_SRC}"
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1

export WORK_DIR
export EXPERIMENT_NAME
export XTUNER_RUN_ID="$RUN_ID"

export MODEL_PATH
export QWEN3P5_VL_MODEL_PATH="$MODEL_PATH"
export MODEL_NAME="xtuner_train_${EXPERIMENT_NAME}_${RUN_ID}"
export RL_LLM_MODEL="$MODEL_NAME"
export RL_LLM_BASE_URL
export XTUNER_ROUTED_API_BASE_URL="$RL_LLM_BASE_URL"

world_size="${NODE_COUNT:-${WORLD_SIZE:-1}}"
rank="${NODE_RANK:-${RANK:-0}}"
ray_master_addr="${MASTER_ADDR:-${RAY_MASTER_ADDR:-127.0.0.1}}"
ray_head_port="${RAY_HEAD_PORT:-6379}"
ray_dashboard_port="${RAY_DASHBOARD_PORT:-8265}"
gpus_per_node="${GPUS_PER_NODE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-$((world_size * gpus_per_node))}"
export LOCALHOST_AGENT_SAMPLE_TIMEOUT_S="${LOCALHOST_AGENT_SAMPLE_TIMEOUT_S:-7200}"

export XTUNER_USE_LMDEPLOY=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export XTUNER_USE_FA3=1
export LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1
export XTUNER_LOG_LEVEL="INFO"
export COMPASS_VERIFIER_V2_HOSTS
export SERPER_MCP_URLS
export JINA_MCP_URLS
export SANDBOX_PROVIDER_KEY

export XTUNER_ASYNCIO_DIAGNOSTICS=1

echo "CONFIG: $CONFIG_PATH"
echo "DATA_PATH: $DATA_PATH"
echo "WORK_DIR: $WORK_DIR"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "RUN_ID: $RUN_ID"
echo "PYTHON_ENV: $PYTHON_ENV"
echo "MODEL: $MODEL_PATH"
echo "MODEL_NAME: $MODEL_NAME"
echo "CRG_PATH: $CRG_PATH"
echo "XTUNER_PATH: $XTUNER_PATH"
echo "LAGENT_PATH: $LAGENT_PATH"
echo "LMDEPLOY_PATH: $LMDEPLOY_PATH"
echo "BASE_WORK_DIR: $BASE_WORK_DIR"
echo "TRAIN: steps=${TOTAL_TRAIN_STEPS:-<cfg default>}, batch=${GLOBAL_BATCH_SIZE:-<cfg default>}, repeat=${PROMPT_REPEAT_K:-<cfg default>}, concurrent=${MAX_CONCURRENT_SAMPLES:-<cfg default>}, localhost_sample_timeout=${LOCALHOST_AGENT_SAMPLE_TIMEOUT_S}s"
echo "RANK/WORLD_SIZE: ${rank}/${world_size}"
echo "GPUS_PER_NODE: $gpus_per_node"
echo "NUM_WORKERS: $NUM_WORKERS"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p "$WORK_DIR"

hash -r
ulimit -n 65536
ray stop --force || true

export LMDEPLOY_LOG_FILE="${WORK_DIR}/lmdeploy_log.txt"
export XTUNER_RL_MEM_DIR="${WORK_DIR}/mem_${RUN_ID}"

if [[ "$rank" -eq 0 ]]; then
  rm -rf /tmp/ray_log
  export RAY_LOG_DIR="${WORK_DIR}/ray_${RUN_ID}"
  mkdir -p "$RAY_LOG_DIR"
  ln -sfn "$RAY_LOG_DIR" /tmp/ray_log
  ray start --head \
    --node-ip-address="$ray_master_addr" \
    --port="$ray_head_port" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="$ray_dashboard_port" \
    --include-dashboard=true \
    --disable-usage-stats \
    --temp-dir="/tmp/ray_log/"
else
  until curl --connect-timeout 2 "http://${ray_master_addr}:${ray_dashboard_port}" >/dev/null 2>&1; do
    echo "Waiting for Ray master at ${ray_master_addr}:${ray_dashboard_port}..."
    sleep 2
  done
  ray start --address="${ray_master_addr}:${ray_head_port}" --block --disable-usage-stats
fi

cp "$RUN_SCRIPT_PATH" "${WORK_DIR}/$(basename "$0")"
cp "$CONFIG_PATH" "${WORK_DIR}/config.py"
LOG_FILE="${WORK_DIR}/training_log_${RUN_ID}.txt"
echo "LOG_FILE: $LOG_FILE"

export NCCL_IB_DISABLE=0 \
NCCL_SOCKET_IFNAME=bond0 \
NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
NCCL_IB_GID_INDEX=3 \
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 \
NVSHMEM_IB_GID_INDEX=3 \
NCCL_DEBUG=INFO

export NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NVSHMEM_IBGDA_NUM_RC_PER_PE=8
export NVSHMEM_IB_TRAFFIC_CLASS=186
export NVSHMEM_DISABLE_NVLs=1
export NCCL_IB_TC=186
export DEEPEP_MAX_TOKENS_PER_RANK=128
python -m xtuner.v1.train.cli.rl --config "$CONFIG_PATH" 2>&1 | tee -a "$LOG_FILE"
