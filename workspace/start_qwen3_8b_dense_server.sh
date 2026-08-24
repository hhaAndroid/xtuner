#!/usr/bin/env bash
set -euo pipefail

LMDEPLOY_PATH=${LMDEPLOY_PATH:-/mnt/shared-storage-user/huanghaian/code/lmdeploy}
if [[ ! -d "${LMDEPLOY_PATH}" ]]; then
    LMDEPLOY_PATH=/mnt/shared-storage-user/duanyanhui/workspace/code/lmdeploy
fi
export PYTHONPATH="${LMDEPLOY_PATH}:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-${QWEN3_PATH:-/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B}}
PORT=${PORT:-24546}
TP_SIZE=${TP_SIZE:-4}
LOG_LEVEL=${LOG_LEVEL:-INFO}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-1024}
SESSION_LEN=${SESSION_LEN:-1536}
SKIP_WARMUP=${SKIP_WARMUP:-1}

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

if [[ "${SKIP_WARMUP}" == "1" ]]; then
    export LMD_SKIP_WARMUP=1
    export LMDEPLOY_SKIP_WARMUP=1
else
    unset LMD_SKIP_WARMUP
    unset LMDEPLOY_SKIP_WARMUP
fi

echo "LMDEPLOY_PATH=${LMDEPLOY_PATH}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PORT=${PORT}"
echo "TP_SIZE=${TP_SIZE}"
echo "SKIP_WARMUP=${SKIP_WARMUP}"

cd "${LMDEPLOY_PATH}"
lmdeploy serve api_server \
    "${MODEL_PATH}" \
    --backend pytorch \
    --server-port "${PORT}" \
    --log-level "${LOG_LEVEL}" \
    --tp "${TP_SIZE}" \
    --session-len "${SESSION_LEN}" \
    --max-batch-size "${MAX_BATCH_SIZE}" \
    --cache-max-entry-count "${GPU_MEMORY_UTILIZATION}" \
    --logprobs-mode raw_logprobs \
    --enable-abort-handling \
    --allow-terminate-by-client
