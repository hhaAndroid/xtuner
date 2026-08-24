#!/usr/bin/env bash

set -ex

LMDEPLOY_REPO="/mnt/shared-storage-user/huanghaian/code/lmdeploy"
MODEL_PATH="/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
MODEL_NAME="${MODEL_NAME:-hha_xtuner_qwen35_35b}"
SERVER_NAME="${SERVER_NAME:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-23333}"
BACKEND="${BACKEND:-pytorch}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3coder}"
REASONING_PARSER="${REASONING_PARSER:-default}"
EXTRA_ARGS=()

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH=/usr/local/nvidia/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${LMDEPLOY_REPO}:${PYTHONPATH:-}"

cd "${LMDEPLOY_REPO}"

exec python -c 'from lmdeploy.cli import run; run()' \
    serve api_server "${MODEL_PATH}" \
    --server-name "${SERVER_NAME}" \
    --server-port "${SERVER_PORT}" \
    --model-name "${MODEL_NAME}" \
    --backend "${BACKEND}" \
    --log-level "${LOG_LEVEL}" \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --reasoning-parser "${REASONING_PARSER}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
