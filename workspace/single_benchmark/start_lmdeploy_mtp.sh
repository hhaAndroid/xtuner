#!/usr/bin/env bash

set -euo pipefail
set -x

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LMDEPLOY_REPO="${LMDEPLOY_REPO:-/mnt/shared-storage-user/huanghaian/code/lmdeploy}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307}"
MODEL_NAME="${MODEL_NAME:-hha_xtuner_qwen35_35b}"
SERVER_NAME="${SERVER_NAME:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-23333}"
API_KEY="${API_KEY:-}"

TP="${TP:-1}"
DP="${DP:-2}"
EP="${EP:-2}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-3}"
SESSION_LEN="${SESSION_LEN:-65536}"
MAX_PREFILL_TOKEN_NUM="${MAX_PREFILL_TOKEN_NUM:-8192}"
CACHE_MAX_ENTRY_COUNT="${CACHE_MAX_ENTRY_COUNT:-0.8}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
UVICORN_LOG_LEVEL="${UVICORN_LOG_LEVEL:-error}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PATH="/usr/local/nvidia/bin/:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${LMDEPLOY_REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export UVICORN_LOG_LEVEL
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# LMDeploy's Ray executor requires one DP rendezvous endpoint for EP/DP > 1.
if [[ -z "${LMDEPLOY_DP_MASTER_ADDR:-}" ]]; then
    if HOST_IP="$(hostname -i 2>/dev/null)" && [[ -n "${HOST_IP}" ]]; then
        export LMDEPLOY_DP_MASTER_ADDR="${HOST_IP}"
    elif HOST_IP="$(hostname -I 2>/dev/null)" && [[ -n "${HOST_IP}" ]]; then
        HOST_IP="${HOST_IP%% *}"
        export LMDEPLOY_DP_MASTER_ADDR="${HOST_IP}"
    else
        export LMDEPLOY_DP_MASTER_ADDR="127.0.0.1"
    fi
fi
export LMDEPLOY_DP_MASTER_PORT="${LMDEPLOY_DP_MASTER_PORT:-29666}"

# Required by DLBlas DeepEP token dispatcher during warmup.
export DEEPEP_MAX_TOKENS_PER_RANK="${DEEPEP_MAX_TOKENS_PER_RANK:-$((MAX_BATCH_SIZE * (1 + SPECULATIVE_NUM_DRAFT_TOKENS)))}"

cd "${LMDEPLOY_REPO}"

ARGS=(
    serve api_server "${MODEL_PATH}"
    --trust-remote-code
    --backend pytorch
    --model-name "${MODEL_NAME}"
    --server-name "${SERVER_NAME}"
    --server-port "${SERVER_PORT}"
    --max-batch-size "${MAX_BATCH_SIZE}"
    --cache-max-entry-count "${CACHE_MAX_ENTRY_COUNT}"
    --speculative-algorithm qwen3_5_mtp
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
    --distributed-executor-backend ray
    --dp "${DP}"
    --ep "${EP}"
    --tp "${TP}"
    --dtype bfloat16
    --session-len "${SESSION_LEN}"
    --max-prefill-token-num "${MAX_PREFILL_TOKEN_NUM}"
    --logprobs-mode raw_logprobs
    --tool-call-parser qwen3coder
    --reasoning-parser default
    --enable-return-routed-experts
    --enable-abort-handling
    --log-level "${LOG_LEVEL}"
)

if [[ -n "${API_KEY}" ]]; then
    ARGS+=(--api-keys "${API_KEY}")
fi

exec python -m lmdeploy "${ARGS[@]}" "$@"
