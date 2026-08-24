#!/usr/bin/env bash

set -euo pipefail
set -x

# 源代码修复了好几个 v0.5.10 问题。
SGLANG_REPO="${SGLANG_REPO:-/mnt/shared-storage-user/huanghaian/code/slime_package/xtuner_sglang/sglang}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307}"
MODEL_NAME="${MODEL_NAME:-hha_xtuner_qwen35_35b}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-23333}"
API_KEY="${API_KEY:-}"

TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-EAGLE}"
SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-${MODEL_PATH}}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-none}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-auto}"
SPECULATIVE_MOE_A2A_BACKEND="${SPECULATIVE_MOE_A2A_BACKEND:-none}"
SPECULATIVE_MOE_RUNNER_BACKEND="${SPECULATIVE_MOE_RUNNER_BACKEND:-auto}"
DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH:-1}"
INCREMENTAL_STREAMING_OUTPUT="${INCREMENTAL_STREAMING_OUTPUT:-0}"
LOG_LEVEL="${LOG_LEVEL:-info}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PATH="/usr/local/nvidia/bin/:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SGLANG_REPO}/python:${SGLANG_REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"

if [[ "${MOE_A2A_BACKEND}" == "deepep" && "${MOE_RUNNER_BACKEND}" == "auto" ]]; then
    cat >&2 <<'EOF'
MOE_A2A_BACKEND=deepep with MOE_RUNNER_BACKEND=auto can hit SGLang's deprecated
forward_deepgemm_masked path for Qwen3.5 bf16 MoE in this fork. Use the default
MOE_A2A_BACKEND=none for the xtuner RL-compatible MTP path, or explicitly choose
a supported EP MoE runner before enabling DeepEP.
EOF
    exit 2
fi

cd "${SGLANG_REPO}"

ARGS=(
    --model-path "${MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --host "${SERVER_HOST}"
    --port "${SERVER_PORT}"
    --trust-remote-code
    --tp-size "${TP_SIZE}"
    --dp-size "${DP_SIZE}"
    --ep-size "${EP_SIZE}"
    --dtype "${DTYPE}"
    --context-length "${CONTEXT_LENGTH}"
    --chunked-prefill-size "${MAX_PREFILL_TOKENS}"
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --speculative-algorithm "${SPECULATIVE_ALGORITHM}"
    --speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL_PATH}"
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
    --mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY}"
    --moe-a2a-backend "${MOE_A2A_BACKEND}"
    --moe-runner-backend "${MOE_RUNNER_BACKEND}"
    --speculative-moe-a2a-backend "${SPECULATIVE_MOE_A2A_BACKEND}"
    --speculative-moe-runner-backend "${SPECULATIVE_MOE_RUNNER_BACKEND}"
    --tool-call-parser "${TOOL_CALL_PARSER}"
    --reasoning-parser "${REASONING_PARSER}"
    --enable-return-routed-experts
    --log-level "${LOG_LEVEL}"
)

if [[ -n "${API_KEY}" ]]; then
    ARGS+=(--api-key "${API_KEY}")
fi

if [[ "${DISABLE_PIECEWISE_CUDA_GRAPH}" == "1" || "${DISABLE_PIECEWISE_CUDA_GRAPH}" == "true" ]]; then
    ARGS+=(--disable-piecewise-cuda-graph)
fi

if [[ "${INCREMENTAL_STREAMING_OUTPUT}" == "1" || "${INCREMENTAL_STREAMING_OUTPUT}" == "true" ]]; then
    ARGS+=(--incremental-streaming-output)
fi

if [[ -n "${SGLANG_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(${SGLANG_EXTRA_ARGS})
    ARGS+=("${EXTRA_ARGS[@]}")
fi

exec python -m sglang.launch_server "${ARGS[@]}" "$@"
