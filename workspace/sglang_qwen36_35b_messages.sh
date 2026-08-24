#!/usr/bin/env bash
set -euo pipefail

# API_KEY= bash xxxxx
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/llmrazor-share/model/Qwen3.6-35B-A3B}"
MODEL_NAME="${MODEL_NAME:-xtuner_qwen3.6-35b-a3b}"
PORT="${PORT:-30001}"
# Set API_KEY="" to disable SGLang bearer-token authentication.
API_KEY="${API_KEY-sk-admin}"
# Claude Code expects plain text blocks on some internal requests. Leave this
# empty by default to avoid emitting Anthropic "thinking" content blocks.
REASONING_PARSER="${REASONING_PARSER:-}"
LOG_FILE="${LOG_FILE:-/tmp/sglang_qwen36_35b_a3b.log}"

API_KEY_ARGS=()
if [[ -n "${API_KEY}" ]]; then
  API_KEY_ARGS=(--api-key "${API_KEY}")
fi

REASONING_PARSER_ARGS=()
if [[ -n "${REASONING_PARSER}" ]]; then
  REASONING_PARSER_ARGS=(--reasoning-parser "${REASONING_PARSER}")
fi

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --served-model-name "${MODEL_NAME}" \
  "${API_KEY_ARGS[@]}" \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --context-length 262144 \
  "${REASONING_PARSER_ARGS[@]}" \
  --tool-call-parser qwen3_coder \
  --enable-fused-qk-norm-rope \
  --skip-server-warmup \
  >"${LOG_FILE}" 2>&1 &

echo "SGLang PID: $!"
echo "Log: ${LOG_FILE}"
echo
echo "curl /v1/messages after the server is ready:"
if [[ -n "${API_KEY}" ]]; then
  AUTH_HEADER="  -H 'Authorization: Bearer ${API_KEY}' \\\\"
else
  AUTH_HEADER="  -H 'Authorization: Bearer any-key-when-auth-is-disabled' \\\\"
fi
cat <<EOF
curl -sS http://10.102.249.52:${PORT}/v1/messages \\
  -H 'Content-Type: application/json' \\
${AUTH_HEADER}
  -d '{
    "model": "${MODEL_NAME}",
    "max_tokens": 256,
    "temperature": 0,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Reply with exactly: pong"}
        ]
      }
    ]
  }'
EOF
