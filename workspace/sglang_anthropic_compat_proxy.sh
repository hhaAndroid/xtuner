#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:-http://127.0.0.1:30001/v1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30002}"
LOG_FILE="${LOG_FILE:-/tmp/sglang_anthropic_compat_proxy.log}"

python workspace/sglang_anthropic_compat_proxy.py \
  --upstream-base-url "${UPSTREAM_BASE_URL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  >"${LOG_FILE}" 2>&1 &

echo "Started SGLang Anthropic compatibility proxy"
echo "  upstream : ${UPSTREAM_BASE_URL}"
echo "  listen   : http://${HOST}:${PORT}/v1"
echo "  log      : ${LOG_FILE}"
echo "  pid      : $!"
