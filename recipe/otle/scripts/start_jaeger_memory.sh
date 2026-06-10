#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VERSION="${JAEGER_VERSION:-2.19.0}"
INSTALL_DIR="${JAEGER_HOME:-/tmp/jaeger}"
SESSION="${JAEGER_TMUX_SESSION:-jaeger-memory}"
CONFIG="${JAEGER_CONFIG:-${RECIPE_DIR}/jaeger/jaeger-memory.yaml}"
BIN="${INSTALL_DIR}/jaeger-${VERSION}-linux-amd64/jaeger"

if [[ ! -x "${BIN}" ]]; then
  "${SCRIPT_DIR}/download_jaeger.sh"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already running: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "env -u OTEL_EXPORTER_OTLP_ENDPOINT -u OTEL_EXPORTER_OTLP_TRACES_ENDPOINT -u OTEL_TRACES_EXPORTER -u OTEL_SERVICE_NAME '${BIN}' --config=file:'${CONFIG}'"

echo "Started Jaeger memory in tmux session: ${SESSION}"
echo "UI:        http://127.0.0.1:16686/"
echo "OTLP HTTP: http://127.0.0.1:14318/v1/traces"
echo "OTLP gRPC: 127.0.0.1:14317"
