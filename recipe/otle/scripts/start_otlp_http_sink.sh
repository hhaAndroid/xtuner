#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ROOT="${OTLP_SINK_ROOT:-/tmp/otelcol}"
HOST="${OTLP_SINK_HOST:-0.0.0.0}"
PORT="${OTLP_SINK_PORT:-4318}"

mkdir -p "${ROOT}"
exec python "${RECIPE_DIR}/tools/otlp_http_sink.py" --root "${ROOT}" --host "${HOST}" --port "${PORT}"
