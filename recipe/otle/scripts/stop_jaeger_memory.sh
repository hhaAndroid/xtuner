#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VERSION="${JAEGER_VERSION:-2.19.0}"
INSTALL_DIR="${JAEGER_HOME:-/tmp/jaeger}"
SESSION="${JAEGER_TMUX_SESSION:-jaeger-memory}"
CONFIG="${JAEGER_CONFIG:-${RECIPE_DIR}/jaeger/jaeger-memory.yaml}"
BIN="${INSTALL_DIR}/jaeger-${VERSION}-linux-amd64/jaeger"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

if [[ -x "${BIN}" ]]; then
  PIDS="$(pgrep -f "${BIN} --config=file:${CONFIG}" || true)"
  if [[ -n "${PIDS}" ]]; then
    echo "Stopping residual Jaeger process(es): ${PIDS}"
    kill -TERM ${PIDS} || true
    for _ in $(seq 1 20); do
      sleep 0.2
      PIDS="$(pgrep -f "${BIN} --config=file:${CONFIG}" || true)"
      [[ -z "${PIDS}" ]] && break
    done
    if [[ -n "${PIDS}" ]]; then
      echo "Residual process(es) still running after TERM: ${PIDS}" >&2
      exit 1
    fi
  fi
fi

echo "Jaeger memory stopped."
