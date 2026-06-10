#!/usr/bin/env bash
set -euo pipefail

SESSION="${JAEGER_TMUX_SESSION:-jaeger-memory}"

echo "tmux:"
tmux list-sessions 2>/dev/null | grep -F "${SESSION}" || true

echo
echo "process:"
pgrep -af "jaeger-.*/jaeger --config=file:" || true

echo
echo "services:"
curl -fsS http://127.0.0.1:16686/api/services || true
echo
