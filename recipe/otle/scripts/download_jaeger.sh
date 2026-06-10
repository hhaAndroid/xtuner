#!/usr/bin/env bash
set -euo pipefail

VERSION="${JAEGER_VERSION:-2.19.0}"
ARCHIVE_NAME="jaeger-${VERSION}-linux-amd64.tar.gz"
INSTALL_DIR="${JAEGER_HOME:-/tmp/jaeger}"
URL="${JAEGER_DOWNLOAD_URL:-https://github.com/jaegertracing/jaeger/releases/download/v${VERSION}/${ARCHIVE_NAME}}"

mkdir -p "${INSTALL_DIR}"

if [[ -x "${INSTALL_DIR}/jaeger-${VERSION}-linux-amd64/jaeger" ]]; then
  echo "Jaeger already exists: ${INSTALL_DIR}/jaeger-${VERSION}-linux-amd64/jaeger"
  exit 0
fi

echo "Downloading ${URL}"
curl -L "${URL}" -o "${INSTALL_DIR}/${ARCHIVE_NAME}"
tar -xzf "${INSTALL_DIR}/${ARCHIVE_NAME}" -C "${INSTALL_DIR}"
echo "Installed: ${INSTALL_DIR}/jaeger-${VERSION}-linux-amd64/jaeger"
