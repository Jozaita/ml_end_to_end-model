#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

uvicorn ml_end_to_end.web_app.server:app \
 --host "${UVICORN_HOST:-0.0.0.0}" --port "${UVICORN_PORT:-8001}" --workers "${UVICORN_WORKERS:-1}" 