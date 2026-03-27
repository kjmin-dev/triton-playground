#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR/packages/worker"
exec .venv/bin/uvicorn api.main:app --reload --host "${WORKER_HOST:-0.0.0.0}" --port "${WORKER_PORT:-8080}"
