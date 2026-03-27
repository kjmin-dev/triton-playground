#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

export MODEL_REPOSITORY_ROOT="${MODEL_REPOSITORY_ROOT:-$ROOT_DIR/model_repository}"
export TRITON_GRPC_URL="${TRITON_GRPC_URL:-localhost:${TRITON_GRPC_PORT:-8001}}"

cd "$ROOT_DIR/packages/worker"
exec .venv/bin/uvicorn api.main:app --reload --host "${WORKER_HOST:-0.0.0.0}" --port "${WORKER_PORT:-8080}"
