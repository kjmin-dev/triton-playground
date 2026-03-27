#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_PYTHON="$ROOT_DIR/packages/worker/.venv/bin/python"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

if ! [[ -x "$WORKER_PYTHON" ]]; then
  echo "worker virtualenv is missing at $WORKER_PYTHON. Run bun install first." >&2
  exit 1
fi

cd "$ROOT_DIR/packages/worker"
exec env PYTHONPATH=. "$WORKER_PYTHON" -u -m pipeline.download_weights "$@"
