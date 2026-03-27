#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_PID=""

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

resolve_path() {
  local value="$1"

  case "$value" in
    /*)
      printf '%s\n' "$value"
      ;;
    ~/*)
      printf '%s\n' "$HOME/${value#~/}"
      ;;
    *)
      printf '%s\n' "$ROOT_DIR/$value"
      ;;
  esac
}

TRITON_CONTAINER_NAME="${TRITON_CONTAINER_NAME:-triton-playground-dev}"

triton_server_available() {
  local triton_server_bin="${TRITON_SERVER_BIN:-tritonserver}"

  if [[ "$triton_server_bin" == */* ]] || [[ "$triton_server_bin" == "~/"* ]]; then
    [[ -x "$(resolve_path "$triton_server_bin")" ]]
    return
  fi

  command -v "$triton_server_bin" >/dev/null 2>&1 && return 0
  command -v docker >/dev/null 2>&1
}

cleanup() {
  if [[ -n "$TRITON_PID" ]] && kill -0 "$TRITON_PID" >/dev/null 2>&1; then
    kill "$TRITON_PID" >/dev/null 2>&1 || true
    wait "$TRITON_PID" || true
  fi
  docker rm -f "$TRITON_CONTAINER_NAME" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

if [[ "${SKIP_TRITON:-0}" != "1" ]]; then
  if triton_server_available; then
    bash "$ROOT_DIR/scripts/dev-triton.sh" &
    TRITON_PID="$!"

    sleep 1

    if ! kill -0 "$TRITON_PID" >/dev/null 2>&1; then
      wait "$TRITON_PID"
    fi
  else
    echo "Triton server binary '${TRITON_SERVER_BIN:-tritonserver}' was not found." >&2
    echo "Starting worker + web without local Triton. Set TRITON_SERVER_BIN to the installed binary, set SKIP_TRITON=1 to suppress this warning, or use docker compose up --build." >&2
  fi
fi

cd "$ROOT_DIR"
npx moon run :dev
