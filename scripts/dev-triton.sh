#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_PYTHON="$ROOT_DIR/packages/worker/.venv/bin/python"

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

TRITON_SERVER_BIN="${TRITON_SERVER_BIN:-tritonserver}"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.05-py3}"
TRITON_CONTAINER_NAME="${TRITON_CONTAINER_NAME:-triton-playground-dev}"
MODEL_REPOSITORY_ROOT="$(resolve_path "${MODEL_REPOSITORY_ROOT:-model_repository}")"
HF_HOME="$(resolve_path "${HF_HOME:-.cache/huggingface}")"
USE_DOCKER=0

export MODEL_REPOSITORY_ROOT
export HF_HOME

if ! [[ -x "$WORKER_PYTHON" ]]; then
  echo "worker virtualenv is missing at $WORKER_PYTHON. Run bun install first." >&2
  exit 1
fi

if ! command -v "$TRITON_SERVER_BIN" >/dev/null 2>&1; then
  if command -v docker >/dev/null 2>&1; then
    USE_DOCKER=1
    echo "Triton binary not found; falling back to Docker ($TRITON_IMAGE)." >&2
  else
    echo "Neither tritonserver binary nor docker was found." >&2
    echo "Install NVIDIA Triton locally, set TRITON_SERVER_BIN, or install Docker." >&2
    exit 1
  fi
fi

mkdir -p "$MODEL_REPOSITORY_ROOT" "$HF_HOME"

if [[ "${TRITON_SKIP_PREPARE:-0}" != "1" ]]; then
  (
    cd "$ROOT_DIR/packages/worker"
    prepare_args=(
      -m pipeline.prepare_models
      --profile "${MODEL_PROFILE:-baseline}"
      --output-root "$MODEL_REPOSITORY_ROOT"
    )

    if [[ "${MATERIALIZE_MANUAL_MODELS:-0}" == "1" ]]; then
      prepare_args+=(--materialize-manual-models)
    fi

    if [[ -n "${MODEL_PREPARE_FLAGS:-}" ]]; then
      # MODEL_PREPARE_FLAGS is intentionally shell-split so operators can pass
      # explicit opt-in flags such as --materialize-manual-models.
      # shellcheck disable=SC2206
      extra_prepare_args=(${MODEL_PREPARE_FLAGS})
      prepare_args+=("${extra_prepare_args[@]}")
    fi

    PYTHONPATH=. "$WORKER_PYTHON" "${prepare_args[@]}"
  )
fi

TRITON_RUNTIME_PROFILE="${TRITON_RUNTIME_PROFILE:-${MODEL_PROFILE:-baseline}}"
TRITON_DEV_IMAGE="triton-playground-dev:${TRITON_RUNTIME_PROFILE}"

if [[ "${MODEL_PROFILE:-baseline}" == "localize" ]]; then
  if [[ "$USE_DOCKER" == "0" ]]; then
    echo "Launching the opt-in localization repository with the host Triton binary." >&2
    echo "Ensure the Triton Python backend environment already has torch, transformers, qwen-tts, and related runtime packages." >&2
  fi
fi

if [[ "$USE_DOCKER" == "1" ]]; then
  if [[ "$TRITON_RUNTIME_PROFILE" != "baseline" ]]; then
    if docker image inspect "$TRITON_DEV_IMAGE" >/dev/null 2>&1; then
      echo "Using existing Triton dev image '$TRITON_DEV_IMAGE'. Run 'docker build -t $TRITON_DEV_IMAGE --build-arg TRITON_RUNTIME_PROFILE=$TRITON_RUNTIME_PROFILE -f packages/triton/Dockerfile .' to rebuild." >&2
    else
      echo "Building Triton dev image for profile '$TRITON_RUNTIME_PROFILE'..." >&2
      docker build -q \
        -t "$TRITON_DEV_IMAGE" \
        --build-arg "TRITON_RUNTIME_PROFILE=$TRITON_RUNTIME_PROFILE" \
        -f "$ROOT_DIR/packages/triton/Dockerfile" \
        "$ROOT_DIR" >&2
    fi
    TRITON_IMAGE="$TRITON_DEV_IMAGE"
  fi

  docker_args=(
    run --rm
    --name "$TRITON_CONTAINER_NAME"
    -v "$MODEL_REPOSITORY_ROOT:/models"
    --shm-size 2g
    -p "${TRITON_HTTP_PORT:-18000}:${TRITON_HTTP_PORT:-18000}"
    -p "${TRITON_GRPC_PORT:-18001}:${TRITON_GRPC_PORT:-18001}"
    -p "${TRITON_METRICS_PORT:-18002}:${TRITON_METRICS_PORT:-18002}"
  )

  if [[ "${TRITON_NO_GPU:-0}" != "1" ]]; then
    docker_args+=(--gpus all)
  fi

  exec docker "${docker_args[@]}" "$TRITON_IMAGE" \
    tritonserver \
    --model-repository=/models \
    --http-port="${TRITON_HTTP_PORT:-18000}" \
    --grpc-port="${TRITON_GRPC_PORT:-18001}" \
    --metrics-port="${TRITON_METRICS_PORT:-18002}" \
    --exit-on-error=false \
    --strict-readiness=false
else
  exec "$TRITON_SERVER_BIN" \
    --model-repository="$MODEL_REPOSITORY_ROOT" \
    --http-port="${TRITON_HTTP_PORT:-18000}" \
    --grpc-port="${TRITON_GRPC_PORT:-18001}" \
    --metrics-port="${TRITON_METRICS_PORT:-18002}" \
    --exit-on-error=false \
    --strict-readiness=false
fi
