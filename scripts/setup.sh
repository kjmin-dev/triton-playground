#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Install toolchains (python, uv) via proto
npx moon setup

# 2. Ensure proto PATH is configured in the user's shell rc file
PROTO_HOME="${PROTO_HOME:-$HOME/.proto}"
MARKER="# proto"
PROTO_PATH_SNIPPET='
# proto
export PROTO_HOME="$HOME/.proto"
export PATH="$PROTO_HOME/bin:$PROTO_HOME/shims:$PATH"'

append_snippet_if_missing() {
  local rc_file="$1"

  mkdir -p "$(dirname "$rc_file")"

  if [[ -f "$rc_file" ]] && grep -qF "$MARKER" "$rc_file"; then
    return
  fi

  echo "$PROTO_PATH_SNIPPET" >> "$rc_file"
  echo "proto PATH added to $rc_file"
}

# Detect shell rc file
case "$(basename "${SHELL:-/bin/bash}")" in
  zsh)  RC_FILE="$HOME/.zshrc" ;;
  bash)
    RC_FILE="$HOME/.bashrc"
    BASH_PROFILE="$HOME/.bash_profile"
    ;;
  fish) RC_FILE="$HOME/.config/fish/config.fish" ;;
  *)    RC_FILE="$HOME/.profile" ;;
esac

if [[ "$(basename "${SHELL:-/bin/bash}")" == "fish" ]]; then
  mkdir -p "$(dirname "$RC_FILE")"
  SNIPPET='
# proto
set -gx PROTO_HOME "$HOME/.proto"
fish_add_path "$PROTO_HOME/bin"
fish_add_path "$PROTO_HOME/shims"'

  if [[ ! -f "$RC_FILE" ]] || ! grep -qF "$MARKER" "$RC_FILE"; then
    echo "$SNIPPET" >> "$RC_FILE"
    echo "proto PATH added to $RC_FILE"
  fi
elif [[ "$(basename "${SHELL:-/bin/bash}")" == "bash" ]]; then
  append_snippet_if_missing "$RC_FILE"
  append_snippet_if_missing "$BASH_PROFILE"
else
  append_snippet_if_missing "$RC_FILE"
fi

export PATH="$PROTO_HOME/bin:$PROTO_HOME/shims:$PATH"

# 3. Bootstrap the worker virtualenv with the toolchain Python so moon tasks
# resolve against the repo-pinned interpreter on first use.
WORKER_DIR="$ROOT_DIR/packages/worker"
if [[ -f "$WORKER_DIR/pyproject.toml" ]] && [[ -x "$PROTO_HOME/bin/uv" ]] && [[ -x "$PROTO_HOME/bin/python" ]]; then
  (
    cd "$WORKER_DIR"

    if [[ ! -d ".venv" ]]; then
      "$PROTO_HOME/bin/uv" venv .venv --python "$PROTO_HOME/bin/python" --no-progress
      echo "bootstrapped $WORKER_DIR/.venv with $("$PROTO_HOME/bin/python" --version 2>&1)"
    fi

    if [[ "${SKIP_WORKER_SYNC:-0}" != "1" ]]; then
      "$PROTO_HOME/bin/uv" sync --python "$PROTO_HOME/bin/python" --no-progress
      echo "synced worker dependencies into $WORKER_DIR/.venv"
    fi
  )
fi

echo "restart your shell or run: source $RC_FILE"
