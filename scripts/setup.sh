#!/usr/bin/env bash
set -euo pipefail

# 1. Install toolchains (python, uv) via proto
npx moon setup

# 2. Ensure proto PATH is configured in the user's shell rc file
PROTO_HOME="${PROTO_HOME:-$HOME/.proto}"
MARKER="# proto"

# Detect shell rc file
case "$(basename "${SHELL:-/bin/bash}")" in
  zsh)  RC_FILE="$HOME/.zshrc" ;;
  bash)
    if [[ "$(uname)" == "Darwin" ]]; then
      RC_FILE="$HOME/.bash_profile"
    else
      RC_FILE="$HOME/.bashrc"
    fi
    ;;
  fish) RC_FILE="$HOME/.config/fish/config.fish" ;;
  *)    RC_FILE="$HOME/.profile" ;;
esac

# Already configured — skip
if [[ -f "$RC_FILE" ]] && grep -qF "$MARKER" "$RC_FILE"; then
  exit 0
fi

# Append PATH config
if [[ "$(basename "${SHELL:-/bin/bash}")" == "fish" ]]; then
  mkdir -p "$(dirname "$RC_FILE")"
  SNIPPET='
# proto
set -gx PROTO_HOME "$HOME/.proto"
fish_add_path "$PROTO_HOME/bin"'
else
  SNIPPET='
# proto
export PROTO_HOME="$HOME/.proto"
export PATH="$PROTO_HOME/bin:$PATH"'
fi

echo "$SNIPPET" >> "$RC_FILE"
echo "proto PATH added to $RC_FILE (restart your shell or run: source $RC_FILE)"
