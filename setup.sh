#!/bin/sh
# DualOCR setup for macOS and Linux: installs everything it needs and builds the app.
# Usage: ./setup.sh [--help for options]
set -eu
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
  else
    echo "==> Installing uv, which sets up Python for DualOCR (https://docs.astral.sh/uv/)"
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://astral.sh/uv/install.sh | sh
    else
      echo "Setup stopped: install curl (or wget) and run ./setup.sh again." >&2
      exit 1
    fi
    PATH="$HOME/.local/bin:$PATH"
  fi
fi

echo "==> Installing Python and DualOCR's Python packages"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"   # slow connections: big packages (Qt) take a while
uv sync --locked --extra build
exec uv run --no-sync python scripts/setup.py "$@"
