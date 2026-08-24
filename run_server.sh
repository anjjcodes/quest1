#!/usr/bin/env bash
# Start the Dialogue Locator backend (FastAPI API + web UI).
#
#   ./run_server.sh                 # http://127.0.0.1:8000
#   ./run_server.sh --port 9000     # custom port
#   ./run_server.sh --host 0.0.0.0  # reachable from other machines
#   ./run_server.sh --reload        # auto-reload on code changes (dev)
#
# Any DL_* environment variable (or a .env file in this directory) overrides config, e.g.
#   DL_WHISPER__FAST_MODEL=base DL_DOWNLOAD__MAX_HEIGHT=360 ./run_server.sh
#
# First run creates .venv (native arm64 Python 3.12 via uv on Apple Silicon) and installs deps.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

HOST="${DL_SERVER__HOST:-127.0.0.1}"
PORT="${DL_SERVER__PORT:-8000}"
RELOAD=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --reload) RELOAD="--reload --reload-dir src"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# --- prerequisites -----------------------------------------------------------
for bin in ffmpeg ffprobe; do
  command -v "$bin" >/dev/null || { echo "ERROR: '$bin' not found. Install FFmpeg (brew install ffmpeg)." >&2; exit 1; }
done

# --- virtualenv --------------------------------------------------------------
if [[ ! -x .venv/bin/python ]]; then
  echo "Creating .venv ..."
  if command -v uv >/dev/null; then
    if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
      # Homebrew/Rosetta Pythons under /usr/local are x86_64; onnxruntime (faster-whisper VAD)
      # has no macOS x86_64 wheels, so force a native arm64 interpreter.
      uv python install cpython-3.12-macos-aarch64-none >/dev/null
      PY=$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*-macos-aarch64-none/bin/python3.12 | tail -1)
      uv venv .venv -p "$PY"
    else
      uv venv .venv -p 3.12
    fi
    uv pip install -p .venv/bin/python -e ".[dev]"
  else
    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"
  fi
fi

# Ensure the package is importable (e.g. after a fresh clone with an old venv).
.venv/bin/python -c "import dialogue_locator" 2>/dev/null || .venv/bin/pip install -q -e ".[dev]"

# --- run ---------------------------------------------------------------------
echo "Dialogue Locator -> http://${HOST}:${PORT}   (docs: /docs, UI: /, Ctrl+C to stop)"
export DL_SERVER__HOST="$HOST" DL_SERVER__PORT="$PORT"
if [[ -n "$RELOAD" ]]; then
  exec .venv/bin/python -m uvicorn dialogue_locator.api.app:create_app --factory \
    --host "$HOST" --port "$PORT" $RELOAD
else
  exec .venv/bin/dialogue-locator-server
fi
