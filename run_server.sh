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
# On Apple Silicon uv is effectively required: it is the only way to fetch a native
# arm64 interpreter. Without it the script stops with an explanation rather than
# building an x86_64 venv that dies later inside an onnxruntime wheel error.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

HOST="${DL_SERVER__HOST:-127.0.0.1}"
PORT="${DL_SERVER__PORT:-8000}"
RELOAD=""

# `--host` with no value would otherwise expand $2 under `set -u` and abort with
# "unbound variable", which reads like a bug in the script rather than a typo.
need_value() { [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; exit 2; }; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) need_value "$@"; HOST="$2"; shift 2 ;;
    --port) need_value "$@"; PORT="$2"; shift 2 ;;
    --reload) RELOAD="--reload --reload-dir src"; shift ;;
    # Print the comment block above, stopping at the first line of code. A fixed
    # line range drifts the moment anything is inserted, and used to print
    # `set -euo pipefail` and the `cd` as if they were help text.
    -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# --- prerequisites -----------------------------------------------------------
for bin in ffmpeg ffprobe; do
  command -v "$bin" >/dev/null || { echo "ERROR: '$bin' not found. Install FFmpeg (brew install ffmpeg)." >&2; exit 1; }
done

# Is this Apple Silicon hardware?
#
# NOT `uname -m`: this script runs under `env bash`, and a Homebrew bash from
# /usr/local is an x86_64 binary, so the whole script is Rosetta-translated and
# `uname -m` answers "x86_64" on an M-series Mac. That silently skipped the
# arm64 branch below and produced a venv whose onnxruntime cannot be imported -
# the exact failure the branch exists to prevent. hw.optional.arm64 asks the
# hardware and is unaffected by translation.
is_apple_silicon() {
  [[ "$(uname -s)" == "Darwin" ]] && [[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" == "1" ]]
}

# Install the package into .venv, whichever tool built it.
#
# `uv venv` creates NO pip binary, so calling .venv/bin/pip there exits 127 and
# `set -e` takes the whole script down - which is what the repair path below used
# to do on every uv-created venv, i.e. the default on Apple Silicon.
# `python -m pip` is used rather than the pip script because the module can be
# present when the console entry point is not.
install_package() {
  if command -v uv >/dev/null; then
    uv pip install -p .venv/bin/python -e ".[dev]"
  elif .venv/bin/python -m pip --version >/dev/null 2>&1; then
    .venv/bin/python -m pip install -e ".[dev]"
  else
    .venv/bin/python -m ensurepip --upgrade >/dev/null
    .venv/bin/python -m pip install -e ".[dev]"
  fi
}

# --- virtualenv --------------------------------------------------------------
if [[ ! -x .venv/bin/python ]]; then
  echo "Creating .venv ..."
  if command -v uv >/dev/null; then
    if is_apple_silicon; then
      # Homebrew/Rosetta Pythons under /usr/local are x86_64; onnxruntime (faster-whisper VAD)
      # has no macOS x86_64 wheels, so force a native arm64 interpreter.
      uv python install cpython-3.12-macos-aarch64-none >/dev/null
      PY=$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*-macos-aarch64-none/bin/python3.12 | tail -1)
      uv venv .venv -p "$PY"
    else
      uv venv .venv -p 3.12
    fi
  else
    # No uv on Apple Silicon means no reliable way to get a native interpreter:
    # `python3` here is usually Homebrew's universal build, and when this script
    # is itself Rosetta-translated it launches x86_64. Stop with an explanation
    # rather than fail minutes later inside an onnxruntime wheel error.
    if is_apple_silicon; then
      echo "ERROR: uv is required on Apple Silicon and was not found." >&2
      echo "       faster-whisper needs onnxruntime, which ships no macOS x86_64 wheels," >&2
      echo "       and only uv can fetch a guaranteed-native arm64 Python here." >&2
      echo "       Fix: brew install uv   (then re-run)" >&2
      exit 1
    fi
    python3 -m venv .venv
  fi
  # onnxruntime has no macOS x86_64 wheels, and a universal python launched from a
  # translated shell resolves to x86_64 - which installs happily and only fails on
  # `import onnxruntime`, deep inside the first transcription. Catch it here.
  if is_apple_silicon && ! .venv/bin/python -c 'import platform,sys; sys.exit(platform.machine()!="arm64")'; then
    echo "ERROR: .venv was built with a non-arm64 Python ($(.venv/bin/python -V 2>&1))." >&2
    echo "       faster-whisper's onnxruntime would install as x86_64 and fail to import." >&2
    echo "       Remove .venv, install uv (brew install uv), and re-run." >&2
    exit 1
  fi
  install_package
fi

# Ensure the package is importable (e.g. after a fresh clone with an old venv).
.venv/bin/python -c "import dialogue_locator" 2>/dev/null || install_package

# --- run ---------------------------------------------------------------------
echo "Dialogue Locator -> http://${HOST}:${PORT}   (docs: /docs, UI: /, Ctrl+C to stop)"
export DL_SERVER__HOST="$HOST" DL_SERVER__PORT="$PORT"
if [[ -n "$RELOAD" ]]; then
  exec .venv/bin/python -m uvicorn dialogue_locator.api.app:create_app --factory \
    --host "$HOST" --port "$PORT" $RELOAD
else
  exec .venv/bin/dialogue-locator-server
fi
