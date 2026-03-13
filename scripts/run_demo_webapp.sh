#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${JANUS_DEMO_HOST:-127.0.0.1}"
PORT="${JANUS_DEMO_PORT:-8000}"
RELOAD=1
START_SPICEDB=0

usage() {
    cat <<'EOF'
Usage: scripts/run_demo_webapp.sh [--with-spicedb] [--host HOST] [--port PORT] [--no-reload]

Starts the Janus demo web app from the repository root.

Options:
  --with-spicedb  Start examples/docker-compose.yml first (needed for Demo 5)
  --host HOST     Bind host for uvicorn (default: 127.0.0.1)
  --port PORT     Bind port for uvicorn (default: 8000)
  --no-reload     Disable uvicorn autoreload
  -h, --help      Show this help text

Install demo dependencies first:
  uv sync --extra langchain --extra dev
  # or: uv sync --extra all --extra dev
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-spicedb)
            START_SPICEDB=1
            ;;
        --host)
            shift
            if [[ $# -eq 0 ]]; then
                echo "--host requires a value." >&2
                exit 1
            fi
            HOST="$1"
            ;;
        --port)
            shift
            if [[ $# -eq 0 ]]; then
                echo "--port requires a value." >&2
                exit 1
            fi
            PORT="$1"
            ;;
        --no-reload)
            RELOAD=0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but not installed." >&2
    exit 1
fi

if [[ ! -d .venv ]]; then
    cat <<'EOF'
No .venv directory found in the repo root.

Install demo dependencies first:
  uv sync --extra langchain --extra dev
  # or: uv sync --extra all --extra dev
EOF
    exit 1
fi

if [[ "$START_SPICEDB" -eq 1 ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "docker is required for --with-spicedb." >&2
        exit 1
    fi

    echo "Starting SpiceDB from examples/docker-compose.yml"
    (
        cd examples
        docker compose up -d
    )
fi

CMD=(uv run uvicorn examples.app:app --host "$HOST" --port "$PORT")
if [[ "$RELOAD" -eq 1 ]]; then
    CMD+=(--reload)
fi

echo "Starting Janus demo web app at http://$HOST:$PORT"
if [[ "$START_SPICEDB" -eq 1 ]]; then
    echo "SpiceDB is available on localhost:50051 for Demo 5."
fi

exec "${CMD[@]}"
