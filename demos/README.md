# Janus Demo Web Application

Split-panel web UI for presenting Janus demos. Left panel shows an agent
without Janus (attacks succeed). Right panel shows the same agent with Janus
(attacks blocked).

## Setup (from scratch after cloning)

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create venv and install dependencies (authzed includes grpcutil; do not install grpcutil separately)
uv venv
uv pip install -e ".[langchain,dev]" fastapi "uvicorn[standard]" websockets pyyaml authzed

# 3. (Demo 5 only) Start SpiceDB via Docker
cd demos && docker compose up -d && cd ..

# 4. Run the web app (must use `uv run` so the project venv is used)
uv run uvicorn demos.app:app --reload

# 5. Open http://localhost:8000 in your browser
```

## Running Tests

```bash
uv run pytest tests/test_examples/ -v
```

## Running Scenarios via CLI (no web UI)

```bash
# Demo 1 unprotected
uv run python -m examples.run demo1_poisoned_readme --unprotected

# Demo 1 protected
uv run python -m examples.run demo1_poisoned_readme --protected

# Demo 5 (requires SpiceDB running)
uv run python -m examples.run demo5_taint_cascade --protected
```

## Architecture

- `app.py` — FastAPI server with WebSocket endpoint
- `static/index.html` — Single-file dark mode UI (no build step)
- `docker-compose.yml` — SpiceDB for PDE-based demos (Demo 5)

The server imports scenarios from `examples/scenarios/` and runs them
concurrently for both panels, streaming events via WebSocket. PDE (SpiceDB + taint)
lives in `janus/policy/pde/`; Demo 5 bootstraps its schema when you run it.
