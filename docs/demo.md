# Demos

The repo includes a **web app** and **CLI** to run scenario-based demos that show Janus blocking attacks. Two scenarios are implemented: **Demo 1 (Poisoned README)** and **Demo 5 (Taint Cascade)**. The agent’s prompts and LLM responses are hardcoded for reproducible playback; network and git tools are mocked. **Janus enforcement is real**: policy and PDE checks run against actual tool arguments and SpiceDB.

## Web app (split-panel)

Left panel: same scenario without Janus (attacks succeed). Right panel: with Janus (blocks disallowed tool calls). Both panels show a chat-style log and a terminal with tool calls and Janus ALLOW/BLOCK decisions.

**Setup and run:**

```bash
uv venv
uv pip install -e ".[langchain,dev]"
uv pip install fastapi "uvicorn[standard]" websockets pyyaml authzed grpcutil

# Demo 5 only: start SpiceDB
cd demos && docker compose up -d && cd ..

uv run uvicorn demos.app:app --reload
```

Open http://localhost:8000, pick a scenario from the dropdown, and click Start Demo.

See `demos/README.md` for full setup (including uv install and tests).

## CLI (no browser)

Run a single scenario, protected or unprotected:

```bash
# Demo 1 — Poisoned README (JSON policy)
uv run python -m examples.run demo1_poisoned_readme --unprotected
uv run python -m examples.run demo1_poisoned_readme --protected

# Demo 5 — Taint Cascade (PDE + SpiceDB); requires SpiceDB up
uv run python -m examples.run demo5_taint_cascade --protected
```

## Implemented scenarios

| Scenario | Description | Engine | What Janus blocks |
|----------|-------------|--------|-------------------|
| **Demo 1 — Poisoned README** | Agent summarizes a repo; README contains hidden instructions to read `.env` and exfiltrate via `fetch_url`. | JSON Schema (`PolicyEnforcer`) | `read_file` on `.env` (pattern), `fetch_url` to non-whitelisted domains. |
| **Demo 5 — Taint Cascade** | Agent fixes a bug from a GitHub issue; issue contains an instruction to add a malicious webhook and push. | SpiceDB + taint (`PDEEnforcer`) | After `fetch_url` (medium risk), taint rises to 40; `git_push` (limit 20) is blocked. |

## Layout

- **`examples/`** — Scenario framework: shared events, mock tools, scripted LLM, scenario runner. Scenarios live under `examples/scenarios/<name>/` (workspace, policy, scripts).
- **`demos/`** — FastAPI app, static UI, `docker-compose.yml` for SpiceDB.
- **`Policy-Discovery-Engine/`** — SpiceDB schema, bootstrap, taint limits, and `GraphInterceptor` used by Demo 5.

More scenarios (Demos 2–4, 6–10) are planned; see `DEMOS.md` and `demos/TODO.md`.
