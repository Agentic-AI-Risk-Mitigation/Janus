# Getting Started

## Install

Requires Python ≥ 3.11. [uv](https://docs.astral.sh/uv/) is the recommended package manager.

```bash
uv add janus-guard
```

Optional extras for providers and adapters:

```bash
uv add "janus-guard[anthropic]"   # Anthropic Claude
uv add "janus-guard[google]"      # Google Gemini
uv add "janus-guard[langchain]"   # LangChain adapter
uv add "janus-guard[adk]"         # Google ADK adapter
uv add "janus-guard[all]"         # Everything
```

Install from source:

```bash
git clone https://github.com/Agentic-AI-Risk-Mitigation/Janus.git
cd janus
uv pip install -e .
```

Copy `.env.example` to `.env` and fill in the API keys for the providers you plan to use:

```bash
cp .env.example .env
```

To build docs locally (from the project root): `uv sync --extra docs` then `uv run mkdocs serve`.

## Quick Demo (Under 5 Minutes)

**Web app (split-panel, recommended):**

```bash
uv pip install -e ".[langchain,dev]"
uv pip install fastapi "uvicorn[standard]" websockets pyyaml authzed
cd examples && docker compose up -d && cd ..   # only needed for PDE-backed scenarios
uv run uvicorn examples.app:app --reload
```

Open http://localhost:8000, select a scenario, and click Start Demo. See [Demo](demo.md) for details.

**CLI (single scenario):**

```bash
uv run python -m examples.run coding_agent_poisoned_readme --protected
```

This runs the Poisoned README scenario: Janus blocks `read_file` on `.env` and `fetch_url` to attacker URLs. See [Demo](demo.md) for all scenarios and the web app.

## Minimal Example

```python
from janus import JanusAgent

agent = JanusAgent(
    model="openai/gpt-4o",
    api_key="sk-...",  # or set OPENAI_API_KEY
    use_builtin_tools=True,
    policy="policies.json",
    system_prompt="You are a helpful coding assistant.",
)

response = agent.run("List the Python files in the project.")
print(response)
```

Create a `policies.json` file that allows the tools your agent needs. See [Policy Reference](policy-reference.md) for format.

## How to Run Examples

Scenarios and the demo framework live under `examples/`. The current catalog includes:

- `ecommerce_ipi`
- `banking_traces`
- `fintech_planning`
- `healthcare_taint`
- `coding_agent_poisoned_readme`
- `coding_agent_supply_chain`
- `coding_agent_taint_cascade`

1. **Install**: From the project root, ensure dependencies are installed (see [Demo](demo.md) or `examples/README.md` for the full list, including `langchain`, `authzed`, `grpcutil` for PDE).

2. **CLI**: Run a scenario via the runner:

   ```bash
   uv run python -m examples.run <scenario_name> [--protected | --unprotected]
   ```

   Example: `uv run python -m examples.run coding_agent_poisoned_readme --protected`

3. **PDE-backed scenarios**: Start SpiceDB first:

   ```bash
   cd examples && docker compose up -d && cd ..
   uv run python -m examples.run coding_agent_taint_cascade --protected
   ```

4. **Web app**: Run the split-panel demo with `uv run uvicorn examples.app:app --reload` and open http://localhost:8000. See [Demo](demo.md) and `examples/README.md`.

5. **Tests**: Run the example test suite with `uv run pytest tests/test_examples/ -v`.
