# Getting Started

## Install

Requires Python ≥ 3.11. [uv](https://docs.astral.sh/uv/) is the recommended package manager.

```bash
uv add janus-security
```

Optional extras for providers and adapters:

```bash
uv add "janus-security[anthropic]"   # Anthropic Claude
uv add "janus-security[google]"      # Google Gemini
uv add "janus-security[langchain]"   # LangChain adapter
uv add "janus-security[adk]"         # Google ADK adapter
uv add "janus-security[all]"         # Everything
```

Install from source:

```bash
git clone https://github.com/your-org/janus
cd janus
uv pip install -e .
```

To build docs locally (from the project root): `uv sync --extra docs` then `uv run mkdocs serve`.

## Quick Demo (Under 5 Minutes)

See the attack-and-block flow:

```bash
export OPENAI_API_KEY="sk-..."
uv run python examples/demo_poisoned_readme.py
```

This runs the poisoned README scenario: malicious instructions in a file, agent attempts exfiltration, Janus blocks. See [Demo](demo.md) for all scenarios.

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

Example scripts and demos live in the `examples/` directory (or as noted in the repo). To run them:

1. **Environment**: Set the API key for your LLM provider, e.g. `OPENAI_API_KEY` for OpenAI.
2. **Policy file**: Examples that use a policy expect a JSON file (e.g. `policies.json`) in the project root or a path you specify.
3. **Command**: From the project root:

   ```bash
   uv run python examples/<script_name>.py
   ```

4. **SpiceDB examples**: For examples using the SpiceDB engine, start SpiceDB first:

   ```bash
   docker compose -f janus/pde/docker-compose.yml up -d
   # Wait for SpiceDB to be ready, then run the example
   uv run python examples/spicedb_demo.py
   ```

5. **E2E integration test**: The full Janus + SpiceDB integration test:

   ```bash
   uv run pytest test_e2e_pde.py -v -s
   ```

Replace `<script_name>` with the actual example filename. If examples are added under a different path, the same pattern applies: `uv run python <path>/<script>.py`.
