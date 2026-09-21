# Getting Started

## Install

Requires Python ≥ 3.10. [uv](https://docs.astral.sh/uv/) is the recommended package manager.

```bash
uv add janus-guard
```

The core install is dependency-light — no SpiceDB/`authzed`, no web server — so
the standalone `PolicyEnforcer` (`from janus.policy import PolicyEnforcer`) can be
used as a tool-call gate without pulling in the optional stacks.

Optional extras for features, providers, and adapters:

```bash
uv add "janus-guard[pde]"         # SpiceDB-backed ReBAC + taint tracking (PDEEnforcer)
uv add "janus-guard[server]"      # Example web demo UI (FastAPI + uvicorn)
uv add "janus-guard[anthropic]"   # Anthropic Claude
uv add "janus-guard[google]"      # Google Gemini
uv add "janus-guard[langchain]"   # LangChain adapter
uv add "janus-guard[adk]"         # Google ADK adapter
uv add "janus-guard[claude]"      # Claude Agent SDK (Claude Code) adapter
uv add "janus-guard[all]"         # Everything
```

The Claude Code **CLI** adapter (`janus-hook`) needs no extra — it ships with the core
install, because a hook has to run wherever `claude` runs.

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
uv pip install -e ".[server,pde,langchain,dev]"
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

5. **Tests**: Run the regression suite with `uv run pytest`. It is fully offline — no LLM, no
   SpiceDB. The live SDK smoke suite is opt-in: `JANUS_LIVE_SMOKE=1 uv run pytest tests/smoke/ -v`
   (needs the `claude` CLI and API credentials).

## Guard Your Interactive Claude Code (Under 5 Minutes)

The `janus-hook` shim enforces a Janus policy on the interactive `claude` CLI via its
`PreToolUse` hook. It ships with the core install — no extra needed.

### The guided route

```bash
pip install janus-guard
janus init          # asks a few questions, writes everything, verifies it
```

`janus init` asks where to guard (this project or the whole machine), what the agent must
never touch, how much network egress it gets, and how strict to be. Every question has a
recommended default, so pressing Enter throughout produces the starter setup below. It
then shows the **exact** settings-file diff and asks before writing anything, backs up any
existing settings file, and finishes by running its decisions through the real hook path —
`curl … | sh` denied, `.env` denied, ordinary reads allowed — so a `PASS` means that policy
denied that call, not that the wizard intended to.

Re-running it updates the existing hook in place rather than adding a second one. Useful
flags: `--dry-run` (show everything, write nothing), `--yes` (accept every default,
for CI), `--scope project|project-local|user`, `--force` (overwrite an existing policy).

### Doing it by hand

The wizard automates exactly these four steps; do them yourself if you would rather.

1. **Self-test the install**:

   ```bash
   janus-hook doctor
   ```

2. **Write a policy** (e.g. `~/.claude/janus/policy.json`). Start from
   `examples/claude_code/policy.starter.json` — secrets-read and pipe-to-shell denies plus
   the built-in tool enumeration that `bypassPermissions` sessions require — or write your
   own: list only the tools Janus should have an opinion about, deny rules first, then an
   unconditional allow so everything else on that tool falls through:

   ```json
   {
     "Read": [
       {"priority": 1, "effect": 1, "fallback": 0,
        "conditions": {"file_path": {"type": "string",
                       "pattern": "(^|/)\\.env(?!\\.example)[^/]*$|/\\.ssh/"}}},
       {"priority": 10, "effect": 0, "conditions": {}, "fallback": 0}
     ]
   }
   ```

3. **Wire the hook** into `~/.claude/settings.json` (or a project's `.claude/settings.json`):

   ```json
   {
     "hooks": {
       "PreToolUse": [
         {"hooks": [{"type": "command",
                     "command": "janus-hook pre --policy ~/.claude/janus/policy.json --mode gate",
                     "timeout": 10}]}
       ]
     }
   }
   ```

   Set the `timeout` explicitly and keep it above the shim's `--deadline` (default 5s).
   The CLI's own hook timeout fails **open** — a deny that arrives after it is discarded
   and the tool runs — so the shim has to reach its deadline first and deny while it
   still can.

4. **Add the backstop** — `janus-hook backstop` prints a `permissions.deny` block to merge
   into the same settings file. It is the only layer that holds if hooks silently stop
   running.

Two behaviors to know before you deploy:

- **Gate mode promotes to strict default-deny under `bypassPermissions`** (including
  `--dangerously-skip-permissions`): abstaining in a session where no human will ever be
  asked would be a silent allow. If you run bypass sessions, the policy must enumerate
  every tool they use — an unlisted tool (built-in or MCP) is denied there, not deferred.
- **Settings-file delivery is not tamper-proof against the agent it guards** — settings
  hooks are re-read from disk, and `Bash` can rewrite any file a `Write`/`Edit` deny rule
  protects. Phase 1 is a policy monitor, not a reachability lockdown; see
  [Adapters → Claude Code CLI](adapters.md#claude-code-cli-interactive-claude) for the
  full security model, and
  [Claude Code Deployment](claude-code-deployment.md) for the delivery-vehicle threat
  model and troubleshooting.
