# Janus Examples

Modular demo scenarios showcasing Janus security enforcement across industries.

## Showcase Scenarios

Four marquee capabilities demonstrated across mixed industries and personas:

| Scenario | Marquee Capability | Industry | Persona | Experience |
|---|---|---|---|---|
| `ecommerce_ipi` | Indirect Prompt Injection Defense | E-commerce | Customer support ops analyst | LangChain Single-Agent |
| `healthcare_taint` | Dynamic Trust (Taint + SpiceDB) | Healthcare | Clinical operations coordinator | ADK Multi-Agent |
| `banking_traces` | Policy Enforcement Traces | Banking | Fraud investigation analyst | ADK Multi-Agent |
| `fintech_planning` | Iterative Tool Looping | Personal Finance | Individual end-user | LangChain Single-Agent |

### Legacy Scenarios

| Scenario | Focus |
|---|---|
| `demo1_poisoned_readme` | Hidden AI_INSTRUCTION in README exfiltrates .env secrets |
| `demo2_supply_chain` | Typosquat package README injects credential-harvesting backdoor |
| `demo5_taint_cascade` | External GitHub issue raises taint, blocks git push |

## Structure

```
examples/
├── config.yaml              # Global config (LLM provider, playback timing)
├── app.py                   # FastAPI + WebSocket server
├── static/index.html        # Split-panel web UI
├── shared/                  # Reusable framework code
│   ├── events.py            # Event types streamed to frontend
│   ├── scripted_llm.py      # Mock LangChain ChatModel (scripted responses)
│   ├── mock_tools.py        # Mock tool handlers (file I/O, network, git)
│   ├── industry_tools.py    # Industry-specific mock tools (ecommerce, healthcare, banking, fintech)
│   ├── tool_defs.py         # ToolDef factories
│   ├── scenario_base.py     # BaseScenario abstract class
│   └── scenario_runner.py   # Orchestrates scenarios, emits events
├── scenarios/               # One subdirectory per demo
│   ├── ecommerce_ipi/       # IPI defense showcase
│   ├── healthcare_taint/    # Dynamic trust showcase
│   ├── banking_traces/      # Policy traces showcase
│   ├── fintech_planning/    # Iterative loops showcase
│   ├── demo1_poisoned_readme/
│   ├── demo2_supply_chain/
│   └── demo5_taint_cascade/
└── run.py                   # CLI runner
```

## Web UI

The web UI features:
- **Experience selector** — filter by LangChain Single-Agent or ADK Multi-Agent
- **Metadata badges** — industry, persona, and marquee capability for each scenario
- **Split-panel comparison** — "Without Janus" vs "With Janus" side-by-side
- **Taint meter** — live taint level visualization for PDE scenarios
- **Enforcement trace** — every tool call shows ALLOW/BLOCK with policy reasoning

```bash
uv run uvicorn examples.app:app --reload
# Open http://localhost:8000
```

## Adding a New Scenario

1. Create `examples/scenarios/<name>/` with:
   - `__init__.py`
   - `scenario.py` — class inheriting `BaseScenario`
   - `scripts.py` — `UNPROTECTED_SCRIPT` and `PROTECTED_SCRIPT` (lists of `AIMessage`)
   - `policy.json` — Janus policy (for `enforcer_type="janus"`)
   - `workspace/` — files the agent reads during the demo

2. In `scenario.py`, implement all abstract methods from `BaseScenario`:
   - `workspace_dir`, `get_tools()`, `get_policy()`
   - `get_unprotected_script()`, `get_protected_script()`
   - `get_system_prompt()` (optional override)
   - Set `experience`, `industry`, `persona`, `marquee_capability` for showcase metadata

3. The scenario is auto-discovered. Test with:
   ```bash
   python -m examples.run <name> --unprotected
   python -m examples.run <name> --protected
   ```

## CLI Usage

```bash
python -m examples.run --list                              # List all scenarios
python -m examples.run ecommerce_ipi --protected           # Run with Janus
python -m examples.run ecommerce_ipi --unprotected         # Run without Janus
python -m examples.run banking_traces --protected           # Banking policy traces
python -m examples.run fintech_planning --protected         # Iterative planning with guardrails
```

## Developer Walkthrough

For a step-by-step live-coding guide that shows how to create an agent,
add Janus policies, and enable taint tracking, see:

**[`docs/healthcare_taint_live_coding_walkthrough.md`](../docs/healthcare_taint_live_coding_walkthrough.md)**

This walkthrough uses the `healthcare_taint` scenario as a running example
and is designed for ~10 minutes of live demo or self-paced reading.

## Switching to Real LLM

In `config.yaml`, set `mode: "live"` and provide your API key env var.
In the scenario class, the `ScriptedChatModel` will be replaced with a real
LLM provider via `JanusLangChainAgent(model="openrouter/openai/gpt-4o")`.
