# Janus Examples

Modular demo scenarios showcasing Janus security enforcement.

## Structure

```
examples/
├── config.yaml              # Global config (LLM provider, playback timing)
├── shared/                  # Reusable framework code
│   ├── events.py            # Event types streamed to frontend
│   ├── scripted_llm.py      # Mock LangChain ChatModel (scripted responses)
│   ├── mock_tools.py        # Mock tool handlers (real file I/O, fake network)
│   ├── tool_defs.py         # ToolDef factories
│   ├── scenario_base.py     # BaseScenario abstract class
│   └── scenario_runner.py   # Orchestrates scenarios, emits events
├── scenarios/               # One subdirectory per demo
│   ├── demo1_poisoned_readme/
│   └── demo5_taint_cascade/
└── run.py                   # CLI runner
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

3. The scenario is auto-discovered. Test with:
   ```bash
   python -m examples.run <name> --unprotected
   python -m examples.run <name> --protected
   ```

## CLI Usage

```bash
python -m examples.run --list                          # List scenarios
python -m examples.run demo1_poisoned_readme --protected    # Run with Janus
python -m examples.run demo1_poisoned_readme --unprotected  # Run without Janus
```

## Switching to Real LLM

In `config.yaml`, set `mode: "live"` and provide your API key env var.
In the scenario class, the `ScriptedChatModel` will be replaced with a real
LLM provider via `JanusLangChainAgent(model="openrouter/openai/gpt-4o")`.
