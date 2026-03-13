# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Janus

Janus (`janus-guard` on PyPI) is a system-level security layer for LLM agents that enforces fine-grained policies on tool calls before execution. It intercepts at the tool-call boundary — not at prompt or output level — to defend against indirect prompt injection, credential exfiltration, and overprivileged agent actions.

## Commands

```bash
# Setup
uv sync --extra dev              # Install with dev dependencies
uv sync --extra all --extra dev  # Install with all providers + dev

# Tests
uv run pytest                    # Run all tests
uv run pytest tests/test_examples/test_demo1_enforcement.py  # Single file
uv run pytest -k "test_read_file"                            # Pattern match

# Lint & Format
uv run ruff check .              # Lint
uv run ruff check --fix .        # Auto-fix
uv run ruff format .             # Format
uv run mypy janus                # Type check

# Docs
uv run mkdocs serve              # Local docs server

# Demos
uv run python -m examples.run demo1_poisoned_readme --protected    # CLI demo
uv run uvicorn examples.app:app --reload                           # Web demo UI
```

## Architecture

The core flow is: **JanusAgent → LLMRunner → ToolRegistry → PolicyEnforcer**

1. `JanusAgent` (`janus/agent.py`) wires together the enforcer, tool registry, LLM runner, and provider
2. `LLMRunner` (`janus/llm/runner.py`) drives the conversation loop: messages → LLM → tool calls → results → repeat
3. `ToolRegistry` (`janus/tools/registry.py`) dispatches tool calls but **always calls `enforcer.enforce()` first**
4. `PolicyEnforcer` (`janus/policy/enforcer.py`) evaluates rules in priority order against JSON Schema conditions

When a tool is blocked, the `PolicyViolation` is caught and returned as a string to the LLM, allowing it to adjust rather than crash.

### Two Enforcement Engines

- **Janus engine** (default, `janus/policy/enforcer.py`): Stateless JSON Schema rule evaluation. Rules have priority, effect (allow/deny), conditions (JSON Schema per argument), and fallback action.
- **PDE engine** (`janus/policy/pde_enforcer.py` + `janus/policy/pde/`): SpiceDB-backed ReBAC with runtime taint tracking. Taint accumulates as the agent reads untrusted sources, progressively disabling higher-risk tools. Config/constants in `pde/config.py`, interceptor in `pde/interceptor.py`, discovery engine in `pde/discovery.py`, bootstrap utilities in `pde/bootstrap.py`.

### Key Design Decisions

- **Single canonical tool representation**: Tools are defined once as `ToolDef`/`ToolParam` and converted to provider-specific schemas via `.to_openai_schema()`, `.to_pydantic_model()`, etc.
- **Default-deny when policy loaded**: Tools not listed in a loaded policy are blocked. Unlisted arguments in conditions are skipped (known issue — see TODO.md #1).
- **No global state**: Every enforcer, registry, and runner is independent and safe for concurrent use.
- **Priority ordering**: Lower priority values evaluate first. Convention: manual rules 1–10, LLM-generated rules 100+.

### Adapters

`janus/adapters/langchain.py` and `janus/adapters/adk.py` wrap framework-native tool execution with Janus enforcement. The shared base (`janus/adapters/_base.py`) provides `resolve_enforcer()` and `make_guarded_handler()`.

## Ruff Config

Line length 100, target Python 3.11, rules: E, F, I, UP. E501 ignored.

## Known Issues

See `TODO.md` for the full backlog. Critical ones:
- Missing argument bypass in `enforcer.py` — if LLM omits a restricted argument, the condition is skipped
- Broad exception catching in `runner.py` — all exceptions become error strings
- Hardcoded SpiceDB token in `pde_enforcer.py`
- No unit tests for `PolicyEnforcer` rule evaluation logic
