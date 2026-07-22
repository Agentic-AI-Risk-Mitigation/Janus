# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Janus

Janus (`janus-guard` on PyPI) is a system-level security layer for LLM agents that enforces fine-grained policies on tool calls before execution. It intercepts at the tool-call boundary — not at prompt or output level — to defend against indirect prompt injection, credential exfiltration, and overprivileged agent actions.

## Commands

```bash
# Setup
uv sync --extra dev              # Core library + lint/type/test tooling
uv sync --extra all --extra dev  # Full demo/provider stack + dev

# Validation
./scripts/run_demo_webapp.sh                                  # Web demo UI from repo root
./scripts/run_demo_webapp.sh --with-spicedb                   # Web demo UI + SpiceDB for PDE scenarios
uv run python -m examples.run --list                         # Example scenarios (requires LangChain extra)
uv run python -m examples.run coding_agent_poisoned_readme --protected
uv run pytest                                                # Run the regression suite in tests/
JANUS_LIVE_SMOKE=1 uv run pytest tests/smoke/ -v -s          # Live SDK smoke suite (real CLI + tokens; add JANUS_SMOKE_SLOW=1 for the hook-timeout experiment)

# Lint & Format
uv run ruff check .              # Lint
uv run ruff check --fix .        # Auto-fix
uv run ruff format .             # Format
uv run mypy janus                # Type check

# Docs
uv run mkdocs serve              # Local docs server

# Demos
uv run python -m examples.run coding_agent_poisoned_readme --protected    # CLI demo
uv run uvicorn examples.app:app --reload                           # Web demo UI
```

The `tests/` tree covers the standalone-enforcer packaging contract, enforcer rule-evaluation semantics, the Claude Agent SDK adapter seams, and replay-style indirect-prompt-injection scenarios (`test_ipi_scenarios.py`). All tests run offline — no LLM or SpiceDB needed — except `tests/smoke/`, the live SDK smoke suite (skipped unless `JANUS_LIVE_SMOKE=1`), which verifies the SDK/CLI-side semantics `janus_options()` depends on against pinned versions; verified runs are recorded in `plans/claude-agent-sdk-hardening.md`.

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
- **Default-deny when policy loaded**: Tools not listed in a loaded policy are blocked. Conditions fail closed on missing arguments (`strict_conditions=True` default): an allow rule conditioning an absent argument does not match, and a per-tool `required_args` option rejects absent/blank arguments outright.
- **No global state**: Every enforcer, registry, and runner is independent and safe for concurrent use.
- **Priority ordering**: Lower priority values evaluate first. Convention: manual rules 1–10, LLM-generated rules 100+.

### Adapters

`janus/adapters/langchain.py` and `janus/adapters/adk.py` wrap framework-native tool execution with Janus enforcement. The shared base (`janus/adapters/_base.py`) provides `resolve_enforcer()` and `make_guarded_handler()`.

`janus/adapters/claude_agent_sdk.py` integrates with the Claude Agent SDK (Claude Code), whose tool loop runs inside the `claude` CLI subprocess. It enforces at the SDK's pre-execution seams rather than in the call path: `janus_options()` (the recommended entry point — generates a locked-down `ClaudeAgentOptions`: `tools=[]`, `strict_mcp_config=True`, `allowed_tools` = policy ∩ mounted, `permission_mode="dontAsk"`, `Task` + built-ins in `disallowed_tools`, guarded overrides that raise unless `unsafe_overrides=True`), `janus_pretooluse_hook()`/`janus_hooks()` (the robust argument-level seam — fires for every call, even allow-listed ones; fails closed on its own exceptions), `make_can_use_tool()` (bypassable by `allowed_tools`/`bypassPermissions` shadowing — documented as such), and `guard_tool_body()` (belt-and-braces). The adapter strips the `mcp__<server>__` tool-name prefix before matching the policy, passes the SDK-internal `StructuredOutput` tool through, supports a per-call `required_args` presence check (delegating to the core `check_required_args`), and via `hook_approved_tools` keeps high-risk sinks off `allowed_tools` so the hook and permission layer must both agree. Verified against `claude-agent-sdk` 0.2.120. Behind the `claude` extra.

## Ruff Config

Line length 100, target Python 3.11, rules: E, F, I, UP. E501 ignored.

## Known Issues

Current high-signal issues:
- Broad exception catching in `runner.py` — all exceptions become error strings
- Hardcoded SpiceDB token defaults in `pde_enforcer.py` and `janus/policy/pde/`
- PDE taint updates are manual (`agent.update_taint()`) and session-scalar — automatic per-source taint derivation is planned (Phase 1 of the IPI roadmap)

Fixed (0.0.6): the missing-argument bypass in `enforcer.py` — conditions now fail closed on absent arguments (`strict_conditions=True` default) and core `required_args` rejects absent/blank arguments; the Claude Agent SDK adapter delegates to the core check.
