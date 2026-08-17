# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It is the single source of guidance for agents; `AGENTS.md` points here.

## What is Janus

Janus (`janus-guard` on PyPI) is a system-level security layer for LLM agents that enforces fine-grained policies on tool calls before execution. It intercepts at the tool-call boundary — not at prompt or output level — to defend against indirect prompt injection, credential exfiltration, and overprivileged agent actions.

## Commands

```bash
# Setup
uv sync --extra dev              # Core library + lint/type/test tooling
uv sync --extra all --extra dev  # Full demo/provider stack + dev (needed for examples/)

# Test, lint, type-check
uv run pytest                    # Regression suite in tests/ (offline)
uv run ruff check .              # Lint (--fix to auto-fix)
uv run ruff format .             # Format
uv run mypy janus                # Type check

# Live SDK smoke suite — real `claude` CLI + API tokens
JANUS_LIVE_SMOKE=1 uv run pytest tests/smoke/ -v -s   # add JANUS_SMOKE_SLOW=1 for the hook-timeout experiment

# Demos
uv run python -m examples.run --list                                      # list scenarios
uv run python -m examples.run coding_agent_poisoned_readme --protected    # CLI
./scripts/run_demo_webapp.sh [--with-spicedb]                             # web UI (SpiceDB for PDE scenarios)

# Docs
uv run mkdocs serve
```

The `tests/` tree covers the standalone-enforcer packaging contract, enforcer rule-evaluation semantics, the Claude Agent SDK adapter seams, and replay-style indirect-prompt-injection scenarios (`test_ipi_scenarios.py`). All tests run offline — no LLM or SpiceDB needed — except `tests/smoke/`, which verifies the SDK/CLI-side semantics `janus_options()` depends on against pinned versions; verified runs are recorded in `plans/claude-agent-sdk-hardening.md`.

## Architecture

The core flow is: **JanusAgent → LLMRunner → ToolRegistry → PolicyEnforcer**

1. `JanusAgent` (`janus/agent.py`) wires together the enforcer, tool registry, LLM runner, and provider
2. `LLMRunner` (`janus/llm/runner.py`) drives the conversation loop: messages → LLM → tool calls → results → repeat
3. `ToolRegistry` (`janus/tools/registry.py`) dispatches tool calls but **always calls `enforcer.enforce()` first**
4. `PolicyEnforcer` (`janus/policy/enforcer.py`) evaluates rules in priority order against JSON Schema conditions

When a tool is blocked, the `PolicyViolation` is caught and returned as a string to the LLM, allowing it to adjust rather than crash.

### Two Enforcement Engines

- **Janus engine** (default, `janus/policy/enforcer.py`): Stateless JSON Schema rule evaluation. Rules have priority, effect (allow/deny), conditions (JSON Schema per argument), and fallback action.
- **PDE engine** (`janus/policy/pde_enforcer.py` + `janus/policy/pde/`): SpiceDB-backed ReBAC with runtime taint tracking. Config/constants in `pde/config.py`, interceptor in `pde/interceptor.py`, discovery engine in `pde/discovery.py`, bootstrap utilities in `pde/bootstrap.py`.

### Two Taint Mechanisms

Don't conflate them — they are independent:

- **PDE taint** (`pde/interceptor.py`): a monotonic session-wide *scalar*, raised manually via `agent.update_taint(risk)`. Higher taint disables tools whose `TOOL_TAINT_LIMIT` it exceeds. Requires SpiceDB.
- **`TaintTracker`** (`janus/policy/taint.py`): framework-agnostic *per-source labels*, derived automatically at a post-execution seam (`record_output`) and gating sinks at the pre-execution seam (`check`). No SpiceDB, no manual calls. Wired into the Claude Agent SDK adapter via `janus_hooks(taint=...)`. This is the path forward for IPI defense (see `docs/taint.md`).

### Key Design Decisions

- **Single canonical tool representation**: Tools are defined once as `ToolDef`/`ToolParam` and converted to provider-specific schemas via `.to_openai_schema()`, `.to_pydantic_model()`, etc.
- **Default-deny when policy loaded**: Tools not listed in a loaded policy are blocked. Conditions fail closed on missing arguments (`strict_conditions=True` default): an allow rule conditioning an absent argument does not match, and a per-tool `required_args` option rejects absent/blank arguments outright.
- **No global state**: Every enforcer, registry, tracker, and runner is independent and safe for concurrent use.
- **Priority ordering**: Lower priority values evaluate first. Convention: manual rules 1–10, LLM-generated rules 100+.

### Adapters

`janus/adapters/langchain.py` and `janus/adapters/adk.py` wrap framework-native tool execution with Janus enforcement. The shared base (`janus/adapters/_base.py`) provides `resolve_enforcer()` and `make_guarded_handler()`.

`janus/adapters/claude_agent_sdk.py` is different in kind: the Claude Agent SDK's tool loop runs inside the `claude` CLI subprocess, so Janus never sees the call in-process and must enforce at the SDK's pre-execution seams. Use `janus_options()` — it builds a locked-down `ClaudeAgentOptions` so that a silently skipped `PreToolUse` hook (which has regressed upstream before) can't escalate to arbitrary `Bash`. Full seam-by-seam reference, including the layering rationale and every knob, is in **`docs/adapters.md`**; verified SDK behaviour is in `plans/claude-agent-sdk-hardening.md`. Behind the `claude` extra.

`janus/adapters/claude_code.py` + `janus/cli/hook.py` (the `janus-hook` console script, core install) target the *interactive* CLI via its `PreToolUse`/`PostToolUse` hooks. Weaker model than the SDK path — a policy monitor backstopped by `permissions.deny`, not a reachability lockdown — and phase 1 is deliberately stateless (static policy per call; no taint or cross-call state until the phase-2 daemon). Gate mode abstains on unlisted tools but auto-promotes to strict default-deny under `bypassPermissions`. The shim fails closed even though CLI hook dispatch fails open on timeout. Reference: the CLI section of `docs/adapters.md`; design and verified CLI probe results: `plans/claude-code-plugin-design.md`.

## Conventions

- **Style**: 4-space indent, type hints on public interfaces, concise docstrings where behavior is non-obvious. `snake_case` modules/functions, `PascalCase` classes, `UPPER_CASE` constants. Ruff config (line length, target version, rule set) lives in `pyproject.toml` — read it there rather than assuming.
- **Tests**: under `tests/`, named `test_*.py`, fixtures close to the tests that use them. Keep them offline; anything needing a live CLI or API key belongs in `tests/smoke/` behind an env guard.
- **Scenarios**: `examples/scenarios/<scenario_name>/scenario.py`.
- **Commits**: short, imperative subjects; add a scope prefix (`docs:`, `ci:`) when it clarifies intent.
- **PRs**: explain the behavioral change, list validation performed (`pytest`, `ruff`, `mypy`, relevant demo flow), link issues, screenshots only for UI/demo changes.
- **Secrets**: start from `.env.example`; never commit real keys. Review `SECURITY.md` before touching provider auth or enforcement flows. Changes to built-in tools or `policy/` must preserve default-deny.

## Code Review

Adopt the role of a seasoned senior programmer with deep experience in Python, security, and AI agent systems. Prioritize, in order:

1. **Code quality** — no unnecessary complexity, clear and readable code, reusable functions, extensible classes, good error handling.
2. **Security**.
3. **Adherence to best practices**.

Always question and analyze the intent and purpose of the code against its functionality, and from a design perspective as an architect.

## Known Issues

- Broad exception catching in `runner.py` — all exceptions become error strings
- Hardcoded SpiceDB token defaults in `pde_enforcer.py` and `janus/policy/pde/`
- PDE taint remains manual and session-scalar; `TaintTracker` supersedes it for new work but the two are not yet unified
- `TaintTracker` is only wired into the Claude Agent SDK adapter — LangChain/ADK adapters have no post-execution seam yet
- The Claude Code CLI adapter is phase-1 stateless: `janus-hook` imports Janus per call and holds no cross-call state, so no taint/provenance on that path until the phase-2 daemon
