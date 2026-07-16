# Changelog

All notable changes to Janus will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **BREAKING — strict condition semantics by default, closing the missing-argument bypass.**
  Previously the enforcer only checked a condition when its argument was *present*, so a rule
  gating `url` was silently skipped if the model omitted `url` entirely — an allow rule could
  vouch for a call it never inspected. Now (`strict_conditions=True`, the default) an allow rule
  whose condition names an absent argument does **not** match, and the call falls through to
  default-deny. Deny rules already failed closed (a condition on an absent argument matches
  vacuously) and are unchanged. Policies that relied on "constrain only if provided" can opt
  back into the legacy behavior with `PolicyEnforcer(strict_conditions=False)`.

### Added

- **Core `required_args`**: `PolicyEnforcer(required_args={"tool": ["arg", …]})` requires named
  arguments to be present and non-empty (rejecting `None` and blank strings) before rule
  evaluation, complementing strict conditions for arguments no condition covers. Promoted from
  the Claude Agent SDK adapter, which now delegates to the shared
  `janus.policy.enforcer.check_required_args` (its per-call `required_args` parameter is
  unchanged).
- **Enforcer semantics regression suite** (`tests/test_enforcer_semantics.py`): priority
  ordering, deny-before-allow tie-break at equal priority, default-deny, strict/legacy
  missing-argument behavior, `required_args`, policy management, and fallback actions.
- **Indirect-prompt-injection scenario suite** (`tests/test_ipi_scenarios.py`): replay-style
  attack/legitimate tool-call sequences under an outreach-pipeline-shaped policy — poisoned
  page → email exfiltration, malicious reply → missing-argument bypass, honeypot log → SSRF /
  attacker-directed scans — asserting attacks are blocked and legitimate twins still flow.
- **CI test workflow** (`.github/workflows/test.yml`): pytest + ruff + mypy on Python 3.10–3.13.

## [0.0.5] — 2026-07-16 (Alpha)

### Added

- **Claude Agent SDK adapter** (`janus.adapters.claude_agent_sdk`, behind the new `janus-guard[claude]` extra): enforce a Janus policy on a [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) / Claude Code tool loop, whose loop runs inside the `claude` CLI subprocess. `janus_pretooluse_hook()` / `janus_hooks()` gate every tool call via a `PreToolUse` hook (the robust seam); `make_can_use_tool()` is a `can_use_tool` alternative (bypassable by `allowed_tools`/`bypassPermissions` shadowing, documented as such); `guard_tool_body()` wraps an in-process `@tool` body as belt-and-braces. The adapter strips the `mcp__<server>__` tool-name prefix, passes the SDK-internal `StructuredOutput` tool through, and backstops the enforcer's absent-argument bypass via `required_args`.
- `tests/test_claude_agent_sdk_adapter.py` (20 offline tests) and `examples/claude_agent_sdk_demo.py` (live end-to-end demo). New `docs/adapters.md` reference page.
- Regression tests (`tests/test_standalone_enforcer.py`) covering standalone `PolicyEnforcer` import/enforcement (callable + JSON-Schema conditions, default-deny, all three fallbacks) with `authzed` unimportable, and the actionable PDE `ImportError`.

### Changed

- **Lightweight core install**: the core package no longer depends on `authzed`, `fastapi`, or `uvicorn`. `PolicyEnforcer` and the static-policy path now import only `jsonschema` + `pydantic` (+ `jinja2`/`python-dotenv`/`openai` for generation and the default provider), so `from janus.policy import PolicyEnforcer` works as a standalone tool-call gate without the SpiceDB/PDE stack.
- **New optional extras**: `janus-guard[pde]` (SpiceDB-backed ReBAC + taint tracking / `PDEEnforcer`) and `janus-guard[server]` (FastAPI/uvicorn demo webapp). `authzed`, `fastapi`, and `uvicorn[standard]` moved out of core into these extras; they are still bundled in `[all]` and `[dev]`.
- **`PDEEnforcer` is now imported lazily** (PEP 562 `__getattr__` in `janus.policy`). Using the PDE engine (`policy_engine="pde"` or `from janus.policy import PDEEnforcer`) without the `pde` extra installed raises a clear, actionable `ImportError` instead of a raw `ModuleNotFoundError` for `authzed`.
- **Python 3.10 supported**: lowered `requires-python` to `>=3.10` (core modules use no 3.11-only features; ruff/mypy targets updated accordingly).

## [0.0.4] — 2026-03-13 (Alpha)

### Changed

- **PDE integration**: Policy-Discovery-Engine has been merged into the main repo. SpiceDB-backed enforcement now lives under `janus/policy/pde/` (config, interceptor, discovery, bootstrap). `PDEEnforcer` imports from `janus.policy.pde.interceptor`; no separate `Policy-Discovery-Engine/` directory. Demos and docs updated accordingly.
- **Demo workflow**: added `scripts/run_demo_webapp.sh` as a repo-root entrypoint for the FastAPI demo UI, with optional local SpiceDB startup for Demo 5.
- **Project docs**: updated AGENTS/CLAUDE/README guidance to match the current `examples/`-based demo layout and the absence of a checked-in `tests/` tree on `main`.
- **Pytest config**: removed the stale `testpaths = ["tests"]` setting so local `pytest` no longer warns about a missing `tests/` directory.

## [0.0.3] — 2026-03-12 (Alpha)

### Added

- **Core engine**: `PolicyEnforcer` with JSON Schema–based argument validation and priority-ordered rule evaluation
- **JanusAgent**: single entry point wrapping LLM conversation loop with policy enforcement
- **10+ LLM providers**: OpenAI, Anthropic, Google Gemini, Azure OpenAI, AWS Bedrock, Ollama, vLLM, Together AI, OpenRouter
- **Framework adapters**: LangChain (3 integration depths) and Google ADK
- **LLM-generated policies**: auto-infer minimum-privilege policies from a user query via `policy="generate"`
- **Policy refinement**: incrementally tighten policies as an agent discovers new information
- **Built-in tools**: `read_file`, `write_file`, `edit_file`, `list_directory`, `run_command`, `fetch_url` — all workspace-scoped with path-traversal rejection
- **Custom tools**: `ToolDef` / `ToolParam` dataclasses for registering and guarding arbitrary tools
- **Three fallback actions**: raise `PolicyViolation`, call `sys.exit()`, or prompt user interactively
- **PDE (janus/policy/pde/)**: SpiceDB-backed graph enforcement with Zanzibar-style ReBAC and runtime taint tracking for IPI defence
- **Demo scenarios**: scripted demos for poisoned-README (IPI) and taint-cascade attacks
- **Web demo app**: FastAPI split-panel UI for live demo playback
- **Documentation site**: MkDocs Material with architecture, policy reference, getting-started, and demo guides
- **Test suite**: E2E PDE/SpiceDB integration tests and unit tests for enforcement, scripted LLM, mock tools, and scenario runner
