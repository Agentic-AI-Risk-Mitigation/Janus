# SpiceDB-Backed Enforcement

*ReBAC + taint tracking for IPI defense*

The SpiceDB engine is an optional enforcement backend that combines relationship-based access control (ReBAC) via SpiceDB with runtime taint tracking. Use it when you need role-based tool access and automatic permission degradation as the agent reads from untrusted data.

## Overview

Enable with `policy_engine="pde"` when creating `JanusAgent`:

```python
agent = JanusAgent(
    model="openai/gpt-4o",
    policy_engine="pde",
    agent_role="coding_agent",
)
```

The engine does not load static JSON policies. Access is governed by the SpiceDB schema and bootstrap relationships, plus a per-session taint level.

## ReBAC Layer

SpiceDB stores a permission graph in the Zanzibar model. Each tool has its own object type (e.g. `tool_view_file`, `tool_run_tests`). ACL edges grant roles the `can_invoke` relation on tools. Agents are members of roles. At tool-call time, SpiceDB is queried: does this agent's role have `invoke` permission on this tool?

Bootstrap assigns the `coding_agent` to roles `readonly`, `developer`, and `executor`. Tools are mapped to roles by tier (see below).

## Taint Tracking

A session taint level starts at 0 and only increases. When the agent reads from a data source, call `agent.update_taint(risk)` with the source's risk level. Higher taint restricts which tools remain accessible.

Risk levels map to taint values:

| Risk | Taint |
|------|-------|
| low | 10 |
| medium | 40 |
| high | 70 |
| critical | 90 |

Example: after reading a web page (critical risk), call `agent.update_taint(90)`. Tools with `TOOL_TAINT_LIMIT` below 90 are then blocked.

## Two-Step Check

Before a tool executes:

1. **Taint gate** (Python): If `current_taint > TOOL_TAINT_LIMIT[tool_name]`, block.
2. **SpiceDB ACL gate**: Query `CheckPermission` for `invoke` on `tool_<name>` for the agent. If no ACL edge exists, block.

Both must pass for the tool to run.

## Tier Structure

Tools are grouped by risk. Tier 4 tools have no ACL edges and are always denied.

| Tier | Tools | Taint limit (approx) |
|------|-------|----------------------|
| 0 (read-only) | view_file, grep, code_search, git_diff, git_log | 90 |
| 1 (disk writes) | edit_file, create_file, delete_file | 50–70 |
| 2 (local exec) | run_tests, run_script, pip_install | 30–40 |
| 3 (remote git) | git_commit, git_push, git_clone | 20–60 |
| 4 (dangerous) | bash_terminal, http_request, read_secret, write_secret, deploy | 5–10, no ACL |

## How to Run

1. Start SpiceDB (Docker):

   ```bash
   cd demos && docker compose up -d && cd ..
   ```

2. Use the engine via the demo scenario or Janus: Demo 5 bootstraps the demo schema and relationships when you run it (web app or `uv run python -m examples.run demo5_taint_cascade --protected`). The PDE code lives in `janus/policy/pde/`; `janus.policy.pde_enforcer.PDEEnforcer` wraps `GraphInterceptor` from `janus.policy.pde.interceptor` and uses a sync gRPC client so it works inside an asyncio event loop (e.g. uvicorn).

3. Call `agent.update_taint(risk)` (or the scenario’s taint callback) when the agent reads from risky sources (e.g. after `fetch_url` or reading untrusted files).

4. Run the example tests: `uv run pytest tests/test_examples/ -v`.

## Layout

The SpiceDB engine lives under `janus/policy/pde/`: `config.py` (schema, taint limits), `interceptor.py` (GraphInterceptor, taint + ACL check), `bootstrap.py` (schema write, relationships), and `discovery.py` (GraphDiscoveryEngine). The Janus adapter is `janus/policy/pde_enforcer.py`, which imports `GraphInterceptor` from `janus.policy.pde.interceptor`. The default engine remains the JSON Schema enforcer; PDE is optional.
