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
   docker compose -f janus/pde/docker-compose.yml up -d
   ```

2. Bootstrap the schema and relationships (run once per SpiceDB instance):

   ```bash
   uv run python -m janus.pde.main
   ```

3. Use `policy_engine="pde"` in `JanusAgent`. Call `agent.update_taint(risk)` when the agent reads from risky sources (e.g. after `read_file` on an untrusted path, or `fetch_url`).

4. E2E test:

   ```bash
   uv run pytest test_e2e_pde.py -v -s
   ```

## Post-Merge Layout

The SpiceDB engine code lives under `janus/pde/` (or `janus/policy/spicedb/`). It remains optional; the default engine is the JSON Schema enforcer. The config key `policy_engine="pde"` may be renamed to `policy_engine="spicedb"` in a future release.
