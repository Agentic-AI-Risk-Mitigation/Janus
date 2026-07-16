# Framework Adapters

Janus enforcement plugs into third-party agent frameworks while the framework itself stays
responsible for the reasoning / execution loop. Each adapter lives in `janus/adapters/` and
reuses the shared helpers in `janus/adapters/_base.py` (`resolve_enforcer()`,
`make_guarded_handler()`), so every adapter accepts the same policy sources: a JSON file path,
a dict, a `PolicyEnforcer` instance, or `None`.

| Framework | Module | Extra | Enforcement point |
|---|---|---|---|
| LangChain | `janus.adapters.langchain` | `langchain` | Guarded `StructuredTool` handlers |
| Google ADK (Gemini) | `janus.adapters.adk` | `adk` | Guarded function-call handlers |
| Claude Agent SDK (Claude Code) | `janus.adapters.claude_agent_sdk` | `claude` | SDK `PreToolUse` hook / `can_use_tool` |

Full, copy-pasteable usage for LangChain and ADK is in the
[README](https://github.com/Agentic-AI-Risk-Mitigation/Janus#framework-adapters). This page
focuses on the Claude Agent SDK adapter, whose enforcement model is different.

## Claude Agent SDK (Claude Code)

Unlike the LangChain and ADK adapters — where Janus wraps the tool handler and the loop calls
your guarded handler directly — the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
runs the tool loop **inside the `claude` CLI subprocess**. Janus never sees the call in-process,
so the adapter enforces at the SDK's pre-execution seams instead.

```bash
uv add "janus-guard[claude]"   # also requires the `claude` CLI on PATH
```

### Primary seam — `PreToolUse` hook

A `PreToolUse` hook fires for **every** tool call, even those in `allowed_tools` and under
`permission_mode="dontAsk"`. This is the seam to use; it cannot be bypassed.

```python
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server
from janus.adapters.claude_agent_sdk import janus_hooks

# Policy keyed on bare tool names (the adapter strips the mcp__<server>__ prefix).
TOOL_POLICY = {
    "web_search": [(1, 0, {"query": {"type": "string", "maxLength": 400}}, 0)],
    "fetch_page": [(1, 0, {"url": my_ssrf_check}, 0)],   # callable condition
}

options = ClaudeAgentOptions(
    mcp_servers={"research": create_sdk_mcp_server(name="research", tools=[...])},
    allowed_tools=["mcp__research__web_search", "mcp__research__fetch_page"],
    permission_mode="dontAsk",
    output_format={"type": "json_schema", "schema": SCHEMA},
    hooks=janus_hooks(TOOL_POLICY, required_args={"fetch_page": ["url"]}),
)
```

On a policy block, the hook returns a `permissionDecision: "deny"` whose reason is fed back to
the model so it can adjust rather than crash — the same "return the violation to the LLM"
philosophy as the standalone enforcer.

`janus_pretooluse_hook()` returns the raw hook callback if you want to build the `HookMatcher`
yourself or set a matcher pattern; `janus_hooks()` is the convenience wrapper that returns a
ready `hooks=` dict.

### Two behaviours the adapter handles for you

- **`StructuredOutput` passthrough.** When you set `output_format`, the SDK delivers the final
  structured result via an internal `StructuredOutput` tool call that the hook also sees. A
  default-deny policy would block it and null out `ResultMessage.structured_output`. The adapter
  passes SDK-internal tools through (default `{"StructuredOutput"}`, configurable via
  `passthrough_tools`).
- **Tool-name mapping.** An in-process `@tool("fetch", …)` on a server named `research` is
  invoked as `mcp__research__fetch`. The adapter maps that back to the bare policy key (`fetch`)
  via `default_resolve_name`. Pass `resolve_name=lambda n: n` if your policy uses the full
  prefixed names.
- **Missing-argument backstop.** The core enforcer fails closed when a conditioned argument is
  omitted (strict conditions, default since 0.0.6). Pass `required_args={"tool": ["arg"]}` to
  additionally reject calls whose named argument is absent or blank — covering arguments no
  condition names, before enforcement runs.

### Alternative seam — `can_use_tool` callback

`make_can_use_tool()` builds a `can_use_tool` callback returning
`PermissionResultAllow` / `PermissionResultDeny`. **It is bypassable**: the SDK auto-approves any
whole-tool `allowed_tools` entry (and everything under `permission_mode="bypassPermissions"`)
*before* the callback runs, and emits a `CanUseToolShadowedWarning`. It also requires the
streaming prompt form (an `AsyncIterable`). Prefer the hook; use this only when nothing shadows
it.

### Belt-and-braces — `guard_tool_body()`

Wrap an in-process `@tool` body so enforcement also runs at execution time, independent of any
SDK permission seam. Even if a hook is misconfigured or shadowed, a blocked call never runs its
body:

```python
from janus.adapters.claude_agent_sdk import guard_tool_body

guarded = guard_tool_body("fetch_page", my_async_body, TOOL_POLICY,
                          required_args={"fetch_page": ["url"]})
```

A runnable end-to-end example is in `examples/claude_agent_sdk_demo.py`.
