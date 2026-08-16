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
| Claude Code CLI (interactive) | `janus.adapters.claude_code` | — (core) | CLI `PreToolUse` / `PostToolUse` hooks |

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

### Recommended entry point — `janus_options()`

The hook seam alone leaves tool-level reachability hostage to the hook firing: upstream
Claude Code releases have shipped regressions where `PreToolUse` hooks silently did not run,
and a skipped hook then executes `Bash(<anything>)`. `janus_options()` generates a locked-down
`ClaudeAgentOptions` in which each layer answers a different question and fails differently:

| Layer | Question | Failure mode |
|---|---|---|
| `tools=[]` + `strict_mcp_config=True` | does the tool *exist*? | CLI-enforced at session start — nothing for a hook to miss |
| `allowed_tools` ∩ policy + `dontAsk` | may it run *unprompted*? | whole-tool granularity only |
| PreToolUse hook (Janus) | may it run *with these arguments*? | depends on hook firing |
| `guard_tool_body` | runs even if all the above lied | in-process tools only |

Locked down, a skipped hook's worst case shrinks from arbitrary `Bash` to *a policy-listed
Janus tool running with arguments the policy would have refused* — bounded to your own tool
surface.

```python
from claude_agent_sdk import create_sdk_mcp_server
from janus.adapters.claude_agent_sdk import janus_options

options = janus_options(
    TOOL_POLICY,
    mcp_servers={"research": create_sdk_mcp_server(name="research", tools=[...])},
    required_args={"fetch_page": ["url"]},
    output_format={"type": "json_schema", "schema": SCHEMA},   # extra kwargs forwarded
)
```

What it generates:

- **`tools=[]`** — built-ins (`Bash`/`Write`/`Edit`/…) don't exist in the session, and
  **`disallowed_tools`** re-denies them (plus `Task`: subagents stay off until `PreToolUse`
  coverage inside them is verified) as defense in depth.
- **`strict_mcp_config=True`** — only the `mcp_servers` you pass exist; `~/.claude.json`,
  `.mcp.json` and project config are ignored. `setting_sources` stays at the SDK default
  (none), so filesystem settings cannot re-enable anything.
- **`allowed_tools` = policy ∩ mounted** — tool names are enumerated from the in-process
  servers and kept only if the policy has a rule for them; mounted tools the policy doesn't
  know are unreachable, not silently allowed. External (stdio/SSE/HTTP) servers can't be
  enumerated and require an explicit `allowed_tools=` merge (still policy-filtered).
- **`permission_mode="dontAsk"`** — anything not allow-listed is denied, not prompted.
- **`hooks=janus_hooks(...)`** — the argument-level seam, wired with the same knobs
  (`required_args`, `taint`, `resolve_name`, `passthrough_tools`).

Three extra knobs:

- **`taint=TaintTracker(...)`** — wires *both* hook seams so session taint is derived
  automatically: a `PostToolUse` hook records untrusted reads, and the `PreToolUse` hook gates
  sinks on them before the static policy runs. See [Taint Tracking](taint.md).

- **`hook_approved_tools={"send_email"}`** — high-risk sinks kept *off* `allowed_tools` even
  though mounted and policy-listed. The Janus hook approves them explicitly on allow, so under
  `dontAsk` the permission layer and the hook must **both** agree before a sink runs — if the
  hook is skipped, the sink is denied rather than silently allowed.
- **`extra_hooks=`** — merge your own hook matchers alongside the Janus wiring (appended after
  ours per event).

Weakening the lockdown is loud: forwarding `tools`, `strict_mcp_config=False`,
`permission_mode="bypassPermissions"`/`"acceptEdits"`, `can_use_tool`, `setting_sources`, or
`hooks` raises `ValueError` unless you pass `unsafe_overrides=True`. `allowed_tools` /
`disallowed_tools` overrides are merged, not replaced — additions can only shrink the
reachable surface, never grow it. Unexpected exceptions inside the Janus hook itself
(enforcer bug, malformed input) return a **deny**, so Janus's own defects fail closed.

The hook-only path below remains the documented integration for sessions that must retain
built-in tools.

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

### Automatic taint — the `PostToolUse` seam

A static policy judges one call at a time, so it cannot express "don't send email *after* reading
an untrusted web page." Pass a `TaintTracker` and both seams get wired: `PostToolUse` derives
taint from tool outputs, `PreToolUse` gates sinks on it *before* the policy runs.

```python
from janus.policy import TaintTracker
from janus.adapters.claude_agent_sdk import janus_hooks

tracker = TaintTracker(
    sources={"fetch_page": "web", "read_email": "email"},
    gates={"send_email": "*"},      # Rule of Two: no outbound send after any untrusted read
)
options = ClaudeAgentOptions(..., hooks=janus_hooks(TOOL_POLICY, taint=tracker))
```

Use one tracker per session and call `tracker.reset()` only at session boundaries. Blocked calls
don't taint the session — only calls that actually returned a response are recorded.
`janus_posttooluse_hook()` returns the raw callback if you are assembling matchers by hand.
Full reference: [Taint Tracking](taint.md).

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

## Claude Code CLI (interactive `claude`)

`janus.adapters.claude_code` targets the **interactive CLI**, not the SDK. It needs no extra —
it is core-install only, because a hook has to run wherever `claude` runs.

### What you get, and what you don't

Read this before deploying it, because the security model is genuinely weaker than the SDK
path's and pretending otherwise is worse than not shipping it:

> **On the CLI, Janus is a policy monitor over a session it does not own, backstopped by
> `permissions.deny`. It is not a reachability lockdown.**

`janus_options()` works because Janus *constructs* the SDK session — no built-in tools, no MCP
leakage, `allowed_tools` = policy ∩ mounted. On the interactive CLI the human constructs the
session, so that layer is simply gone:

| Layer | SDK path | CLI path |
|---|---|---|
| does the tool exist? | `tools=[]` + `strict_mcp_config` | **gone** — the session is the user's |
| may it run unprompted? | `allowed_tools` ∩ policy + `dontAsk` | `permissions.deny` (+ managed settings) |
| may it run with these args? | Janus `PreToolUse` hook (fails closed on timeout) | Janus `PreToolUse` hook (CLI dispatch fails **open** on timeout) |
| runs even if all above lied | `guard_tool_body()` | **gone** — tool bodies are the CLI's |

Two consequences worth internalizing. First, the `permissions.deny` backstop is not optional
decoration: it is the only layer that holds with zero hooks running, which is exactly the
failure mode upstream hook-dispatch regressions produce. Print it with `janus-hook backstop`
and paste it into your settings. Second, settings-file hook delivery is **not a security
boundary against the agent it guards** — settings are re-read from disk, so one `Edit` of
`~/.claude/settings.json` disarms the guard mid-session. That is fine for evaluation and for
catching accidents; a plugin (hooks are snapshotted per session) is the minimum for "the
session I started stays guarded", and managed settings are the minimum for "the machine stays
guarded". Those ship in later phases.

### Gate mode vs. policy mode

`mode="gate"` (default) has Janus enforce the tools it has an opinion about — policy rules,
taint gates, required-args entries — and return `{}` for everything else. `{}` is *not* a
silent allow: it means "no opinion, fall through to the CLI permission flow and the human".
That downstream authority is what makes abstention defensible here and not on the SDK seam,
where default-deny remains correct.

`mode="policy"` is strict default-deny, identical to the library and SDK paths. Use it for
headless and managed deployments, where the tool surface is known and no human is watching.

**Abstention is only as good as the authority it defers to.** Under
`permission_mode="bypassPermissions"` nothing will ever ask a human, so abstention degrades
to a real silent allow. Gate mode therefore auto-promotes to policy mode under those modes.
Note this is specifically about *abstention*, not about hooks losing: a hook `deny` and a
hook `ask` were both verified to still block under `--dangerously-skip-permissions`. An
abstention just isn't a decision, so there is nothing for the CLI to honor.

Note also that the payload cannot tell you a session is headless — a `claude -p` run reports
`permission_mode: "default"` exactly like an interactive one — so pass `--headless` when wiring
hooks into a non-interactive deployment.

### Escalation uses `ask`, and the spelling matters

A taint-gate hit resolves to the CLI's `ask` decision: the call is blocked and the Janus
reason (including its `(audit id …)` suffix) is surfaced, so a human's approval is informed.
Static policy denies stay `deny` — the operator already decided those calls are wrong, and
prompting would only train click-through.

`ask` is not an arbitrary choice of word. Probed against CLI 2.1.233, an **unrecognized**
`permissionDecision` does not raise an error — the hook output is ignored and the tool
runs. `escalate`, which reads like the natural name, behaves exactly like a misspelling:

| emitted decision | `claude -p` | `--dangerously-skip-permissions` |
|---|---|---|
| `deny` | blocked | blocked |
| `ask` | blocked, reason reached the model | blocked |
| `escalate` | **ran** | — |
| `totally-bogus-value` | **ran** | — |

A gate emitting `escalate` would have silently allowed every hit — the worst available
failure for the mechanism whose whole job is stopping consequential actions after untrusted
input. If you extend the decision vocabulary, re-run that experiment rather than trusting a
doc.

### Wiring it (phase 1: settings file, stateless)

```bash
janus-hook backstop > /tmp/backstop.json   # the permissions.deny block; merge into settings
```

```json
{
  "hooks": {
    "PreToolUse": [
      { "hooks": [{ "type": "command",
                    "command": "janus-hook pre --policy /etc/janus/policy.json --mode gate" }] }
    ]
  }
}
```

The shim reads the payload on stdin and prints the CLI's hook JSON. It fails **closed**: an
unreachable policy file, an unparseable payload, or a bug inside Janus all produce a deny,
because the CLI's own dispatch failure mode is to proceed. `janus-hook doctor` self-tests the
install.

It also owns a `--deadline` (default 5s), and that is not belt-and-braces. The CLI's hook
timeout was verified to fail open on 2.1.233 — a hook configured with `"timeout": 3` that
slept 10s before denying had its deny **discarded and the tool ran**. Whatever stalls (a
policy file on a hung mount, a pathological condition regex, the janus import itself), the
shim has to hit its own limit first and deny while it still can. Keep `--deadline` well
under whatever `timeout` you set on the hook entry.

Phase 1 is deliberately the **degraded mode**: there is no daemon, so the shim holds no
cross-call state — static policy evaluation only, with no taint, no provenance, and no
`PreToolUse`/`PostToolUse` cross-check. It is genuinely useful (argument-level enforcement of a
static policy, today) and it is not the recommended deployment. Configuration arrives as
explicit argv flags rather than env vars so that a plugin's `userConfig` can slot into
exec-form `args` unchanged later.

### Using the adapter directly

```python
from janus.adapters.claude_code import handle_cli_payload, normalize_cli_event, decide_cli_event

output = handle_cli_payload(payload, "policy.json", mode="gate", session=session)
```

`normalize_cli_event` is the load-bearing piece: it reads `tool_response` **or** `tool_output`
(CLI 2.1.233 sends the former, the docs say the latter), never raises on shape drift, and keeps
the untouched payload in `.raw` for audit. `normalize_cli_events` additionally fans out a
`PostToolBatch` envelope, which carries a `tool_calls` array and no `tool_name` at all. Every
one of these behaviours is pinned by verbatim payload captures in
`tests/fixtures/claude_code_payloads/` — where the fixtures and the docs disagree, the fixtures
win.

`claude_code_resolve_name(name, known_servers=...)` maps `mcp__<server>__<tool>` (and the
plugin form `mcp__plugin_<plugin>_<server>__<tool>`) to the bare policy key. Supply
`known_servers`: the CLI has no `strict_mcp_config`, so an unsanctioned server would otherwise
inherit an allow rule written for a same-named tool elsewhere. Unknown servers resolve to a
reserved sentinel that no policy key can match.
