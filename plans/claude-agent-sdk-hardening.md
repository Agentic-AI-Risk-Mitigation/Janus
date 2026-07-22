# Claude Agent SDK adapter hardening

Tracking doc for closing out the acceptance criteria in
[issue #1](https://github.com/Agentic-AI-Risk-Mitigation/Janus/issues/1)
("Track Claude Agent SDK support and upstream permission-hook blockers").

**State as of 2026-07-22:** `janus/adapters/claude_agent_sdk.py` enforces at the
`PreToolUse` hook seam (verified to fire for every call, unshadowable by
`allowed_tools`/`permission_mode`), with `make_can_use_tool()` documented as
bypassable and `guard_tool_body()` as belt-and-braces for in-process `@tool`
bodies. What it does **not** yet do is implement the lockdown design from the
issue #1 comment thread. That gap is the plan below; the remaining hardening
work is captured as follow-ups.

## Verified SDK facts this plan builds on

Checked against installed `claude-agent-sdk` **0.2.120** (source-inspected, not
assumed). The SDK has moved since issue #1 was filed — the `extra_args`
escape hatch from the issue thread is obsolete:

- `tools: list[str] | ToolsPreset | None` — **availability**. `[]` disables all
  built-in tools ("don't exist", not "need permission"). Distinct from
  `allowed_tools`, which is auto-approval.
- `strict_mcp_config: bool` — maps to `--strict-mcp-config`; only the
  `mcp_servers` passed in options exist, `~/.claude.json` / `.mcp.json` /
  project config are ignored.
- `disallowed_tools: list[str]` — tools are *removed from the model's context*
  and cannot be used; strictly stronger than not-allow-listed.
- `allowed_tools: list[str]` — auto-approval (no prompt). Under
  `permission_mode="dontAsk"`, anything not listed is denied rather than
  prompted.
- `setting_sources` default (`None`) loads **no** filesystem settings, so
  project/user `settings.json` cannot re-enable anything.

### The layered model

Each layer answers a different question and fails differently:

| Layer | Question | Failure mode |
|---|---|---|
| `tools=[]` + `strict_mcp_config=True` | does the tool *exist*? | CLI-enforced at session start — nothing for a hook to miss |
| `allowed_tools` ∩ policy + `dontAsk` | may it run *unprompted*? | whole-tool granularity only |
| PreToolUse hook (Janus) | may it run *with these arguments*? | depends on hook firing |
| `guard_tool_body` | runs even if all the above lied | in-process tools only |

Security consequence: hook-only, a skipped hook executes
`Bash(<exfil command>)`. Locked down, a skipped hook's worst case is a
*policy-listed Janus tool running with arguments the policy would have
refused* — bounded to the project's own tool surface. Argument-level
enforcement stays hook-dependent (follow-ups 2 and 5); tool-level reachability
stops depending on hooks entirely. That split is what lets us close issue #1
criterion 5 honestly.

---

## Plan: `janus_options()` hardened options builder

Closes issue #1 acceptance criteria 1, 2, and 4 (built-ins unavailable; only
Janus-owned tools reachable; no MCP config leakage) by generating the locked-down
`ClaudeAgentOptions` from the policy + servers, instead of leaving six fields of
boilerplate as tribal knowledge.

### API sketch

```python
def janus_options(
    policy: PolicySource,
    *,
    mcp_servers: dict[str, Any],
    required_args: RequiredArgs | None = None,
    taint: TaintTracker | None = None,
    resolve_name: NameResolver = default_resolve_name,
    passthrough_tools: frozenset[str] = DEFAULT_PASSTHROUGH_TOOLS,
    hook_approved_tools: set[str] | None = None,   # sinks kept OFF allowed_tools
    unsafe_overrides: bool = False,
    **overrides: Any,          # forwarded to ClaudeAgentOptions
) -> ClaudeAgentOptions:
```

### Generated configuration

1. **`tools=[]`** — no built-ins. `Bash`/`Write`/`Edit` do not exist in the
   session.
2. **`strict_mcp_config=True`** — only the `mcp_servers` passed here; no
   config leakage.
3. **`allowed_tools` = policy ∩ mounted tools** — enumerate tool names from
   `mcp_servers` (in-process `create_sdk_mcp_server` instances expose them),
   prefix as `mcp__<server>__<name>`, keep only those whose
   `resolve_name(...)`-bare name has a rule in the loaded policy. Tools the
   policy doesn't know are mounted-but-unreachable, not silently allowed.
4. **High-risk sinks stay off `allowed_tools`** — any tool named in
   `hook_approved_tools` (e.g. `send_email`) is excluded from auto-approval
   even though mounted and policy-listed. Under `dontAsk` this makes the
   permission layer and the Janus hook *both* required to agree before a sink
   runs.
5. **`permission_mode="dontAsk"`** — not-allow-listed ⇒ denied, not prompted.
   (Overridable, but see guardrails.)
6. **`disallowed_tools=[...built-ins..., "Task"]`** — defense in depth behind
   `tools=[]` if its semantics ever shift, plus **`Task` denied explicitly**:
   whether PreToolUse fires for tool calls inside subagents is unverified
   (follow-up 6), so subagents are off until the smoke suite proves the hook
   covers them.
7. **`setting_sources` untouched** (SDK default: none) — filesystem settings
   cannot re-enable anything.
8. **`hooks=janus_hooks(policy, required_args=…, taint=…, …)`** — the
   argument-level seam, wired with the same knobs.

### Guardrails

Weakening the lockdown must be loud. `**overrides` is forwarded to
`ClaudeAgentOptions`, but these raise `ValueError` unless
`unsafe_overrides=True`:

- `tools` (anything but omitted)
- `strict_mcp_config=False`
- `permission_mode="bypassPermissions"` (or `"acceptEdits"`)
- `can_use_tool` (documented-bypassable seam; don't let it masquerade as a layer)
- `setting_sources` (re-introduces filesystem config)
- `hooks` (would replace the Janus wiring; merging user hooks *alongside* ours
  is fine and supported — replacing PreToolUse is not)

`disallowed_tools` and `allowed_tools` overrides are merged, not replaced:
user additions can only shrink the reachable surface, never grow it.

### Steps

1. Implement `janus_options()` in `janus/adapters/claude_agent_sdk.py`. SDK
   import inside the function (module must keep importing without the SDK,
   matching `janus_hooks`).
2. Tool-name enumeration from `mcp_servers` values; if a server type can't be
   enumerated (external stdio/HTTP server config), fail with an instructive
   error asking for an explicit `allowed_tools` merge rather than silently
   allowing `mcp__<server>__*`.
3. Fold in **follow-up 3** (fail-closed on hook exception) — same file, small
   diff, and the layered story is only honest if Janus's own bugs deny rather
   than pass.
4. Tests (offline, alongside existing adapter seam tests in `tests/`):
   - generated options: `tools == []`, `strict_mcp_config is True`,
     `permission_mode == "dontAsk"`, hooks wired (Pre + Post when `taint=`),
     `"Task"` in `disallowed_tools`;
   - `allowed_tools` is exactly policy ∩ mounted, minus `hook_approved_tools`;
   - unknown-to-policy mounted tool ⇒ not in `allowed_tools`;
   - each guarded override raises without `unsafe_overrides=True`; merges
     (`disallowed_tools`) shrink-only;
   - non-enumerable server ⇒ instructive error;
   - hook body raising an unexpected exception ⇒ `permissionDecision: "deny"`;
   - module imports without the SDK installed.
5. Docs: new `docs/adapters.md` section presenting `janus_options()` as the
   recommended entry point, with the layer table above; hook-only path kept as
   the documented integration for sessions that must retain built-ins.
6. Update the adapter paragraph in `CLAUDE.md` and the
   `claude-agent-sdk-integration` memory note (0.2.120 facts).
7. Comment on issue #1 mapping acceptance criteria → implementation:
   1/2/4 closed by lockdown; 3 already held (hook denies pre-execution);
   5 closed at reachability level, argument level tracked by follow-ups 2/5.

---

## Follow-ups

### 2. PostToolUse cross-check — detect a skipped PreToolUse hook

**Problem:** upstream history (claude-code #6305, #10814) shows PreToolUse
hooks silently not executing on some versions. Argument-level enforcement then
fails open with no signal.

**Proposal:** extend `janus_posttooluse_hook` (or a sibling sharing session
state) to assert every `tool_use_id` seen at PostToolUse was decided at
PreToolUse. On a miss: error-level log, optionally deny-all for the rest of the
session (configurable). This is m13v's post-execution verification pattern from
the issue thread — it can't prevent the first bypassed call, but converts
silent fail-open into a detected incident. With the lockdown in place, the
blast radius it guards is "own tool, bad arguments", not "arbitrary Bash".

### 3. Fail closed on hook exception — **folded into the plan above (step 3)**

`janus_pretooluse_hook` currently catches only `PolicyViolation`; an unexpected
exception in the hook body (malformed `tool_input`, enforcer bug, taint bug)
errors the hook and the CLI proceeds — fail-open on Janus's own defects. Wrap
the body, deny on unexpected error; same for `make_can_use_tool`.

### 4. Server-aware name resolution (prefix-collision)

**Problem:** `default_resolve_name` maps `mcp__<any-server>__fetch_page` →
`fetch_page`, so a tool on an unexpected server inherits a same-named trusted
tool's rules. `strict_mcp_config=True` closes the leak at the source; this is
defense in depth behind it, and matters most for hook-only (non-lockdown)
integrations.

**Proposal:** `make_resolve_name(servers=…)` denying any `mcp__` name whose
server segment isn't in the known set (resolves to a never-allowed sentinel, or
raises → deny via the fail-closed wrapper). Default it inside
`janus_options()`, where the server set is known.

### 5. Pinned-version live smoke test

**Problem:** offline tests verify Janus's side of the contract, not the SDK's.
Hook behavior regressed upstream before (Claude Code 2.0.30/2.0.31); the
0.2.120 field semantics above could also shift.

**Proposal:** a small live suite (manual / nightly, excluded from the offline
default run) against pinned SDK + CLI versions asserting:

- PreToolUse fires for every call (in-process MCP, allow-listed, `dontAsk`);
- `tools=[]` really removes built-ins (ask the agent to run `Bash` — it must
  not exist);
- `strict_mcp_config=True` really ignores a planted `.mcp.json`;
- deny path holds in multi-turn / continued sessions;
- `StructuredOutput` passthrough still works;
- the follow-up 6 experiments below.

Record the verified SDK+CLI version pair here and in the integration memory
note on every run.

### 6. Unverified SDK semantics (experiments for the smoke suite)

- **Subagent tool calls:** does PreToolUse fire inside a `Task`/subagent spawn?
  Until proven, `janus_options()` keeps `Task` in `disallowed_tools`; if the
  answer is yes on the pinned version, make subagents opt-in.
- **Hook timeout:** if the Janus hook is slow (future SpiceDB-backed check),
  does the CLI time out and proceed (fail open)? Related to
  claude-agent-sdk-python #304. If it fails open, document a hard latency
  budget for hook-side checks.
- **PostToolUse reliability:** taint derivation and the follow-up 2 cross-check
  inherit hook-firing risk — if PostToolUse skips, the session under-taints and
  the Rule-of-Two gate silently weakens.
