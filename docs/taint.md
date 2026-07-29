# Taint Tracking

*Automatic per-source taint for indirect-prompt-injection defense*

An agent that reads a web page, an inbound email, or a log written by someone else has taken
untrusted instructions into its context. No policy on a single tool call can tell that the
`send_email` it is now attempting was the user's idea or the web page's. `TaintTracker`
(`janus.policy.taint`) closes that gap by remembering *which untrusted sources a session has
read* and gating outbound or state-changing tools on it — Meta's "Rule of Two", enforced
mechanically.

`TaintTracker` ships in the **core** install — no SpiceDB, no extra.

!!! note "Two taint mechanisms, deliberately separate"
    `TaintTracker` is not the [SpiceDB engine's](spicedb-enforcement.md) taint. That one is a
    monotonic session-wide **scalar** you raise by hand with `agent.update_taint(risk)`, and it
    needs SpiceDB. `TaintTracker` is framework-agnostic, uses **per-source labels**, and derives
    them **automatically** from tool outputs. New integrations should prefer `TaintTracker`.

## Model

- **Sources** — `{tool_name: label}`. When a listed tool returns, the session is tainted with
  that label. Unlisted tools are taint-neutral.
- **Gates** — `{tool_name: labels}`. The tool is denied once any listed label has tainted the
  session. `"*"` gates on *any* taint at all (strictest Rule of Two: after reading anything
  untrusted, this tool needs out-of-band approval).
- **Monotonic within a session** — labels are never removed by ordinary operation. Once
  untrusted content is in the context window it cannot be un-read. Only `reset()` clears them,
  at a session boundary.
- **Source-granular, not per-datum** — when the agent loop runs inside a vendor subprocess
  (Claude Code), Janus sees pre/post tool hooks only; it cannot prove which bytes of a tool
  output influenced which argument. Labeling at the source level and gating sinks
  conservatively is the sound retrofit.
- **Auditable** — every taint introduction and gate denial is appended to `events`, so any
  decision traces back to the tool call that caused it.

## Standalone use

Two seams: record after a tool runs, check before the next one does.

```python
from janus.policy import TaintTracker

tracker = TaintTracker(
    sources={"fetch_page": "web", "read_email": "email"},
    gates={"send_email": {"web", "email"}, "run_scan": "*"},
)

tracker.check("send_email")           # None — nothing untrusted read yet
tracker.record_output("fetch_page")   # -> ["web"]; session now tainted by "web"

reason = tracker.check("send_email")  # -> deny reason naming the causing tool call
if reason:
    raise PolicyViolation(reason)

tracker.check("git_diff")             # None — not a gated tool
```

`check()` returns a human-readable deny reason (suitable to hand back to the model) or `None`.
It does not raise, so you decide the failure mode.

### Content-aware labels

The static source map can be extended with a classifier run on every recorded output — for
labels that depend on what came back, not just where from:

```python
def classify(tool_name: str, output) -> str | None:
    if tool_name == "fetch_page" and "@" in str(output):
        return "contains_pii"
    return None

tracker = TaintTracker(
    sources={"fetch_page": "web"},
    gates={"send_email": {"contains_pii"}},
    classify=classify,
)
```

### Introspection

| Member | Meaning |
|---|---|
| `tainted_by` | `frozenset` of labels that have tainted this session |
| `is_tainted()` | whether any label is present |
| `events` | ordered audit trail of taint introductions and gate denials |
| `taint(label, reason=…)` | manually add a label (escape hatch; prefer `record_output`) |
| `reset()` | clear all taint and the audit trail — session boundaries only |

Instances are independent (one per agent session) and lock-protected, so concurrent hook
callbacks can share one safely.

## Automatic derivation with the Claude Agent SDK

Pass `taint=` to [`janus_hooks()`](adapters.md) or `janus_options()` and both seams are wired
for you — a `PostToolUse` hook calls `record_output()`, and the `PreToolUse` hook runs
`check()` **before** the static policy. No manual calls anywhere:

```python
from janus.policy import TaintTracker
from janus.adapters.claude_agent_sdk import janus_options

tracker = TaintTracker(
    sources={"fetch_page": "web"},
    gates={"send_email": "*"},        # no outbound send after any untrusted read
)

options = janus_options(
    TOOL_POLICY,
    mcp_servers={"research": server},
    taint=tracker,
    hook_approved_tools={"send_email"},   # sink must also clear the permission layer
)
```

Taint gating is whole-tool and argument-independent: once the session is tainted, the sink is
denied no matter how innocuous the arguments look. That is the point — the arguments are
exactly what an injected instruction controls.

Denied calls do not taint the session: the `PostToolUse` hook only records calls that actually
produced a response, so a blocked read is not mistaken for a completed one.

Use `hook_approved_tools` for gated sinks. It keeps them off `allowed_tools`, so under
`permission_mode="dontAsk"` the sink runs only if the Janus hook affirmatively approves it —
meaning a skipped hook denies the sink instead of letting it through ungated.

## Limitations

- **Whole-tool gating.** A gated sink is all-or-nothing once tainted; there is no notion of
  "this argument came from a trusted source." Provenance-typed conditions are future work.
- **Source classification is yours.** Janus does not guess which tools are untrusted — an
  unlisted source taints nothing.
- **Session-scoped.** A long-running service must construct one tracker per session (or
  `reset()` at boundaries), or taint from one user's request will gate the next one's.
- **Only wired into the Claude Agent SDK adapter.** The LangChain and ADK adapters have no
  post-execution seam yet; use `record_output()` manually there.
