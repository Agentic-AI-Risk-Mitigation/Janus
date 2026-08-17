# Claude Code CLI hook payloads — pinned fixtures

Verbatim stdin payloads received by `command` hooks in a live `claude -p` session.
**Do not hand-edit these files** — their value is being the bytes the CLI actually
sent, not shapes derived from documentation. To refresh, re-run the capture (below)
against a newer CLI and update this provenance block.

## Provenance

- Captured: 2026-08-15
- CLI: **2.1.233** (Claude Code), Linux
- Method: `--settings` file wiring a dump script (`cat > <event>.<ns>.json`) as a
  `command` hook for every event; one `claude -p` run exercising built-in `Read` +
  `Bash` + a stdio MCP tool (`mcp__janusfix__echo`), one run spawning a
  general-purpose subagent via the `Agent` tool. Design context:
  `plans/claude-code-plugin-design.md` §10.

## Findings the fixtures pin (where they contradict the docs, the fixtures win)

- **`PostToolUse` carries `tool_response`, NOT `tool_output`, on CLI 2.1.233** —
  the hooks docs (as read 2026-08-15) say `tool_output`. The normalizer must read
  both keys and take whichever is present; a payload with neither is the
  regression signal.
- MCP tool `tool_response` is a **raw JSON string** (`"{\"result\":...}"`), not
  MCP content blocks; built-in tools return dicts (`Bash`: `stdout`/`stderr`/
  `interrupted`/…; `Read`: `type`/`file`). Unwrapping must handle both.
- `agent_id` / `agent_type` are present **only** on payloads from inside a
  subagent (absent, not null, at top level). The subagent spawn tool is named
  **`Agent`** (not `Task`) — matching the SDK-path smoke finding.
- `PostToolBatch` fires (payload has a `tool_calls` array, no `tool_name`), even
  for single-call "batches".
- Extra keys beyond the documented set: `effort`, `prompt_id` (most events),
  `duration_ms` (PostToolUse); `SubagentStop` is rich (`agent_transcript_path`,
  `last_assistant_message`, `stop_hook_active`, …). `SessionStart`/`SessionEnd`
  omit `permission_mode`.

## Decision vocabulary (probed 2026-08-15, CLI 2.1.233)

Separate experiment, same method: a `PreToolUse` hook emits a candidate
`permissionDecision` for `Bash`, and *whether `PostToolUse` fires* tells us
whether the tool ran. `pretooluse.bypass-permissions.json` is the payload
captured during the `--dangerously-skip-permissions` leg.

| emitted `permissionDecision` | `claude -p` | `--dangerously-skip-permissions` |
|---|---|---|
| `deny` | blocked | blocked |
| `ask` | blocked, reason reached the model | blocked |
| `escalate` | **ran** | — |
| `totally-bogus-value` | **ran** | — |

Two findings, both load-bearing:

- **`escalate` is not in the CLI's vocabulary** — it is indistinguishable from a
  misspelling, and an unrecognized decision does not error, it falls through and
  the tool runs. `ask` is the real value. A taint gate emitting `escalate` would
  have silently allowed every hit.
- **Hooks are honored under `bypassPermissions`** — both `deny` and `ask` still
  block there. Hook decisions win over the permission mode; what *doesn't* win
  is an abstention (`{}`), which is not a decision at all.

## Hook timeout (probed 2026-08-15, CLI 2.1.233)

A `PreToolUse` hook configured with `"timeout": 3` that slept 10s before emitting
a `deny`: the deny was **discarded and the tool ran** (`PostToolUse` fired). The
same hook denying immediately blocked. **The CLI's hook timeout fails open**, as
documented — so the shim must own a deadline well under it and deny while it
still can.

## PostToolUseFailure (captured 2026-08-15)

`posttooluse-failure.bash.json`. A failed call does **not** emit `PostToolUse` —
it emits `PostToolUseFailure` instead, with **no `tool_response`/`tool_output`**
at all. It carries `error` (a string: exit code plus stderr), `is_interrupt`, and
`duration_ms`. Consequences: taint derivation sees nothing for a failed call
(correct — an error message is not fetched content), and any PostToolUse-based
cross-check must subscribe to this event too or it will simply never see failed
calls.

## Not yet captured (known gaps)

- Plugin-MCP tool names (`mcp__plugin_<plugin>_<server>__<tool>`) — needs an
  installed plugin.
- `PermissionRequest`, `PermissionDenied`, `PreCompact`.
- Payloads in interactive (non `-p`) sessions. `bypassPermissions` is now
  captured; `plan` / `acceptEdits` are not.
- What `ask` does in an *interactive* session (it should prompt; only its
  headless behaviour is verified) and whether an approval there produces a
  `PermissionRequest` payload rich enough to correlate back to the escalated
  `tool_use_id`.
