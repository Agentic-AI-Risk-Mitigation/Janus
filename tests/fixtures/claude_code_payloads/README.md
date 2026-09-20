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

Three files are a **third capture, on a newer CLI** —
`pretooluse.subagent-handback.json`, `posttooluse.subagent-handback.json` and
`posttooluse.agent-result.handback.json`:

- Captured: 2026-09-20
- CLI: **2.1.278** (Claude Code), Linux
- Method: same — a `command` hook appending stdin to a file, one `claude -p` run
  spawning a general-purpose subagent via the `Agent` tool.

They are *additions*, not replacements: every other fixture here is still the
2.1.233 capture. Where the two versions disagree, both shapes are live in the
field and the code handles each — see "A taint source moved" below.

`pretooluse.windows-read.json` is a **second capture, on a different platform**:

- Captured: 2026-08-31
- CLI: **2.1.246** (Claude Code), Windows 11
- Method: same — a `command` hook appending stdin to a file, one `claude -p` run
  reading a file in the project directory.

## Paths use the host's native separator

The Windows capture exists because every fixture above it is from Linux, and
that gap hid a live bug: `tool_input.file_path` arrives as
`C:\Users\...\README.md` on Windows, so a policy pattern anchored on `/` alone
matches **nothing** there. Against the pre-fix starter policy, reads of `.env`,
`~/.ssh/id_rsa`, `~/.aws/credentials` and `~/.claude/.credentials.json` were all
*allowed* on Windows, as was writing `.claude/settings.json` — the rule meant to
stop an agent disarming the guard. Only `\.pem$` held, because it is the one
pattern needing no separator.

Path patterns must therefore use a separator **class** (`[/\\]`), not a slash;
`janus.cli.starter_policy.SEP` exists for this. The same applies to anything
that constructs a probe or test path: build it with the native separator
(`str(Path)`), never `as_posix()`, or the test will pass against a string the
deployment never sees.

The environment a `command` hook runs with also carries `CLAUDE_PROJECT_DIR`
(verified on 2.1.246; on Windows its value uses *forward* slashes), which is
what makes the `$CLAUDE_PROJECT_DIR/...` form in a POSIX project-scoped hook
command resolve.

## A taint source moved (2.1.233 → 2.1.278)

The single most consequential difference between the two captures, and the
reason the three handback fixtures exist.

On **2.1.233**, a subagent's report arrived in the *parent's* result:
`posttooluse.agent-result.json` has `tool_response.content[0].text` holding the
subagent's actual words.

On **2.1.278**, that field is a placeholder
(`posttooluse.agent-result.handback.json`):

> "This agent's report was delivered to you as a message from `<agent_id>`
> (its SubagentHandback call). Read it there; it is not repeated here."

alongside a new `handback: "send"` key. The content now travels in
`PostToolUse[SubagentHandback].tool_input.message` — note **`tool_input`**, not
`tool_response`; the handback's response is a delivery receipt
(`{"success": true, "message": "Report delivered to your caller."}`).

Why this is the dangerous kind of drift: a tracker that lists `Agent` as a taint
source keeps working, keeps recording, and keeps reporting success — while
recording a fixed placeholder sentence instead of subagent-derived content. No
exception, no missing key, no failing test. Every downstream sink stays open.
It is the same failure shape as the Windows `/`-anchored path patterns and the
`as_posix()` probes: **a mechanism that reports green while measuring nothing.**

`SubagentHandback` is consequently both a decision-seam passthrough (denying it
only strands the subagent's work) and a recording-seam source. See
`DEFAULT_CLI_PASSTHROUGH_TOOLS` and `CLI_INPUT_SOURCE_TOOLS` in
`janus/adapters/claude_code.py`.

Still unknown: whether `handback` takes values other than `"send"`, i.e. whether
some sessions still inline the report in the `Agent` result. If so both paths are
live at once and the recorder must handle each without double-counting.

## Findings the fixtures pin (where they contradict the docs, the fixtures win)

- **`PostToolUse` carries `tool_response`, NOT `tool_output`, on CLI 2.1.233** —
  the hooks docs (as read 2026-08-15) say `tool_output`. The normalizer must read
  both keys and take whichever is present; a payload with neither is the
  regression signal.
- MCP tool `tool_response` has carried **two different dialects across CLI
  versions**, so unwrapping must handle both — and must not assume either:
  - On **2.1.233** (the fixtures here): a **raw JSON string**
    (`"{\"result\":...}"`), *not* MCP content blocks.
  - On **2.1.278** (re-probed 2026-09-20, same `mcp__janusfix__echo` method;
    fixtures not regenerated): a **content-block list**,
    `[{"type": "text", "text": "{\"result\":...}"}]` — the shape the earlier
    capture explicitly ruled out. `unwrap_cli_response` already absorbs it
    (verified: yields `{"result": "echo: fixture"}`), because it falls through
    to the SDK's block unwrapper and then JSON-parses the inner text.

  Built-in tools return dicts in `PostToolUse` (`Bash`: `stdout`/`stderr`/
  `interrupted`/…; `Read`: `type`/`file`); that has not changed.
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
